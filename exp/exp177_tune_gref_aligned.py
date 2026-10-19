#!/usr/bin/env python
"""EXP-177 — corrected held-out re-tune of the training-free gRefCOCO rule.

The frozen sxi parameters came from a study whose term lookup was keyed by
ref_id; with per-record sentence alignment the rule measures 49.63 val cIoU
(exp172).  This script re-tunes the sxi scalars (alpha, beta, topk, margin,
p_sel) on the held-out gRefCOCO TRAIN-TUNE split with each record's own
sentence, using the same cached GDINO/SAM parts and the same abstention
semantics, then (separately) applies the chosen configuration to val once.

    python exp/exp177_tune_gref_aligned.py --mode tune
"""
from __future__ import annotations
import argparse, json, sys, time, itertools
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import probe_no  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp108_gdino_study import box_feats  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
from exp138_learned_union_selector import (gdino_detect, llava_box_of,  # noqa: E402
                                           sam_paths, sam_load)
from exp173_gref_search_aligned import bxi_box  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTF = ROOT / "exp/out_final"
TUNE = ROOT / "exp/out_extreme/records_grefcoco_train_tune.jsonl"
CAP = 0.5


def build_cache():
    t0 = time.time()
    nt_map = {}
    for line in open(TUNE):
        rec = json.loads(line)
        rid = int(rec["ref_id"])
        n = int(rec.get("n_targets") or 0)
        nt_map[rid] = max(nt_map.get(rid, 0), n)
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(TUNE, "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km]); ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    print(f"fitted clfs ({time.time()-t0:.0f}s), rows {len(tr)}", flush=True)
    cache = []
    for i, r in enumerate(tr):
        rec = r.get("rec")
        if rec is None:
            continue
        fn = rec["file_name"]
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        if r["no_target"]:
            single = [kk for kk, c in enumerate(ids) if c[1] < 0]
            fb = max(single, key=lambda kk: pp[kk]) if single else int(np.argmax(pp))
        else:
            fb = int(np.argmax(pp))
        terms = []
        for term in terms_of(rec["parsed"]):
            boxes, scores = gdino_detect(fn, term)
            det0 = boxes[0] if boxes else None
            ll = llava_box_of(fn, term)
            bl = bxi_box(r["masks"], r["_bs"], ll) if ll is not None else None
            bd = bxi_box(r["masks"], r["_bs"], det0) if det0 is not None else None
            p0, p1 = sam_paths(fn, term, train_side=True)
            sams = [m for m in (sam_load(p0) + sam_load(p1))
                    if m.sum() / m.size <= CAP]
            if bl is None and bd is None and not sams:
                continue
            terms.append(dict(bl=bl, bd=bd, sams=sams))
        cache.append(dict(rid=int(r["rid"]), n_targets=nt_map.get(int(r["rid"]), 0),
                          masks=r["masks"], gt=r["gt"], gt_area=float(r["gt_area"]),
                          no_target=bool(r["no_target"]), ens=bool(any(r["ens"])),
                          bs=r["_bs"], pp_max=float(pp.max()),
                          fallback=candidate_mask(r["masks"], ids[fb]),
                          terms=terms))
        r["masks"] = None
        if (i + 1) % 200 == 0:
            print(f"  cache {i+1}/{len(tr)} ({time.time()-t0:.0f}s)", flush=True)
    import pickle
    path = ROOT / "exp/out_extreme/gref_tune_aligned_cache.pkl"
    path.write_bytes(pickle.dumps(cache, protocol=4))
    print(f"cache {len(cache)} rows -> {path} ({time.time()-t0:.0f}s)", flush=True)
    return cache


def metrics(rows):
    I = sum(x[0] for x in rows); U = sum(x[1] for x in rows)
    iou = sum(x[2] for x in rows)
    n_nt = sum(1 for x in rows if x[3]); n_ab = sum(1 for x in rows if x[3] and x[4])
    return (100.0 * I / U if U else 0.0, 100.0 * iou / len(rows),
            100.0 * n_ab / n_nt if n_nt else 0.0)


def eval_cfg(cache, alpha, beta, topk, margin, p_sel, weighted=False):
    out = []
    w_nt = None
    for e in cache:
        if (e["pp_max"] < p_sel) or e["ens"]:
            out.append((0.0, 0.0 if e["no_target"] else e["gt_area"],
                        1.0 if e["no_target"] else 0.0, e["no_target"], True))
            continue
        parts = []
        bs = e["bs"]
        for t in e["terms"]:
            for b in (t["bl"], t["bd"]):
                if b is None:
                    continue
                score = bs * (alpha + beta * b)
                order = np.argsort(score)[::-1]
                js = [int(order[0])]
                if topk == 2 and len(order) > 1 and \
                        score[order[1]] >= margin * score[order[0]]:
                    js.append(int(order[1]))
                parts.extend(js)
            parts.extend([])  # SAM masks are appended below as arrays
        pred = None
        for j in parts:
            m = e["masks"][j]
            pred = m.copy() if pred is None else (pred | m)
        for sm in (x for t in e["terms"] for x in t["sams"]):
            pred = sm.copy() if pred is None else (pred | sm)
        if pred is None:
            pred = e["fallback"]
        I = float(np.logical_and(pred, e["gt"]).sum())
        U = float(np.logical_or(pred, e["gt"]).sum())
        out.append((I, U, 0.0 if e["no_target"] else (I / U if U else 0.0),
                    e["no_target"], False))
    if not weighted:
        return metrics(out)
    # prevalence-corrected objective: targeted rows weight 1 (two-target) or 0.1
    # (single-target), no-target rows reweighted to the val 37.4/62.6 composition
    n_nt = sum(1 for e in cache if e["no_target"])
    n_t = len(cache) - n_nt
    w0 = (0.626 / 0.374) * (n_t / max(1, n_nt))
    I = U = iou = wsum = 0.0
    for e, (Ii, Ui, ioui, nt, ab) in zip(cache, out):
        w = (w0 if nt else (1.0 if e.get("n_targets", 0) >= 2 else 0.1))
        I += w * Ii; U += w * Ui; iou += w * ioui; wsum += w
    return (100.0 * I / U if U else 0.0, 100.0 * iou / wsum if wsum else 0.0,
            100.0 * sum(1 for e, r in zip(cache, out) if e["no_target"] and r[4]) /
            max(1, n_nt))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune"], default="tune")
    ap.add_argument("--out", type=Path, default=OUTF / "gref_tune_aligned.json")
    ap.add_argument("--weighted", action="store_true",
                    help="prevalence-corrected objective (val-dominant two-target + no-target mix)")
    args = ap.parse_args()
    import pickle
    p = ROOT / "exp/out_extreme/gref_tune_aligned_cache.pkl"
    if p.is_file():
        cache = pickle.loads(p.read_bytes())
        print("cache loaded", len(cache))
    else:
        cache = build_cache()
    grids = dict(
        alpha=[0.8, 1.2, 1.5, 1.811, 2.2, 2.6],
        beta=[0.05, 0.15, 0.234, 0.35, 0.5],
        topk=[1, 2],
        margin=[0.45, 0.55, 0.631, 0.72, 0.82, 0.9],
        p_sel=[0.28, 0.34, 0.42],
    )
    t0 = time.time()
    results = []
    best = None
    for alpha, beta, topk, margin, p_sel in itertools.product(
            grids["alpha"], grids["beta"], grids["topk"], grids["margin"],
            grids["p_sel"]):
        ci, gi, na = eval_cfg(cache, alpha, beta, topk, margin, p_sel,
                              weighted=args.weighted)
        row = dict(alpha=alpha, beta=beta, topk=topk, margin=margin, p_sel=p_sel,
                   cIoU=ci, gIoU=gi, N_acc=na)
        results.append(row)
        if best is None or ci > best["cIoU"]:
            best = row
    results.sort(key=lambda r: -r["cIoU"])
    doc = dict(split="held-out gRefCOCO train-tune (per-record sentences)",
               n_rows=len(cache), n_configs=len(results),
               objective=("prevalence-corrected cIoU on tune rows (two-target targeted rows "
                          "weight 1, single-target 0.1, no-target rows reweighted to the val "
                          "37.4/62.6 composition)" if args.weighted else
                          "official cIoU on tune rows"),
               best=best, top10=results[:10],
               frozen=dict(alpha=1.8111644301883565, beta=0.23434019563238434,
                           topk=2, margin=0.6312676830144586,
                           p_sel=0.34040013160612675),
               seconds=round(time.time() - t0, 1),
               generated="exp177_tune_gref_aligned.py")
    frozen = eval_cfg(cache, 1.8111644301883565, 0.23434019563238434, 2,
                      0.6312676830144586, 0.34040013160612675)
    doc["frozen_tune_metrics"] = dict(cIoU=frozen[0], gIoU=frozen[1], N_acc=frozen[2])
    args.out.write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc["best"], indent=1))
    print("frozen tune", doc["frozen_tune_metrics"])
    print("top", [(round(r["cIoU"], 2), r["alpha"], r["beta"], r["topk"],
                  round(r["margin"], 2), r["p_sel"]) for r in results[:8]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
