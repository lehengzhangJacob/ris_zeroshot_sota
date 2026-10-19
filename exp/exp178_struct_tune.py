#!/usr/bin/env python
"""EXP-178 — corrected held-out search over the gRefCOCO rule's structural knobs.

The exp117 study that chose score source, pick box, SAM sources and area cap
ran under the ref_id-keyed term map.  This script rebuilds the tune-side
structures with per-record sentences (unfiltered SAM masks, base/sc/gem
scores) and runs a random search (800 configurations) over the structural
space plus the sxi scalars, scored with the prevalence-corrected cIoU that
matches the val targeted/no-target mix.  The winner is applied to val once.

    python exp/exp178_struct_tune.py --mode search
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
from exp138_learned_union_selector import (gdino_detect, llava_box_of,  # noqa: E402
                                           sam_paths, sam_load)
from exp173_gref_search_aligned import bxi_box  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTF = ROOT / "exp/out_final"
TUNE = ROOT / "exp/out_extreme/records_grefcoco_train_tune.jsonl"
CACHE = ROOT / "exp/out_extreme/gref_struct_tune_cache.pkl"


def build_cache():
    t0 = time.time()
    nt_map = {}
    for line in open(TUNE):
        rec = json.loads(line)
        rid = int(rec["ref_id"])
        nt_map[rid] = max(nt_map.get(rid, 0), int(rec.get("n_targets") or 0))
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    clf = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(TUNE, "train", clf)
    clf.fit(np.concatenate([r["X"] for r in tr]), np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf.predict_proba(r["X"])[:, 1]
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km]); ys.append(y[km])
    pair = make_clf("logreg", dict(C=g8["C"]))
    pair.fit(np.concatenate(Xs), np.concatenate(ys))
    print(f"fitted clfs ({time.time()-t0:.0f}s)", flush=True)
    cache = []
    for i, r in enumerate(tr):
        rec = r.get("rec")
        if rec is None:
            continue
        fn = rec["file_name"]
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair.predict_proba(F)[:, 1]
        if r["no_target"]:
            single = [kk for kk, c in enumerate(ids) if c[1] < 0]
            fb = max(single, key=lambda kk: pp[kk]) if single else int(np.argmax(pp))
        else:
            fb = int(np.argmax(pp))
        terms = []
        for term in terms_of(rec["parsed"]):
            boxes, _ = gdino_detect(fn, term)
            d0 = boxes[0] if boxes else None
            ll = llava_box_of(fn, term)
            bll = bxi_box(r["masks"], r["_bs"], ll) if ll is not None else None
            bd0 = bxi_box(r["masks"], r["_bs"], d0) if d0 is not None else None
            p0, p1 = sam_paths(fn, term, train_side=True)
            s0 = sam_load(p0); s1 = sam_load(p1)
            if bll is None and bd0 is None and not s0 and not s1:
                continue
            terms.append(dict(bll=bll, bd0=bd0, s0=s0, s1=s1))
        cache.append(dict(rid=int(r["rid"]), n_targets=nt_map.get(int(r["rid"]), 0),
                          masks=r["masks"], gt=r["gt"], gt_area=float(r["gt_area"]),
                          no_target=bool(r["no_target"]), ens=bool(any(r["ens"])),
                          bs=r["_bs"], sc=np.asarray(r["_sc"], np.float32),
                          gem=np.asarray(r["_gem"], np.float32),
                          pp_max=float(pp.max()),
                          fallback=candidate_mask(r["masks"], ids[fb]), terms=terms))
        r["masks"] = None
        if (i + 1) % 200 == 0:
            print(f"  cache {i+1}/{len(tr)} ({time.time()-t0:.0f}s)", flush=True)
    CACHE.write_bytes(pickle.dumps(cache, protocol=4))
    print(f"cache {len(cache)} rows -> {CACHE} ({time.time()-t0:.0f}s)", flush=True)
    return cache


def eval_cfg(cache, cfg):
    src, pb, sam, cap = cfg["src"], cfg["pick_box"], cfg["sam"], cfg["cap"]
    alpha, beta = cfg["alpha"], cfg["beta"]
    topk, margin, p_sel = cfg["topk"], cfg["margin"], cfg["p_sel"]
    out = []
    for e in cache:
        if (e["pp_max"] < p_sel) or e["ens"]:
            out.append((0.0, 0.0 if e["no_target"] else e["gt_area"],
                        1.0 if e["no_target"] else 0.0, e["no_target"], True))
            continue
        base = e["bs"] if src == "bs" else (e["sc"] if src == "sc" else e["gem"])
        parts = []
        for t in e["terms"]:
            boxes = []
            if pb in ("ll", "ll+d0") and t["bll"] is not None:
                boxes.append(t["bll"])
            if pb in ("d0", "ll+d0") and t["bd0"] is not None:
                boxes.append(t["bd0"])
            for b in boxes:
                score = base * (alpha + beta * b)
                order = np.argsort(score)[::-1]
                js = [int(order[0])]
                if topk == 2 and len(order) > 1 and \
                        score[order[1]] >= margin * score[order[0]]:
                    js.append(int(order[1]))
                parts.extend(e["masks"][j] for j in js)
            for sm in ((t["s0"] + (t["s1"] if sam == "d0+d1" else [])) if sam != "none" else []):
                if sm.sum() / sm.size <= cap:
                    parts.append(sm)
        pred = None
        for m in parts:
            pred = m.copy() if pred is None else (pred | m)
        if pred is None:
            pred = e["fallback"]
        I = float(np.logical_and(pred, e["gt"]).sum())
        U = float(np.logical_or(pred, e["gt"]).sum())
        out.append((I, U, 0.0 if e["no_target"] else (I / U if U else 0.0),
                    e["no_target"], False))
    n_nt = sum(1 for e in cache if e["no_target"])
    n_t = len(cache) - n_nt
    w0 = (0.626 / 0.374) * (n_t / max(1, n_nt))
    I = U = iou = wsum = 0.0
    for e, r in zip(cache, out):
        w = w0 if e["no_target"] else (1.0 if e.get("n_targets", 0) >= 2 else 0.1)
        I += w * r[0]; U += w * r[1]; iou += w * r[2]; wsum += w
    return (100.0 * I / U if U else 0.0,
            100.0 * iou / wsum if wsum else 0.0,
            100.0 * sum(1 for e, r in zip(cache, out) if e["no_target"] and r[4]) /
            max(1, n_nt))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["search"], default="search")
    ap.add_argument("--configs", type=int, default=800)
    ap.add_argument("--out", type=Path, default=OUTF / "gref_struct_tune.json")
    args = ap.parse_args()
    if CACHE.is_file():
        cache = pickle.loads(CACHE.read_bytes()); print("cache loaded", len(cache))
    else:
        cache = build_cache()
    rng = np.random.RandomState(20261019)
    configs = []
    for _ in range(args.configs):
        configs.append(dict(
            src=str(rng.choice(["bs", "sc", "gem"])),
            pick_box=str(rng.choice(["ll", "d0", "ll+d0"])),
            sam=str(rng.choice(["d0", "d0+d1", "none"])),
            cap=float(rng.choice([0.2, 0.3, 0.5, 0.7])),
            alpha=float(rng.uniform(0.4, 3.2)),
            beta=float(rng.uniform(0.0, 0.55)),
            topk=int(rng.choice([1, 2])),
            margin=float(rng.uniform(0.4, 1.0)),
            p_sel=float(rng.uniform(0.24, 0.45)),
        ))
    # include the shipped/current config families as anchors
    for a, b, t, m, p in [(1.811, 0.234, 2, 0.631, 0.340), (0.8, 0.234, 1, 0.45, 0.28),
                          (2.2, 0.35, 1, 0.45, 0.28)]:
        configs.append(dict(src="bs", pick_box="ll+d0", sam="d0+d1", cap=0.5,
                            alpha=a, beta=b, topk=t, margin=m, p_sel=p))
    t0 = time.time()
    results = []
    for cfg in configs:
        ci, gi, na = eval_cfg(cache, cfg)
        results.append(dict(**cfg, cIoU=ci, gIoU=gi, N_acc=na))
    results.sort(key=lambda r: -r["cIoU"])
    doc = dict(split="held-out gRefCOCO train-tune (per-record sentences)",
               n_rows=len(cache), n_configs=len(results),
               objective=("prevalence-corrected cIoU (two-target targeted rows weight 1, "
                          "single-target 0.1, no-target rows reweighted to the val mix)"),
               search="random search, numpy RandomState(20261019) over src/pick_box/sam/cap/"
                      "alpha/beta/topk/margin/p_sel plus three anchor configs",
               best=results[0], top10=results[:10], seconds=round(time.time() - t0, 1),
               generated="exp178_struct_tune.py")
    args.out.write_text(json.dumps(doc, indent=2))
    print(json.dumps(results[0], indent=1))
    for r in results[:6]:
        print(round(r["cIoU"], 3), r["src"], r["pick_box"], r["sam"], r["cap"],
              round(r["alpha"], 2), round(r["beta"], 2), r["topk"], round(r["p_sel"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
