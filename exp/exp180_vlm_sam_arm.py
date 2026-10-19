#!/usr/bin/env python
"""EXP-180 — DINO-free gRefCOCO arm with SAM box-prompted VLM masks.

Parts per targeted record: (a) sxi pool picks inside each VLM clause/noun box,
and (b) SAM ViT-H box-prompted masks from those same VLM boxes (exp179 cache),
selected by SAM's predicted IoU and an area cap.  Abstention and the pair/
union fallback are the shipped ones; no GroundingDINO anywhere.

Modes:
  tune  build the train-tune cache and random-search the held-out config
  val   one official-val pass with the selected config

    python exp/exp180_vlm_sam_arm.py --mode tune
    python exp/exp180_vlm_sam_arm.py --mode val --params exp/out_final/vlm_sam_tune.json
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
from exp108_gdino_study import lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
from exp173_gref_search_aligned import bxi_box  # noqa: E402
from exp179_vlm_sam import cache_path as sam_cache_path  # noqa: E402
from exp169_gref_rescue import base_box_path  # noqa: E402
from method_gres import clauses  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTF = ROOT / "exp/out_final"
TUNE = ROOT / "exp/out_extreme/records_grefcoco_train_tune.jsonl"
VAL = ROOT / "exp/out_extreme/records_grefcoco_14229.jsonl"
CACHE = ROOT / "exp/out_extreme/gref_vlmsam_tune_cache.pkl"


def load_sam(source, fn, phrase):
    p = sam_cache_path(source, fn, phrase)
    if not p.is_file():
        return [], []
    with np.load(p, allow_pickle=False) as z:
        shape = tuple(int(x) for x in z["shape"])
        bits = np.unpackbits(z["bits"], count=int(np.prod(shape))).reshape(shape)
        scores = z["scores"].astype(np.float32)
    return [bits[i].astype(bool) for i in range(shape[0])], scores.tolist()


def term_structs(r, rec, source):
    fn = rec["file_name"]
    sent = (rec["parsed"].get("sentence") or "").strip()
    phrases = clauses(sent) if source == "clause" else terms_of(rec["parsed"])
    out = []
    for ph in phrases:
        ph = (ph or "").strip()
        if not ph:
            continue
        p = base_box_path(fn, ph) if source == "clause" else lpath(fn, ph)
        if not p.is_file():
            continue
        box = json.loads(p.read_text()).get("box")
        if box is None:
            continue
        b = bxi_box(r["masks"], r["_bs"], box)
        if b is None:
            continue
        sm, sc = load_sam(source, fn, ph)
        out.append(dict(b=b, sam=sm, scores=sc))
    return out


def build_cache():
    t0 = time.time()
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
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair.predict_proba(F)[:, 1]
        if r["no_target"]:
            single = [kk for kk, c in enumerate(ids) if c[1] < 0]
            fb = max(single, key=lambda kk: pp[kk]) if single else int(np.argmax(pp))
        else:
            fb = int(np.argmax(pp))
        cache.append(dict(
            rid=int(r["rid"]), no_target=bool(r["no_target"]), ens=bool(any(r["ens"])),
            masks=r["masks"], gt=r["gt"], gt_area=float(r["gt_area"]),
            bs=r["_bs"], sc=np.asarray(r["_sc"], np.float32),
            gem=np.asarray(r["_gem"], np.float32), pp_max=float(pp.max()),
            fallback=candidate_mask(r["masks"], ids[fb]),
            clause=term_structs(r, rec, "clause"),
            noun=term_structs(r, rec, "noun")))
        r["masks"] = None
        if (i + 1) % 200 == 0:
            print(f"  cache {i+1}/{len(tr)} ({time.time()-t0:.0f}s)", flush=True)
    CACHE.write_bytes(pickle.dumps(cache, protocol=4))
    print(f"cache {len(cache)} rows ({time.time()-t0:.0f}s)", flush=True)
    return cache


def evaluate(cache, cfg):
    src = cfg["source"]
    alpha, beta = cfg["alpha"], cfg["beta"]
    topk, margin, p_sel = cfg["topk"], cfg["margin"], cfg["p_sel"]
    cap, sam_k, smin = cfg["cap"], cfg["sam_topk"], cfg["sam_score_min"]
    out = []
    for e in cache:
        if (e["pp_max"] < p_sel) or e["ens"]:
            out.append((0.0, 0.0 if e["no_target"] else e["gt_area"],
                        1.0 if e["no_target"] else 0.0, e["no_target"], True))
            continue
        base = e["bs"] if cfg.get("base", "bs") == "bs" else (
            e["sc"] if cfg.get("base") == "sc" else e["gem"])
        terms = []
        if src in ("clause", "both"):
            terms += e["clause"]
        if src in ("noun", "both"):
            terms += e["noun"]
        masks = []
        for t in terms:
            score = np.asarray(base, np.float32) * (alpha + beta * t["b"])
            order = np.argsort(score)[::-1]
            js = [int(order[0])]
            if topk == 2 and len(order) > 1 and \
                    score[order[1]] >= margin * score[order[0]]:
                js.append(int(order[1]))
            masks.extend(e["masks"][j] for j in js)
            if t["sam"]:
                sc = np.asarray(t["scores"], np.float32)
                keep = [k for k in np.argsort(sc)[::-1][:sam_k] if sc[k] >= smin]
                for k in keep:
                    m = t["sam"][k]
                    if m.sum() / m.size <= cap:
                        masks.append(m)
        pred = None
        for m in masks:
            pred = m.copy() if pred is None else (pred | m)
        if pred is None:
            pred = e["fallback"]
        I = float(np.logical_and(pred, e["gt"]).sum())
        U = float(np.logical_or(pred, e["gt"]).sum())
        out.append((I, U, 0.0 if e["no_target"] else (I / U if U else 0.0),
                    e["no_target"], False))
    I = sum(x[0] for x in out); U = sum(x[1] for x in out)
    iou = sum(x[2] for x in out)
    n_nt = sum(1 for e in cache if e["no_target"])
    n_ab = sum(1 for e, r in zip(cache, out) if e["no_target"] and r[4])
    return (100.0 * I / U if U else 0.0, 100.0 * iou / len(out),
            100.0 * n_ab / max(1, n_nt))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "val"], required=True)
    ap.add_argument("--configs", type=int, default=800)
    ap.add_argument("--params", type=Path, default=OUTF / "vlm_sam_tune.json")
    ap.add_argument("--out", type=Path, default=OUTF / "vlm_sam_val.json")
    args = ap.parse_args()

    if args.mode == "tune":
        cache = pickle.loads(CACHE.read_bytes()) if CACHE.is_file() else build_cache()
        if CACHE.is_file():
            print("cache loaded", len(cache))
        rng = np.random.RandomState(20261020)
        cfgs = []
        for _ in range(args.configs):
            cfgs.append(dict(
                source=str(rng.choice(["clause", "noun", "both"])),
                base="bs",
                alpha=float(rng.uniform(0.4, 3.2)), beta=float(rng.uniform(0.0, 0.55)),
                topk=int(rng.choice([1, 2])), margin=float(rng.uniform(0.4, 1.0)),
                cap=float(rng.choice([0.2, 0.3, 0.5, 0.7])),
                sam_topk=int(rng.choice([1, 2, 3])),
                sam_score_min=float(rng.choice([0.0, 0.5, 0.6, 0.7, 0.8])),
                p_sel=float(rng.uniform(0.24, 0.45))))
        cfgs += [dict(source="clause", base="bs", alpha=1.811, beta=0.234, topk=2,
                      margin=0.631, cap=0.5, sam_topk=3, sam_score_min=0.0, p_sel=0.34),
                 dict(source="both", base="bs", alpha=0.8, beta=0.234, topk=1,
                      margin=0.45, cap=0.5, sam_topk=2, sam_score_min=0.6, p_sel=0.28)]
        t0 = time.time()
        results = []
        for cfg in cfgs:
            ci, gi, na = evaluate(cache, cfg)
            results.append(dict(**cfg, cIoU=ci, gIoU=gi, N_acc=na))
        results.sort(key=lambda r: -r["cIoU"])
        doc = dict(split="held-out gRefCOCO train-tune (per-record sentences)",
                   n_rows=len(cache), n_configs=len(results),
                   objective="official cIoU on tune rows (abstention included)",
                   search="random search (RandomState 20261020) over source/base/sxi/"
                          "cap/SAM top-k/SAM score floor/p_sel plus two anchors",
                   best=results[0], top10=results[:10],
                   seconds=round(time.time() - t0, 1),
                   generated="exp180_vlm_sam_arm.py")
        args.params.write_text(json.dumps(doc, indent=2))
        print(json.dumps(results[0], indent=1))
        for r in results[:6]:
            print(round(r["cIoU"], 3), r["source"], r["cap"], r["sam_topk"],
                  r["sam_score_min"], round(r["alpha"], 2), round(r["beta"], 2),
                  r["topk"], round(r["p_sel"], 2))
        return 0

    # ---- one val pass -----------------------------------------------------
    cfg = json.loads(args.params.read_text())["best"]
    print("frozen config:", json.dumps(cfg), flush=True)
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
    del tr
    print("clfs fitted; loading val", flush=True)
    val = load_rows(VAL, "val", clf)
    for r in val:
        r["_bs"] = clf.predict_proba(r["X"])[:, 1]
    print(f"val rows {len(val)}", flush=True)
    I = U = iou = 0.0
    n_nt = n_ab = 0
    TP = FN = TN = FP = 0
    t0 = time.time()
    for k, r in enumerate(val):
        rec = r["rec"]
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair.predict_proba(F)[:, 1]
        abstain = (pp.max() < cfg["p_sel"]) or any(r["ens"])
        gt = r["gt"]
        if abstain:
            U += 0.0 if r["no_target"] else r["gt_area"]
            iou += 1.0 if r["no_target"] else 0.0
            if r["no_target"]:
                n_nt += 1; n_ab += 1; TP += 1
            else:
                FP += 1
            continue
        terms = []
        if cfg["source"] in ("clause", "both"):
            terms += term_structs(r, rec, "clause")
        if cfg["source"] in ("noun", "both"):
            terms += term_structs(r, rec, "noun")
        masks = []
        base = r["_bs"] if cfg.get("base", "bs") == "bs" else (
            r["_sc"] if cfg.get("base") == "sc" else r["_gem"])
        for t in terms:
            score = np.asarray(base, np.float32) * (cfg["alpha"] + cfg["beta"] * t["b"])
            order = np.argsort(score)[::-1]
            js = [int(order[0])]
            if cfg["topk"] == 2 and len(order) > 1 and \
                    score[order[1]] >= cfg["margin"] * score[order[0]]:
                js.append(int(order[1]))
            masks.extend(r["masks"][j] for j in js)
            if t["sam"]:
                sc = np.asarray(t["scores"], np.float32)
                keep = [kk for kk in np.argsort(sc)[::-1][:cfg["sam_topk"]]
                        if sc[kk] >= cfg["sam_score_min"]]
                for kk in keep:
                    m = t["sam"][kk]
                    if m.sum() / m.size <= cfg["cap"]:
                        masks.append(m)
        pred = None
        for m in masks:
            pred = m.copy() if pred is None else (pred | m)
        if pred is None:
            if r["no_target"]:
                single = [kk for kk, c in enumerate(ids) if c[1] < 0]
                kk = max(single, key=lambda kk: pp[kk]) if single else int(np.argmax(pp))
            else:
                kk = int(np.argmax(pp))
            pred = candidate_mask(r["masks"], ids[kk])
        ii = float(np.logical_and(pred, gt).sum())
        uu = float(np.logical_or(pred, gt).sum())
        I += ii; U += uu
        iou += 0.0 if r["no_target"] else (ii / uu if uu else 0.0)
        if r["no_target"]:
            n_nt += 1; FN += 1
        else:
            TN += 1
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(val)} ({time.time()-t0:.0f}s)", flush=True)
    n = len(val)
    doc = dict(n=n, cIoU=100.0 * I / U if U else 0.0, gIoU=100.0 * iou / n,
               N_acc=100.0 * n_ab / max(1, n_nt), n_no_target=n_nt,
               T_acc=100.0 * TN / max(1, TN + FP), TP=TP, FN=FN, TN=TN, FP=FP,
               config=cfg, alignment="per-record sentence",
               generated="exp180_vlm_sam_arm.py --mode val")
    args.out.write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
