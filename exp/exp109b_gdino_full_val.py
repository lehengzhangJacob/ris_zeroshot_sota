#!/usr/bin/env python
"""EXP-109b — full-val pass of the frozen GDINO+SAM termbox rule (change 13).

Mirrors exp89 (the shipped termbox val pass) with the prediction step
replaced by the frozen study winner (exp108, optuna_study_gref12.json):

  per term: box = top-1 GDINO detection if score >= gd_thresh else the cached
            LLaVA nounbox (fallback); pool pick via the frozen picker
            (refit on the study's FIT refs only, exactly as in exp108);
  SAM:      box-prompted 3-mask outputs from the top-1 detection, cached by
            exp109a (sam_mode largest/all3/evid, area-capped);
  cover2:   optional pool pick in the 2nd detection box;
  pred = union of all selected masks; pair/union fallback if nothing selected.

Abstention, no-target handling, metrics and the val-prevalence weighting are
byte-identical to exp89.  Val is read only here; nothing is fitted on it.

    CUDA_VISIBLE_DEVICES=1 python exp/exp109b_gdino_full_val.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO, probe_no  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import spath, terms_of  # noqa: E402
from vlm_boxes import IMAGES, MODEL_TAG  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--study", type=Path,
                    default=Path("exp/out_final/optuna_study_gref12.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/gref_gdino_full_val.json"))
    args = ap.parse_args()

    study = json.loads(args.study.read_text())
    cfg = dict(study["best_params"])
    cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
    print(f"frozen GDINO+SAM rule: {cfg}", flush=True)

    t0 = time.time()
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]

    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))

    # frozen picker: refit on the study's FIT refs only (same seed/frac as exp108)
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    dedup = {r["rid"]: r for r in tr}
    rows = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]
    rng = np.random.RandomState(0)
    refs = sorted({r["rid"] for r in rows})
    rng.shuffle(refs)
    n_fit = int(0.7 * len(refs))
    fit_refs = set(refs[:n_fit])
    fit_rows = [r for r in rows if r["rid"] in fit_refs]

    from gdino import Gdino
    g = Gdino(device="cuda:0")

    def box_for(fn, term):
        """Exact exp108 semantics: box_src 'llava' prefers the LLaVA nounbox
        (GDINO det0 fallback); 'gdino' takes det0 when its score >= thresh
        (LLaVA fallback).  Returns (box, second_box_or_None)."""
        boxes, scores = g.detect(fn, term)
        det0 = boxes[0] if boxes else None
        b2 = boxes[1] if len(boxes) > 1 and scores[1] >= cfg["gd_thresh"] else None
        if cfg["box_src"] == "gdino":
            if det0 is not None and scores[0] >= cfg["gd_thresh"]:
                return det0, b2
            b = llava_box_of(fn, term)
            return ((b, None) if b is not None else (det0, None))
        b = llava_box_of(fn, term)
        return ((b, b2) if b is not None else (det0, b2))

    def llava_box_of(fn, term):
        p = lpath(fn, term)
        if not p.is_file():
            return None
        return json.loads(p.read_text()).get("box")

    Xf, yf = [], []
    for r in fit_rows:
        rec = recs_by_rid[r["rid"]]
        iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
        for term in rec["terms"]:
            b, _ = box_for(rec["file_name"], term)
            if b is None:
                continue
            F, _ = box_feats(r["masks"], r["_bs"], b)
            if F is None:
                continue
            Xf.append(F)
            yf.append((iou_gt >= 0.5).astype(np.float32))
    if cfg.get("pool_rule") == "sxi":
        pick_clf = None
        print("pool_rule=sxi: no picker fit needed (hand rule)", flush=True)
    else:
        pick_clf = make_clf("logreg", dict(C=cfg["C"]))
        pick_clf.fit(np.concatenate(Xf), np.concatenate(yf))
        print(f"picker fit on {len(Xf)} term matrices over {len(fit_rows)} fit rows "
              f"({time.time()-t0:.0f}s)", flush=True)

    def sam_masks(fn, term):
        p = spath(fn, term)
        if not p.is_file() or p.stat().st_size == 0:
            return None
        with np.load(p) as z:
            shape = tuple(int(x) for x in z["shape"])
            bits = np.unpackbits(z["bits"], count=int(np.prod(shape))).reshape(shape)
        return [bits[i].astype(bool) for i in range(shape[0])]

    def pick_masks(r, F, topk, margin):
        if cfg.get("pool_rule") == "sxi":
            score = F[:, 0] * (cfg["alpha"] + cfg["beta"] * F[:, 1])
        else:
            score = pick_clf.predict_proba(F)[:, 1]
        order = np.argsort(score)[::-1]
        js = [int(order[0])]
        if topk == 2 and len(order) > 1 and score[order[1]] >= margin * score[order[0]]:
            js.append(int(order[1]))
        return [r["masks"][j] for j in js]

    # ---- full val pass -----------------------------------------------------
    val = load_rows(args.records, args.split, clf_base)
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    terms_by_rid = {}
    for line in open(args.records):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        terms_by_rid[rec["ref_id"]] = (rec["file_name"], terms_of(rec["parsed"]))
    print(f"val rows {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    # SAM cache coverage on val terms
    val_pairs = set()
    for rid, (fn, ts) in terms_by_rid.items():
        for t in ts:
            val_pairs.add((fn, t))
    have = sum(1 for fn, t in val_pairs if spath(fn, t).is_file())
    print(f"val SAM cache: {have}/{len(val_pairs)} term pairs present", flush=True)

    p_sel = g8["p_sel"]
    masses = dict(S_U=0.0, I_tot=0.0, G_U=0.0, FP_U=0.0, A_U=0.0)
    acc_iou = acc_I = acc_U = 0.0
    tp = fn_c = tn = fp = 0
    n_gdino = n_fallback = 0
    per_row = []
    t1 = time.time()
    for k, r in enumerate(val):
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        abstain = (pp.max() < p_sel) or any(r["ens"])
        if abstain:
            per_row.append((r["no_target"], True, 0.0, 0.0, r["gt_area"]))
            if r["no_target"]:
                tp += 1
                acc_iou += 1.0
            else:
                fp += 1
                acc_U += r["gt_area"]
                masses["A_U"] += r["gt_area"]
            continue
        pred_parts = []
        fn_, ts = terms_by_rid.get(r["rid"], (None, []))
        if fn_ is not None and ts:
            for term in ts:
                b, b2 = box_for(fn_, term)
                if b is None:
                    continue
                Fb, _ = box_feats(r["masks"], r["_bs"], b)
                if Fb is not None:
                    pred_parts.extend(pick_masks(r, Fb, cfg["topk"], cfg["margin"]))
                if cfg["cover2"] and b2 is not None:
                    F2, _ = box_feats(r["masks"], r["_bs"], b2)
                    if F2 is not None:
                        pred_parts.extend(pick_masks(r, F2, cfg["topk"], cfg["margin"]))
                if cfg["sam_mode"] != "off":
                    ss = sam_masks(fn_, term)
                    if ss:
                        areas = r["masks"].sum((1, 2)).astype(np.float64)
                        cap = cfg["sam_area_cap"]
                        if cfg["sam_mode"] == "largest":
                            sm = ss[int(np.argmax([m.sum() for m in ss]))]
                            if sm.sum() / sm.size <= cap:
                                pred_parts.append(sm)
                        elif cfg["sam_mode"] == "all3":
                            pred_parts.extend(s for s in ss
                                              if s.sum() / s.size <= cap)
                        elif cfg["sam_mode"] == "evid":
                            best, bestv = None, -1.0
                            for sm in ss:
                                if sm.sum() / sm.size > cap:
                                    continue
                                inter = np.logical_and(r["masks"], sm[None]).sum((1, 2)).astype(np.float64)
                                ov = inter / np.maximum(areas + sm.sum() - inter, 1.0)
                                v = float((r["_bs"] * ov).max())
                                if v > bestv:
                                    best, bestv = sm, v
                            if best is not None:
                                pred_parts.append(best)
        pred = None
        if pred_parts:
            pred = pred_parts[0].copy()
            for x in pred_parts[1:]:
                pred |= x
            n_gdino += 1
        if pred is None:
            kk = int(np.argmax(pp))
            pred = candidate_mask(r["masks"], ids[kk])
            n_fallback += 1
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        masses["S_U"] += float(pred.sum())          # r-denominator (wall convention)
        masses["I_tot"] += I
        if not r["no_target"]:
            masses["G_U"] += r["gt_area"]           # GT mass over predicted-targeted
        else:
            masses["FP_U"] += float(pred.sum())     # no-target predicted target
        per_row.append((r["no_target"], False, I, U, r["gt_area"]))
        if r["no_target"]:
            fn_c += 1
            acc_U += U
        else:
            tn += 1
            acc_iou += (I / U if U else 0.0)
            acc_I += I
            acc_U += U
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(val)} ({time.time()-t1:.0f}s)", flush=True)

    n = len(val)
    m = dict(n=n, cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
             gIoU=100.0 * acc_iou / n,
             N_acc=100.0 * tp / (tp + fn_c) if (tp + fn_c) else 0.0,
             T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0,
             TP=tp, FN=fn_c, TN=tn, FP=fp,
             gdino_used=n_gdino, fallback_used=n_fallback)
    m["masses"] = dict(masses)
    n_nt = sum(1 for row in per_row if row[0])
    n_t = len(per_row) - n_nt
    w_nt = (0.6256 / 0.3744) * (n_t / max(1, n_nt))
    wI = wU = wiou = wsum = 0.0
    for nt, ab, I, U, gta in per_row:
        w = w_nt if nt else 1.0
        wsum += w
        if ab:
            if nt:
                wiou += w
            else:
                wU += w * gta
            continue
        if nt:
            wU += w * U
        else:
            wiou += w * (I / U if U else 0.0)
            wI += w * I
            wU += w * U
    m["cIoU_val_prevalence"] = 100.0 * wI / wU if wU else 0.0
    m["gIoU_val_prevalence"] = 100.0 * wiou / wsum if wsum else 0.0
    print(f"FULL VAL gdino+sam: cIoU {m['cIoU']:.2f} gIoU {m['gIoU']:.2f} "
          f"N {m['N_acc']:.1f} T {m['T_acc']:.1f} "
          f"(gdino {n_gdino}, fallback {n_fallback}) ({time.time()-t0:.0f}s)",
          flush=True)
    out = dict(model=dict(kind="GDINO-box + SAM-mask + pool-pick termbox union",
                          study_params=cfg, source_study=str(args.study)),
               metrics=m,
               history={"termbox_llava": {"cIoU": 40.9593, "gIoU": 71.1064},
                        "pair_union": {"cIoU": 36.7453, "gIoU": 69.6279}},
               generated="exp109b_gdino_full_val.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
