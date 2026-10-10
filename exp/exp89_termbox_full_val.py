#!/usr/bin/env python
"""EXP-89 — frozen termbox rule: one full-val pass (change 10, protocol-compliant).

The exp87 study selected the per-term picker (logreg on 8 per-term features,
top-2 with margin 0.815, C=1.07e-3) on the val-matched held-out population:
calib r 0.5506 vs the shipped pair rule's 0.4268 (+0.124).  This refits that
configuration on ALL 262 val-matched held-out rows, freezes every scalar, and
evaluates the official protocol once over the full 14,220-record val split:

  - abstention: unchanged (shipped pair-rule gate + 3-probe ensemble);
  - selection for predicted records: union of per-term picks (each parsed term's
    box -> learned picker -> top-2 with margin); if no term boxes, the shipped
    pair-rule candidate (fallback).

No val number influences any choice.

    python exp/exp89_termbox_full_val.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp84_nounbox_sample import npath  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def term_features(r, rec):
    """Per-term feature matrices + boxes in stride-2 coords for one record."""
    masks = r["masks"]
    bs = r["_bs"]
    areas = masks.sum((1, 2)).astype(np.float64)
    out = []
    for ti, term in enumerate(rec["terms"]):
        p = npath(rec["file_name"], term)
        if not p.is_file():
            continue
        box = json.loads(p.read_text()).get("box")
        if box is None:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
        x1, y1 = min(masks.shape[2], -(-x1 // STRIDE)), min(masks.shape[1], -(-y1 // STRIDE))
        if x1 <= x0 or y1 <= y0:
            continue
        area_box = float((x1 - x0) * (y1 - y0))
        inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
        bxi = inb / (areas + area_box - inb + 1e-9)
        inbf = inb / np.maximum(areas, 1.0)
        n = len(bs)
        F = np.stack([
            bs, bxi, inbf, areas / masks[0].size,
            np.argsort(np.argsort(bs)) / max(1, n - 1),
            np.full(n, ti, np.float32),
            np.full(n, len(rec["terms"]), np.float32),
            np.full(n, bxi.max(), np.float32),
        ], 1).astype(np.float32)
        out.append(F)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/gref_termbox_full_val.json"))
    ap.add_argument("--restrict-calib", action="store_true",
                    help="restrict evaluation to the sample refs' calib split (seed 0, 0.3)")
    args = ap.parse_args()
    study = json.loads((OUTF / "optuna_study_gref9.json").read_text())
    bp = study["best_params"]
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    topk, margin, C = bp["topk"], bp["margin"], bp["C"]
    print(f"frozen termbox rule: kind={bp['kind']} topk={topk} margin={margin:.3f} "
          f"C={C:.2e}", flush=True)

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]

    # pair-rule fallback classifier (shipped)
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))

    # per-term picker fitted on the val-matched held-out population
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    dedup = {r["rid"]: r for r in tr}
    Xf, yf = [], []
    for rid, rec in recs_by_rid.items():
        r = dedup.get(rid)
        if r is None:
            continue
        iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
        for F in term_features(r, rec):
            Xf.append(F)
            yf.append((iou_gt >= 0.5).astype(np.float32))
    pick_clf = make_clf("logreg", dict(C=C))
    pick_clf.fit(np.concatenate(Xf), np.concatenate(yf))
    print(f"term picker fit on {len(Xf)} term matrices ({time.time()-t0:.0f}s)", flush=True)

    # ---- full val pass -----------------------------------------------------
    import hashlib
    from vlm_boxes import CACHE_DIR, MODEL_TAG
    from exp67_ensemble_eval import probe_no

    val = load_rows(args.records, args.split, clf_base)
    if args.restrict_calib:
        rng = np.random.RandomState(0)
        refs = sorted({rec["ref_id"] for rec in sample})
        rng.shuffle(refs)
        calib_refs = set(refs[int(0.7 * len(refs)):])
        val = [r for r in val if r["rid"] in calib_refs]
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    # attach terms per record (from the records file, aligned by ref_id matching)
    terms_by_rid = {}
    for line in open(args.records):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue          # a ref's no-target row shares its ref_id and would
                              # clobber the targeted row's own (file_name, terms)
        ts, seen = [], set()
        for t in [rec["parsed"].get("noun_phrase") or ""] + list(rec["parsed"].get("nouns") or []):
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                ts.append(t)
        terms_by_rid[rec["ref_id"]] = (rec["file_name"], ts[:3])
    print(f"val rows {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    p_sel = g8["p_sel"]
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0
    n_termbox = n_fallback = 0
    per_row = []          # (no_target, abstained, I, U, gt_area) for weighting
    for r in val:
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
            continue
        pred = None
        fn_, ts = terms_by_rid.get(r["rid"], (None, []))
        if fn_ is not None and ts:
            picks = []
            for F_t in term_features(r, {"file_name": fn_, "terms": ts}):
                pt = pick_clf.predict_proba(F_t)[:, 1]
                order = np.argsort(pt)[::-1]
                picks.append(int(order[0]))
                if topk == 2 and len(order) > 1 and pt[order[1]] >= margin * pt[order[0]]:
                    picks.append(int(order[1]))
            picks = list(dict.fromkeys(picks))
            if picks:
                pred = r["masks"][picks[0]].copy()
                for j in picks[1:]:
                    pred = np.logical_or(pred, r["masks"][j])
                n_termbox += 1
        if pred is None:
            k = int(np.argmax(pp))
            pred = candidate_mask(r["masks"], ids[k])
            n_fallback += 1
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        per_row.append((r["no_target"], False, I, U, r["gt_area"]))
        if r["no_target"]:
            fn += 1
            acc_U += U
        else:
            tn += 1
            acc_iou += (I / U if U else 0.0)
            acc_I += I
            acc_U += U
    n = len(val)
    m = dict(n=n, cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
             gIoU=100.0 * acc_iou / n,
             N_acc=100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
             T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0,
             TP=tp, FN=fn, TN=tn, FP=fp,
             termbox_used=n_termbox, fallback_used=n_fallback)
    # val-prevalence weighted variant (for held-out transfer statements)
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
    print(f"FULL VAL termbox: cIoU {m['cIoU']:.2f} gIoU {m['gIoU']:.2f} "
          f"N {m['N_acc']:.1f} T {m['T_acc']:.1f} "
          f"(termbox {n_termbox}, fallback {n_fallback}) ({time.time()-t0:.0f}s)",
          flush=True)
    out = dict(model=dict(kind="per-term box picker + pair fallback",
                          study_params=bp, source_study="optuna_study_gref9.json"),
               metrics=m,
               history={"pair_union": {"cIoU": 36.7453, "gIoU": 69.6279},
                        "ensemble": {"cIoU": 34.9419, "gIoU": 69.7711},
                        "single_probe": {"cIoU": 35.0884, "gIoU": 68.3729}},
               generated="exp89_termbox_full_val.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
