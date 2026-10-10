#!/usr/bin/env python
"""EXP-85 — diagnose LLaVA per-noun boxes: quality and deployed proxy.

Reads the exp84 sample (300 multi-object targeted held-out-train records) and
its cached per-term boxes (namespace `nounbox`), then measures:

  (a) per-term box vs the best-matching GT object mask: mean IoU, % >= 0.5;
  (b) the deployed proxy: per-term box -> the pool mask with max bbox-IoU vs
      that box -> union of picks -> the sample's cumulative targeted r,
      compared against the shipped rule's r on the same records and against
      the exp83 GT-box reference (0.7151 on the full held-out targeted set —
      different row set, so compare like-for-like within the sample: the
      bboxIoU-pick-with-GT-boxes r is recomputed here on the sample rows).

    python exp/exp85_nounbox_diagnose.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import load_record  # noqa: E402
from exp67_ensemble_eval import REPRO, probe_no  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp83_two_box_oracle import object_masks  # noqa: E402
from exp84_nounbox_sample import npath  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def main() -> int:
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]

    gd = GreferDataset(REPRO / "data/refer", "train")
    by_id = {r["ref_id"]: r for r in gd.refs}

    # pair selector on the sample rows: r_shipped_sample
    clf_base = make_clf_safe(st5)
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
    from exp62_clause_selector import make_clf
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))

    rid_set = {rec["ref_id"] for rec in sample}
    rows = [r for r in tr if r["rid"] in rid_set]
    print(f"sample rows matched: {len(rows)}/{len(sample)}", flush=True)
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}

    acc = dict(shI=0.0, shU=0.0, nbI=0.0, nbU=0.0, gtI=0.0, gtU=0.0)
    ious, sel50 = [], []
    n_terms_used = 0
    t0 = time.time()
    for r in rows:
        rec = recs_by_rid[r["rid"]]
        masks, gt = r["masks"], r["gt"]
        objs = object_masks(gd, by_id[r["rid"]])
        ys_all, xs_all = np.nonzero(gt)
        y0, y1 = ys_all.min(), ys_all.max() + 1
        x0g, x1g = xs_all.min(), xs_all.max() + 1

        def pick_bboxIoU(box):
            x0, y0b, x1, y1b = box
            area = float((x1 - x0) * (y1b - y0b))
            if area <= 0:
                return None
            inb = masks[:, y0b:y1b, x0:x1].sum((1, 2)).astype(np.float64)
            bxi = inb / (masks.sum((1, 2)) + area - inb + 1e-9)
            return int(np.argmax(bxi))

        # shipped pair selector prediction
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        k = int(np.argmax(pair_clf.predict_proba(F)[:, 1]))
        pred_sh = candidate_mask(r["masks"], ids[k])
        acc["shI"] += float(np.logical_and(pred_sh, gt).sum())
        acc["shU"] += float(np.logical_or(pred_sh, gt).sum())

        # nounbox picks
        picks = []
        for term in rec["terms"]:
            p = npath(rec["file_name"], term)
            if not p.is_file():
                continue
            d = json.loads(p.read_text())
            box = d.get("box")
            if box is None:
                continue
            x0, y0b, x1, y1b = [int(round(v)) for v in box]
            x0, y0b = max(0, x0 // STRIDE), max(0, y0b // STRIDE)
            x1, y1b = min(masks.shape[2], -(-x1 // STRIDE)), min(masks.shape[1], -(-y1b // STRIDE))
            if x1 <= x0 or y1b <= y0b:
                continue
            j = pick_bboxIoU((x0, y0b, x1, y1b))
            if j is not None:
                picks.append(j)
                n_terms_used += 1
            # quality vs the best-matching object
            best = 0.0
            for om in objs:
                inter = float(np.logical_and(masks[j], om).sum()) if j is not None else 0.0
                uni = float(np.logical_or(masks[j], om).sum()) if j is not None else 0.0
                best = max(best, inter / max(1.0, uni))
            ious.append(best)
            sel50.append(1.0 if best >= 0.5 else 0.0)
        if picks:
            m = masks[picks[0]].copy()
            for j in picks[1:]:
                m = np.logical_or(m, masks[j])
            acc["nbI"] += float(np.logical_and(m, gt).sum())
            acc["nbU"] += float(np.logical_or(m, gt).sum())

        # GT-box reference on the same rows
        picks_gt = []
        for om in objs:
            ys, xs = np.nonzero(om)
            if not len(ys):
                continue
            j = pick_bboxIoU((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            if j is not None:
                picks_gt.append(j)
        if picks_gt:
            m = masks[picks_gt[0]].copy()
            for j in picks_gt[1:]:
                m = np.logical_or(m, masks[j])
            acc["gtI"] += float(np.logical_and(m, gt).sum())
            acc["gtU"] += float(np.logical_or(m, gt).sum())

    out = dict(
        sample="300 multi-object targeted held-out-train records (exp84)",
        rows_matched=len(rows),
        nounbox_box_vs_best_object_iou_mean=float(np.mean(ious)) if ious else 0.0,
        nounbox_box_select_ge50_pct=100.0 * float(np.mean(sel50)) if sel50 else 0.0,
        n_term_picks=n_terms_used,
        r_shipped_pair_on_sample=acc["shI"] / max(1.0, acc["shU"]),
        r_nounbox_pick_on_sample=acc["nbI"] / max(1.0, acc["nbU"]),
        r_gtbox_pick_reference_on_sample=acc["gtI"] / max(1.0, acc["gtU"]),
        note=("deployed proxy: per-term LLaVA box -> pool mask with max bbox-IoU vs box -> "
              "union. Compare r_nounbox_pick vs r_shipped_pair (the gain, deployable) and vs "
              "r_gtbox_pick_reference (what perfect boxes would give on these rows)"),
        generated="exp85_nounbox_diagnose.py")
    (OUTF / "gref_nounbox_sample.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


def make_clf_safe(st5):
    from exp62_clause_selector import make_clf
    return make_clf("logreg", dict(C=st5["C"]))


if __name__ == "__main__":
    raise SystemExit(main())
