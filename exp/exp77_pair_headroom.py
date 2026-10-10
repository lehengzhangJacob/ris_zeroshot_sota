#!/usr/bin/env python
"""EXP-77 — is the union-structure headroom real on the selector's own pool?

The wall's ladder: oracle SINGLE mask cumulative r = 0.581, oracle UNION r =
0.884, and the shipped threshold-union delivers 0.4695 — BELOW even the
single-mask oracle.  That pattern says the failure is in HOW unions are
constructed (threshold-union adds area that dilutes IoU), not only in which
masks score high.  Per-proposal labels are also pathological on 2-object
targets: a mask covering exactly one of two equal-area objects has
IoU(mask, GT-union) ~ 0.5, straddling the >=0.5 label boundary, so the
selector is taught to ignore the smaller object.

This measures, per targeted record over its FULL pool (what the selector
actually sees): cumulative IoU of best-single, best-pair, and greedy-union
selections, plus the single-object-mask label band.  If pair >> single >> the
delivered rule, a LEARNED pair/union selector is the next model change; if the
pool's pairs add nothing over singles, that path is dead too.

Fit-side only (held-out train); val untouched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp62_clause_selector import load_record, make_clf, score_rows  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    st = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    t0 = time.time()
    gd = GreferDataset(REPRO / "data/refer", "train")
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    rows = []
    for line in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        rec = json.loads(line)
        r = load_record(rec, gd, by_id, mc, gc)
        if r is not None and not r["no_target"]:
            rows.append(r)
    print(f"targeted rows {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    clf = make_clf("logreg", dict(C=st["C"]))
    clf.fit(np.concatenate([r["X"] for r in rows]),
            np.concatenate([r["y"] for r in rows]))
    score_rows(rows, clf)

    S_gt = S_single = S_pair = S_greedy = S_delivered = 0.0
    I_single = I_pair = I_greedy = I_delivered = 0.0
    band = n_prop = n_cover1 = 0     # single-object-mask label band
    two_obj = 0
    for r in rows:
        gt = r["gt"]
        masks = r["masks"]
        gt_mass = float(gt.sum())
        # per-proposal IoU vs the GT union
        inter = np.logical_and(masks, gt).sum((1, 2)).astype(np.float64)
        union = np.logical_or(masks, gt).sum((1, 2)).astype(np.float64)
        iou = inter / np.maximum(union, 1.0)
        order = np.argsort(iou)[::-1]
        # best single
        i1 = order[0]
        S_single += union[i1]; I_single += inter[i1]
        # best pair (union of two) over the top-12 by IoU (exhaustive on 66 pairs)
        top = order[:12]
        best = (iou[i1], i1, -1)
        for a in range(len(top)):
            ka = top[a]
            for b in range(a + 1, len(top)):
                kb = top[b]
                m = np.logical_or(masks[ka], masks[kb])
                I = float(np.logical_and(m, gt).sum())
                U = float(np.logical_or(m, gt).sum())
                v = I / max(U, 1.0)
                if v > best[0]:
                    best = (v, ka, kb)
        if best[1] >= 0:
            m = np.logical_or(masks[best[1]], masks[best[2]])
        else:
            m = masks[i1]
        I_pair += float(np.logical_and(m, gt).sum())
        S_pair += float(np.logical_or(m, gt).sum())
        # greedy union: add the mask that most increases union-IoU
        cur = masks[i1].copy()
        for _ in range(4):
            base_i = float(np.logical_and(cur, gt).sum())
            base_u = float(np.logical_or(cur, gt).sum())
            gained = False
            bi, bv = None, base_i / max(base_u, 1.0)
            for k in range(len(masks)):
                if np.array_equal(masks[k], cur):
                    continue
                m2 = np.logical_or(cur, masks[k])
                v = float(np.logical_and(m2, gt).sum()) / max(1.0, float(np.logical_or(m2, gt).sum()))
                if v > bv + 1e-6:
                    bv, bi, gained = v, k, True
            if not gained or bi is None:
                break
            cur = np.logical_or(cur, masks[bi])
        I_greedy += float(np.logical_and(cur, gt).sum())
        S_greedy += float(np.logical_or(cur, gt).sum())
        # what the shipped threshold-union delivers
        sel = np.flatnonzero(r["scores"] >= st["p_inc"])
        if sel.size:
            pred = masks[sel].any(0)
            I_delivered += float(np.logical_and(pred, gt).sum())
            S_delivered += float(np.logical_or(pred, gt).sum())
        # label band + coverage structure
        inb = np.logical_and(iou >= 0.40, iou <= 0.60).sum()
        band += int(inb)
        n_prop += len(iou)
        # does any single mask cover >= 95% of the GT union area?
        cov = inter / max(gt_mass, 1.0)
        if (cov >= 0.95).any():
            n_cover1 += 1
        # two-object requirement: no single mask covers > 70% of GT
        if not (cov >= 0.70).any():
            two_obj += 1
        S_gt += gt_mass

    n = len(rows)
    out = dict(
        n_targeted=n,
        note=("cumulative targeted IoU (sum I / sum U) over the held-out train "
              "targeted records, per selection over the FULL selector pool"),
        cumulative=dict(
            best_single=I_single / S_single,
            best_pair=I_pair / S_pair,
            greedy_union=I_greedy / S_greedy,
            shipped_threshold_union=(I_delivered / S_delivered if S_delivered else 0.0),
        ),
        label_pathology=dict(
            proposals_total=n_prop,
            in_iou_band_040_060_pct=100.0 * band / max(1, n_prop),
            rows_with_a_ge95pct_single_mask_pct=100.0 * n_cover1 / n,
            rows_where_no_single_mask_covers_70pct_pct=100.0 * two_obj / n,
        ),
        interpretation=("if best_pair >> best_single and shipped << best_single, the union "
                        "construction is the defect and a learned pair/union selector is "
                        "the model change to build next"),
        generated="exp77_pair_headroom.py")
    (OUTF / "gref_pair_headroom.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["cumulative"], indent=1))
    print(json.dumps(out["label_pathology"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
