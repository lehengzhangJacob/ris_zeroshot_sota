#!/usr/bin/env python
"""EXP-102 — the termbox rule's MEAN targeted IoU under per-object GT boxes.

The wall's stronger-localizer closure still quotes the ensemble-era
threshold-union numbers (required m 0.6432; perfect-box rule m 0.547), which
are stale for the shipped termbox rule (required m 0.6212, delivered 0.5228)
and never measured the TERMBOX rule under GT boxes at all.  This measures, on
the val-matched held-out population (262 rows), with the shipped picker and
the shipped top-2 margin:

  m_llava : mean targeted IoU of the shipped rule with LLaVA per-term boxes
  m_gtbox : same with per-object GT boxes (the perfect-localizer mean)
  r_gtbox : cumulative counterpart (for continuity with exp94)

If m_gtbox clears the required 0.6212, the gIoU bind is localizer quality
(box-bound), not the construction; if not, it is foreclosed even with a
perfect grounder.  Held-out only.

    python exp/exp102_termbox_perfectbox_m.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp78_pair_selector import precompute_row  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402
from exp83_two_box_oracle import object_masks  # noqa: E402
from exp84_nounbox_sample import npath  # noqa: E402
from exp89_termbox_full_val import term_features  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def main() -> int:
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")
    ref_by_id = {x["ref_id"]: x for x in gd.refs}

    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    rows = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]

    Xf, yf = [], []
    for rid, rec in recs_by_rid.items():
        r = dedup.get(rid)
        if r is None:
            continue
        iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
        for F in term_features(r, rec):
            Xf.append(F); yf.append((iou_gt >= 0.5).astype(np.float32))
    pick_clf = make_clf("logreg", dict(C=g9["C"]))
    pick_clf.fit(np.concatenate(Xf), np.concatenate(yf))

    acc = dict(li=0.0, lu=0.0, gi=0.0, gu=0.0, lm=0.0, gm=0.0, n=0)
    for r in rows:
        rec = recs_by_rid[r["rid"]]
        masks, gt = r["masks"], r["gt"]

        def pick_in_box(box):
            x0, y0, x1, y1 = box
            F_t = None
            return None

        def best_pick(box):
            x0, y0, x1, y1 = box
            area_box = float((x1 - x0) * (y1 - y0))
            areas = masks.sum((1, 2)).astype(np.float64)
            inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
            bxi = inb / (areas + area_box - inb + 1e-9)
            n = len(gt if False else masks)
            F = None
            return None

        # shipped picker per term with LLaVA boxes
        picks_l = []
        for F_t in term_features(r, rec):
            pt = pick_clf.predict_proba(F_t)[:, 1]
            order = np.argsort(pt)[::-1]
            picks_l.append(int(order[0]))
            if g9["topk"] == 2 and len(order) > 1 and pt[order[1]] >= g9["margin"] * pt[order[0]]:
                picks_l.append(int(order[1]))
        # GT per-object boxes: for each object, mimic the same picker by
        # building the per-term feature rows against the GT box
        picks_g = []
        for om in object_masks(gd, ref_by_id[r["rid"]]):
            ys, xs = np.nonzero(om)
            if not len(ys):
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max() + 1), int(ys.max() + 1)
            area_box = float((x1 - x0) * (y1 - y0))
            areas = masks.sum((1, 2)).astype(np.float64)
            inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
            bxi = inb / (areas + area_box - inb + 1e-9)
            inbf = inb / np.maximum(areas, 1.0)
            n = len(areas)
            F = np.stack([
                r["_bs"], bxi, inbf, areas / masks[0].size,
                np.argsort(np.argsort(r["_bs"])) / max(1, n - 1),
                np.full(n, 0, np.float32), np.full(n, 1, np.float32),
                np.full(n, bxi.max(), np.float32),
            ], 1).astype(np.float32)
            pt = pick_clf.predict_proba(F)[:, 1]
            order = np.argsort(pt)[::-1]
            picks_g.append(int(order[0]))
            if g9["topk"] == 2 and len(order) > 1 and pt[order[1]] >= g9["margin"] * pt[order[0]]:
                picks_g.append(int(order[1]))

        def cum(picks, key):
            if not picks:
                return
            m = masks[picks[0]].copy()
            for p_ in picks[1:]:
                m = np.logical_or(m, masks[p_])
            I = float(np.logical_and(m, gt).sum())
            U = float(np.logical_or(m, gt).sum())
            acc[key + "i"] += I
            acc[key + "u"] += U
            return I / U if U else 0.0

        iou_l = cum(picks_l, "l")
        iou_g = cum(picks_g, "g")
        if iou_l is not None:
            acc["lm"] += iou_l
        if iou_g is not None:
            acc["gm"] += iou_g
        acc["n"] += 1

    out = dict(
        rows=acc["n"],
        m_llava_box=acc["lm"] / max(1, acc["n"]),
        m_gt_box=acc["gm"] / max(1, acc["n"]),
        r_llava_box=acc["li"] / max(1.0, acc["lu"]),
        r_gt_box=acc["gi"] / max(1.0, acc["gu"]),
        required_m_at_shipped_operating_point=0.6212,
        note=("means are over rows (per-record IoU), the gIoU-relevant m; the shipped "
              "picker + top-2 margin used in both variants"),
        generated="exp102_termbox_perfectbox_m.py")
    (OUTF / "gref_termbox_perfectbox_m.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
