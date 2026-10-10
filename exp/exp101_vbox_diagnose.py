#!/usr/bin/env python
"""EXP-101 — LLaVA-Video per-term box quality vs LLaVA-OneVision.

Same measurements as exp85/86, on the val-matched sample (250 matched rows):
  (a) per-term box vs best-matching object: mean IoU, % >= 0.50
      (LLaVA-OneVision reference: 0.580 / 63.5%)
  (b) deployed proxy: per-term vbox -> pool mask with max bbox-IoU vs the box
      -> union.  LLaVA-OneVision reference: sxi (base x bbox-IoU) 0.5187,
      bbox-IoU 0.4161, shipped rule 0.4819 on 250 rows.

    python exp/exp101_vbox_diagnose.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp83_two_box_oracle import object_masks  # noqa: E402
from exp100_vbox_sample import vpath  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def main() -> int:
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
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
    print(f"rows {len(rows)}", flush=True)

    ious, sel50 = [], []
    acc = dict(bi=0.0, bu=0.0, si=0.0, su=0.0, ri=0.0, ru=0.0)
    for r in rows:
        rec = recs_by_rid[r["rid"]]
        masks, gt = r["masks"], r["gt"]
        objs = object_masks(gd, ref_by_id[r["rid"]]) if r["rid"] in ref_by_id else []
        picks_b, picks_s, picks_r = [], [], []
        for term in rec["terms"]:
            p = vpath(rec["file_name"], term)
            if not p.is_file():
                continue
            d = json.loads(p.read_text())
            box = d.get("box")
            if box is None:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
            x1, y1 = min(masks.shape[2], -(-x1 // STRIDE)), min(masks.shape[1], -(-y1 // STRIDE))
            if x1 <= x0 or y1 <= y0:
                continue
            area = float((x1 - x0) * (y1 - y0))
            areas = masks.sum((1, 2)).astype(np.float64)
            inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
            bxi = inb / (areas + area - inb + 1e-9)
            j = int(np.argmax(bxi))
            picks_b.append(j)
            picks_r.append(int(np.argmax(r["_bs"] * (0.2 + bxi))))
            picks_s.append(j)
            best = 0.0
            for om in objs:
                inter = float(np.logical_and(masks[j], om).sum())
                uni = float(np.logical_or(masks[j], om).sum())
                best = max(best, inter / max(1.0, uni))
            ious.append(best)
            sel50.append(1.0 if best >= 0.5 else 0.0)
        def cum(picks, key):
            if not picks:
                return
            m = masks[picks[0]].copy()
            for p_ in picks[1:]:
                m = np.logical_or(m, masks[p_])
            acc[key + "i"] += float(np.logical_and(m, gt).sum())
            acc[key + "u"] += float(np.logical_or(m, gt).sum())
        cum(picks_b, "b")
        cum(picks_s, "s")
        cum(picks_r, "r")

    out = dict(
        sample=f"{len(rows)} val-matched rows",
        vbox_box_vs_best_object_iou_mean=float(np.mean(ious)) if ious else 0.0,
        vbox_box_select_ge50_pct=100.0 * float(np.mean(sel50)) if sel50 else 0.0,
        n_term_picks=len(ious),
        r_vbox_bboxiou_on_sample=acc["bi"] / max(1.0, acc["bu"]),
        r_vbox_sxi_on_sample=acc["ri"] / max(1.0, acc["ru"]),
        llava_ov_reference=dict(box_iou_mean=0.580, select_ge50_pct=63.5,
                                sxi_r_250rows=0.5187, bboxiou_r_250rows=0.4161,
                                shipped_rule_250rows=0.4819),
        generated="exp101_vbox_diagnose.py")
    (OUTF / "gref_vbox_sample.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
