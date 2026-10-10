#!/usr/bin/env python
"""EXP-86 — which per-term-box PICK RULE realizes the two-box construction?

exp85: per-term LLaVA boxes are good (IoU 0.580 vs their object; 63.5%
select>=0.50) and GT boxes give r 0.764 on the sample, but the bbox-IoU pick
collapses to r 0.416 < the shipped pair rule's 0.482 on the same rows.  The
pick rule is the bottleneck.  This sweeps deployable pick rules on the cached
boxes (no new queries), same 250 sample rows, same union-of-picks structure:

  bxi     : max bbox-IoU vs the term box                  (exp85: 0.416)
  score   : max base-classifier score                     (box ignored)
  sxi     : max base score * bbox-IoU
  inb50   : max base score among masks with in-box >= .5
  inb30   : max base score among masks with in-box >= .3
  bxi50   : max bbox-IoU among masks with bbox-IoU >= .5  (fallback bxi)

Reference points on the same rows: shipped pair selector 0.4819; GT-box
bbox-IoU pick 0.7639 (construction ceiling).

    python exp/exp86_nounbox_pickrules.py
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
from exp84_nounbox_sample import npath  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def main() -> int:
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")

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

    rid_set = {rec["ref_id"] for rec in sample}
    rows = [r for r in tr if r["rid"] in rid_set]
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    print(f"rows {len(rows)}", flush=True)

    RULES = ["bxi", "score", "sxi", "inb50", "inb30", "bxi50"]
    acc = {k: [0.0, 0.0] for k in RULES}
    acc["shipped"] = [0.0, 0.0]
    for r in rows:
        rec = recs_by_rid[r["rid"]]
        masks, gt = r["masks"], r["gt"]
        bs = r["_bs"]
        areas = masks.sum((1, 2)).astype(np.float64)

        # shipped pair reference
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        k = int(np.argmax(pair_clf.predict_proba(F)[:, 1]))
        pred = candidate_mask(masks, ids[k])
        acc["shipped"][0] += float(np.logical_and(pred, gt).sum())
        acc["shipped"][1] += float(np.logical_or(pred, gt).sum())

        picks = {k: [] for k in RULES}
        for term in rec["terms"]:
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
            area = float((x1 - x0) * (y1 - y0))
            inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
            bxi = inb / (areas + area - inb + 1e-9)
            inbf = inb / np.maximum(areas, 1.0)
            picks["bxi"].append(int(np.argmax(bxi)))
            picks["score"].append(int(np.argmax(bs)))
            picks["sxi"].append(int(np.argmax(bs * (0.2 + bxi))))
            m50 = inbf >= 0.5
            picks["inb50"].append(int(np.argmax(np.where(m50, bs, -1))) if m50.any()
                                  else int(np.argmax(bs)))
            m30 = inbf >= 0.3
            picks["inb30"].append(int(np.argmax(np.where(m30, bs, -1))) if m30.any()
                                  else int(np.argmax(bs)))
            b50 = bxi >= 0.5
            picks["bxi50"].append(int(np.argmax(np.where(b50, bxi, -1))) if b50.any()
                                  else int(np.argmax(bxi)))
        for rule, ps in picks.items():
            if not ps:
                continue
            m = masks[ps[0]].copy()
            for j in ps[1:]:
                m = np.logical_or(m, masks[j])
            acc[rule][0] += float(np.logical_and(m, gt).sum())
            acc[rule][1] += float(np.logical_or(m, gt).sum())

    out = {k: round(v[0] / max(1.0, v[1]), 4) for k, v in acc.items()}
    out["_refs"] = {"gt_box_bboxIoU_ceiling_same_rows": 0.7639,
                    "deployed_nounbox_bxi": 0.4161}
    (OUTF / "gref_nounbox_pickrules.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if not k.startswith("_"):
            print(f"{k:9s}: r {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
