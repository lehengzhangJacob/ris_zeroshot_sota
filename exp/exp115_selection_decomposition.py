#!/usr/bin/env python
"""EXP-115 — where the selection loss lives: the 2x2 decomposition (held-out).

The corrected cIoU framing says "the training-free gap is one of selection
quality" (both oracles ~0.88 r; requirement ~94% of either; delivered ~64%).
That claim is ceiling arithmetic; this attaches mechanism on the 262-row
val-matched held-out sample: per record, {delivered boxes, per-object GT
boxes} x {shipped pick, oracle pick over the same candidates}, reporting
row-mean and per-ref mean targeted IoU.

  shipped pick  = sxi pool pick (top-2 margin) in the box + SAM all-3 masks
                  prompted by the box  (the delivered mask choice)
  oracle pick   = per term, the single candidate (pool mask inside the box, or
                  one of SAM's 3 outputs) with the best GT IoU; union of the
                  per-term bests  (the delivered mask decision made perfectly)

Held-out only; no val reads.

    CUDA_VISIBLE_DEVICES=1 python exp/exp115_selection_decomposition.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402
from exp83_two_box_oracle import object_masks  # noqa: E402
from exp103_sam_boxp import downsample  # noqa: E402
from exp105_box_ensemble import term_features_box  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import IMAGES  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def main() -> int:
    study = json.loads((OUTF / "optuna_study_gref12.json").read_text())
    cfg = dict(study["best_params"])
    cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([x["X"] for x in tr]),
                 np.concatenate([x["y"] for x in tr]))
    for x in tr:
        x["_bs"] = clf_base.predict_proba(x["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    rows = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]
    gd = GreferDataset(REPRO / "data/refer", "train")
    ref_by_id = {x["ref_id"]: x for x in gd.refs}

    from gdino import Gdino
    g = Gdino(device="cuda:0")
    import torch
    from PIL import Image
    sys.path.insert(0, str(REPRO))
    from glod.vision import load_upstream
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](
        checkpoint=str(REPRO / "checkpoints/sam_vit_h_4b8939.pth")).to("cuda").eval()
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    def sam3(box):
        ms, _, _ = predictor.predict(box=np.asarray(box, np.float32), multimask_output=True)
        return [downsample(m.astype(bool)) for m in ms]

    def tocand(box):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
        x1, y1 = min(320, -(-x1 // STRIDE)), min(240, -(-y1 // STRIDE))
        return x0, y0, x1, y1

    last = None
    for r in rows:
        rec = recs_by_rid[r["rid"]]
        fn = rec["file_name"]
        if fn != last:
            predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
            last = fn
        objs = object_masks(gd, ref_by_id[r["rid"]]) if r["rid"] in ref_by_id else []
        gt_boxes = []
        for om in objs[:3]:
            ys, xs = np.nonzero(om)
            if len(ys):
                gt_boxes.append((xs.min() * STRIDE, ys.min() * STRIDE,
                                 (xs.max() + 1) * STRIDE, (ys.max() + 1) * STRIDE))
        terms = []
        for ti, term in enumerate(rec["terms"]):
            boxes, scores = g.detect(fn, term)
            ll = None
            p = lpath(fn, term)
            if p.is_file():
                ll = json.loads(p.read_text()).get("box")
            db = ll if ll is not None else (boxes[0] if boxes else None)
            terms.append(dict(llava=ll,
                              gdino=boxes[0] if boxes else None,
                              dsam=sam3(db) if db is not None else None,
                              gtsam=sam3(gt_boxes[ti]) if ti < len(gt_boxes) else None))
        r["_terms2"] = dict(terms=terms, gt_boxes=gt_boxes)
    print(f"precompute done ({time.time()-t0:.0f}s)", flush=True)

    def sxi_pick(r, box):
        F, _ = box_feats(r["masks"], r["_bs"], box)
        if F is None:
            return []
        sc = F[:, 0] * (cfg["alpha"] + cfg["beta"] * F[:, 1])
        order = np.argsort(sc)[::-1]
        js = [int(order[0])]
        if cfg["topk"] == 2 and len(order) > 1 and sc[order[1]] >= cfg["margin"] * sc[order[0]]:
            js.append(int(order[1]))
        return js

    def delivered_box(r, tb):
        return tb["llava"] if tb["llava"] is not None else tb["gdino"]

    def cell(r, box_mode, pick_mode):
        """box_mode: 'del'|'gt'; pick_mode: 'shipped'|'oracle'."""
        gt = r["gt"]
        parts = []
        t2 = r["_terms2"]
        for ti, tb in enumerate(t2["terms"]):
            box = delivered_box(r, tb) if box_mode == "del" else \
                (t2["gt_boxes"][ti] if ti < len(t2["gt_boxes"]) else None)
            if box is None:
                continue
            sam_m = tb["gtsam"] if box_mode == "gt" else tb["dsam"]
            sam_m = [m for m in (sam_m or []) if m.sum() / m.size <= cfg["sam_area_cap"]]
            if pick_mode == "shipped":
                js = sxi_pick(r, box)
                parts.extend(r["masks"][j] for j in js)
                parts.extend(sam_m)
            else:
                x0, y0, x1, y1 = tocand(box)
                cand = []
                if x1 > x0 and y1 > y0:
                    inbox = r["masks"][:, y0:y1, x0:x1].sum((1, 2))
                    for i in np.flatnonzero(inbox > 0):
                        cand.append(r["masks"][int(i)])
                cand.extend(sam_m)
                if not cand:
                    continue
                best, bv = None, -1.0
                for m in cand:
                    I = float(np.logical_and(m, gt).sum())
                    U = float(np.logical_or(m, gt).sum())
                    v = I / U if U else 0.0
                    if v > bv:
                        best, bv = m, v
                parts.append(best)
        if not parts:
            return None
        m = parts[0].copy()
        for x in parts[1:]:
            m |= x
        I = float(np.logical_and(m, gt).sum())
        U = float(np.logical_or(m, gt).sum())
        return I / U if U else 0.0

    out = {}
    for box_mode in ("del", "gt"):
        for pick_mode in ("shipped", "oracle"):
            key = f"{box_mode}_{pick_mode}"
            vals, rids = [], []
            for r in rows:
                v = cell(r, box_mode, pick_mode)
                if v is None:
                    continue
                vals.append(v)
                rids.append(r["rid"])
            v = np.asarray(vals)
            per_rid = {}
            for rid, val in zip(rids, vals):
                per_rid.setdefault(rid, []).append(val)
            pr = np.array([np.mean(x) for x in per_rid.values()])
            out[key] = dict(row_mean=float(v.mean()), per_rid_mean=float(pr.mean()),
                            n=len(v))
            print(f"{key:14s} row-mean {v.mean():.4f}  per-rid {pr.mean():.4f}", flush=True)

    out["decomposition"] = dict(
        box_loss_row=out["gt_shipped"]["row_mean"] - out["del_shipped"]["row_mean"],
        mask_loss_row=out["del_oracle"]["row_mean"] - out["del_shipped"]["row_mean"],
        joint=out["gt_oracle"]["row_mean"] - out["del_shipped"]["row_mean"],
        note=("box_loss = GT boxes with the shipped pick; mask_loss = shipped boxes with the "
              "oracle pick; oracle pick = per-term best (pool-in-box u SAM) candidate by GT IoU"))
    (OUTF / "gref_selection_decomposition.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
