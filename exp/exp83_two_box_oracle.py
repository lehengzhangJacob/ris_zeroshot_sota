#!/usr/bin/env python
"""EXP-83 — does the two-object-box construction have headroom? (GT-only, no queries)

The union is 2 objects, and the localizer is far better at single-object boxes
(RefCOCO-family: 62% of boxes select a mask with IoU>=0.5) than at union boxes
(gRefCOCO: IoU 0.329).  So: give EACH object its own box, select a mask inside
each box independently, output the union.  Before spending any VLM queries, this
measures the construction's ceiling from GT alone, on the held-out train
targeted rows:

  r_two_box_oracle   : per object, the pool mask with max IoU vs THAT object;
                       union of picks. The construction's oracle ceiling.
  r_two_box_boxiou   : per object, the pool mask with max bbox-IoU vs the
                       object's GT box (a deployable feature given a perfect
                       per-object box). What the CURRENT selector feature would
                       reach under perfect boxes.
  r_shipped_calib    : the shipped pair selector's calib r (0.5538) as the bar.

If the oracle ceiling is far above the shipped r, the two-box construction is
worth a per-noun-box extraction; if it is ~the same, the construction is dead.

    python exp/exp83_two_box_oracle.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import load_record, make_clf, score_rows  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def object_masks(gd, ref):
    """Per-annotation masks at stride 2 (same decode path as ground_truth)."""
    from pycocotools import mask as mask_utils
    image = gd.images[ref["image_id"]]
    h, w = image["height"], image["width"]
    out = []
    for ann_id in gd.target_ann_ids(ref):
        segmentation = gd.annotations[ann_id]["segmentation"]
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, h, w))
        elif isinstance(segmentation["counts"], list):
            rle = mask_utils.frPyObjects(segmentation, h, w)
        else:
            rle = segmentation
        m = mask_utils.decode(rle).astype(bool)
        out.append(m[::STRIDE, ::STRIDE])
    return out


def main() -> int:
    st = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    t0 = time.time()
    gd = GreferDataset(REPRO / "data/refer", "train")
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    rows = []
    for line in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        r = load_record(rec, gd, by_id, mc, gc)
        if r is None:
            continue
        ref = by_id[r["rid"]]
        objs = object_masks(gd, ref)
        if not objs:
            continue
        r["objs"] = objs
        r["n_obj"] = len(objs)
        rows.append(r)
    clf = make_clf("logreg", dict(C=st["C"]))
    clf.fit(np.concatenate([r["X"] for r in rows]),
            np.concatenate([r["y"] for r in rows]))
    score_rows(rows, clf)
    print(f"targeted rows {len(rows)} ({time.time()-t0:.0f}s) "
          f"multi-obj {sum(1 for r in rows if r['n_obj']>1)}", flush=True)

    acc = dict(orcI=0.0, orcU=0.0, bxiI=0.0, bxiU=0.0, orcSI=0.0, orcSU=0.0)
    for r in rows:
        masks, gt = r["masks"], r["gt"]
        picks_o, picks_b, picks_s = [], [], []
        for om in r["objs"]:
            ys, xs = np.nonzero(om)
            if not len(ys):
                continue
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            box_area = float((y1 - y0) * (x1 - x0))
            inter_gt = np.logical_and(masks, om).sum((1, 2)).astype(np.float64)
            union_gt = np.logical_or(masks, om).sum((1, 2)).astype(np.float64)
            iou_obj = inter_gt / np.maximum(union_gt, 1.0)
            inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
            bxi = inb / (masks.sum((1, 2)) + box_area - inb + 1e-9)
            picks_o.append(int(np.argmax(iou_obj)))          # per-object oracle
            picks_b.append(int(np.argmax(bxi)))              # box-bbox IoU pick
            picks_s.append(int(np.argmax(inb)))              # in-box mass pick
        if not picks_o:
            continue
        def cum(picks, key):
            m = masks[picks[0]].copy()
            for p in picks[1:]:
                m = np.logical_or(m, masks[p])
            acc[key + "I"] += float(np.logical_and(m, gt).sum())
            acc[key + "U"] += float(np.logical_or(m, gt).sum())
        cum(picks_o, "orc")
        cum(picks_b, "bxi")
        cum(picks_s, "orcS")

    out = dict(
        measured_on="held-out train targeted rows, pool = full selector pool",
        two_box_oracle_r=acc["orcI"] / max(1.0, acc["orcU"]),
        two_box_boxiou_pick_r=acc["bxiI"] / max(1.0, acc["bxiU"]),
        two_box_inboxmass_pick_r=acc["orcSI"] / max(1.0, acc["orcSU"]),
        shipped_pair_calib_r=0.5538,
        single_box_oracle_note="exp81/exp77: best-single 0.696, best-pair 0.825 (oracle-ranked)",
        interpretation=("oracle ceiling of the per-object-box construction vs the shipped "
                        "calib r 0.5538; the deployable-pick variants show what the current "
                        "box features would reach GIVEN perfect per-object boxes"),
        generated="exp83_two_box_oracle.py")
    (OUTF / "gref_two_box_oracle.json").write_text(json.dumps(out, indent=2))
    print(f"two-box oracle r   {out['two_box_oracle_r']:.4f}")
    print(f"two-box bboxIoU r  {out['two_box_boxiou_pick_r']:.4f}")
    print(f"two-box inbox r    {out['two_box_inboxmass_pick_r']:.4f}")
    print(f"(shipped pair calib r 0.5538)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
