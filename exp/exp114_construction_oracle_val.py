#!/usr/bin/env python
"""EXP-114 — the GT-box SAM construction oracle, measured on FULL VAL.

The corrected cIoU framing (after review) brackets two oracles:
  pool selection oracle   r 0.8842 (measured, exp77-era) — requirement 0.8305
                          is BELOW it: realization-bound relative to the pool;
  construction oracle     previously EXTRAPOLATED (~0.82 r from exp103's
                          mean-IoU 0.885) and stated as below the requirement:
                          foreclosed relative to the measured construction.
The second bracket is load-bearing for the paper's structural claim ("the
construction family itself would have to change") and rests on an
extrapolation.  This measures it directly: for every val targeted record,
per-object GT boxes -> SAM's 3 outputs each -> best union (over the 3^n
combinations, n<=3) vs the record GT, accumulated through the same mass
identity as exp109b.  No parameters are selected here; val is read for
measurement only, and nothing downstream re-tunes on it.

Reports the construction's cIoU ceiling directly comparable to the floor:
  cIoU_orc = 100 * sum(I) / (sum(union(pred,GT)) + FP_U + A_U)
with FP_U/A_U held at the shipped rule's measured values (best case: the
abstention axis unchanged).  If this lands below 68.94, the construction
foreclosure is measured; if above, it flips back to realization-bound.

    CUDA_VISIBLE_DEVICES=1 python exp/exp114_construction_oracle_val.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp83_two_box_oracle import object_masks  # noqa: E402
from exp103_sam_boxp import downsample  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import IMAGES  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="gref_construction_oracle.json")
    args = ap.parse_args()

    gd = GreferDataset(REPRO / "data/refer", "val")
    by_id = {r["ref_id"]: r for r in gd.refs}
    jobs = []
    for line in open('exp/out_extreme/records_grefcoco.jsonl'):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        ref = by_id.get(rec["ref_id"])
        if ref is None:
            continue
        jobs.append((rec["ref_id"], rec["file_name"], ref))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"val targeted records: {len(jobs)}, unique images: {len({j[1] for j in jobs})}",
          flush=True)

    import torch
    from PIL import Image
    sys.path.insert(0, str(REPRO))
    from glod.vision import load_upstream
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](
        checkpoint=str(REPRO / "checkpoints/sam_vit_h_4b8939.pth")).to("cuda").eval()
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    acc = dict(I=0.0, U=0.0, P=0.0, gt=0.0, n=0, missing=0)
    mean_iou = 0.0
    last = None
    t0 = time.time()
    for k, (rid, fn, ref) in enumerate(jobs):
        objs = object_masks(gd, ref)
        gt = gd.ground_truth(ref)[::STRIDE, ::STRIDE]
        if not objs or gt is None:
            acc["missing"] += 1
            continue
        if fn != last:
            predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
            last = fn
        per_obj = []
        for om in objs[:3]:
            ys, xs = np.nonzero(om)
            if not len(ys):
                continue
            b = np.asarray([xs.min() * STRIDE, ys.min() * STRIDE,
                            (xs.max() + 1) * STRIDE, (ys.max() + 1) * STRIDE], np.float32)
            ms, _, _ = predictor.predict(box=b, multimask_output=True)
            per_obj.append([downsample(m.astype(bool)) for m in ms])
        if not per_obj:
            acc["missing"] += 1
            continue
        best = (0.0, None)
        for combo in itertools.product(*[range(len(p)) for p in per_obj]):
            m = per_obj[0][combo[0]].copy()
            for p, ci in zip(per_obj[1:], combo[1:]):
                m |= p[ci]
            I = float(np.logical_and(m, gt).sum())
            U = float(np.logical_or(m, gt).sum())
            v = I / U if U else 0.0
            if v > best[0]:
                best = (v, (I, U, float(m.sum())))
        I, U, P = best[1]
        acc["I"] += I
        acc["U"] += U
        acc["P"] += P
        acc["gt"] += float(gt.sum())
        mean_iou += best[0]
        acc["n"] += 1
        if (k + 1) % 250 == 0:
            print(f"  {k+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    ship = json.load(open(OUTF / "gref_gdino_full_val.json"))["metrics"]
    FP_U = ship["masses"]["FP_U"]
    A_U = ship["masses"]["A_U"]
    G_U = ship["masses"]["G_U"]
    r_orc = acc["I"] / acc["U"] if acc["U"] else 0.0
    cIoU_orc = 100.0 * acc["I"] / (acc["U"] + FP_U + A_U) if acc["U"] else 0.0
    out = dict(
        n_records=acc["n"], missing=acc["missing"],
        construction="per-object GT box -> SAM 3 masks -> best union over 3^n combos",
        mean_targeted_IoU_oracle=mean_iou / max(1, acc["n"]),
        r_construction_oracle=r_orc,
        cIoU_construction_oracle=cIoU_orc,
        floor=68.94,
        r_required_measured=0.8305,
        cIoU_requirement_masses=dict(FP_U=FP_U, A_U=A_U, G_U=G_U),
        verdict=("construction foreclosure MEASURED (cIoU oracle %.2f < 68.94)" % cIoU_orc
                 if cIoU_orc < 68.94 else
                 "construction ceiling ABOVE the floor (%.2f) - cIoU flips to realization-bound "
                 "for the construction too" % cIoU_orc),
        note=("FP_U/A_U held at the shipped rule's measured values (best case: abstention unchanged); "
              "oracle uses at most 3 GT objects per record"),
        generated="exp114_construction_oracle_val.py")
    (OUTF / args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
