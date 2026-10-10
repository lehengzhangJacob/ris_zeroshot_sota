#!/usr/bin/env python
"""EXP-128 — Qwen2.5-VL-7B as the per-term box source (reopen trigger #2, part A).

The clean Qwen2.5-VL-7B (models/qwen2.5-vl-7b-clean, sha256-verified, functional
smoke test) reopens the stronger-grounder route (gref_wall.json
terminal_declaration trigger 2). Pre-registered first step: bound the new
grounder's box quality on the two-object held-out slice BEFORE any val pass or
300-trial study.

Same protocol as exp106 (GDINO 0.7046 top-1 / 71.6% hit; LLaVA-OneVision
reference 0.580 / 63.5%): the 262-row val-matched held-out sample, 487 unique
(file, term) pairs; one Qwen box per term (vlm_boxes PROMPT), scored as best
IoU against the best GT object box. Queries cache in exp/out_vlm_boxes under
the "qwen2.5vl|" namespace, so reruns and follow-ups are free re-analyses.

  QWEN_MODEL_ID=/home/msj_team/Jacob/models/qwen2.5-vl-7b-clean \
  CUDA_VISIBLE_DEVICES=1 python exp/exp128_qwen_terms.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp83_two_box_oracle import object_masks  # noqa: E402
from exp105_box_ensemble import best_object_iou_box  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    gd = GreferDataset(REPRO / "data/refer", "train")
    ref_by_id = {x["ref_id"]: x for x in gd.refs}
    pairs = list(dict.fromkeys((r["file_name"], t)
                               for r in sample for t in r["terms"]))
    rid_of = {(r["file_name"], t): r["ref_id"] for r in sample for t in r["terms"]}
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"unique (file, term) pairs: {len(pairs)}", flush=True)

    from vlm_boxes import Qwen2Boxer
    boxer = Qwen2Boxer(device="cuda:0")

    ious, hits, failed = [], [], 0
    t0 = time.time()
    for k, (fn, term) in enumerate(pairs):
        try:
            box, raw = boxer.box(fn, term)
        except Exception as e:  # noqa: BLE001 — keep going, one bad query is not fatal
            print(f"  ERR {fn} {term!r}: {e}", flush=True)
            failed += 1
            continue
        rid = rid_of[(fn, term)]
        objs = object_masks(gd, ref_by_id[rid]) if rid in ref_by_id else []
        iou = best_object_iou_box(box, objs)
        ious.append(float(iou or 0.0))
        hits.append(1.0 if (iou or 0.0) >= 0.5 else 0.0)
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(pairs)} ({el:.0f}s, {60*(k+1)/max(el,1):.0f}/min) "
                  f"mean_iou_so_far={np.mean(ious):.4f}", flush=True)

    out = dict(unique_pairs=len(pairs), pairs_measured=len(ious), failed=failed,
               qwen_box_iou_mean=float(np.mean(ious)) if ious else 0.0,
               qwen_hit50_pct=100.0 * float(np.mean(hits)) if hits else 0.0,
               llava_reference=dict(box_iou_mean=0.580, hit50_pct=63.5),
               gdino_reference=dict(top1_box_iou_mean=0.7046, top1_hit50_pct=71.6,
                                    oracle_box_iou_mean=0.8419),
               gt_box_rule_ceiling=0.802,
               generated="exp128_qwen_terms.py")
    (OUTF / "gref_qwen_terms.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
