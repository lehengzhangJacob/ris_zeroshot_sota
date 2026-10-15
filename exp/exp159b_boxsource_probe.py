#!/usr/bin/env python
"""EXP-159b — reconcile the quoted box-over-full-pool attribution.

The operator directive quotes box-over-full-pool at 50.20/40.24/49.48 mIoU
(RefCOCO/+/g) while our re-analysis of the shipped configuration (LLaVA
sentence box, iou overlap, argmax over the full pool) measures 69.99/65.23/
64.83.  Candidate explanations to test on full refcoco val: the box phrase
(noun phrase vs full sentence) and the box source (LLaVA-OneVision vs
Qwen2.5-VL).  Deterministic re-analysis over cached records.

    CUDA_VISIBLE_DEVICES=2 python exp/exp159b_boxsource_probe.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_map import build_map, cache_file  # noqa: E402
from vlm_boxes import LlavaBoxer, Qwen2Boxer  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
RECORDS = "exp/out_extreme/records_refcoco.jsonl"


def box_overlap(masks, box, mode="iou"):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = masks.shape[1], masks.shape[2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(len(masks), np.float32)
    a_in = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float32)
    box_area = float((y1 - y0) * (x1 - x0))
    tot = masks.sum((1, 2)).astype(np.float32)
    if mode == "cover":
        return a_in / box_area
    if mode == "inside":
        return a_in / np.maximum(tot, 1)
    return a_in / (tot + box_area - a_in + 1e-9)


def main() -> int:
    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", "refcoco", "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    build_map([ds.images[r["image_id"]]["file_name"] for r in ds.refs])
    recs = [json.loads(l) for l in open(RECORDS)]
    llava = LlavaBoxer.__new__(LlavaBoxer)
    qwen = Qwen2Boxer.__new__(Qwen2Boxer)

    configs = [("llava_sent", llava, "sent"), ("llava_noun", llava, "noun"),
               ("qwen2_sent", qwen, "sent"), ("qwen2_noun", qwen, "noun")]
    acc = {c[0]: 0.0 for c in configs}
    n_used = {c[0]: 0 for c in configs}
    store, order = {}, []

    def masks_of(fn):
        if fn in store:
            return store[fn]
        p = cache_file(fn)
        with np.load(p, allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["mask_shape"])
            m = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                              ).reshape(shape).astype(bool)
        store[fn] = m
        order.append(fn)
        while len(order) > 12:
            store.pop(order.pop(0), None)
        return m

    for i, r in enumerate(recs):
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        masks = masks_of(fn)
        if len(masks) != len(r["gt_iou"]):
            continue
        gt = np.asarray(r["gt_iou"], np.float64)
        for name, boxer, phrase in configs:
            ph = (r["parsed"]["sentence"] if phrase == "sent"
                  else r["parsed"]["noun_phrase"]).strip()
            bp = boxer._path(fn, ph)
            if not bp.is_file():
                continue
            box = json.loads(bp.read_text())["box"]
            if box is None:
                continue
            loc = box_overlap(masks, box, "iou")
            j = int(np.argmax(loc))
            acc[name] += float(gt[j])
            n_used[name] += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)
    out = {}
    for name in acc:
        if n_used[name]:
            out[name] = dict(n=n_used[name], mIoU=round(100.0 * acc[name] / n_used[name], 4))
    print(json.dumps(out, indent=2))
    Path("exp/out_final/head_rebuild_parts/boxsource_probe.json").write_text(
        json.dumps(dict(dataset="refcoco", probe=out, generated="exp159b_boxsource_probe.py"),
                   indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
