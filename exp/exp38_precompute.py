#!/usr/bin/env python
"""EXP-38 — flatten a records file + box cache into arrays for fast tuning.

Optuna needs thousands of cheap evaluations; loading masks per trial is not
viable.  This pass computes, once per record: the per-proposal box-overlap
vectors for every overlap mode, plus every score vector the tuner may use.
Ragged per-record arrays are stored concatenated with offsets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_map import cache_file  # noqa: E402
from vlm_boxes import QwenBoxer  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
MODES = ("cover", "inside", "iou")


def box_overlap(masks, box, mode):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = masks.shape[1], masks.shape[2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(len(masks), np.float32)
    sub = masks[:, y0:y1, x0:x1]
    a_in = sub.sum((1, 2)).astype(np.float32)
    box_area = float((y1 - y0) * (x1 - x0))
    tot = masks.sum((1, 2)).astype(np.float32)
    if mode == "cover":
        return a_in / box_area
    if mode == "inside":
        return a_in / np.maximum(tot, 1)
    return a_in / (tot + box_area - a_in + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--phrase-source", default="noun", choices=["noun", "sent"])
    ap.add_argument("--refs-file", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split,
                    limit=args.limit, seed=args.seed)
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    if args.refs_file is not None:
        wanted = set(json.loads(args.refs_file.read_text())["ref_ids"])
        ref_by_id = {k: v for k, v in ref_by_id.items() if k in wanted}
        print(f"restricted to {len(ref_by_id)} refs from {args.refs_file}", flush=True)
    recs = [json.loads(l) for l in open(args.records)]
    boxer = QwenBoxer.__new__(QwenBoxer)     # cache-only use, no model load

    cols = {k: [] for k in ("gt_iou", "sc", "sc_noun", "sc_strip", "gem", "area",
                            "ps_idx", "cx", "cy", "has_box")}
    for m in MODES:
        cols[f"loc_{m}"] = []
        cols[f"locq_{m}"] = []
    offsets = [0]
    store, order = {}, []

    def masks_of(fn):
        if fn in store:
            return store[fn]
        with np.load(cache_file(fn), allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["mask_shape"])
            mm = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                               ).reshape(shape).astype(bool)
        store[fn] = mm
        order.append(fn)
        while len(order) > 10:
            store.pop(order.pop(0), None)
        return mm

    used = skipped = 0
    for r in recs:
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            skipped += 1
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        masks = masks_of(fn)
        n = len(masks)
        if n != len(r["gt_iou"]):
            skipped += 1
            continue
        ph = r["parsed"]["sentence"] if args.phrase_source == "sent" else r["parsed"]["noun_phrase"]
        bp = boxer._path(fn, ph.strip())
        box = None
        if bp.is_file():
            box = json.loads(bp.read_text())["box"]
        from vlm_boxes import Qwen2Boxer
        qb = Qwen2Boxer.__new__(Qwen2Boxer)
        qp = qb._path(fn, ph.strip())
        qbox = json.loads(qp.read_text())["box"] if qp.is_file() else None
        used += 1
        cols["gt_iou"].append(np.asarray(r["gt_iou"], np.float32))
        for k, src in (("sc", "sc"), ("sc_noun", "sc_noun"), ("sc_strip", "sc_strip"),
                       ("gem", "score_gem"), ("area", "area"), ("cx", "cx"), ("cy", "cy"),
                       ("ps_idx", "ps_idx")):
            cols[k].append(np.asarray(r[src][:n], np.float32))
        # per-proposal (not per-record) so the offsets slicing stays uniform
        cols["has_box"].append(np.ones(n, np.float32) * (box is not None))
        cols.setdefault("has_qbox", []).append(np.ones(n, np.float32) * (qbox is not None))
        for m in MODES:
            v = box_overlap(masks, box, m) if box is not None else np.zeros(n, np.float32)
            cols[f"loc_{m}"].append(v)
            vq = box_overlap(masks, qbox, m) if qbox is not None else np.zeros(n, np.float32)
            cols[f"locq_{m}"].append(vq)
        offsets.append(offsets[-1] + n)
        if used % 200 == 0:
            print(f"  {used} used / {skipped} skipped", flush=True)

    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["offsets"] = np.asarray(offsets, np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output}: {used} records, {len(out['gt_iou'])} proposals", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
