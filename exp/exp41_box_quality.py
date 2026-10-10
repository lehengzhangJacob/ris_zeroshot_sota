#!/usr/bin/env python
"""EXP-41 — how good are the VLM's boxes?

Before trusting the box prior, measure it: coverage (how often the VLM answers
with a box), box-vs-GT-mask IoU, and whether a good box is actually what the
selector needs (fraction of records where the mask the box selects has IoU>=0.5
with GT).
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--phrase-source", default="noun", choices=["noun", "sent"])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split,
                    limit=args.limit, seed=args.seed)
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(args.records)]
    boxer = QwenBoxer.__new__(QwenBoxer)

    n = box_ok = none_ans = 0
    ious, cover_hits, miss_examples = [], 0, []
    store = {}

    for r in recs:
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        ph = r["parsed"]["sentence"] if args.phrase_source == "sent" else r["parsed"]["noun_phrase"]
        bp = boxer._path(fn, ph.strip())
        if not bp.is_file():
            continue
        d = json.loads(bp.read_text())
        box = d["box"]
        n += 1
        if box is None:
            none_ans += 1
            continue
        box_ok += 1
        if fn not in store:
            with np.load(cache_file(fn), allow_pickle=False) as z:
                shape = tuple(int(x) for x in z["mask_shape"])
                store[fn] = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                                          ).reshape(shape).astype(bool)
        masks = store[fn]
        gt = np.asarray(r["gt_iou"], np.float32)
        orc = int(r["orc"])
        gm = masks[orc]
        rows = np.flatnonzero(gm.any(1))
        cols = np.flatnonzero(gm.any(0))
        if not len(rows):
            continue
        gbox = np.array([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1], float)
        b = np.asarray(box, float)
        ix0, iy0 = max(b[0], gbox[0]), max(b[1], gbox[1])
        ix1, iy1 = min(b[2], gbox[2]), min(b[3], gbox[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        ua = (b[2] - b[0]) * (b[3] - b[1]) + (gbox[2] - gbox[0]) * (gbox[3] - gbox[1]) - inter
        ious.append(inter / ua if ua > 0 else 0.0)
        # does the box at least select a mask with IoU>=0.5?
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, x1 = max(0, x0), min(masks.shape[2], x1)
        y0, y1 = max(0, y0), min(masks.shape[1], y1)
        if x1 > x0 and y1 > y0:
            sub = masks[:, y0:y1, x0:x1]
            a_in = sub.sum((1, 2)).astype(np.float32)
            cov = a_in / float((y1 - y0) * (x1 - x0))
            j = int(np.argmax(cov))
            if gt[j] >= 0.5:
                cover_hits += 1
            elif len(miss_examples) < 5:
                miss_examples.append(dict(sent=r["sent"][:90], box=[round(v) for v in box],
                                          best_iou=round(float(gt[j]), 3)))

    res = dict(
        n_checked=n, with_box=box_ok, none_answer=none_ans,
        coverage=round(100 * box_ok / max(1, n), 2),
        none_rate=round(100 * none_ans / max(1, n), 2),
        box_iou_mean=round(100 * float(np.mean(ious)), 2) if ious else None,
        box_iou_ge50=round(100 * float(np.mean(np.asarray(ious) >= 0.5)), 2) if ious else None,
        box_selects_ge50=round(100 * cover_hits / max(1, box_ok), 2),
        miss_examples=miss_examples,
    )
    print(json.dumps(res, indent=2))
    if args.output:
        args.output.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
