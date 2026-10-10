#!/usr/bin/env python
"""EXP-37 — box-prior selection architecture, evaluated offline.

Architecture under test (GLOD-RIS's shape, dense boxes instead of a 3x3 grid):
    S_dec  = released semantic score (or category score)
    S_loc  = proposal <-> box agreement, from an external box for the referent
    select = argmax over top-k(S_dec) of S_loc   (or S_dec * S_loc, or S_loc alone)
then optionally the containment-growth repair.

`--box-source gt` uses the ground-truth mask's bounding box: that is the
CEILING of any box-prior pipeline (a perfect locator), and it is what decides
whether the architecture is worth a VLM/GPU pass.  `jitter` deliberately
degrades the box to measure robustness to localization error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_map import cache_file  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

RECORDS = {
    "refcoco": "exp/out_extreme/records_refcoco.jsonl",
    "refcoco+": "exp/out_extreme/records_refcocoplus.jsonl",
    "refcocog": "exp/out_extreme/records_refcocog.jsonl",
}


class MaskStore:
    """LRU of unpacked masks, keyed by file_name."""

    def __init__(self, cap=12):
        self.cap = cap
        self.d = {}
        self.order = []

    def masks(self, file_name, fn_of=None):
        if file_name in self.d:
            return self.d[file_name]
        with np.load(cache_file(file_name), allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["mask_shape"])
            m = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                              ).reshape(shape).astype(bool)
        if fn_of is not None:
            fn_of[file_name] = m
        self.d[file_name] = m
        self.order.append(file_name)
        while len(self.order) > self.cap:
            self.d.pop(self.order.pop(0), None)
        return m


def box_overlap(masks, box, mode):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = masks.shape[1], masks.shape[2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(len(masks), np.float32)
    sub = masks[:, y0:y1, x0:x1]
    a_mask = sub.sum((1, 2)).astype(np.float32)
    box_area = float((y1 - y0) * (x1 - x0))
    if mode == "cover":            # how much of the box the mask explains
        return a_mask / box_area
    if mode == "inside":           # how much of the mask lies in the box
        tot = masks.sum((1, 2)).astype(np.float32)
        return a_mask / np.maximum(tot, 1)
    if mode == "iou":
        return a_mask / (masks.sum((1, 2)) + box_area - a_mask + 1e-9)
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco+", choices=list(RECORDS))
    ap.add_argument("--box-source", default="gt", choices=["gt", "jitter", "vlm"])
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--mode", default="cover", choices=["cover", "inside", "iou"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", args.dataset, "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(args.records or RECORDS[args.dataset])]
    if args.limit:
        recs = recs[: args.limit]
    vlm = None
    if args.box_source == "vlm":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vlm_boxes import LlavaBoxer
        vlm = LlavaBoxer.__new__(LlavaBoxer)   # cache-only use
    print(f"{len(recs)} records ({args.dataset}), box={args.box_source}, mode={args.mode}", flush=True)

    store = MaskStore()
    rng = np.random.RandomState(0)
    acc, used, names = {}, 0, None
    for i, r in enumerate(recs):
        ref = ref_by_id[r["ref_id"]]
        fn = ds.images[ref["image_id"]]["file_name"]
        masks = store.masks(fn)
        n = len(masks)
        if n != len(r["gt_iou"]):
            continue
        gt = np.asarray(r["gt_iou"], dtype=np.float32)
        sc = np.asarray(r["sc"][:n], dtype=np.float32)
        sc_cat = np.asarray(r["sc_strip"][:n], dtype=np.float32)   # spatial-stripped
        area = np.asarray(r["area"][:n], dtype=np.float32)

        # ---- the box prior -------------------------------------------------
        if args.box_source == "vlm":
            bp = vlm._path(fn, r["parsed"]["noun_phrase"].strip())
            box = json.loads(bp.read_text())["box"] if bp.is_file() else None
            if box is None:
                # no box: fall back to the released selection
                box = None
        else:
            # oracle-proposal bbox = the ceiling of an ideal locator
            orc = int(r["orc"])
            m = masks[orc]
            rows = np.flatnonzero(m.any(1))
            cols = np.flatnonzero(m.any(0))
            if len(rows) == 0:
                continue
            box = [cols[0], rows[0], cols[-1] + 1, rows[-1] + 1]
        if args.box_source == "jitter":
            H, W = m.shape
            bw, bh = box[2] - box[0], box[3] - box[1]
            dx, dy = rng.uniform(-args.jitter, args.jitter) * bw, rng.uniform(-args.jitter, args.jitter) * bh
            sx, sy = 1 + rng.uniform(-args.jitter, args.jitter), 1 + rng.uniform(-args.jitter, args.jitter)
            cx, cy = (box[0] + box[2]) / 2 + dx, (box[1] + box[3]) / 2 + dy
            box = [cx - bw * sx / 2, cy - bh * sy / 2, cx + bw * sx / 2, cy + bh * sy / 2]
        if box is None:
            # no localization: fall back to the released pipeline for every arm
            rel = int(r["final"])
            variants = {k: rel for k in ("loc_only", "loc_cover_x_sc_top3",
                                         "loc_cover_x_sc_top5", "loc_cover_x_sc_top10",
                                         "loc_cover_x_cat_top5", "loc_cover_x_cat_top10")}
            variants["released_final"] = rel
            variants["released_snap"] = int(r["ps_idx"][rel])
            if names is None:
                names = list(variants)
                acc = {k: 0.0 for k in names}
            used += 1
            for k, j in variants.items():
                acc[k] += float(gt[j])
            continue
        loc = box_overlap(masks, box, args.mode)

        def pick(idx):
            return int(idx[int(np.argmax(loc[idx]))])

        variants = {
            "loc_only": int(np.argmax(loc)),
            "loc_cover_x_sc_top3": pick(np.argsort(sc)[::-1][:3]),
            "loc_cover_x_sc_top5": pick(np.argsort(sc)[::-1][:5]),
            "loc_cover_x_sc_top10": pick(np.argsort(sc)[::-1][:10]),
            "loc_cover_x_cat_top5": pick(np.argsort(sc_cat)[::-1][:5]),
            "loc_cover_x_cat_top10": pick(np.argsort(sc_cat)[::-1][:10]),
            "released_final": int(r["final"]),
            "released_snap": int(r["ps_idx"][int(r["final"])]),
        }
        # growth: promote to the smallest strict container
        for tag in ("loc_cover_x_sc_top5", "loc_cover_x_cat_top10"):
            j = variants[tag]
            mj = masks[j]
            cont = [k for k in range(n) if k != j and (masks[k] & mj).sum() == mj.sum()
                    and masks[k].sum() > mj.sum()]
            if cont:
                kbest = min(cont, key=lambda k: masks[k].sum())
                variants[tag + "+grow"] = kbest
            else:
                variants[tag + "+grow"] = j
        if names is None:
            names = list(variants)
            acc = {k: 0.0 for k in names}
        used += 1
        for k, j in variants.items():
            acc[k] += float(gt[j])
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    res = {k: round(100 * v / used, 3) for k, v in acc.items()}
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:26s} {v:7.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(dataset=args.dataset, n=used,
                                               box_source=args.box_source,
                                               mode=args.mode, mIoU=res), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
