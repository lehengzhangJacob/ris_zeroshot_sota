#!/usr/bin/env python
"""EXP-49 — is the VLM box or the selection rule the bottleneck?

Diagnostic only (no rule selection happens here).  On targeted records with a
cached box it reports:

  box_iou        IoU(VLM box, GT union)          — is the box right?
  setcover       mean IoU of the deployed greedy box-cover union
  orc_pool       greedy IoU oracle over the SAME shortlist, box-free
  orc_inbox      greedy IoU oracle restricted to masks mostly inside the box
  orc_full       greedy IoU oracle over the full pool (exp31 number)

orc_inbox is the ceiling any box-conditioned RULE can reach when the box is
correct; comparing it with orc_pool separates "the box is wrong" from "the
rule is weak".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gref_setselect import (load_masks, select_set, candidates, iou_iu, rank01)  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")


def box_for(fn, ph):
    key = hashlib.sha1(f"{MODEL_TAG}|{fn}|{ph.lower()}".encode()).hexdigest()
    p = CACHE_DIR / (key + ".json")
    return (json.loads(p.read_text())["box"] if p.is_file() else None), p.is_file()


def greedy_oracle(masks, cand, gt, max_steps=6):
    cur = np.zeros_like(gt)
    best = 0.0
    for _ in range(max_steps):
        bi, biou, bu = -1, best, None
        for j in range(len(cand)):
            u = cur | masks[cand[j]]
            iou = np.logical_and(u, gt).sum() / max(1, np.logical_or(u, gt).sum())
            if iou > biou:
                biou, bi, bu = iou, j, u
        if bi < 0:
            break
        cur = bu
        best = biou
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-rank", type=int, default=15)
    ap.add_argument("--gem-mix", type=float, default=1.0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.records)]
    rng = random.Random(args.seed)
    nt = [r for r in recs if r["no_target"]]
    tg = [r for r in recs if not r["no_target"]]
    sample = (rng.sample(nt, args.n_per_class) + rng.sample(tg, args.n_per_class)
              if args.n_per_class else nt + tg)
    gd = GreferDataset(REPRO / "data/refer", "val")
    by_id = {r["ref_id"]: r for r in gd.refs}

    params = dict(lam=0.5, max_k=4, k_rank=args.k_rank, gem_mix=args.gem_mix,
                  dec_source="sc_x_gem", min_gain=0.0)
    rows = []
    for i, r in enumerate(sample):
        if r["no_target"]:
            continue
        gt = gd.ground_truth(by_id[r["ref_id"]])
        masks = load_masks(r["cache_file"])
        ph = (r["parsed"].get("sentence") or "").strip()
        box, cached = box_for(r["file_name"], ph)
        if not cached or box is None:
            continue
        H, W = masks.shape[1], masks.shape[2]
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        box_mask = np.zeros((H, W), bool)
        if x1 > x0 and y1 > y0:
            box_mask[y0:y1, x0:x1] = True
        bi = np.logical_and(box_mask, gt).sum() / max(1, np.logical_or(box_mask, gt).sum())

        cand = np.asarray(candidates(r, params)[0])
        idxs, _ = select_set(r, masks, box, params)
        _, _, iou_sc = iou_iu(masks, idxs, gt)
        orc_pool = greedy_oracle(masks, cand, gt)
        # masks mostly inside the box (>=60% of the mask lies in the box)
        def in_box(bb):
            bx0, by0, bx1, by1 = [int(round(v)) for v in bb]
            bx0, by0 = max(0, bx0), max(0, by0)
            bx1, by1 = min(W, bx1), min(H, by1)
            if bx1 <= bx0 or by1 <= by0:
                return np.array([], int)
            f = masks[cand][:, by0:by1, bx0:bx1].sum((1, 2)) / np.maximum(
                masks[cand].sum((1, 2)), 1)
            return cand[f >= 0.6]

        inbox = in_box(box)
        orc_inbox = greedy_oracle(masks, inbox, gt) if len(inbox) else 0.0
        # reference ceiling: a PERFECT box (the GT union's bounding box)
        ys, xs = np.nonzero(gt)
        gtbox = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        gtb = in_box(gtbox)
        orc_in_gtbox = greedy_oracle(masks, gtb, gt) if len(gtb) else 0.0
        rows.append(dict(box_iou=bi, setcover=iou_sc, orc_pool=orc_pool,
                         orc_inbox=orc_inbox, orc_in_gtbox=orc_in_gtbox,
                         n_cand=len(cand), n_inbox=len(inbox)))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)

    a = {k: float(np.mean([r[k] for r in rows])) for k in
         ("box_iou", "setcover", "orc_pool", "orc_inbox", "orc_in_gtbox")}
    print(f"\nn_targeted={len(rows)}")
    for k, v in a.items():
        print(f"  {k:10s} {v:.4f}")
    out = dict(n=len(rows), means=a, k_rank=args.k_rank)
    Path("exp/out_final/gref_box_ceiling.json").write_text(json.dumps(out, indent=2))
    print("wrote exp/out_final/gref_box_ceiling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
