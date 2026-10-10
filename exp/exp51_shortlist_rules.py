#!/usr/bin/env python
"""EXP-51 — how much of the deep-shortlist oracle can a deployable rule realize?

exp51 showed the oracle over the top-50 CLIP+GEM shortlist reaches r = 0.881
(localization is NOT the wall at k=50); the question is what rules without GT
can reach.  Diagnostic grid over

    rule  in {box_cover, top2_disjoint, greedy_novel, top2_plain}
    k     in {15, 50}
    box   in {vlm, gt}

reporting targeted mean IoU and cumulative r for each cell.  Any actual rule
SELECTION is deferred to the held-out gRefCOCO train study.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gref_setselect import load_masks, select_set, iou_iu, rank01  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")


def box_for(fn, ph):
    key = hashlib.sha1(f"{MODEL_TAG}|{fn}|{ph.lower()}".encode()).hexdigest()
    p = CACHE_DIR / (key + ".json")
    return (json.loads(p.read_text())["box"] if p.is_file() else None)


def fused(r):
    n = len(r["gt_iou"])
    return rank01(r["sc"][:n]) + rank01(r["score_gem"][:n])


def masks_in_box_frac(masks, cand, box, H, W, thr=0.6):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(len(cand), bool)
    return (masks[cand][:, y0:y1, x0:x1].sum((1, 2))
            / np.maximum(masks[cand].sum((1, 2)), 1)) >= thr


def rule_top2_plain(masks, cand, score, k, **kw):
    return list(cand[:2])


def rule_top2_disjoint(masks, cand, score, k, **kw):
    sel = [int(cand[0])]
    m1 = masks[sel[0]]
    for j in cand[1:]:
        m = masks[j]
        uni = np.logical_or(m, m1).sum()
        if uni and np.logical_and(m, m1).sum() / uni < 0.5:
            sel.append(int(j))
            break
    return sel


def rule_greedy_novel(masks, cand, score, k, max_k=3, max_iou=0.5, min_rank=0.3):
    """Highest-scoring mask, then highest score*novelty while it stays ranked."""
    order = np.argsort(score[cand])[::-1]
    cand = np.asarray(cand)[order]
    s = score[cand] / max(1e-9, score[cand].max())
    sel = [int(cand[0])]
    cur = masks[cand[0]].copy()
    for j in range(1, len(cand)):
        m = masks[cand[j]]
        inter = np.logical_and(m, cur).sum()
        uni = np.logical_or(m, cur).sum()
        if uni and inter / uni < max_iou:
            sel.append(int(cand[j]))
            cur |= m
            if len(sel) >= max_k:
                break
    return sel


RULES = dict(box_cover=None, top2_disjoint=rule_top2_disjoint,
             greedy_novel=rule_greedy_novel, top2_plain=rule_top2_plain)


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/gref_rules.json"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open("exp/out_extreme/records_grefcoco.jsonl")]
    rng = random.Random(args.seed)
    nt = [r for r in recs if r["no_target"]]
    tg = [r for r in recs if not r["no_target"]]
    # draw in the SAME order as the existence pilot so its cached VLM boxes
    # (queried for exactly these records) apply
    sample = rng.sample(nt, args.n) + rng.sample(tg, args.n)
    sample = [r for r in sample if not r["no_target"]]
    gd = GreferDataset(REPRO / "data/refer", "val")
    by_id = {r["ref_id"]: r for r in gd.refs}

    cells = {(rn, k, bs): [0.0, 0, 0] for rn in RULES for k in (15, 50)
             for bs in ("vlm", "gt")}
    n_vlm = 0
    for i, r in enumerate(sample):
        ref = by_id[r["ref_id"]]
        gt = gd.ground_truth(ref)
        masks = load_masks(r["cache_file"])
        H, W = masks.shape[1], masks.shape[2]
        sc = fused(r)
        order = np.argsort(sc)[::-1]
        ph = (r["parsed"].get("sentence") or "").strip()
        boxes = {"vlm": box_for(r["file_name"], ph)}
        ys, xs = np.nonzero(gt)
        boxes["gt"] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        n_vlm += boxes["vlm"] is not None
        for k in (15, 50):
            cand = order[:min(k, len(order))]
            for bs, box in boxes.items():
                for rn, fn in RULES.items():
                    if rn == "box_cover":
                        if box is None:      # only the box-dependent rule needs one
                            continue
                        sel, _ = select_set(dict(r, sc=r["sc"]), masks, box,
                                            dict(lam=0.5, max_k=4, k_rank=k,
                                                 gem_mix=1.0, dec_source="sc_x_gem"))
                        sel = list(sel)
                    else:
                        sel = fn(masks, cand, sc, k)
                    I, U, iou = iou_iu(masks, sel, gt)
                    c = cells[(rn, k, bs)]
                    c[0] += iou
                    c[1] += I
                    c[2] += U
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.n}", flush=True)

    print(f"\nn={len(sample)} targeted ({n_vlm} with cached VLM boxes);  mean IoU | cumulative r")
    out = {}
    for (rn, k, bs), (s, I, U) in sorted(cells.items()):
        row = dict(mean_iou=s / len(sample), r=I / U if U else 0.0)
        out[f"{rn}|k{k}|{bs}"] = row
        print(f"  {rn:14s} k={k:2d} box={bs:3s}  mean={row['mean_iou']:.4f}  r={row['r']:.4f}")
    args.out.write_text(json.dumps(dict(n=len(sample), n_vlm=n_vlm, seed=args.seed,
                                        cells=out), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
