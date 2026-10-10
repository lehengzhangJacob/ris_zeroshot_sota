#!/usr/bin/env python
"""EXP-48 — end-to-end gRefCOCO estimate of the set-valued + abstention method.

Runs the full pipeline offline on any record subset that already has VLM boxes
cached (the pilot's 300-record sample today, the full val after extraction):

  none answer -> abstain (no-target decision)
  a box       -> greedy box-cover union of candidate masks

and reports the official cIoU / gIoU / N_acc / T_acc, alongside the
no-abstention control (box ignored, CLIP top-1) so the value of the existence
branch is visible.
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
from gref_setselect import (load_masks, select_set, candidates, iou_iu,  # noqa: E402
                            official_metrics, rank01)
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")


def box_for(file_name, phrase):
    key = hashlib.sha1(f"{MODEL_TAG}|{file_name}|{phrase.lower()}".encode()).hexdigest()
    p = CACHE_DIR / (key + ".json")
    if not p.is_file():
        return None, False
    d = json.loads(p.read_text())
    return d["box"], True


def single_pick(rec, params):
    """RefCOCO-family rule: top-k by fused score, best IoU with the box."""
    n = len(rec["gt_iou"])
    base = rank01(rec["sc"][:n]) + params.get("gem_mix", 1.0) * rank01(rec["score_gem"][:n])
    k = min(int(params.get("k_rank", 15)), n)
    return np.argsort(base)[::-1][:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phrase-source", default="sent", choices=["sent", "noun"])
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--k-rank", type=int, default=15)
    ap.add_argument("--gem-mix", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/gref_estimate.json"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.records)]
    rng = random.Random(args.seed)
    nt = [r for r in recs if r["no_target"]]
    tg = [r for r in recs if not r["no_target"]]
    if args.n_per_class and args.n_per_class < len(nt):
        sample = rng.sample(nt, args.n_per_class) + rng.sample(tg, args.n_per_class)
    else:
        sample = nt + tg
    print(f"evaluating {len(sample)} records", flush=True)

    gd = GreferDataset(REPRO / "data/refer", "val")
    ref_by_id = {r["ref_id"]: r for r in gd.refs}
    gt_cache = {}
    mask_cache = {}

    params = dict(lam=args.lam, max_k=args.max_k, k_rank=args.k_rank,
                  gem_mix=args.gem_mix, dec_source="sc_x_gem", min_gain=0.0)
    rows = {k: [] for k in ("set+abstain", "single+abstain", "set_noabstain")}
    n_box = n_none = n_miss = 0

    for i, r in enumerate(sample):
        rid = r["ref_id"]
        if rid not in gt_cache:
            gt_cache[rid] = gd.ground_truth(ref_by_id[rid])
        gt = gt_cache[rid]
        cf = r["cache_file"]
        if cf not in mask_cache:
            mask_cache[cf] = load_masks(cf)
            while len(mask_cache) > 6:
                mask_cache.pop(next(iter(mask_cache)))
        masks = mask_cache[cf]
        key = "sentence" if args.phrase_source == "sent" else "noun_phrase"
        ph = (r["parsed"].get(key) or "").strip()
        box, cached = box_for(r["file_name"], ph)
        n_box += box is not None
        n_none += cached and box is None
        n_miss += (not cached)

        gt_nt = bool(r["no_target"])
        gt_area = float(r["gt_area"]) * masks.shape[1] * masks.shape[2]

        # --- ours: box-cover set, abstain when the VLM says none
        idxs, _ = select_set(r, masks, box, params)
        I, U, iou = iou_iu(masks, idxs, gt)
        rows["set+abstain"].append(dict(gt_nt=gt_nt, abstained=(not idxs),
                                        I=I, U=U, iou=iou, gt_area=gt_area))
        # --- control 1: single mask (best box IoU inside the shortlist)
        if box is not None:
            cand = single_pick(r, params)
            H, W = masks.shape[1], masks.shape[2]
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 > x0 and y1 > y0:
                sub = masks[cand][:, y0:y1, x0:x1].sum((1, 2)).astype(np.float32)
                tot = masks[cand].sum((1, 2)).astype(np.float32)
                box_area = float((y1 - y0) * (x1 - x0))
                # same box_overlap(mode="iou") the RefCOCO-family rule uses
                score = sub / (tot + box_area - sub + 1e-9)
                j = int(cand[int(np.argmax(score))])
                I1, U1, iou1 = iou_iu(masks, [j], gt)
            else:
                I1 = U1 = iou1 = 0.0
            rows["single+abstain"].append(dict(gt_nt=gt_nt, abstained=False,
                                               I=I1, U=U1, iou=iou1, gt_area=gt_area))
        else:
            rows["single+abstain"].append(dict(gt_nt=gt_nt, abstained=True,
                                               I=0, U=0, iou=0.0, gt_area=gt_area))
        # --- control 2: never abstain (box ignored when none -> CLIP top-1)
        if idxs:
            rows["set_noabstain"].append(dict(gt_nt=gt_nt, abstained=False,
                                              I=I, U=U, iou=iou, gt_area=gt_area))
        else:
            cand = np.asarray(candidates(r, params)[0])
            j = int(cand[0]) if len(cand) else 0
            In, Un, ioun = iou_iu(masks, [j], gt)
            rows["set_noabstain"].append(dict(gt_nt=gt_nt, abstained=False,
                                              I=In, U=Un, iou=ioun, gt_area=gt_area))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)

    print(f"\nbox answers: {n_box} boxes, {n_none} none, {n_miss} not cached")
    out = dict(n=len(sample), phrase_source=args.phrase_source, params=params,
               n_box=n_box, n_none=n_none, n_uncached=n_miss, rows={})
    print(f"{'row':<16}{'cIoU':>8}{'gIoU':>8}{'N_acc':>8}{'T_acc':>8}")
    for name in rows:
        m = official_metrics(rows[name])
        out["rows"][name] = m
        print(f"{name:<16}{m['cIoU']:>8.2f}{m['gIoU']:>8.2f}{m['N_acc']:>8.1f}{m['T_acc']:>8.1f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
