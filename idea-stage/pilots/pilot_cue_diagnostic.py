#!/usr/bin/env python
"""Pilot 3 — Is the selection failure reducible by re-scoring, or is it a feature wall?

Pilot 1: the argmax reaches ~0.45 mIoU against a single-proposal oracle of ~0.81.
Pilot 2: naive recalibrations (fg-bg contrast, area penalty, generic negatives)
         did not help on a 10-expression sample.

This pilot settles the question on a real sample by measuring, for every
expression, the *ceiling* of any score built from the cues the pipeline already
has for free:

    c1 = cos(F_fg,   t)        appearance of the region
    c2 = cos(F_bg,   t)        how much the complement also matches the text
    c3 = cos(F_fuse, t)        GLOD-RIS DOD score
    c4 = log area              size
    c5 = compactness           perimeter^2 / area (mask-shape regularity)

Reported per expression:
  * Spearman rank correlation between each cue and true IoU
  * the rank of the best-IoU proposal under each cue
  * best achievable mIoU for a *single global* linear weighting of the cues
    (upper bound for any fixed hyper-parameter choice)
  * best achievable mIoU when the weighting may adapt *per expression*
    (upper bound for any instance-adaptive training-free calibration)

If the per-expression ceiling is close to the single-proposal oracle while the
global ceiling is not, the winning idea is instance-adaptive calibration.
If both ceilings are low, re-scoring cannot fix it and the wall is the feature.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage  # noqa: F401  (optional; compactness falls back)

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

from glod.core import Config, decouple  # noqa: E402
from glod.data import RefDataset  # noqa: E402
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402


def iou(pred, gt):
    union = int(np.logical_or(pred, gt).sum())
    return 1.0 if not union else int(np.logical_and(pred, gt).sum()) / union


def spearman(a, b):
    """Rank correlation; ties get average ranks."""
    n = len(a)
    if n < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def perimeter_compactness(mask):
    m = mask.astype(bool)
    er = ndimage.binary_erosion(m, iterations=1, border_value=0)
    per = int((m & ~er).sum())
    area = int(m.sum())
    return (per * per) / area if area else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("idea-stage/pilots/out_cues"))
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    config = Config()
    encoder = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                            REPRO / "checkpoints/ViT-B-16.pt",
                            device=args.device, cache_dir=REPRO / "cache/vision")

    per_expr = []          # cue matrices + ious, kept for the grid search
    rows = []
    t0 = time.perf_counter()

    for ref in dataset.refs:
        image_path = dataset.image_path(ref)
        if not image_path.exists():
            if not args.allow_download:
                continue
            try:
                dataset.download_image(ref)
            except Exception:  # noqa: BLE001
                continue
        try:
            image = Image.open(image_path).convert("RGB")
            gt = dataset.ground_truth(ref)
        except Exception:  # noqa: BLE001
            continue
        masks, fg, bg, _ = encoder.extract(image, file_sha256(image_path))
        if masks.shape[1:] != gt.shape:
            continue

        fg_t, bg_t = torch.from_numpy(fg).float(), torch.from_numpy(bg).float()
        fused, _, _ = decouple(fg_t, bg_t, masks.mean((1, 2)), config)
        expression = ref["sentences"][0]["sent"]
        text = F.normalize(encoder.text(expression).float().reshape(-1), dim=0)

        c1 = (F.normalize(fg_t, dim=-1) @ text).numpy()
        c2 = (F.normalize(bg_t, dim=-1) @ text).numpy()
        c3 = (F.normalize(fused, dim=-1) @ text).numpy()
        area = masks.reshape(len(masks), -1).mean(1)
        c4 = np.log(area + 1e-8)
        c5 = np.array([perimeter_compactness(m) for m in masks])
        ious = np.array([iou(masks[i], gt) for i in range(len(masks))])

        # consensus cues: is the semantically top-ranked mask also one that the
        # other plausible masks agree with? (a training-free medoid / stability
        # signal, computed on masks alone, no extra forward passes)
        top10 = np.argsort(-c1, kind="stable")[:10]
        flat = masks.reshape(len(masks), -1)
        sub = flat[top10]
        inter = sub.astype(np.float32) @ sub.astype(np.float32).T      # [10,10]
        union = sub.sum(1)[:, None] + sub.sum(1)[None, :] - inter
        agree = np.zeros(len(masks), np.float32)
        agree[top10] = (inter / np.maximum(union, 1)).mean(1)
        c6 = agree
        # coverage: how much of the top-10 union does mask i cover
        top_union = sub.any(0)
        cov = (flat & top_union).sum(1) / np.maximum(flat.sum(1), 1)
        c7 = cov

        cues = np.stack([c1, c2, c3, c4, c5, c6, c7], 1)
        per_expr.append((cues, ious))
        rows.append(dict(
            ref_id=int(ref["ref_id"]), n=len(masks), oracle=float(ious.max()),
            iou_argmax_c1=float(ious[int(np.argmax(c1))]),
            iou_argmax_c3=float(ious[int(np.argmax(c3))]),
            rank_best_under_c1=int(np.argsort(np.argsort(-c1))[int(np.argmax(ious))]),
            rank_best_under_c3=int(np.argsort(np.argsort(-c3))[int(np.argmax(ious))]),
            rho_c1=spearman(c1, ious), rho_c3=spearman(c3, ious),
            rho_area=spearman(area, ious), rho_compact=spearman(c5, ious),
            rho_c2=spearman(c2, ious),
            iou_argmax_c6=float(ious[int(np.argmax(c6))]),
            iou_argmax_c7=float(ious[int(np.argmax(c7))]),
        ))
        if len(rows) % 50 == 0:
            print(f"[{len(rows)}] {time.perf_counter()-t0:.0f}s", flush=True)

    if not rows:
        print("no records", flush=True)
        return 1

    # --- standardise cues within each expression, then grid-search weights -----
    grids = {
        "c1": [1.0], "c2": [0.0, -1.0, -0.5], "c3": [0.0, 0.5, 1.0],
        "c4": [0.0, -0.3, -0.1, 0.1], "c5": [0.0, 0.2, -0.2],
        "c6": [0.0, 0.5, 1.0], "c7": [0.0, 0.3, -0.3],
    }
    KEYS = ("c1", "c2", "c3", "c4", "c5", "c6", "c7")
    combos = list(itertools.product(*[grids[k] for k in KEYS]))
    zs = []
    for cues, ious in per_expr:
        z = (cues - cues.mean(0)) / (cues.std(0) + 1e-8)
        zs.append((z, ious))

    def eval_weights(w):
        tot = []
        for z, ious in zs:
            s = z @ np.asarray(w)
            tot.append(ious[int(np.argmax(s))])
        return float(np.mean(tot))

    scored = sorted(((eval_weights(w), w) for w in combos), reverse=True, key=lambda x: x[0])
    best_global, best_w = scored[0]

    # per-expression upper bound (what an instance-adaptive rule could reach)
    per_expr_best = []
    for z, ious in zs:
        vals = [ious[int(np.argmax(z @ np.asarray(w)))] for w in combos]
        per_expr_best.append(max(vals))

    def m(k):
        v = [r[k] for r in rows if r[k] is not None and not np.isnan(r[k])]
        return float(np.mean(v)) if v else None

    summary = dict(
        dataset=args.dataset, split=args.split, limit=args.limit, seed=args.seed,
        n_expressions=len(rows), elapsed_s=round(time.perf_counter() - t0, 1),
        mIoU_oracle_single=m("oracle"),
        mIoU_argmax_dod=m("iou_argmax_c3"),
        mIoU_argmax_cosfg=m("iou_argmax_c1"),
        mIoU_best_global_weighting=best_global,
        best_global_weights=dict(zip(KEYS, best_w)),
        mIoU_per_expression_ceiling=float(np.mean(per_expr_best)),
        oracle_recovery_argmax=m("iou_argmax_c3") / m("oracle"),
        oracle_recovery_best_global=best_global / m("oracle"),
        oracle_recovery_per_expr_ceiling=float(np.mean(per_expr_best)) / m("oracle"),
        mean_rank_of_best_under_dod=m("rank_best_under_c3"),
        mean_rank_of_best_under_cosfg=m("rank_best_under_c1"),
        mean_spearman=dict(c1_fg=m("rho_c1"), c2_bg=m("rho_c2"), c3_fused=m("rho_c3"),
                           area=m("rho_area"), compactness=m("rho_compact")),
        mIoU_argmax_consensus=m("iou_argmax_c6"),
        mIoU_argmax_coverage=m("iou_argmax_c7"),
        top5_global_weightings=[dict(mIoU=s, weights=dict(zip(KEYS, w))) for s, w in scored[:5]],
    )
    with (args.output / "records.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with (args.output / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
