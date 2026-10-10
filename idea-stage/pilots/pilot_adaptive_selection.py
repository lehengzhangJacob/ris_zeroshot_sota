#!/usr/bin/env python
"""Pilot 4 — Can an UNSUPERVISED per-expression cue weighting capture the gap?

Pilot 3 measured, on 250 RefCOCO-val expressions:
  * oracle single-proposal mIoU        0.8265
  * GLOD-RIS DOD argmax mIoU           0.3532   (42.7 % of oracle)
  * best GLOBAL cue weighting          0.3798   (46.0 % of oracle)   <- fixed weights cap out
  * per-EXPRESSION best weighting      0.5215   (63.1 % of oracle)   <- the real headroom

So the missing ingredient is per-instance calibration, and it cannot be tuned in
as a constant. This pilot tests one training-free mechanism for it: weight each
cue by how much it AGREES with the other cues on this particular expression
(an unsupervised reliability estimate), instead of by a constant.

Everything is computed from the frozen pipeline's own features. Ground truth is
used only to score the result. No training, no MLLM, no parser.

Also introduces a parser-free spatial cue: direction words are detected by
comparing the expression's CLIP text embedding with canonical phrases
("on the left", "on the right", ...), and the resulting preference is applied to
proposal centroids. This is the training-free, MLLM-free replacement candidate
for GLOD-RIS's 3x3 grid, and it composes with any ordinal structure because it
acts on continuous centroid coordinates rather than nine cells.
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

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

from glod.core import Config, decouple  # noqa: E402
from glod.data import RefDataset  # noqa: E402
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402

DIRECTIONS = {
    "left": "on the left side of the image",
    "right": "on the right side of the image",
    "top": "at the top of the image",
    "bottom": "at the bottom of the image",
    "middle": "in the middle of the image",
}


def iou(pred, gt):
    union = int(np.logical_or(pred, gt).sum())
    return 1.0 if not union else int(np.logical_and(pred, gt).sum()) / union


def centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.5, 0.5
    return float(xs.mean()) / mask.shape[1], float(ys.mean()) / mask.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("idea-stage/pilots/out_adaptive"))
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    config = Config()
    encoder = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                            REPRO / "checkpoints/ViT-B-16.pt",
                            device=args.device, cache_dir=REPRO / "cache/vision")
    dir_feats = {k: F.normalize(encoder.text(v).float().reshape(-1), dim=0) for k, v in DIRECTIONS.items()}

    per_expr, rows = [], []
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
        t = F.normalize(encoder.text(expression).float().reshape(-1), dim=0)

        c1 = (F.normalize(fg_t, dim=-1) @ t).numpy()
        c2 = (F.normalize(bg_t, dim=-1) @ t).numpy()
        c3 = (F.normalize(fused, dim=-1) @ t).numpy()
        area = masks.reshape(len(masks), -1).mean(1)
        c4 = np.log(area + 1e-8)

        # --- parser-free spatial cue -----------------------------------------
        dir_sim = {k: float(t @ v) for k, v in dir_feats.items()}
        best_dir = max(dir_sim, key=dir_sim.get)
        # a direction is "present" if its similarity clearly leads the others
        vals = np.array(list(dir_sim.values()))
        lead = dir_sim[best_dir] - np.sort(vals)[-2] if len(vals) > 1 else 0.0
        spatial_on = lead > 0.01
        cents = np.array([centroid(m) for m in masks])           # [N,2] in [0,1]
        if spatial_on and best_dir in ("left", "right"):
            pref = 1.0 - cents[:, 0] if best_dir == "left" else cents[:, 0]
        elif spatial_on and best_dir in ("top", "bottom"):
            pref = 1.0 - cents[:, 1] if best_dir == "top" else cents[:, 1]
        elif spatial_on:
            pref = -np.abs(cents[:, 0] - 0.5) - np.abs(cents[:, 1] - 0.5)
        else:
            pref = np.zeros(len(masks))
        c8 = pref

        ious = np.array([iou(masks[i], gt) for i in range(len(masks))])
        cues = np.stack([c1, c2, c3, c4, c8], 1)
        per_expr.append((cues, ious, spatial_on))
        rows.append(dict(ref_id=int(ref["ref_id"]), expression=expression, n=len(masks),
                         oracle=float(ious.max()), spatial_on=bool(spatial_on),
                         best_dir=best_dir if spatial_on else None,
                         iou_c1=float(ious[int(np.argmax(c1))]),
                         iou_c3=float(ious[int(np.argmax(c3))]),
                         iou_spatial_only=float(ious[int(np.argmax(c8))]) if spatial_on else None))
        if len(rows) % 50 == 0:
            print(f"[{len(rows)}] {time.perf_counter()-t0:.0f}s", flush=True)

    if not rows:
        print("no records", flush=True)
        return 1

    # ---- aggregation strategies ----------------------------------------------
    def zscores(cues):
        return (cues - cues.mean(0)) / (cues.std(0) + 1e-8)

    def ranks(cues):
        r = np.empty_like(cues, dtype=float)
        for j in range(cues.shape[1]):
            r[:, j] = np.argsort(np.argsort(cues[:, j]))
        return r / max(len(cues) - 1, 1)

    def agreement_weights(z):
        """Unsupervised per-expression reliability = mean |corr| with the other cues."""
        m = z.shape[1]
        if m < 2:
            return np.ones(m)
        cm = np.corrcoef(z.T)
        cm = np.nan_to_num(np.abs(cm))
        np.fill_diagonal(cm, 0.0)
        w = cm.mean(1)
        return w / (w.sum() + 1e-8) if w.sum() > 0 else np.ones(m) / m

    def evaluate(fn):
        return float(np.mean([fn(cues, ious) for cues, ious, _ in per_expr]))

    def pick(score, ious):
        return float(ious[int(np.argmax(score))])

    glob_grid = list(itertools.product([1.0], [0.0, -0.5, -1.0], [0.0, 0.5, 1.0],
                                       [0.0, -0.1, 0.1], [0.0, 0.2, 0.5]))

    def global_w(w):
        return lambda cues, ious: pick(zscores(cues) @ np.asarray(w), ious)

    scored = sorted(((evaluate(global_w(w)), w) for w in glob_grid), reverse=True, key=lambda x: x[0])

    results = {
        "oracle_single": float(np.mean([r["oracle"] for r in rows])),
        "argmax_dod": float(np.mean([r["iou_c3"] for r in rows])),
        "argmax_cosfg": float(np.mean([r["iou_c1"] for r in rows])),
        "best_global_weighting": scored[0][0],
        "best_global_weights": dict(zip(("appearance", "bg", "dod", "logarea", "spatial"), scored[0][1])),
        # unsupervised, per-expression, no tuning on labels:
        "agreement_weighted_zsum": evaluate(lambda c, i: pick(zscores(c) @ agreement_weights(zscores(c)), i)),
        "equal_weight_zsum": evaluate(lambda c, i: pick(zscores(c).mean(1), i)),
        "agree_weighted_borda": evaluate(lambda c, i: pick(ranks(c) @ agreement_weights(ranks(c)), i)),
        "equal_borda": evaluate(lambda c, i: pick(ranks(c).mean(1), i)),
        "spatial_only_on_spatial_expr": float(np.mean(
            [r["iou_spatial_only"] for r in rows if r["iou_spatial_only"] is not None])) if any(
            r["iou_spatial_only"] is not None for r in rows) else None,
        "frac_expressions_with_spatial_cue": float(np.mean([r["spatial_on"] for r in rows])),
        "mean_c1_on_spatial_expr": float(np.mean([r["iou_c1"] for r in rows if r["spatial_on"]])) if any(
            r["spatial_on"] for r in rows) else None,
    }
    results["oracle_recovery"] = {k: (v / results["oracle_single"] if isinstance(v, float) else None)
                                  for k, v in results.items() if k.startswith(("argmax", "best_global", "agreement",
                                                                               "equal_", "spatial_only"))}
    with (args.output / "records.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with (args.output / "summary.json").open("w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
