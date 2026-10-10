#!/usr/bin/env python
"""Pilot 2 — Which training-free score actually recovers the single-proposal oracle?

Pilot 1 showed the bottleneck is *selection*, not composition: the DOD semantic
argmax reaches ~0.45 mIoU while the best single proposal reaches ~0.81. This
pilot asks whether a better *scoring rule* over the SAME frozen features can
close that gap, and which rule wins.

Every variant below is training-free, uses only features the GLOD/HybridGL
pipeline already computes (foreground feature, background feature, text
feature, mask area), and is evaluated as a pure re-ranking of the identical
proposal pool. Ground truth is used only to score the result, never to select.
"""
from __future__ import annotations

import argparse
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

NEGATIVE_TEXTS = [
    "the background of an image",
    "an empty scene",
    "a blank surface",
    "nothing in particular",
]


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = int(np.logical_or(pred, gt).sum())
    if not union:
        return 1.0
    return int(np.logical_and(pred, gt).sum()) / union


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("idea-stage/pilots/out_calibration"))
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    config = Config()
    encoder = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                            REPRO / "checkpoints/ViT-B-16.pt",
                            device=args.device, cache_dir=REPRO / "cache/vision")

    neg_feats = torch.stack([encoder.text(t) for t in NEGATIVE_TEXTS])  # [M,512]
    neg_feats = F.normalize(neg_feats.float(), dim=-1)

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
        fused, orthogonal, _ = decouple(fg_t, bg_t, masks.mean((1, 2)), config)
        expression = ref["sentences"][0]["sent"]
        text = F.normalize(encoder.text(expression).float().reshape(-1), dim=0)

        fgn = F.normalize(fg_t, dim=-1)
        bgn = F.normalize(bg_t, dim=-1)
        fusedn = F.normalize(fused, dim=-1)
        orthn = F.normalize(orthogonal, dim=-1)

        area = masks.reshape(len(masks), -1).mean(1)
        cos_fg = (fgn @ text).numpy()
        cos_bg = (bgn @ text).numpy()
        cos_fused = (fusedn @ text).numpy()
        cos_orth = (orthn @ text).numpy()
        neg_fg = (fgn @ neg_feats.T).numpy().max(1)   # best generic-negative similarity
        neg_bg = (bgn @ neg_feats.T).numpy().max(1)

        variants = {
            # GLOD-RIS DOD score, reproduced (reference point)
            "dod_cos_fused": cos_fused,
            # plain foreground CLIP score (no decoupling)
            "cos_fg": cos_fg,
            # orthogonal residual only
            "cos_orthogonal": cos_orth,
            # foreground/background contrast on the SAME text: a mask whose
            # "background" is also described by the text is not discriminative
            "fg_minus_bg": cos_fg - cos_bg,
            # symmetric ratio
            "fg_over_bg": cos_fg - 0.5 * cos_bg,
            # contrast against generic negative texts (per-mask)
            "fg_minus_generic": cos_fg - neg_fg,
            "fg_minus_bg_and_generic": cos_fg - cos_bg - neg_fg + neg_bg,
            # area controls: does the score just prefer big masks?
            "cos_fg_minus_area": cos_fg - area,
            "cos_fg_over_sqrt_area": cos_fg - np.sqrt(area),
            # centering within the candidate pool (self-calibration, no extra text)
            "centered_fg": cos_fg - cos_fg.mean(),
            "centered_fused": cos_fused - cos_fused.mean(),
            "zscore_fg": (cos_fg - cos_fg.mean()) / (cos_fg.std() + 1e-6),
        }

        single_ious = np.array([iou(masks[i], gt) for i in range(len(masks))])
        oracle = float(single_ious.max())
        rec = dict(ref_id=int(ref["ref_id"]), expression=expression, n_masks=len(masks),
                   oracle=oracle, gt_area_frac=float(gt.mean()))
        for name, score in variants.items():
            idx = int(np.argmax(score))
            rec[name] = float(single_ious[idx])
            rec[name + "__iou"] = float(single_ious[idx])
        # oracle recovery of the DOD score, and of the best variant
        rows.append(rec)
        if len(rows) % 50 == 0:
            print(f"[{len(rows)}] {time.perf_counter()-t0:.0f}s", flush=True)

    def m(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    variant_names = [k for k in rows[0] if k.endswith("__iou")]
    summary = dict(
        dataset=args.dataset, split=args.split, limit=args.limit, seed=args.seed,
        n_expressions=len(rows), elapsed_s=round(time.perf_counter() - t0, 1),
        oracle_single=m("oracle"),
        results=sorted(
            [dict(variant=k.replace("__iou", ""), mIoU=m(k),
                  oracle_recovery=m(k) / m("oracle") if m("oracle") else None)
             for k in variant_names],
            key=lambda d: -d["mIoU"]),
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
