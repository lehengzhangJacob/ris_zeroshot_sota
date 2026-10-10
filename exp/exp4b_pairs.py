#!/usr/bin/env python
"""EXP-4b — exact per-image pairwise mask IoU for the val split.

The selector needs a true disjointness test between competing candidates. This
pass stores the full N x N mask-IoU matrix for every reference's image, keyed by
ref_id, so the analysis stays exact without re-running the encoder per rule.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
from glod.data import RefDataset  # noqa: E402
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp4"))
    ap.add_argument("--tag", default="pairs_guided_val")
    args = ap.parse_args()

    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split)
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                        device=args.device, cache_dir=REPRO / "cache/vision")
    out = {}
    t0 = time.perf_counter()
    done = 0
    for ref in ds.refs:
        ip = ds.image_path(ref)
        if not ip.exists():
            continue
        try:
            im = Image.open(ip).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        masks, _, _, _ = enc.extract(im, file_sha256(ip))
        n = len(masks)
        mf = torch.from_numpy(masks.reshape(n, -1)).to(enc.device, torch.float32)
        inter = mf @ mf.T
        ar = mf.sum(1)
        iou = (inter / (ar[:, None] + ar[None, :] - inter).clamp(min=1)).cpu().numpy().astype(np.float16)
        out[str(int(ref["ref_id"]))] = iou
        done += 1
        if done % 500 == 0:
            print(f"[{done}] {time.perf_counter()-t0:.0f}s", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{args.tag}.npz", **out)
    print(f"saved {done} refs to {args.output / (args.tag + '.npz')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
