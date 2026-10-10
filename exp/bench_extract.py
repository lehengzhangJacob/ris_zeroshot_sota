#!/usr/bin/env python
"""Benchmark frozen-pool extraction (SAM ViT-H + HybridGL G2L) on fresh images."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
from glod.data import RefDataset  # noqa: E402
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402

device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
n_target = int(sys.argv[2]) if len(sys.argv) > 2 else 5

ds = RefDataset(REPRO / "data/refer", "refcoco", "val", limit=400, seed=303)
enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                    device=device, cache_dir=REPRO / "cache/vision")
times, seen = [], set()
for ref in ds.refs:
    ip = ds.image_path(ref)
    if not ip.exists() or str(ip) in seen:
        continue
    seen.add(str(ip))
    im = Image.open(ip).convert("RGB")
    h = file_sha256(ip)
    # force a miss by using a throwaway cache dir
    enc.cache_dir = Path("/tmp/bench_cache")
    t0 = time.perf_counter()
    masks, fg, bg, info = enc.extract(im, h + "-bench")
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"{ip.name} masks={masks.shape[0]} {dt:.2f}s", flush=True)
    if len(times) >= n_target:
        break
import shutil  # noqa: E402
shutil.rmtree("/tmp/bench_cache", ignore_errors=True)
print(f"mean {np.mean(times):.2f}s/image -> 1500 images = {np.mean(times)*1500/3600:.2f} GPU-hours")
