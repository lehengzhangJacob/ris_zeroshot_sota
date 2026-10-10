#!/usr/bin/env python
"""EXP-2 prep — materialise the full RefCOCO val split.

Downloads only the COCO train2014 images the val split actually references
(no fresh full image set), then extracts the frozen SAM ViT-H pool + HybridGL
G2L features once, into the shared cache. Every later experiment reads the
cache, so the pool is computed exactly once.

Resumable: already-cached images are skipped, so a crash costs only the
in-flight image.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp2"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split)
    refs = ds.refs
    uniq = {}
    for r in refs:
        p = ds.image_path(r)
        uniq.setdefault(str(p), r)
    have = [p for p in uniq if Path(p).exists()]
    miss = [p for p in uniq if not Path(p).exists()]
    print(f"refs={len(refs)} unique_images={len(uniq)} present={len(have)} missing={len(miss)}", flush=True)

    t0 = time.perf_counter()
    if miss:
        errs = []

        def fetch(p):
            try:
                ds.download_image(uniq[p])
                return None
            except Exception as exc:  # noqa: BLE001
                return f"{Path(p).name}: {type(exc).__name__}"

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for k, e in enumerate(pool.map(fetch, miss)):
                if e:
                    errs.append(e)
                if (k + 1) % 200 == 0:
                    print(f"  downloaded {k+1}/{len(miss)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        print(f"download done in {time.perf_counter()-t0:.0f}s errors={len(errs)}", flush=True)
        if errs:
            (args.output / "download_errors.json").write_text(json.dumps(errs[:200], indent=2))

    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                        device=args.device, cache_dir=REPRO / "cache/vision")
    t1 = time.perf_counter()
    done = skipped = failed = 0
    for p, ref in uniq.items():
        ip = Path(p)
        if not ip.exists():
            failed += 1
            continue
        try:
            with Image.open(ip) as im:
                rgb = im.convert("RGB")
                h = file_sha256(ip)
                if (enc.cache_dir / (h and __import__("glod.io", fromlist=["digest"]).digest(
                        dict(image_sha256=h, settings=enc.settings)) + ".npz")).exists():
                    skipped += 1
                    continue
                enc.extract(rgb, h)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {ip.name}: {type(exc).__name__}", flush=True)
        if (done + skipped) % 100 == 0 and done:
            rate = (time.perf_counter() - t1) / max(done, 1)
            print(f"  extract {done+skipped}/{len(uniq)} ({rate:.2f}s/img)", flush=True)

    print(json.dumps(dict(unique_images=len(uniq), extracted=done, cached=skipped, failed=failed,
                          extract_seconds=round(time.perf_counter() - t1, 1)), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
