#!/usr/bin/env python
"""Build the held-out tuning split: a seeded sample of RefCOCO-family TRAIN refs.

The official validation splits are reserved for the SOTA claim; hyperparameters
are chosen on this split only (see tune_study.py).  Downloads any missing
COCO train2014 images and records the exact ref ids + sampling parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
OUT = Path("exp/tuning")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["refcoco", "refcoco+", "refcocog"])
    ap.add_argument("--per-dataset", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--download", action="store_true", default=True)
    args = ap.parse_args()

    from glod.data import RefDataset
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        ds = RefDataset(REPRO / "data/refer", name, "train",
                        limit=args.per_dataset, seed=args.seed)
        print(f"{name}: {len(ds.refs)} sampled refs", flush=True)
        ref_ids, files, t0, missing = [], set(), time.time(), 0
        for i, r in enumerate(ds.refs):
            fn = ds.images[r["image_id"]]["file_name"]
            if fn not in files:
                files.add(fn)
                if args.download and not ds.image_path(r).exists():
                    try:
                        ds.download_image(r)
                        missing += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"  download failed {fn}: {exc}", flush=True)
            ref_ids.append(r["ref_id"])
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(ds.refs)} images={len(files)} new={missing} "
                      f"{time.time()-t0:.0f}s", flush=True)
        (OUT / f"{name}_refs.json").write_text(json.dumps(dict(
            dataset=name, split="train", limit=args.per_dataset, seed=args.seed,
            ref_ids=ref_ids, n_images=len(files)), indent=2))
        print(f"{name}: done, {len(files)} images ({missing} downloaded) "
              f"{time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
