#!/usr/bin/env python
"""EXP-53 — box-or-none extraction for the held-out gRefCOCO TRAIN sample.

Reads exp/tuning/grefcoco_refs.json, expands each ref into its sentences
(the evaluation unit), and runs the batched box query.  Shardable across GPUs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import LlavaBoxer8bit  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", type=Path, default=Path("exp/tuning/grefcoco_refs.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ds = GreferDataset(REPRO / "data/refer", args.split)
    wanted = set(json.loads(args.refs.read_text())["ref_ids"])
    refs = [r for r in ds.refs if r["ref_id"] in wanted]
    pairs, seen = [], set()
    for r in refs:
        for s in r["sentences"]:
            key = (r["file_name"], s["sent"].lower())
            if key not in seen and s["sent"].strip():
                seen.add(key)
                pairs.append((r["file_name"], s["sent"].strip()))
    print(f"{len(refs)} refs -> {len(pairs)} unique (image, sentence) pairs", flush=True)
    pairs = [p for i, p in enumerate(pairs) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(pairs)} pairs", flush=True)

    boxer = LlavaBoxer8bit(device=args.device)
    t0 = time.time()
    out = boxer.box_many(pairs, batch_size=args.batch_size)
    el = time.time() - t0
    n_none = sum(1 for b, _ in out if b is None)
    print(f"done {len(out)} pairs in {el:.0f}s, {n_none} none", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
