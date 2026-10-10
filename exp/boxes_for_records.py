#!/usr/bin/env python
"""Populate the Qwen2.5-VL box cache for a records file.

    python exp/boxes_for_records.py --records exp/tuning/records_tune_refcoco.jsonl \
        --dataset refcoco --split train --limit 400 --seed 7 --phrase-source noun

--phrase-source sent queries the full expression instead of the parsed noun
phrase (GLOD-RIS prompts its MLLM with the whole expression).  Shardable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import QwenBoxer, LlavaBoxer, LlavaBoxer8bit, CACHE_DIR  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--phrase-source", default="noun", choices=["noun", "sent"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--load-8bit", action="store_true", default=False)
    args = ap.parse_args()

    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split,
                    limit=args.limit, seed=args.seed)
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(args.records)]
    print(f"{len(recs)} records from {args.records}", flush=True)

    pairs, seen = [], set()
    for r in recs:
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        ph = r["parsed"]["sentence"] if args.phrase_source == "sent" else r["parsed"]["noun_phrase"]
        ph = (ph or "").strip()
        if not ph:
            continue
        key = (fn, ph.lower())
        if key not in seen:
            seen.add(key)
            pairs.append((fn, ph))
    pairs = [p for i, p in enumerate(pairs) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(pairs)} (image, phrase) pairs "
          f"[{args.phrase_source}]", flush=True)

    b = (LlavaBoxer8bit if args.load_8bit else LlavaBoxer)(device=args.device)
    t0 = time.time()
    n_new = n_none = 0
    for i, (fn, ph) in enumerate(pairs):
        before = CACHE_DIR.joinpath  # noqa: F841  (kept for symmetry)
        box, raw = b.box(fn, ph)
        n_none += (box is None)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(pairs)} none={n_none} {el:.0f}s "
                  f"({el/(i+1):.2f}s/call)", flush=True)
    print(f"done: {len(pairs)} pairs, {n_none} none, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
