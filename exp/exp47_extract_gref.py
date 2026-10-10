#!/usr/bin/env python
"""EXP-47 — batched, sharded box-or-none extraction for gRefCOCO val.

One VLM query per (image, sentence): the answer is either a box (the
referent's extent) or the word `none` (the referent is absent).  Both signals
are needed downstream: the box drives mask *selection*, the none drives
*abstention* (the no-target decision).

    python exp/exp47_extract_gref.py --shard 0 --nshards 2 --device cuda:0 \
        --load-8bit --batch-size 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import LlavaBoxer, LlavaBoxer8bit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--phrase-source", default="sent", choices=["sent", "noun"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--load-8bit", action="store_true", default=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.records)]
    seen, pairs = set(), []
    for r in recs:
        key = "sentence" if args.phrase_source == "sent" else "noun_phrase"
        ph = (r["parsed"].get(key) or "").strip()
        if not ph:
            continue
        k = (r["file_name"], ph.lower())
        if k in seen:
            continue
        seen.add(k)
        pairs.append((r["file_name"], ph))
    print(f"{len(recs)} records -> {len(pairs)} unique (image, phrase) pairs", flush=True)
    pairs = [p for i, p in enumerate(pairs) if i % args.nshards == args.shard]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"shard {args.shard}/{args.nshards}: {len(pairs)} pairs to check", flush=True)

    boxer = (LlavaBoxer8bit if args.load_8bit else LlavaBoxer)(device=args.device)
    t0 = time.time()
    out = boxer.box_many(pairs, batch_size=args.batch_size)
    el = time.time() - t0
    n_none = sum(1 for b, _ in out if b is None)
    print(f"done {len(out)} pairs in {el:.0f}s ({60 * len(out) / max(el, 1):.1f}/min), "
          f"{n_none} none", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
