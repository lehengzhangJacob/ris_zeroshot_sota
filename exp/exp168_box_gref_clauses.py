#!/usr/bin/env python
"""EXP-168a — gRefCOCO clause-text boxes + presence probes (no-detector rescue).

The no-detector gRefCOCO path so far used cached LLaVA noun-phrase boxes.  The
rescue directive asks for per-CLAUSE full-text boxes (the full-expression
advantage this run measured on the RefCOCO family was never applied to the
gRefCOCO term path) and presence-verified term filtering.  This script fills
the two caches for the unique (image, clause) pairs of the complete val split:

  * box    under the base namespace          (MODEL_TAG|fn|clause)
  * exists under the existence namespace     (MODEL_TAG|exists|fn|clause)

with the same prompts/parsing as the shipped localizer (8-bit).

    CUDA_VISIBLE_DEVICES=0 python exp/exp168_box_gref_clauses.py --shard 0 --nshards 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from method_gres import clauses  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_14229.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from vlm_boxes import ExistsBoxer8bit, LlavaBoxer8bit

    recs = [json.loads(l) for l in open(args.records)]
    pairs, seen = [], set()
    for r in recs:
        fn = r.get("file_name")
        sent = ((r.get("parsed") or {}).get("sentence") or "").strip()
        if not fn or not sent:
            continue
        for c in clauses(sent):
            if (fn, c) not in seen:
                seen.add((fn, c))
                pairs.append((fn, c))
    print(f"{len(recs)} records -> {len(pairs)} unique (image, clause) pairs")
    pairs = [p for i, p in enumerate(sorted(pairs)) if i % args.nshards == args.shard]
    if args.limit:
        pairs = pairs[: args.limit]

    boxer = LlavaBoxer8bit(device=args.device)
    exister = ExistsBoxer8bit(device=args.device)
    # which are missing in each namespace
    todo = []
    for fn, c in pairs:
        if not boxer._path(fn, c).is_file() or not exister._path(fn, c).is_file():
            todo.append((fn, c))
    print(f"shard {args.shard}/{args.nshards}: {len(pairs)} pairs, "
          f"{len(todo)} to query", flush=True)
    if not todo:
        return 0

    t0 = time.time()
    n_done = 0
    for fn, c in todo:
        try:
            if not boxer._path(fn, c).is_file():
                boxer.box(fn, c)
            if not exister._path(fn, c).is_file():
                exister.box(fn, c)      # yes/no answer cached under the exists tag
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {fn} {c[:40]!r}: {type(exc).__name__} {exc}", flush=True)
        n_done += 1
        if n_done % 100 == 0:
            rate = n_done / (time.time() - t0)
            print(f"  {n_done}/{len(todo)} ({rate:.2f}/s, eta "
                  f"{(len(todo)-n_done)/max(rate,1e-9)/60:.1f} min)", flush=True)
    print(f"done {n_done} in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
