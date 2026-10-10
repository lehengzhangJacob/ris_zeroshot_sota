#!/usr/bin/env python
"""EXP-75 — Qwen2.5-VL boxes for gRefCOCO (model change 9: swap the localizer).

The wall is perceptual: the LLaVA-OneVision localizer overlaps a 2-object
union at IoU 0.329, cannot reject in yes/no, and is at chance in forced choice
(exp72-74).  The RefCOCO-family line of this run already uses Qwen2.5-VL-7B
boxes (exp38/exp40); gRefCOCO never did.  This extracts Qwen boxes for the
held-out train gate so the box-quality diagnostic can decide whether a
stronger grounder moves the wall.

    python exp/exp75_qwen_boxes.py --shard 0 --nshards 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import IMAGES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image
    from vlm_boxes import Qwen2Boxer

    recs = [json.loads(l) for l in open(args.records)]
    if args.limit:
        recs = recs[:args.limit]
    # unique (file, phrase) pairs in this shard
    pairs, seen = [], set()
    for i, rec in enumerate(recs):
        if i % args.nshards != args.shard:
            continue
        ph = (rec["parsed"].get("sentence") or "").strip()
        if not ph:
            continue
        key = (rec["file_name"], ph.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs.append((rec["file_name"], ph))

    # drop cached
    boxer = Qwen2Boxer(device=args.device)
    todo = [(fn, ph) for fn, ph in pairs if not boxer._path(fn, ph).is_file()]
    print(f"shard {args.shard}/{args.nshards}: {len(todo)} qwen queries "
          f"({len(pairs)-len(todo)} cached)", flush=True)
    t0 = time.time()
    n = 0
    for fn, ph in todo:
        try:
            boxer.box(fn, ph)
        except Exception as e:                      # noqa: BLE001 — keep going
            print(f"  ERR {fn}: {e}", flush=True)
            continue
        n += 1
        if n % 100 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(todo)} ({el:.0f}s, {60*n/el:.0f}/min)", flush=True)
    print(f"done {n} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
