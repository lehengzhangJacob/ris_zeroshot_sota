#!/usr/bin/env python
"""EXP-33 — populate the GroundingDINO detection cache for a dataset split.

Detection is per (image, phrase); phrases are the parsed noun phrases of the
split's expressions (plus their constituent nouns).  Shardable across GPUs:
    python exp33_gdino_detect.py --dataset refcoco+ --shard 0 --nshards 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdino import Gdino  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
RECORDS = {
    "refcoco": "exp/out_extreme/records_refcoco.jsonl",
    "refcoco+": "exp/out_extreme/records_refcocoplus.jsonl",
    "refcocog": "exp/out_extreme/records_refcocog.jsonl",
}


def phrases_for(rec):
    p = rec["parsed"]
    out = [p["noun_phrase"]]
    for n in p.get("nouns", []):
        out.append(n if n.startswith(("a ", "the ")) else n)
    # de-dup preserving order, drop empties/duplicates of the head
    seen, res = set(), []
    for s in out:
        s = (s or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            res.append(s)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco+", choices=list(RECORDS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", args.dataset, "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(RECORDS[args.dataset])]
    if args.limit:
        recs = recs[: args.limit]

    pairs, seen = [], set()
    for r in recs:
        ref = ref_by_id[r["ref_id"]]
        fn = ds.images[ref["image_id"]]["file_name"]
        for ph in phrases_for(r):
            key = (fn, ph.lower())
            if key not in seen:
                seen.add(key)
                pairs.append((fn, ph))
    pairs = [p for i, p in enumerate(pairs) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(pairs)} (image, phrase) pairs", flush=True)

    g = Gdino(device=args.device)
    t0 = time.time()
    n_new = 0
    for i, (fn, ph) in enumerate(pairs):
        before = g._cache_path(fn, ph).is_file()
        g.detect(fn, ph)
        n_new += (not before)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pairs)} new={n_new} {time.time()-t0:.0f}s "
                  f"({(time.time()-t0)/(i+1):.2f}s/pair)", flush=True)
    print(f"done: {len(pairs)} pairs, {n_new} new, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
