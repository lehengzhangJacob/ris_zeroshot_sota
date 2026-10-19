#!/usr/bin/env python
"""EXP-169b — merge the sharded rescue runs into gref_rescue.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUTF = Path("exp/out_final")


def main() -> int:
    parts = sorted(OUTF.glob("gref_rescue_part*.json"))
    if not parts:
        print("no shard parts found")
        return 1
    docs = [json.loads(p.read_text()) for p in parts]
    arms = list(docs[0]["arms"].keys())
    merged = {}
    for a in arms:
        I = sum(d["arms"][a]["_raw"]["I"] for d in docs)
        U = sum(d["arms"][a]["_raw"]["U"] for d in docs)
        iou = sum(d["arms"][a]["_raw"]["iou"] for d in docs)
        n = sum(d["arms"][a]["_raw"]["n"] for d in docs)
        n_nt = sum(d["arms"][a]["_raw"]["n_nt"] for d in docs)
        n_ab = sum(d["arms"][a]["_raw"]["n_nt_ab"] for d in docs)
        merged[a] = dict(cIoU=100.0 * I / U if U else 0.0,
                         gIoU=100.0 * iou / n if n else 0.0,
                         N_acc=100.0 * n_ab / n_nt if n_nt else 0.0,
                         n=n,
                         _raw=dict(I=I, U=U, iou=iou, n=n, n_nt=n_nt,
                                   n_nt_ab=n_ab))
    stats = {}
    for d in docs:
        for k, v in d["stats"].items():
            stats[k] = stats.get(k, 0) + v
    doc = dict(n=sum(d["n"] for d in docs), arms=merged, stats=stats,
               shards=len(docs), generated="exp169b_merge_rescue.py",
               note=docs[0].get("note", ""))
    (OUTF / "gref_rescue.json").write_text(json.dumps(doc, indent=2))
    for a, v in merged.items():
        print(f"  {a:24s} cIoU {v['cIoU']:7.3f}  gIoU {v['gIoU']:7.3f}  "
              f"N-acc {v['N_acc']:6.2f}")
    print(f"wrote {OUTF / 'gref_rescue.json'} from {len(docs)} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
