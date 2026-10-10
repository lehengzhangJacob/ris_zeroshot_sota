#!/usr/bin/env python
"""EXP-58 — how far can a direct existence probe push N_acc?

The wall analysis (gref_wall.json) says gIoU needs N_acc ~0.85 given the
measurable mask quality, and the shipped existence signal (box-or-none +
learned max_p) reaches 0.684 on val.  The box prompt *forces* a box, which may
understate the VLM's presence judgment.  This probes two dedicated existence
formulations on a balanced held-out TRAIN sample (never val):

  exists : "Is the {phrase} present in this image? yes/no"      (binary)
  conf   : "How likely is the {phrase} present? 0-100"          (graded ROC)

and reports N_acc at several T_acc levels against the same-sample baseline
(the shipped box-or-none answer).

    python exp/exp58_exists_probe.py --shard 0 --nshards 2 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import (CACHE_DIR, MODEL_TAG, ExistsBoxer8bit,  # noqa: E402
                       ConfBoxer8bit, LlavaBoxer8bit)


def split_pairs(pairs, shard, nshards):
    return [p for i, p in enumerate(pairs) if i % nshards == shard]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--phase", default="conf", choices=["exists", "conf"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.records)]
    import hashlib

    def box_cached(r):
        ph = (r["parsed"].get("sentence") or "").strip()
        k = hashlib.sha1(f"{MODEL_TAG}|{r['file_name']}|{ph.lower()}".encode()).hexdigest()
        return (CACHE_DIR / (k + ".json")).is_file()

    recs = [r for r in recs if r.get("parsed") and box_cached(r)]
    rng = random.Random(args.seed)
    nt = [r for r in recs if r["no_target"]]
    tg = [r for r in recs if not r["no_target"]]
    sample = rng.sample(nt, min(args.n_per_class, len(nt))) + \
        rng.sample(tg, min(args.n_per_class, len(tg)))
    pairs, meta = [], []
    for r in sample:
        ph = (r["parsed"].get("sentence") or "").strip()
        if ph:
            pairs.append((r["file_name"], ph))
            meta.append(bool(r["no_target"]))
    print(f"{len(pairs)} pairs ({sum(meta)} no-target)", flush=True)
    pairs = split_pairs(pairs, args.shard, args.nshards)
    meta = split_pairs(meta, args.shard, args.nshards)
    print(f"shard {args.shard}/{args.nshards}: {len(pairs)}", flush=True)

    cls = {"exists": ExistsBoxer8bit, "conf": ConfBoxer8bit}[args.phase]
    boxer = cls(device=args.device)
    t0 = time.time()
    out = boxer.box_many(pairs, batch_size=1)
    print(f"done in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for (fn, ph), nt_flag, (box, raw) in zip(pairs, meta, out):
        t = (raw or "").strip().lower()
        if args.phase == "exists":
            m = re.search(r"\b(yes|no)\b", t)
            val = (1.0 if m.group(1) == "yes" else 0.0) if m else None
        else:
            m = re.search(r"\d+(?:\.\d+)?", t)
            val = float(m.group(0)) if m else None
        rows.append(dict(no_target=nt_flag, raw=raw[:60], val=val))
    dst = Path(f"exp/out_final/gref_exists_probe_sh{args.shard}.json")
    dst.write_text(json.dumps(rows, indent=1))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
