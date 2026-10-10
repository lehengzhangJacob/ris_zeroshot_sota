#!/usr/bin/env python
"""PILOT: does the VLM's `none` answer separate gRefCOCO no-target from targeted?

The shipped no-target decision is a threshold on CLIP scores; the ceiling
analysis capped that at N_acc 0.175.  The VLM prompt already asks for the word
`none` when the object is absent, so its *generative* existence judgment is a
different signal.  This pilot measures it on a small stratified sample before
committing to the full 14,220-record extraction.

Reports, per phrase source (sentence / noun):
    N_acc = P(answer none | no_target)     T_acc = P(box | targeted)
and distinguishes an explicit `none` from an unparseable answer.
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
from vlm_boxes import LlavaBoxer, LlavaBoxer8bit, Qwen2Boxer  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))


def classify(raw: str):
    """explicit none | parsed box | unparseable (a parse failure is NOT an answer)"""
    t = raw.strip()
    if re.search(r"\bnone\b", t, re.I):
        return "none"
    if re.search(r"\d", t):
        return "box"
    return "garbage"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--model", default="llava", choices=["llava", "qwen"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/gref_pilot.json"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.records)]
    rng = random.Random(args.seed)
    nt = [r for r in recs if r["no_target"]]
    tg = [r for r in recs if not r["no_target"]]
    sample = rng.sample(nt, args.n_per_class) + rng.sample(tg, args.n_per_class)
    # group by image so the per-image feature cache is warm
    sample.sort(key=lambda r: r["file_name"])

    pairs = []
    for r in sample:
        for src in ("sent", "noun"):
            ph = (r["parsed"]["sentence"] if src == "sent"
                  else r["parsed"]["noun_phrase"]) or ""
            ph = ph.strip()
            if ph:
                pairs.append((r["file_name"], ph, r["no_target"], src))
    print(f"{len(sample)} records -> {len(pairs)} (image, phrase) pairs", flush=True)
    pairs = [p for i, p in enumerate(pairs) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(pairs)} pairs", flush=True)

    if args.model == "qwen":
        boxer = Qwen2Boxer(device=args.device)
    else:
        boxer = (LlavaBoxer8bit if args.load_8bit else LlavaBoxer)(device=args.device)

    rows = []
    t0 = time.time()
    for i, (fn, ph, no_t, src) in enumerate(pairs):
        box, raw = boxer.box(fn, ph)
        rows.append(dict(file_name=fn, phrase=ph, no_target=no_t, src=src,
                         kind=classify(raw), box=box, raw=raw[:120]))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(pairs)}  {el:.0f}s ({el/(i+1):.2f}s/pair)", flush=True)

    print(f"\n=== existence separation ({time.time()-t0:.0f}s) ===")
    for src in ("sent", "noun"):
        s = [r for r in rows if r["src"] == src]
        notarget = [r for r in s if r["no_target"]]
        targeted = [r for r in s if not r["no_target"]]
        n_none = sum(r["kind"] == "none" for r in notarget)
        t_none = sum(r["kind"] == "none" for r in targeted)
        n_garb = sum(r["kind"] == "garbage" for r in notarget)
        t_garb = sum(r["kind"] == "garbage" for r in targeted)
        print(f"  [{src}] no-target n={len(notarget)}: none={n_none} ({100*n_none/len(notarget):.1f}%) "
              f"garbage={n_garb}")
        print(f"  [{src}] targeted  n={len(targeted)}: none={t_none} ({100*t_none/len(targeted):.1f}%) "
              f"garbage={t_garb}")
        print(f"  [{src}] N_acc={n_none/len(notarget):.3f}  T_acc(box)="
              f"{1 - t_none/len(targeted):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows=rows, model=args.model), indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
