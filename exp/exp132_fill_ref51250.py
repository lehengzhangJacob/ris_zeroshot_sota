#!/usr/bin/env python
"""EXP-132 — cache fill for the one gRefCOCO val ref missing from the records.

Ref 51250 (9 no-target sentences, image COCO_train2014_000000320432.jpg) was
absent from records_grefcoco.jsonl because its image was missing from the
repro image dir.  The image has been restored; this script fills the VLM
caches the shipped rule reads for those 9 sentences:

  * sentence-level box   (base cache namespace, exp47-style query)
  * existence probes     (exists / exists2 / exists3 tags, exp58-style)

Terms/GDINO/SAM are not queried: every record of this ref is no-target, and
the shipped rule never consults term boxes on no-target records
(exp124: terms_by_rid skips no_target).  One model load, four cache phases.

    CUDA_VISIBLE_DEVICES=1 python exp/exp132_fill_ref51250.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import (PROMPT, PROMPT_EXISTS, PROMPT_EXISTS2,  # noqa: E402
                       PROMPT_EXISTS3, LlavaBoxer8bit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_ref51250.jsonl"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.records)]
    sentences = []
    for r in recs:
        ph = (r["parsed"].get("sentence") or "").strip()
        if ph:
            sentences.append((r["file_name"], ph))
    print(f"{len(recs)} records -> {len(sentences)} (image, sentence) pairs", flush=True)

    b = LlavaBoxer8bit(device=args.device)
    phases = [
        (None, PROMPT, "box"),
        ("exists", PROMPT_EXISTS, "exists"),
        ("exists2", PROMPT_EXISTS2, "exists2"),
        ("exists3", PROMPT_EXISTS3, "exists3"),
    ]
    summary = {}
    for tag, prompt, label in phases:
        b.TAG = tag
        b.PROMPT_TEXT = prompt
        t0 = time.time()
        out = b.box_many(sentences, batch_size=1)
        raws = [raw.strip() for _, raw in out]
        summary[label] = raws
        print(f"[{label}] {len(out)} answered in {time.time()-t0:.0f}s", flush=True)
        for (fn, ph), raw in zip(sentences, raws):
            print(f"   {label}: {ph[:48]!r} -> {raw[:60]!r}", flush=True)

    out_path = Path("exp/out_final/gref_ref51250_cache_fill.json")
    out_path.write_text(json.dumps(
        dict(records=str(args.records), image=sentences[0][0], summary=summary),
        indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
