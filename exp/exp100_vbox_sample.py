#!/usr/bin/env python
"""EXP-100 — LLaVA-Video per-term boxes on the val-matched sample (change 12).

The hand-rolled LLaVA-Video assembly (exp99) passes the coherence test and
returns plausible normalized boxes.  This extracts boxes for the 291-row
val-matched held-out sample's terms into the `vbox` cache namespace
(single-query decode; bf16 path batches safely but keep it simple and
resumable).  Then exp101 measures box quality vs the object GT and compares
against LLaVA-OneVision's 0.580 mean / 63.5% select>=0.50.

    CUDA_VISIBLE_DEVICES=3 python exp/exp100_vbox_sample.py --shard 0 --nshards 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import CACHE_DIR  # noqa: E402

OUTF = Path("exp/out_final")
SAMPLE = OUTF / "nounbox_sample_records.json"
MODEL_TAG_V = "llava-video-7b-qwen2"


def vpath(file_name, term):
    key = hashlib.sha1(
        f"{MODEL_TAG_V}|vbox|{file_name}|{term.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from exp99_llavavideo import build, generate, IMG
    from PIL import Image
    from vlm_boxes import parse_box

    recs = json.loads(SAMPLE.read_text())
    pairs = []
    for i, rec in enumerate(recs):
        for t in rec["terms"]:
            if not vpath(rec["file_name"], t).is_file():
                h = int(hashlib.sha1(f"{rec['file_name']}|{t.lower()}".encode()
                                     ).hexdigest()[:8], 16)
                if h % args.nshards == args.shard:
                    pairs.append((rec["file_name"], t))
    pairs = list(dict.fromkeys(pairs))
    print(f"shard {args.shard}/{args.nshards}: {len(pairs)} vbox queries",
          flush=True)
    if not pairs:
        return 0

    vision, proj, llm, inl, tok, cfg = build(args.device)
    t0 = time.time()
    n = 0
    img_cache = {}
    for fn, term in pairs:
        img = img_cache.get(fn)
        if img is None:
            img = Image.open(f"{IMG}/{fn}").convert("RGB")
            img_cache = {fn: img}
        q = ("Locate the " + term + " in this image. Reply with ONLY the bounding "
             "box as [x0,y0,x1,y1] in pixel coordinates measured from the "
             "top-left corner, or the single word none if it does not appear in "
             "the image.")
        try:
            raw = generate(vision, proj, llm, inl, tok, cfg, img, q,
                           args.device, max_new_tokens=48)
            tail = raw.split("assistant")[-1].strip()
            box = parse_box(tail, img.width, img.height)
            vpath(fn, term).write_text(json.dumps(
                dict(box=box, raw=tail[:200], w=img.width, h=img.height)))
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {fn} {term!r}: {type(e).__name__}", flush=True)
            continue
        if n % 50 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(pairs)} ({el:.0f}s, {60*n/max(1,el):.1f}/min)", flush=True)
    print(f"done {n} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
