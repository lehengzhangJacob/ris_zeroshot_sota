#!/usr/bin/env python
"""EXP-119 — val-side SAM(det1) precompute for the change-14 config.

Config-independent artifacts for the full-val pass:
  (a) GDINO detections per (file_name, term)  -> exp/gdino.py cache (shared)
  (b) SAM box-prompt 3-mask outputs from the top-1 GDINO detection box,
      stored stride-2 packbits npz under exp/out_val_gdino_sam/, keyed by
      sha1(MODEL_TAG|gdino_sam|file|term)

Terms: noun_phrase + parsed nouns (deduped, max 3) — the exp84 term list, on
val records (exp/out_extreme/records_grefcoco.jsonl), targeted records only
(no-target records never consult boxes in the rule).

Resumable; sharded over records.  Nothing here reads the val GT beyond what
the shipped pipeline already does; no tuning happens in this script.

    CUDA_VISIBLE_DEVICES=1 python exp/exp119_precompute_val_sam_d1.py [--shard i --nshards n]
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
from exp67_ensemble_eval import REPRO  # noqa: E402
from vlm_boxes import IMAGES, MODEL_TAG  # noqa: E402

OUTF = Path("exp/out_final")
OUTDIR = Path("exp/out_val_gdino_sam_d1")
STRIDE = 2
RECORDS = Path("exp/out_extreme/records_grefcoco.jsonl")


def terms_of(parsed):
    out, seen = [], set()
    for t in [parsed.get("noun_phrase") or ""] + list(parsed.get("nouns") or []):
        t = t.strip()
        k = t.lower()
        if t and k not in seen:
            seen.add(k)
            out.append(t)
    return out[:3]


def spath(file_name, term):
    key = hashlib.sha1(
        f"{MODEL_TAG}|gdino_sam_d1|{file_name}|{term.lower()}".encode()).hexdigest()
    return OUTDIR / (key + ".npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    recs = []
    for line in open(RECORDS):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        ts = terms_of(rec["parsed"])
        if ts:
            recs.append((rec["file_name"], ts))
    # dedupe (file, term) pairs preserving order
    pairs, seen = [], set()
    for fn, ts in recs:
        for t in ts:
            if (fn, t) not in seen:
                seen.add((fn, t))
                pairs.append((fn, t))
    if args.limit:
        pairs = pairs[: args.limit]
    pairs = [p for i, p in enumerate(pairs) if i % args.nshards == args.shard]
    todo_gd = [p for p in pairs if True]  # Gdino caches internally
    todo_sam = [p for p in pairs if not spath(*p).is_file()]
    print(f"shard {args.shard}/{args.nshards}: pairs {len(pairs)} sam-todo {len(todo_sam)}",
          flush=True)

    from gdino import Gdino
    g = Gdino(device=args.device)

    t0 = time.time()
    n_det = 0
    for fn, term in pairs:
        g.detect(fn, term)
        n_det += 1
        if n_det % 500 == 0:
            print(f"  detections {n_det}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"detections done ({time.time()-t0:.0f}s)", flush=True)

    if not todo_sam:
        return 0

    import torch
    from PIL import Image
    sys.path.insert(0, str(REPRO))
    from glod.vision import load_upstream
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](
        checkpoint=str(REPRO / "checkpoints/sam_vit_h_4b8939.pth")).to("cuda").eval()
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    last_img = None
    done = 0
    t1 = time.time()
    for fn, term in todo_sam:
        boxes, scores = g.detect(fn, term)
        if not boxes:
            spath(fn, term).write_bytes(b"")  # negative marker handled by reader
            continue
        if fn != last_img:
            predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
            last_img = fn
        if len(boxes) < 2:
            # no second detection: write a marker so the reader skips it
            spath(fn, term).write_bytes(b"")
            continue
        ms, _, _ = predictor.predict(box=np.asarray(boxes[1], np.float32),
                                     multimask_output=True)
        ms = np.stack([m.astype(bool)[::STRIDE, ::STRIDE] for m in ms])
        shape = ms.shape
        np.savez_compressed(spath(fn, term),
                            bits=np.packbits(ms), shape=np.asarray(shape))
        done += 1
        if done % 200 == 0:
            print(f"  sam {done}/{len(todo_sam)} ({time.time()-t1:.0f}s)", flush=True)
    print(f"sam done {done} in {time.time()-t1:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
