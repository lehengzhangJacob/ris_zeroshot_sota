#!/usr/bin/env python
"""EXP-137 — box-prompted SAM caches for the held-out gRefCOCO TRAIN rows.

The held-out tuning population has no SAM caches (the studies computed those
masks live).  A selector trained on the train population must see the same
candidate parts as the deployed rule, so this mirrors exp109a/exp119 exactly
on `records_grefcoco_train_tune.jsonl`:

  det0: SAM's 3 masks from the top-1 GroundingDINO detection box
        -> exp/out_train_gdino_sam/<sha1(MODEL_TAG|gdino_sam|file|term)>.npz
  det1: same from the 2nd detection box (score >= 0.25)
        -> exp/out_train_gdino_sam_d1/<sha1(MODEL_TAG|gdino_sam_d1|file|term)>.npz

Resumable (skips existing cache files).  No fitting, no val records.

    CUDA_VISIBLE_DEVICES=1 python exp/exp137_train_side_caches.py
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
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
from vlm_boxes import IMAGES, MODEL_TAG  # noqa: E402

RECORDS = Path("exp/out_extreme/records_grefcoco_train_tune.jsonl")
OUT0 = Path("exp/out_train_gdino_sam")
OUT1 = Path("exp/out_train_gdino_sam_d1")
STRIDE = 2


def spath0(fn, term):
    key = hashlib.sha1(f"{MODEL_TAG}|gdino_sam|{fn}|{term.lower()}".encode()).hexdigest()
    return OUT0 / (key + ".npz")


def spath1(fn, term):
    key = hashlib.sha1(f"{MODEL_TAG}|gdino_sam_d1|{fn}|{term.lower()}".encode()).hexdigest()
    return OUT1 / (key + ".npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT0.mkdir(parents=True, exist_ok=True)
    OUT1.mkdir(parents=True, exist_ok=True)

    pairs, seen = [], set()
    for line in open(RECORDS):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        for t in terms_of(rec["parsed"]):
            k = (rec["file_name"], t)
            if k not in seen:
                seen.add(k)
                pairs.append(k)
    print(f"{len(pairs)} (file, term) pairs", flush=True)

    from gdino import Gdino
    g = Gdino(device=args.device)

    import torch
    from PIL import Image
    sys.path.insert(0, "/home/msj_team/Jacob/sys/glod_ris_repro")
    from glod.vision import load_upstream
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](checkpoint="/home/msj_team/Jacob/sys/glod_ris_repro/checkpoints/sam_vit_h_4b8939.pth")
    sam = sam.to("cuda").eval()
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    last_img = None
    done0 = done1 = 0
    t0 = time.time()
    for fn, term in pairs:
        boxes, scores = g.detect(fn, term)
        need0 = not spath0(fn, term).is_file()
        need1 = not spath1(fn, term).is_file() and len(boxes) > 1 and scores[1] >= 0.25
        if not (need0 or need1):
            continue
        if not boxes:
            if need0:
                spath0(fn, term).write_bytes(b"")
            continue
        if fn != last_img:
            predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
            last_img = fn
        if need0:
            ms, _, _ = predictor.predict(box=np.asarray(boxes[0], np.float32),
                                         multimask_output=True)
            ms = np.stack([m.astype(bool)[::STRIDE, ::STRIDE] for m in ms])
            np.savez_compressed(spath0(fn, term), bits=np.packbits(ms),
                                shape=np.asarray(ms.shape))
            done0 += 1
        if need1:
            ms, _, _ = predictor.predict(box=np.asarray(boxes[1], np.float32),
                                         multimask_output=True)
            ms = np.stack([m.astype(bool)[::STRIDE, ::STRIDE] for m in ms])
            np.savez_compressed(spath1(fn, term), bits=np.packbits(ms),
                                shape=np.asarray(ms.shape))
            done1 += 1
        if (done0 + done1) % 200 == 0 and (done0 + done1):
            print(f"  det0 {done0} det1 {done1} ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: det0 {done0}, det1 {done1} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
