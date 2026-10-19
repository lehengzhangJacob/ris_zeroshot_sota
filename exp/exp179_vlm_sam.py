#!/usr/bin/env python
"""EXP-179 — SAM box-prompted masks from VLM term/clause boxes (DINO-free).

The detector-bearing gRefCOCO path gets most of its mask quality from SAM
box-prompted masks on detector boxes.  The DINO-free path has never had
box-prompted SAM at all: it only selected from the precomputed SAM-H pool.
This script prompts SAM ViT-H with the *VLM* boxes (LLaVA clause boxes for
``clause``, LLaVA noun-term boxes for ``noun``) and caches the three
multimask outputs per (image, phrase), downsampled by 2 to match the records.

    CUDA_VISIBLE_DEVICES=0 python exp/exp179_vlm_sam.py \
        --records exp/out_extreme/records_grefcoco_train_tune.jsonl \
        --source clause --shard 0 --nshards 1
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp108_gdino_study import lpath  # noqa: E402
from exp169_gref_rescue import base_box_path  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
from method_gres import clauses  # noqa: E402
from vlm_boxes import IMAGES, MODEL_TAG  # noqa: E402

STRIDE = 2
OUT = {"clause": Path("exp/out_vlm_sam_clause"),
       "noun": Path("exp/out_vlm_sam_noun")}


def cache_path(source, fn, phrase):
    key = hashlib.sha1(
        f"{MODEL_TAG}|vlm_sam_{source}|{fn}|{phrase.lower()}".encode()).hexdigest()
    return OUT[source] / (key + ".npz")


def pairs_for(records, source):
    seen = set()
    out = []
    for line in open(records):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        fn = rec["file_name"]
        sent = (rec["parsed"].get("sentence") or "").strip()
        phrases = clauses(sent) if source == "clause" else terms_of(rec["parsed"])
        for ph in phrases:
            ph = (ph or "").strip()
            if not ph:
                continue
            k = (fn, ph.lower())
            if k in seen:
                continue
            p = base_box_path(fn, ph) if source == "clause" else lpath(fn, ph)
            if not p.is_file():
                continue
            box = json.loads(p.read_text()).get("box")
            if box is None:
                continue
            seen.add(k)
            if cache_path(source, fn, ph).is_file():
                continue
            out.append((fn, ph, box))
    # group by image
    by_img = {}
    for fn, ph, box in out:
        by_img.setdefault(fn, []).append((ph, box))
    return by_img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--source", choices=["clause", "noun"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    OUT[args.source].mkdir(parents=True, exist_ok=True)
    by_img = pairs_for(args.records, args.source)
    items = sorted(by_img.items())
    items = items[args.shard::args.nshards]
    n_pairs = sum(len(v) for _, v in items)
    print(f"{args.source}: {len(items)} images, {n_pairs} pairs to prompt "
          f"(shard {args.shard}/{args.nshards})", flush=True)
    if not items:
        return 0

    import torch
    from PIL import Image
    sys.path.insert(0, str(REPRO))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp/literature/repos/sam"))
    from glod.vision import load_upstream
    from segment_anything import SamPredictor
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](
        checkpoint=str(REPRO / "checkpoints/sam_vit_h_4b8939.pth")).to("cuda").eval()
    predictor = SamPredictor(sam)

    t0 = time.time()
    for i, (fn, plist) in enumerate(items):
        img = np.asarray(Image.open(IMAGES / fn).convert("RGB"))
        predictor.set_image(img)
        for ph, box in plist:
            ms, scores, _ = predictor.predict(
                box=np.asarray(box, np.float32), multimask_output=True)
            ms = ms.astype(bool)[:, ::STRIDE, ::STRIDE]
            shape = ms.shape
            bits = np.packbits(ms.reshape(shape[0], -1), axis=1)
            np.savez_compressed(cache_path(args.source, fn, ph),
                                shape=np.asarray(shape), bits=bits,
                                box=np.asarray(box, np.float32),
                                scores=np.asarray(scores, np.float32))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            done = sum(len(v) for _, v in items[:i + 1])
            print(f"  {i+1}/{len(items)} images, {done}/{n_pairs} pairs "
                  f"({el:.0f}s, {done/max(el,1):.1f}/s)", flush=True)
    print(f"done {n_pairs} pairs in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
