#!/usr/bin/env python
"""EXP-164 — smaller-VLM box extraction (backbone audit arm a).

Runs Qwen2.5-VL-3B-Instruct (strictly smaller than both the shipped
LLaVA-OneVision-7B and the rival's cloud MLLM) over the full official val
splits AND the RefCOCO-family held-out tune refs, with the SAME prompt
template and parsing as the shipped boxer.  Boxes land in their own cache
namespace (qwen2.5vl3b|...) under exp/out_vlm_boxes/, one json per
(image, sentence) pair; existing 7B boxes are never touched.

    CUDA_VISIBLE_DEVICES=0 python exp/exp164_boxer3b.py --dataset refcoco --split val --shard 0 --nshards 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_DIR = "/home/msj_team/Jacob/models/qwen2.5-vl-3b-clean"
VAL_RECORDS = {
    "refcoco": "exp/out_extreme/records_refcoco.jsonl",
    "refcocoplus": "exp/out_extreme/records_refcocoplus.jsonl",
    "refcocog": "exp/out_extreme/records_refcocog.jsonl",
}
TUNE_RECORDS = {
    "refcoco": "exp/tuning/records_tune_refcoco.jsonl",
    "refcocoplus": "exp/tuning/records_tune_refcoco+.jsonl",
    "refcocog": "exp/tuning/records_tune_refcocog.jsonl",
}
REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

NS3B = "qwen2.5vl3b"


def path3b(file_name, phrase):
    from vlm_boxes import CACHE_DIR
    key = hashlib.sha1(f"{NS3B}|{file_name}|{phrase.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(VAL_RECORDS))
    ap.add_argument("--split", default="val", choices=["val", "tune"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from glod.data import RefDataset
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cache_map import _image_path
    from vlm_boxes import PROMPT, parse_box

    ds_name = {"refcoco": "refcoco", "refcocoplus": "refcoco+",
               "refcocog": "refcocog"}[args.dataset]
    split = "val" if args.split == "val" else "train"
    ds = RefDataset(REPRO / "data/refer", ds_name, split)
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    records_path = VAL_RECORDS[args.dataset] if args.split == "val" \
        else TUNE_RECORDS[args.dataset]

    recs = [json.loads(l) for l in open(records_path)]
    pairs, seen = [], set()
    for r in recs:
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        ph = (r["parsed"]["sentence"] or "").strip()
        if not ph or (fn, ph) in seen:
            continue
        seen.add((fn, ph))
        pairs.append((fn, ph))
    pairs = [p for i, p in enumerate(pairs) if i % args.nshards == args.shard]
    todo = [(fn, ph) for fn, ph in pairs if not path3b(fn, ph).is_file()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.split} {args.dataset}: {len(pairs)} pairs, "
          f"{len(todo)} to query (shard {args.shard}/{args.nshards})", flush=True)
    if not todo:
        return 0

    os.environ["QWEN_MODEL_ID"] = MODEL_DIR   # Qwen2Boxer.__init__ reads this
    from PIL import Image
    import torch
    from vlm_boxes import Qwen2Boxer

    boxer = Qwen2Boxer(device=args.device)
    t0 = time.time()
    n_done = n_parked = 0
    for fn, ph in todo:
        p = path3b(fn, ph)
        image = Image.open(_image_path(fn)).convert("RGB")
        box, raw = None, ""
        for attempt in range(2):
            try:
                if attempt == 1:
                    # pin the pixel bounds: the default processor can produce
                    # a token grid that mismatches the vision features on
                    # some aspect ratios (Qwen2.5-VL processor quirk)
                    boxer.proc = type(boxer.proc).from_pretrained(
                        boxer.model_id, min_pixels=4 * 28 * 28,
                        max_pixels=1280 * 28 * 28)
                msgs = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT.format(phrase=ph)}]}]
                prompt = boxer.proc.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                inp = boxer.proc(text=[prompt], images=[image],
                                 return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    out = boxer.model.generate(**inp, max_new_tokens=48,
                                               do_sample=False)
                raw = boxer.proc.batch_decode(
                    out[:, inp["input_ids"].shape[1]:],
                    skip_special_tokens=True)[0]
                box = parse_box(raw, image.width, image.height)
                break
            except Exception as exc:  # noqa: BLE001
                raw = f"error:{type(exc).__name__}"
        parked = box is None
        n_parked += int(parked)
        p.write_text(json.dumps(dict(box=box, raw=raw[:200],
                                     w=image.width, h=image.height,
                                     parked=parked)))
        n_done += 1
        if parked:
            print(f"  PARKED {fn} :: {ph[:40]!r} ({raw[:60]})", flush=True)
        if n_done % 250 == 0:
            rate = n_done / (time.time() - t0)
            print(f"  {n_done}/{len(todo)} ({rate:.2f}/s, eta "
                  f"{(len(todo)-n_done)/max(rate,1e-9)/60:.1f} min)", flush=True)
    print(f"done {n_done} ({n_parked} parked) in {time.time()-t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
