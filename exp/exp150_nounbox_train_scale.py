#!/usr/bin/env python
"""EXP-150 — per-term LLaVA boxes for the scaled gRefCOCO TRAIN population.

Mirrors exp84's extraction loop byte-for-byte (same `nounbox` cache namespace,
same prompt, same 8-bit model), but reads arbitrary records files so the
6,000-reference train-side scale-up (exp146) gets the same box evidence the
val-side rule sees.  Resumable; shardable across GPUs.

    CUDA_VISIBLE_DEVICES=2 python exp/exp150_nounbox_train_scale.py \
        --records exp/out_train_scale/records_train_scale_s0.jsonl \
                  exp/out_train_scale/records_train_scale_s1.jsonl \
        --shard 0 --nshards 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import CACHE_DIR, IMAGES, MODEL_TAG  # noqa: E402


def npath(file_name, term):
    key = hashlib.sha1(
        f"{MODEL_TAG}|nounbox|{file_name}|{term.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def terms_of(parsed):
    out, seen = [], set()
    for t in [parsed.get("noun_phrase") or ""] + list(parsed.get("nouns") or []):
        t = t.strip()
        k = t.lower()
        if t and k not in seen:
            seen.add(k)
            out.append(t)
    return out[:3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    pairs, seen = [], set()
    for path in args.records:
        for line in open(path):
            rec = json.loads(line)
            if rec.get("no_target"):
                continue
            for t in terms_of(rec["parsed"]):
                key = (rec["file_name"], t.lower())
                if key in seen:
                    continue
                seen.add(key)
                if not npath(rec["file_name"], t).is_file():
                    pairs.append((rec["file_name"], t))
    pairs = [p for i, p in enumerate(pairs) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(pairs)} box queries to run", flush=True)
    if not pairs:
        return 0

    import torch
    from PIL import Image
    from vlm_boxes import LlavaBoxer8bit, PROMPT, parse_box

    boxer = LlavaBoxer8bit(device=args.device)
    t0 = time.time()
    current_fn, image = None, None
    for i, (fn, term) in enumerate(pairs):
        if fn != current_fn:
            image = Image.open(IMAGES / fn).convert("RGB")
            current_fn = fn
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT.format(phrase=term)}]}]
        prompt = boxer.proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = boxer.proc(images=image, text=prompt, return_tensors="pt").to(boxer.device)
        inp = {k: (v.to(torch.bfloat16) if v.dtype.is_floating_point else v)
               for k, v in inp.items()}
        with torch.inference_mode():
            out = boxer.model.generate(**inp, max_new_tokens=32, do_sample=False)
        raw = boxer.proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]
        box = parse_box(raw, image.width, image.height)
        npath(fn, term).write_text(json.dumps(
            dict(box=box, raw=raw[:200], w=image.width, h=image.height)))
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(pairs)} ({el:.0f}s, {60*(i+1)/el:.0f}/min)", flush=True)
    print(f"done {len(pairs)} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
