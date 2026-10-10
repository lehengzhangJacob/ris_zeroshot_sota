#!/usr/bin/env python
"""EXP-84 — per-noun boxes from the parse: feasibility on a sample.

exp83 showed the two-object-box construction has headroom (perfect per-object
boxes + the current bbox-IoU pick feature reach cumulative targeted r 0.7151
vs the shipped rule's 0.5538 on the same rows).  The remaining unknown is
whether LLaVA can produce per-noun boxes good enough.  This queries, on a
300-record sample of MULTI-OBJECT targeted held-out-train records, a box per
parsed term (noun_phrase + parsed nouns, deduped, max 3), cache namespace
`nounbox`, then measures:

  (a) per-term box vs the best-matching object mask (IoU, select rate);
  (b) the deployed proxy: per-term box -> pool mask with max bbox-IoU vs that
      box -> union of picks -> cumulative targeted r on the sample, against
      the shipped rule's r on the same sample.

Shardable; resumes from cache.  Sample only — the full extraction follows only
if the proxy shows the gain.

    CUDA_VISIBLE_DEVICES=0 python exp/exp84_nounbox_sample.py --shard 0 --nshards 4
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
from vlm_boxes import CACHE_DIR, IMAGES, MODEL_TAG  # noqa: E402

OUTF = Path("exp/out_final")
SAMPLE = OUTF / "nounbox_sample_records.json"


def npath(file_name, term):
    key = hashlib.sha1(
        f"{MODEL_TAG}|nounbox|{file_name}|{term.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def terms_of(parsed):
    """noun_phrase + parsed nouns, deduped case-insensitively, max 3."""
    out, seen = [], set()
    for t in [parsed.get("noun_phrase") or ""] + list(parsed.get("nouns") or []):
        t = t.strip()
        k = t.lower()
        if t and k not in seen:
            seen.add(k)
            out.append(t)
    return out[:3]


def build_sample():
    recs = []
    for line in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        n_obj = int(rec.get("n_targets") or 0)
        ts = terms_of(rec["parsed"])
        if n_obj >= 2 and len(ts) >= 2:
            recs.append({"file_name": rec["file_name"],
                         "sentence": rec["parsed"].get("sentence") or "",
                         "terms": ts, "ref_id": rec["ref_id"],
                         "cache_file": rec.get("cache_file"),
                         "sc": rec.get("sc"), "score_gem": rec.get("score_gem"),
                         "gt_area": rec.get("gt_area")})
    recs = recs[:300]
    SAMPLE.write_text(json.dumps(recs))
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    recs = json.loads(SAMPLE.read_text()) if SAMPLE.is_file() else build_sample()
    jobs = []
    for i, rec in enumerate(recs):
        if i % args.nshards != args.shard:
            continue
        todo = [(rec["file_name"], t) for t in rec["terms"]
                if not npath(rec["file_name"], t).is_file()]
        if todo:
            jobs.append((rec, todo))
    n_q = sum(len(t) for *_, t in jobs)
    print(f"shard {args.shard}/{args.nshards}: {n_q} term queries over {len(jobs)} records",
          flush=True)
    if not jobs:
        return 0

    import torch
    from PIL import Image
    from vlm_boxes import LlavaBoxer8bit, PROMPT

    boxer = LlavaBoxer8bit(device=args.device)
    t0 = time.time()
    n_done = 0
    for rec, todo in jobs:
        image = Image.open(IMAGES / rec["file_name"]).convert("RGB")
        for fn, term in todo:
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
            from vlm_boxes import parse_box
            box = parse_box(raw, image.width, image.height)
            npath(fn, term).write_text(json.dumps(
                dict(box=box, raw=raw[:200], w=image.width, h=image.height)))
            n_done += 1
            if n_done % 100 == 0:
                el = time.time() - t0
                print(f"  {n_done}/{n_q} ({el:.0f}s, {60*n_done/el:.0f}/min)", flush=True)
    print(f"done {n_done} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
