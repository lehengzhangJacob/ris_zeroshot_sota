#!/usr/bin/env python
"""EXP-88 — per-term (nounbox) extraction over the full gRefCOCO val (change 10).

The termbox study (exp87) won its held-out gate: calib r 0.5506 vs the shipped
pair rule's 0.4268 on the val-matched population.  This extracts the per-term
boxes for every TARGETED val record (union of noun_phrase + parsed nouns, max
3 terms, deduped); no-target records never need them (abstention runs first,
the pair rule is the fallback when term boxes are absent/empty).

Shardable by stable pair hash, resumes from cache.  Namespace `nounbox`
(shared with the held-out extraction).

    CUDA_VISIBLE_DEVICES=1 python exp/exp88_nounbox_val.py --shard 0 --nshards 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import IMAGES  # noqa: E402


def npath(file_name, term):
    from vlm_boxes import CACHE_DIR, MODEL_TAG
    key = hashlib.sha1(
        f"{MODEL_TAG}|nounbox|{file_name}|{term.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only-uncovered", action="store_true",
                    help="restrict to records with NO usable term box yet")
    args = ap.parse_args()

    pairs = set()
    for line in open(args.records):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        ts = [rec["parsed"].get("noun_phrase") or ""] + list(rec["parsed"].get("nouns") or [])
        seen = set()
        for t in ts:
            t = t.strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            if len(seen) > 3:
                break
            pairs.add((rec["file_name"], t))
    def _useful(fn, tt):
        p = npath(fn, tt)
        if not p.is_file():
            return False
        try:
            d = json.loads(p.read_text())
        except Exception:
            return False
        return not (d.get("box") is None and (d.get("raw") or "") == "")

    if args.only_uncovered:
        need_record = set()
        for line in open(args.records):
            r = json.loads(line)
            if r.get("no_target"):
                continue
            ts, seen = [], set()
            for tt in [r["parsed"].get("noun_phrase") or ""] + list(r["parsed"].get("nouns") or []):
                tt = tt.strip()
                if tt and tt.lower() not in seen:
                    seen.add(tt.lower()); ts.append(tt)
            ts = ts[:3]
            if not any(_useful(r["file_name"], tt) for tt in ts):
                for tt in ts:
                    need_record.add((r["file_name"], tt))
        pairs = {p for p in pairs if p in need_record}

    todo = []
    for fn, t in sorted(pairs):
        if not _useful(fn, t):
            h = int(hashlib.sha1(f"{fn}|{t.lower()}".encode()).hexdigest()[:8], 16)
            if h % args.nshards == args.shard:
                todo.append((fn, t))
    print(f"shard {args.shard}/{args.nshards}: {len(todo)} queries", flush=True)
    if not todo:
        return 0

    import torch
    from PIL import Image
    from vlm_boxes import LlavaBoxer8bit, PROMPT, parse_box

    boxer = LlavaBoxer8bit(device=args.device)
    t0 = time.time()
    n = 0
    for fn, term in todo:
        try:
            image = Image.open(IMAGES / fn).convert("RGB")
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
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {fn} {term!r}: {type(e).__name__}", flush=True)
            continue
        if n % 200 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(todo)} ({el:.0f}s, {60*n/max(1,el):.0f}/min)", flush=True)
    print(f"done {n} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
