#!/usr/bin/env python
"""EXP-151 — repair degenerate LLaVA box cache entries for the test columns.

A fraction of the test-split box queries returned degenerate greedy decodes
(the model looping on '!'), which the parser correctly reports as box=None.
Greedy decoding is deterministic, so a plain re-run reproduces the failure;
the repair re-queries exactly those entries with a guarded decode
(repetition_penalty=1.15, max_new_tokens=48) and records the change so the
paper's scope statement stays honest (same model + prompt; decoder differs on
the repaired subset only).

    CUDA_VISIBLE_DEVICES=1 python exp/exp151_fix_degenerate_boxes.py \
        --records exp/out_test_columns/records_tc_refcoco_testA.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hashlib

from vlm_boxes import CACHE_DIR, IMAGES, MODEL_TAG, PROMPT, parse_box  # noqa: E402


def npath(file_name, phrase):
    """The sentence-prompt box cache used by boxes_for_records (default namespace)."""
    key = hashlib.sha1(f"{MODEL_TAG}|{file_name}|{phrase.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def degenerate(raw: str) -> bool:
    r = (raw or "").strip()
    if not r:
        return True
    if set(r) <= set("!"):
        return True
    return bool(re.fullmatch(r"(.{1,3})\1{6,}", r))  # short-cycle loops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from pathlib import Path as _P

    sys.path.insert(0, "/home/msj_team/Jacob/sys/glod_ris_repro")
    from glod.data import RefDataset

    ds = RefDataset(_P("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer"),
                    args.dataset, args.split)
    by_id = {r["ref_id"]: r for r in ds.refs}
    bad = []
    pairs, seen = set(), set()
    for path in args.records:
        for line in open(path):
            rec = json.loads(line)
            if rec.get("no_target"):
                continue
            ref = by_id.get(rec["ref_id"])
            if ref is None:
                continue
            fn = ds.images[ref["image_id"]]["file_name"]
            phrase = (rec["parsed"].get("sentence") or "").strip()
            if not phrase:
                continue
            key = (fn, phrase.lower())
            if key in pairs:
                continue
            pairs.add(key)
            p = npath(fn, phrase)
            if p.is_file():
                d = json.loads(p.read_text())
                if d.get("box") is None and degenerate(d.get("raw", "")):
                    bad.append((fn, phrase, p))
    missing = sum(1 for fn, t in pairs if not npath(fn, t).is_file())
    print(f"{len(bad)} degenerate entries to repair; {missing} still missing", flush=True)
    if not bad:
        return 0

    import torch
    from PIL import Image
    from vlm_boxes import LlavaBoxer

    boxer = LlavaBoxer(device=args.device)
    t0 = time.time()
    fixed = 0
    cur_fn, image = None, None
    for fn, term, path in bad:
        if fn != cur_fn:
            image = Image.open(IMAGES / fn).convert("RGB")
            cur_fn = fn
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT.format(phrase=term)}]}]
        prompt = boxer.proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = boxer.proc(images=image, text=prompt, return_tensors="pt").to(boxer.device)
        inp = {k: (v.to(torch.bfloat16) if v.dtype.is_floating_point else v)
               for k, v in inp.items()}
        with torch.inference_mode():
            out = boxer.model.generate(**inp, max_new_tokens=48, do_sample=False,
                                       repetition_penalty=1.15)
        raw = boxer.proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]
        box = parse_box(raw, image.width, image.height)
        old = json.loads(path.read_text())
        path.write_text(json.dumps(dict(
            box=box, raw=raw[:200], w=image.width, h=image.height,
            repaired=dict(previous_raw=(old.get("raw") or "")[:120],
                          decoder="repetition_penalty=1.15,max_new_tokens=48"))))
        fixed += box is not None
        if (fixed + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {fixed} repaired of {len(bad)} tried ({el:.0f}s)", flush=True)
    print(f"repaired {fixed}/{len(bad)} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
