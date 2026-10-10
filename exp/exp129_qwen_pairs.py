#!/usr/bin/env python
"""EXP-129 — Qwen2.5-VL forced-choice on the two-object held-out slice (trigger #2, part B).

exp128 measured Qwen's TERM-BOX quality (part A). Part B tests the capability
that closed the LLaVA route: instance disambiguation under forced choice
(exp72 protocol; LLaVA measured at chance — 29.8% agreement with the GT-best
proposal vs 25% baseline, exp73 diagnostic). Same job construction as exp72
(record -> load_record -> shipped study-5 classifier top-4 -> 6 pairs x 2
orders, mask-on-grey crops via exp72.render_crop), Qwen backend, cache
namespace "qwen2.5vl|pair|..." so re-analysis is free. Records: the 262-row
two-object sample (nounbox_sample_records.json) — the pre-registered slice.

  QWEN_MODEL_ID=/home/msj_team/Jacob/models/qwen2.5-vl-7b-clean \
  CUDA_VISIBLE_DEVICES=1 python exp/exp129_qwen_pairs.py [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp72_pairwise_extract import PROMPT_PAIR, STRIDE, TOPK, render_crop  # noqa: E402
from vlm_boxes import CACHE_DIR  # noqa: E402

OUTF = Path("exp/out_final")


def qpath(file_name, i, j, phrase):
    key = hashlib.sha1(
        f"qwen2.5vl|pair|{file_name}|{i}|{j}|{phrase.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from exp62_clause_selector import load_record, make_clf
    from grefer_data import GreferDataset

    st = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    train_rows = [r for r in (load_record(json.loads(l), gd, by_id, dict(mc), {})
                              for l in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
                  if r is not None]
    clf = make_clf("logreg", dict(C=st["C"]))
    clf.fit(np.concatenate([r["X"] for r in train_rows]),
            np.concatenate([r["y"] for r in train_rows]))
    print(f"classifier refit on {len(train_rows)} train rows", flush=True)

    tune_by_rid = {}
    for l in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        tr = json.loads(l)
        tune_by_rid[tr["ref_id"]] = tr
    recs = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    if args.limit:
        recs = recs[:args.limit]
    jobs = []
    for rec in recs:
        tr = tune_by_rid.get(rec["ref_id"])
        if tr is None:
            continue
        row = load_record(tr, gd, by_id, mc, gc)
        if row is None:
            continue
        p = clf.predict_proba(row["X"])[:, 1]
        top = [int(j) for j in np.argsort(p)[::-1][:TOPK]]
        ph = (tr["parsed"].get("sentence") or "").strip()
        todo = []
        for a, b in combinations(top, 2):
            for i2, j2 in ((a, b), (b, a)):
                if not qpath(rec["file_name"], i2, j2, ph).is_file():
                    todo.append((i2, j2))
        if todo:
            jobs.append((rec, row, ph, todo))
    n_q = sum(len(t) for *_, t in jobs)
    print(f"{n_q} pair queries over {len(jobs)} records ({len(recs)} in sample)", flush=True)
    if not jobs:
        print("nothing to do", flush=True)
        return 0

    from vlm_boxes import Qwen2Boxer
    from vlm_boxes import IMAGES
    boxer = Qwen2Boxer(device="cuda:0")
    t0 = time.time()
    n_done = 0
    for rec, row, ph, todo in jobs:
        image = Image.open(IMAGES / rec["file_name"]).convert("RGB")
        masks = row["masks"]
        crops = {}
        for i2, j2 in todo:
            for idx in (i2, j2):
                if idx not in crops:
                    m_full = np.kron(masks[idx], np.ones((STRIDE, STRIDE), bool))
                    m_full = m_full[: image.height, : image.width]
                    crops[idx] = render_crop(image, m_full)
            cl, cr = crops[i2], crops[j2]
            if cl is None or cr is None or min(cl.width, cl.height, cr.width,
                                               cr.height) < 28:
                # Qwen's smart_resize rejects any side < 28 px; record the skip
                qpath(rec["file_name"], i2, j2, ph).write_text(
                    json.dumps(dict(raw="", skip="empty")))
                continue
            conv = [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"},
                {"type": "text", "text": PROMPT_PAIR.format(phrase=ph)}]}]
            prompt = boxer.proc.apply_chat_template(conv, tokenize=False,
                                                    add_generation_prompt=True)
            inp = boxer.proc(text=[prompt], images=[cl, cr],
                             return_tensors="pt").to(boxer.device)
            with torch.inference_mode():
                out = boxer.model.generate(**inp, max_new_tokens=8, do_sample=False)
            raw = boxer.proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[0]
            qpath(rec["file_name"], i2, j2, ph).write_text(
                json.dumps(dict(raw=raw[:80])))
            n_done += 1
            if n_done % 50 == 0:
                el = time.time() - t0
                print(f"  {n_done}/{n_q} ({el:.0f}s, {60*n_done/max(el,1):.0f}/min)",
                      flush=True)
    print(f"done {n_done} queries in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
