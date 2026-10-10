#!/usr/bin/env python
"""EXP-72 — forced-choice pairwise VLM comparison of candidate regions.

The diagnosed failure is instance disambiguation: on descriptive expressions
the frozen scores pick the wrong instance.  Region verification (exp65/66)
failed because a crop-level yes/no question can confirm but cannot reject —
the VLM answers yes to any plausible instance of the category.  A forced
choice removes the escape hatch: show two candidate crops (mask region on grey
background) and ask which one shows the phrase.  No "none" option exists, so
the answer is a preference even when both crops are plausible.

Pairs are the 6 unordered pairs of the classifier's top-4 proposals; each pair
is queried in BOTH orders (position-bias control).  Answers cache per ordered
pair, so the selection study (exp73) is a free re-analysis.  Shardable,
resumes from cache.

    python exp72_pairwise_extract.py --split train --shard 0 --nshards 4
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
from vlm_boxes import CACHE_DIR, IMAGES, MODEL_TAG  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
STRIDE = 2
TOPK = 4
PROMPT_PAIR = ("The first image and the second image are two candidate regions "
               "from the same photo. Which one shows the {phrase}? "
               "Answer with only the word first or the word second.")


def ppath(file_name, i, j, phrase):
    key = hashlib.sha1(
        f"{MODEL_TAG}|pair|{file_name}|{i}|{j}|{phrase.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def render_crop(image, mask_full, pad=0.08):
    """Mask region on grey background, cropped to the mask bbox (exp65 path)."""
    ys, xs = np.nonzero(mask_full)
    if not len(ys):
        return None
    h, w = mask_full.shape
    y0, y1 = max(0, ys.min()), min(h, ys.max() + 1)
    x0, x1 = max(0, xs.min()), min(w, xs.max() + 1)
    ph, pw = int((y1 - y0) * pad), int((x1 - x0) * pad)
    y0, y1 = max(0, y0 - ph), min(h, y1 + ph)
    x0, x1 = max(0, x0 - pw), min(w, x1 + pw)
    arr = np.asarray(image).copy()
    sel = mask_full[y0:y1, x0:x1]
    sub = arr[y0:y1, x0:x1]
    grey = np.full_like(sub, 127)
    sub = np.where(sel[..., None], sub, grey)
    from PIL import Image as PILImage
    return PILImage.fromarray(sub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--classifier", type=Path,
                    default=Path("exp/out_final/optuna_study_gref5.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from exp62_clause_selector import load_record, make_clf
    from grefer_data import GreferDataset
    from vlm_boxes import LlavaBoxer8bit

    st = json.loads(args.classifier.read_text())["best_params"]
    params = dict(C=st["C"]) if st["clf"] == "logreg" else dict(
        n_estimators=st["n_estimators"], max_depth=st["max_depth"],
        learning_rate=st["learning_rate"])

    gd_tr = GreferDataset(REPRO / "data/refer", "train")
    by_tr = {r["ref_id"]: r for r in gd_tr.refs}
    gd = GreferDataset(REPRO / "data/refer", args.split)
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    train_rows = [r for r in (load_record(json.loads(l), gd_tr, by_tr, dict(mc), {})
                              for l in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
                  if r is not None]
    clf = make_clf(st["clf"], params)
    clf.fit(np.concatenate([r["X"] for r in train_rows]),
            np.concatenate([r["y"] for r in train_rows]))
    print(f"classifier refit on {len(train_rows)} train rows", flush=True)

    recs = [json.loads(l) for l in open(args.records)]
    if args.limit:
        recs = recs[:args.limit]
    jobs = []          # (record, phrase, [ordered queries (i,j) missing])
    for i, rec in enumerate(recs):
        if i % args.nshards != args.shard:
            continue
        row = load_record(rec, gd, by_id, mc, gc)
        if row is None:
            continue
        p = clf.predict_proba(row["X"])[:, 1]
        top = [int(j) for j in np.argsort(p)[::-1][:TOPK]]
        ph = (rec["parsed"].get("sentence") or "").strip()
        todo = []
        for a, b in combinations(top, 2):
            for i2, j2 in ((a, b), (b, a)):
                if not ppath(rec["file_name"], i2, j2, ph).is_file():
                    todo.append((i2, j2))
        if todo:
            jobs.append((rec, row, ph, todo))
    n_q = sum(len(t) for *_, t in jobs)
    print(f"shard {args.shard}/{args.nshards}: {n_q} pair queries over {len(jobs)} records",
          flush=True)
    if not jobs:
        return 0

    boxer = LlavaBoxer8bit(device=args.device)
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
            if cl is None or cr is None:
                ppath(rec["file_name"], i2, j2, ph).write_text(
                    json.dumps(dict(raw="", skip="empty")))
                continue
            conv = [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"},
                {"type": "text", "text": PROMPT_PAIR.format(phrase=ph)}]}]
            prompt = boxer.proc.apply_chat_template(conv, add_generation_prompt=True)
            inp = boxer.proc(images=[cl, cr], text=prompt, return_tensors="pt").to(
                boxer.device)
            inp = {k: (v.to(torch.bfloat16) if v.dtype.is_floating_point else v)
                   for k, v in inp.items()}
            with torch.inference_mode():
                out = boxer.model.generate(**inp, max_new_tokens=8, do_sample=False)
            raw = boxer.proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[0]
            ppath(rec["file_name"], i2, j2, ph).write_text(
                json.dumps(dict(raw=raw[:60])))
            n_done += 1
            if n_done % 200 == 0:
                el = time.time() - t0
                print(f"  {n_done}/{n_q} ({el:.0f}s, {60*n_done/el:.0f}/min)", flush=True)
    print(f"done {n_done} queries in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
