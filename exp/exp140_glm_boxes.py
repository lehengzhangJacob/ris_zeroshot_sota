#!/usr/bin/env python
"""EXP-140 — GLM-4.1V-Thinking-Flash box cache (component-matched GLOD-RIS MLLM).

Operator instruction 2026-10-10: evaluate the shipped box-prior path with
GLOD-RIS's exact MLLM (GLM-4.1V-Thinking-Flash, Zhipu API) in place of the
local LLaVA prior.  Same prompt template; only the box source changes.

  * model   glm-4.1v-thinking-flash (endpoint open.bigmodel.cn, paas/v4)
  * prompt  byte-identical to the LLaVA template (vlm_boxes.PROMPT)
  * images  base64 data URIs
  * answer  choices[0].message.content, boxes wrapped in <|begin_of_box|>...
            coordinates are 0-1000 normalised (verified against cached LLaVA
            boxes: IoU ~0.65 after scaling, 0.0 unscaled)
  * cache   exp/out_glm_boxes/<sha1(glm|file|phrase)>.json (raw + box + usage)
            -- every response is cached; the run is resumable
  * sample  per dataset a seeded 2,000-expression sample of the official val
            records (seed 20261010), written as records_<ds>_glm2000.jsonl so
            the evaluation re-runs exp40 on exactly these rows

    python exp/exp140_glm_boxes.py --dataset refcoco
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import PROMPT  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
IMAGES = REPRO / "data/refer/images/train2014"
OUT = Path("exp/out_glm_boxes")
OUT.mkdir(parents=True, exist_ok=True)
KEY = Path("/home/msj_team/Jacob/sys/glm_key").read_text().strip()
MODEL = "glm-4.1v-thinking-flash"
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
RECORDS = {
    "refcoco": Path("exp/out_extreme/records_refcoco.jsonl"),
    "refcocoplus": Path("exp/out_extreme/records_refcocoplus.jsonl"),
    "refcocog": Path("exp/out_extreme/records_refcocog.jsonl"),
}
SAMPLE_N = 2000
SEED = 20261010


def cache_path(fn, phrase):
    key = hashlib.sha1(f"glm|{fn}|{phrase.lower()}".encode()).hexdigest()
    return OUT / (key + ".json")


def parse_box(text, w, h):
    if not text:
        return None
    m = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", text, re.S)
    seg = m.group(1) if m else text
    nums = re.findall(r"-?\d+\.?\d*", seg)
    if len(nums) < 4:
        if re.search(r"\bnone\b", text, re.I):
            return None
        return None
    x0, y0, x1, y1 = [float(v) for v in nums[:4]]
    # 0-1000 normalised (verified); scale to pixels
    if max(x0, y0, x1, y1) > max(w, h) + 1:
        x0, x1 = x0 * w / 1000.0, x1 * w / 1000.0
        y0, y1 = y0 * h / 1000.0, y1 * h / 1000.0
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [x0, y0, x1, y1]


def query(fn, phrase):
    p = cache_path(fn, phrase)
    if p.is_file():
        return "cached"
    from PIL import Image
    img = Image.open(IMAGES / fn).convert("RGB")
    w, h = img.width, img.height
    b64 = base64.b64encode((IMAGES / fn).read_bytes()).decode()
    body = {"model": MODEL, "temperature": 0.0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": PROMPT.format(phrase=phrase)}]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    last = None
    for attempt in range(4):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=240) as r:
                resp = json.loads(r.read())
            msg = resp["choices"][0]["message"]
            content = msg.get("content") or ""
            box = parse_box(content, w, h)
            p.write_text(json.dumps(dict(
                box=box, raw=content[:400], reasoning=(msg.get("reasoning_content") or "")[:200],
                w=w, h=h, usage=resp.get("usage"), latency=round(time.time() - t0, 2),
                model=MODEL), ensure_ascii=False))
            return "new"
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (attempt + 1))
    return f"fail:{type(last).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    from glod.data import RefDataset
    ds_name = {"refcocoplus": "refcoco+"}.get(args.dataset, args.dataset)
    ds = RefDataset(REPRO / "data/refer", ds_name, "val")
    by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(RECORDS[args.dataset])]
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(recs)), min(SAMPLE_N, len(recs))))
    sample = [recs[i] for i in idx]
    out_path = Path(f"exp/out_extreme/records_{args.dataset}_glm2000.jsonl")
    with out_path.open("w") as fh:
        for r in sample:
            fh.write(json.dumps(r) + "\n")
    print(f"{args.dataset}: sampled {len(sample)} of {len(recs)} -> {out_path}", flush=True)

    pairs, seen = [], set()
    for r in sample:
        ph = (r["parsed"].get("sentence") or "").strip()
        ref = by_id.get(r["ref_id"])
        if not ph or ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        k = (fn, ph.lower())
        if k in seen:
            continue
        seen.add(k)
        pairs.append((fn, ph))
    todo = [p for p in pairs if not cache_path(*p).is_file()]
    print(f"{len(pairs)} unique pairs, {len(todo)} to query, {args.workers} workers", flush=True)

    n_new = n_fail = n_none = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(query, fn, ph): (fn, ph) for fn, ph in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res == "new":
                n_new += 1
            elif res.startswith("fail"):
                n_fail += 1
                print(f"  FAIL {futs[fut]}: {res}", flush=True)
            if i % 100 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(todo)} ({el:.0f}s, {i/el:.1f}/s, fails {n_fail})", flush=True)
    print(f"done: new {n_new}, fails {n_fail} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
