#!/usr/bin/env python
"""EXP-171 — full-validation GLM-4.1V box extraction (gate directive 2026-10-18).

Extends exp140's 2,000-expression samples to every unique (image, full
sentence) pair of the three official val splits.  Same cache namespace and
prompt, so the 5,914 sample responses are reused.  Resumable; responses are
cached before anything else runs.

  python exp/exp171_glm_full.py --dataset refcoco --workers 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp140_glm_boxes as base  # noqa: E402

base.OUT.mkdir(parents=True, exist_ok=True)


def query(fn, phrase):
    p = base.cache_path(fn, phrase)
    if p.is_file():
        return "cached"
    import base64
    from PIL import Image
    img = Image.open(base.IMAGES / fn).convert("RGB")
    w, h = img.width, img.height
    b64 = base64.b64encode((base.IMAGES / fn).read_bytes()).decode()
    body = {"model": base.MODEL, "temperature": 0.0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": base.PROMPT.format(phrase=phrase)}]}]}
    req = urllib.request.Request(base.URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {base.KEY}",
                                          "Content-Type": "application/json"})
    last = None
    for attempt in range(8):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=240) as r:
                resp = json.loads(r.read())
            msg = resp["choices"][0]["message"]
            content = msg.get("content") or ""
            box = base.parse_box(content, w, h)
            p.write_text(json.dumps(dict(
                box=box, raw=content[:400], reasoning=(msg.get("reasoning_content") or "")[:200],
                w=w, h=h, usage=resp.get("usage"), latency=round(time.time() - t0, 2),
                model=base.MODEL, source="exp171-full-val"), ensure_ascii=False))
            return "new"
        except urllib.error.HTTPError as exc:
            last = exc
            # 429/5xx: long bounded backoff, rate limits are the expected failure
            time.sleep(min(30.0 * (attempt + 1), 120.0) if exc.code in (429, 500, 502, 503, 504)
                       else 3.0 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(5.0 * (attempt + 1), 40.0))
    return f"fail:{type(last).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    from glod.data import RefDataset
    ds_name = {"refcocoplus": "refcoco+"}.get(args.dataset, args.dataset)
    ds = RefDataset(base.REPRO / "data/refer", ds_name, "val")
    by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(base.RECORDS[args.dataset])]
    pairs, seen = [], set()
    for r in recs:
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
    todo = [p for p in pairs if not base.cache_path(*p).is_file()]
    print(f"{args.dataset}: {len(pairs)} unique pairs, {len(todo)} to query, "
          f"{args.workers} workers", flush=True)

    n_new = n_fail = 0
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
            if i % 200 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(todo)} ({el:.0f}s, {i/el:.2f}/s, fails {n_fail})", flush=True)
    print(f"done: new {n_new}, fails {n_fail} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
