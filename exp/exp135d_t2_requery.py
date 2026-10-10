#!/usr/bin/env python
"""EXP-135d — T2 determinism: identical-prompt greedy re-query.

Re-queries 12 cached (image, sentence) pairs with the identical box prompt in a
throwaway cache namespace and compares the raw answers with the cached ones.
Merges the result into exp/out_final/stability.json (T2 entry).

    CUDA_VISIBLE_DEVICES=1 python exp/exp135d_t2_requery.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import CACHE_DIR, MODEL_TAG, LlavaBoxer8bit, PROMPT  # noqa: E402

RECORDS = Path("exp/out_extreme/records_grefcoco_train_tune.jsonl")
OUTF = Path("exp/out_final")


def main() -> int:
    recs = [json.loads(l) for l in open(RECORDS)]
    pairs = []
    seen = set()
    for r in recs:
        ph = (r["parsed"].get("sentence") or "").strip()
        if not ph:
            continue
        k = (r["file_name"], ph.lower())
        if k in seen:
            continue
        bp = CACHE_DIR / (hashlib.sha1(
            f"{MODEL_TAG}|{r['file_name']}|{ph.lower()}".encode()).hexdigest() + ".json")
        if bp.is_file():
            seen.add(k)
            pairs.append((r["file_name"], ph))
        if len(pairs) >= 12:
            break
    print(f"{len(pairs)} cached prompts selected for re-query", flush=True)

    b = LlavaBoxer8bit(device="cuda:0")
    b.TAG = "requery"          # throwaway namespace; base cache untouched
    b.PROMPT_TEXT = PROMPT     # byte-identical prompt text
    out = b.box_many(pairs, batch_size=2)

    n_match = n_match_box = 0
    diffs = []
    for (fn, ph), (box2, raw2) in zip(pairs, out):
        bp = CACHE_DIR / (hashlib.sha1(f"{MODEL_TAG}|{fn}|{ph.lower()}".encode()).hexdigest() + ".json")
        d = json.loads(bp.read_text())
        raw1, box1 = (d.get("raw") or "").strip(), d.get("box")
        if raw1 == (raw2 or "").strip():
            n_match += 1
        if box1 == box2:
            n_match_box += 1
        else:
            diffs.append(dict(file=fn, phrase=ph, cached_raw=raw1, requery_raw=(raw2 or "").strip()))
    print(f"raw identical {n_match}/{len(pairs)}; boxes identical {n_match_box}/{len(pairs)}", flush=True)

    p = OUTF / "stability.json"
    st = json.loads(p.read_text())
    for t in st["tests"]:
        if t["name"].startswith("T2"):
            t["measurements"]["greedy_requery_prompts"] = len(pairs)
            t["measurements"]["greedy_requery_raw_identical"] = n_match
            t["measurements"]["greedy_requery_boxes_identical"] = n_match_box
            t["measurements"]["greedy_requery_diffs"] = diffs
    p.write_text(json.dumps(st, indent=2, ensure_ascii=False))
    print(f"updated {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
