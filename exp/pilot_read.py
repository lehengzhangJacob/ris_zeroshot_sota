#!/usr/bin/env python
"""Read the gRefCOCO existence pilot straight from the box cache.

The pilot writes one json per (image, phrase) as it goes, so this reports the
N_acc / T_acc confusion on whatever fraction has landed — no need to wait for
the full 600-pair run.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import CACHE_DIR  # noqa: E402
import hashlib  # noqa: E402


def kind_of(raw: str) -> str:
    t = (raw or "").strip()
    if re.search(r"\bnone\b", t, re.I):
        return "none"
    if re.search(r"\d", t):
        return "box"
    return "garbage"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.records)]
    rng = random.Random(args.seed)
    nt = [r for r in recs if r["no_target"]]
    tg = [r for r in recs if not r["no_target"]]
    sample = rng.sample(nt, args.n_per_class) + rng.sample(tg, args.n_per_class)

    def path_for(fn, ph):
        key = hashlib.sha1(f"llava-onevision-qwen2-7b-ov|{fn}|{ph.lower()}".encode()).hexdigest()
        return CACHE_DIR / (key + ".json")

    stats = {}
    for src in ("sent", "noun"):
        for cls in ("nt", "tg"):
            stats[(src, cls)] = dict(n=0, none=0, box=0, garbage=0)
    miss = 0
    for r in sample:
        cls = "nt" if r["no_target"] else "tg"
        for src, key in (("sent", "sentence"), ("noun", "noun_phrase")):
            ph = (r["parsed"].get(key) or "").strip()
            if not ph:
                continue
            p = path_for(r["file_name"], ph)
            if not p.is_file():
                miss += 1
                continue
            d = json.loads(p.read_text())
            k = kind_of(d.get("raw", ""))
            s = stats[(src, cls)]
            s["n"] += 1
            s[k] += 1
    print(f"pending/unqueried: {miss}")
    for src in ("sent", "noun"):
        a, b = stats[(src, "nt")], stats[(src, "tg")]
        if a["n"] == 0 or b["n"] == 0:
            print(f"[{src}] too few: nt={a['n']} tg={b['n']}")
            continue
        n_acc = a["none"] / a["n"]
        t_acc = 1 - b["none"] / b["n"]
        print(f"[{src}] NT n={a['n']:4d} none={a['none']:4d} garb={a['garbage']:3d} "
              f"| TG n={b['n']:4d} none={b['none']:4d} garb={b['garbage']:3d} "
              f"| N_acc={n_acc:.3f} T_acc={t_acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
