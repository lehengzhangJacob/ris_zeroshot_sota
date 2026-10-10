#!/usr/bin/env python
"""Resilient extractor: run the box (or exists) probe over the full val split.

Waits for a GPU with enough free memory, runs the sharded extractor (which
skips already-cached pairs, so every restart resumes for free), and loops
until nothing remains.  GPU health is re-tested each round, so the dead
context on GPU3 and fluctuating shared memory are handled automatically.

    python exp/extract_loop.py --kind box       # box-or-none probe
    python exp/extract_loop.py --kind exists    # yes/no presence probe
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

RECORDS = Path("exp/out_extreme/records_grefcoco.jsonl")
NEED_MB = 11_500


def cache_path(kind, fn, ph):
    name = MODEL_TAG if kind == "box" else f"{MODEL_TAG}|{kind}"
    return CACHE_DIR / (hashlib.sha1(f"{name}|{fn}|{ph.lower()}".encode()).hexdigest() + ".json")


def remaining(kind, shard=0, nshards=1):
    """Uncached pairs, restricted to this shard's slice of the pair list."""
    seen, need, i = set(), 0, 0
    for line in open(RECORDS):
        r = json.loads(line)
        ph = (r["parsed"].get("sentence") or "").strip()
        if not ph:
            continue
        k = (r["file_name"], ph.lower())
        if k in seen:
            continue
        seen.add(k)
        if i % nshards == shard and not cache_path(kind, r["file_name"], ph).is_file():
            need += 1
        i += 1
    return need


def gpu_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        i, m = line.split(",")
        gpus.append((int(i), int(m.strip().split()[0])))
    return sorted(gpus, key=lambda t: -t[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["box", "exists"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--poll", type=int, default=420)
    ap.add_argument("--gpu", type=int, default=-1,
                    help="pin to this GPU index (-1 = any)")
    args = ap.parse_args()

    while True:
        nb = remaining("box", args.shard, args.nshards)
        ne = remaining("exists", args.shard, args.nshards)
        if nb + ne == 0:
            print("[loop] both probes complete", flush=True)
            return 0
        args.kind = "exists" if ne > 0 else "box"
        need = ne if ne > 0 else nb
        print(f"[loop] remaining: box={nb} exists={ne} -> working on {args.kind}",
              flush=True)
        allg = gpu_free()
        if args.gpu >= 0:
            pinned = [(g, m) for g, m in allg if g == args.gpu and m >= NEED_MB]
            cands = [g for g, m in pinned] or [g for g, m in allg if m >= NEED_MB][:1]
        else:
            cands = [g for g, m in allg if m >= NEED_MB]
        print(f"[{args.kind}] {need} pairs remain; GPUs with >= {NEED_MB} MB: {cands}",
              flush=True)
        if not cands:
            time.sleep(args.poll)
            continue
        gpu = cands[0]
        cmd = ([sys.executable, "-u", "exp/exp47_extract_gref.py", "--records", str(RECORDS),
                "--phrase-source", "sent", "--shard", str(args.shard),
                "--nshards", str(args.nshards),
                "--batch-size", "1", "--device", "cuda:0"]
               if args.kind == "box" else
               [sys.executable, "-u", "/tmp/run_exists_full.py", "0", "1"])
        env = {"CUDA_VISIBLE_DEVICES": str(gpu),
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "PATH": "/usr/bin:/bin",
               "PYTHONPATH": "/home/msj_team/Jacob/sys/glod_ris_repro"}
        print(f"[{args.kind}] launching on GPU{gpu}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, env=env, cwd="/home/msj_team/Jacob/research/ris").returncode
        print(f"[{args.kind}] worker exited rc={rc} after {time.time()-t0:.0f}s", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
