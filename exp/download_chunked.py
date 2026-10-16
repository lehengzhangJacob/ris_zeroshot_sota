#!/usr/bin/env python
"""Parallel chunked downloader for the backbone-audit checkpoints.

The mirror throttles single streams to ~0.4 MB/s; N range requests in parallel
restore aggregate throughput.  Chunks are stored as <dest>.part<N> and
assembled in order when all are complete; re-running resumes from the parts.

    python exp/download_chunked.py URL DEST [--threads 10] [--chunk-mb 64]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end, out):
    out = Path(out)
    want = end - start + 1
    if out.is_file() and out.stat().st_size == want:
        return out
    for attempt in range(6):
        try:
            done = out.stat().st_size if out.is_file() else 0
            if done > want:
                out.unlink()
                done = 0
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start+done}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(out, "ab") as fh:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
            if out.stat().st_size == want:
                return out
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None)
            wait = 30 + 20 * attempt if code == 429 else 2 + 3 * attempt
            time.sleep(wait)
            print(f"  retry {out.name} (HTTP {code}): sleeping {wait}s", flush=True)
    raise RuntimeError(f"chunk failed: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--chunk-mb", type=int, default=64)
    args = ap.parse_args()

    dest = Path(args.dest)
    if dest.is_file():
        print(f"{dest} already present ({dest.stat().st_size} bytes)")
        return 0
    total = head_size(args.url)
    chunk = args.chunk_mb << 20
    ranges = [(i, min(i + chunk - 1, total - 1))
              for i in range(0, total, chunk)]
    print(f"{dest.name}: {total/1e9:.2f} GB in {len(ranges)} chunks "
          f"x{args.threads} threads", flush=True)
    t0 = time.time()
    parts = [str(dest) + f".part{i}" for i in range(len(ranges))]
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(fetch_range, args.url, s, e, p)
                for (s, e), p in zip(ranges, parts)]
        for k, f in enumerate(futs):
            f.result()
            if (k + 1) % 5 == 0:
                got = sum(Path(p).stat().st_size for p in parts
                          if Path(p).is_file())
                rate = got / (time.time() - t0) / 1e6
                print(f"  {k+1}/{len(futs)} chunks, {got/1e9:.2f} GB, "
                      f"{rate:.1f} MB/s", flush=True)
    with open(dest, "wb") as out:
        for p in parts:
            with open(p, "rb") as fh:
                while True:
                    buf = fh.read(1 << 22)
                    if not buf:
                        break
                    out.write(buf)
            os.unlink(p)
    print(f"wrote {dest} ({dest.stat().st_size} bytes, "
          f"{(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
