#!/usr/bin/env python
"""EXP-158 — cluster bootstrap (over images) for the shipped-vs-record val delta.

Review must-fix (2026-10-13): the shipped learned-recipe row (52.7061/75.0817)
beats the training-free termbox-union record (51.3182/74.9098) by +1.39 cIoU /
+0.17 gIoU. gIoU in particular is small; a cluster bootstrap over images (the
661 val images, samples within an image correlated) gives a CI for both deltas.

Inputs (jsonl per-row dumps written by the frozen val passes):
  * shipped arm : exp138 --apply-val --dump-rows ... (the exact shipped command)
  * record arm  : exp124 --records records_grefcoco_14229.jsonl
                  --study optuna_study_gref17.json --dump-rows ...
Each row: fn, rid, no_target, abstain, I, U, iou, gt_area — I/U are the cIoU
contributions (abstained targeted rows contribute gt_area to U), iou is the
gIoU contribution (correct empty no-target = 1).

    python exp/exp158_bootstrap_shipped_vs_record.py \
        --shipped exp/out_final/dump_shipped_rows.jsonl \
        --record exp/out_final/dump_record_rows.jsonl \
        --out exp/out_final/bootstrap_shipped_vs_record.json --reps 10000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RIS = Path(__file__).resolve().parent.parent


def load_rows(path):
    rows = [json.loads(line) for line in open(path)]
    return rows


def metrics(rows, idx):
    r = [rows[i] for i in idx]
    I = sum(x["I"] for x in r)
    U = sum(x["U"] for x in r)
    iou = sum(x["iou"] for x in r)
    n = len(r)
    ciou = 100.0 * I / U if U else 0.0
    giou = 100.0 * iou / n if n else 0.0
    return ciou, giou


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    sh = load_rows(args.shipped)
    rc = load_rows(args.record)
    assert len(sh) == len(rc), f"row count mismatch: {len(sh)} vs {len(rc)}"
    mismatch = sum(1 for a, b in zip(sh, rc)
                   if a["rid"] != b["rid"] or a["fn"] != b["fn"])
    assert mismatch == 0, f"{mismatch} row identity mismatches"

    n = len(sh)
    idx_all = list(range(n))
    c_sh, g_sh = metrics(sh, idx_all)
    c_rc, g_rc = metrics(rc, idx_all)
    print(f"shipped: cIoU {c_sh:.4f} gIoU {g_sh:.4f}")
    print(f"record : cIoU {c_rc:.4f} gIoU {g_rc:.4f}")
    print(f"delta  : cIoU {c_sh - c_rc:+.4f} gIoU {g_sh - g_rc:+.4f}")

    by_image = defaultdict(list)
    for i, row in enumerate(sh):
        by_image[row["fn"]].append(i)
    images = sorted(by_image)
    print(f"images: {len(images)}  rows: {n}")

    rng = np.random.RandomState(args.seed)
    dc = np.empty(args.reps)
    dg = np.empty(args.reps)
    for b in range(args.reps):
        pick = rng.randint(0, len(images), len(images))
        idx = []
        for p in pick:
            idx.extend(by_image[images[p]])
        cs, gs = metrics(sh, idx)
        cr, gr = metrics(rc, idx)
        dc[b] = cs - cr
        dg[b] = gs - gr
    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return round(float(a.mean()), 4), round(float(lo), 4), round(float(hi), 4)

    out = dict(
        n_rows=n, n_images=len(images), reps=args.reps, seed=args.seed,
        shipped=dict(cIoU=round(c_sh, 4), gIoU=round(g_sh, 4)),
        record=dict(cIoU=round(c_rc, 4), gIoU=round(g_rc, 4)),
        delta=dict(cIoU=round(c_sh - c_rc, 4), gIoU=round(g_sh - g_rc, 4)),
        bootstrap=dict(
            cIoU=dict(mean=ci(dc)[0], lo=ci(dc)[1], hi=ci(dc)[2],
                      excludes_zero=bool(np.percentile(dc, 2.5) > 0)),
            gIoU=dict(mean=ci(dg)[0], lo=ci(dg)[1], hi=ci(dg)[2],
                      excludes_zero=bool(np.percentile(dg, 2.5) > 0)),
        ),
        note=("cluster bootstrap over the val images (resample images with "
              "replacement, keep all their rows); 2.5/97.5 percentiles of the "
              "delta distribution"),
        sources=dict(shipped=str(args.shipped), record=str(args.record)),
    )
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["bootstrap"], indent=1))
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
