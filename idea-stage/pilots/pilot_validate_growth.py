#!/usr/bin/env python
"""Pilot 6 — Validate the connected score-gated growth rule out of sample.

Pilot 5 selected the rule and its epsilon on 250 first-sentence expressions of
RefCOCO val (seed 101). Selecting a hyper-parameter and reporting the number on
the same sample is exactly the mistake this pilot must avoid, so here the rule
is fixed and re-measured on data the selection never saw:

  * every sentence of the same references  (same images, new expressions)
  * a fresh random sample with a new seed   (new images and expressions)

Reports both mIoU (per-expression mean) and oIoU (dataset-level), so the gain
can be read against the numbers the papers actually publish.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

from glod.core import Config, decouple  # noqa: E402
from glod.data import RefDataset  # noqa: E402
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402

EPS_SWEEP = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
RULES = ["argmax", "growth", "gated_union", "largest_gate"]


def iou(pred, gt):
    union = int(np.logical_or(pred, gt).sum())
    return 1.0 if not union else int(np.logical_and(pred, gt).sum()) / union


def evaluate(c3, masks, area, eps):
    """Return the selected mask for each rule at a given relative epsilon."""
    rng = float(c3.max() - c3.min())
    thr = c3.max() - eps * rng
    cand = np.flatnonzero(c3 >= thr)
    am = int(np.argmax(c3))

    growth = masks[am].copy()
    for i in cand:
        if np.logical_and(masks[i], growth).any():
            np.logical_or(growth, masks[i], out=growth)

    gated_union = masks[cand[0]].copy()
    for i in cand[1:]:
        np.logical_or(gated_union, masks[i], out=gated_union)

    largest_gate = masks[cand[np.argmax(area[cand])]]

    return {"argmax": masks[am], "growth": growth,
            "gated_union": gated_union, "largest_gate": largest_gate}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--all-sentences", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("idea-stage/pilots/out_validate"))
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--score", choices=["c1", "c3"], default="c3",
                    help="c1 = plain HybridGL G2L appearance score; c3 = GLOD-RIS DOD score")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    config = Config()
    encoder = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                            REPRO / "checkpoints/ViT-B-16.pt",
                            device=args.device, cache_dir=REPRO / "cache/vision")

    inter = {r: {e: 0 for e in EPS_SWEEP} for r in RULES}
    union = {r: {e: 0 for e in EPS_SWEEP} for r in RULES}
    ious = {r: {e: [] for e in EPS_SWEEP} for r in RULES}
    oracle_ious, n = [], 0
    t0 = time.perf_counter()

    for ref in dataset.refs:
        ip = dataset.image_path(ref)
        if not ip.exists():
            if not args.allow_download:
                continue
            try:
                dataset.download_image(ref)
            except Exception:  # noqa: BLE001
                continue
        try:
            im = Image.open(ip).convert("RGB")
            gt = dataset.ground_truth(ref)
        except Exception:  # noqa: BLE001
            continue
        masks, fg, bg, _ = encoder.extract(im, file_sha256(ip))
        if masks.shape[1:] != gt.shape:
            continue

        fg_t, bg_t = torch.from_numpy(fg).float(), torch.from_numpy(bg).float()
        fused, _, _ = decouple(fg_t, bg_t, masks.mean((1, 2)), config)
        area = masks.reshape(len(masks), -1).mean(1)

        sentences = [s["sent"] for s in ref["sentences"]] if args.all_sentences \
            else [ref["sentences"][0]["sent"]]
        for sent in sentences:
            t = F.normalize(encoder.text(sent).float().reshape(-1), dim=0)
            base_feat = F.normalize(fg_t, dim=-1) if args.score == "c1" else F.normalize(fused, dim=-1)
            c3 = (base_feat @ t).numpy()
            oracle_ious.append(max(iou(masks[i], gt) for i in range(len(masks))))
            for e in EPS_SWEEP:
                preds = evaluate(c3, masks, area, e)
                for r in RULES:
                    v = iou(preds[r], gt)
                    ious[r][e].append(v)
                    inter[r][e] += int(np.logical_and(preds[r], gt).sum())
                    union[r][e] += int(np.logical_or(preds[r], gt).sum())
            n += 1
        if len(oracle_ious) % 100 == 0:
            print(f"[{len(oracle_ious)} expr] {time.perf_counter()-t0:.0f}s", flush=True)

    out = dict(tag=args.tag, dataset=args.dataset, split=args.split, seed=args.seed,
               limit=args.limit, all_sentences=args.all_sentences, n_expressions=n,
               elapsed_s=round(time.perf_counter() - t0, 1),
               oracle_mIoU=float(np.mean(oracle_ious)), score=args.score,
               table=[dict(rule=r, eps=e, mIoU=float(np.mean(ious[r][e])),
                           oIoU=inter[r][e] / union[r][e] if union[r][e] else None)
                      for r in RULES for e in EPS_SWEEP])
    with (args.output / f"summary_{args.tag}.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in ("tag", "n_expressions", "oracle_mIoU")}, indent=2), flush=True)
    for row in sorted(out["table"], key=lambda d: -d["mIoU"])[:12]:
        print(f"  {row['rule']:14s} eps={row['eps']:<5} mIoU={row['mIoU']:.4f} oIoU={row['oIoU']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
