#!/usr/bin/env python
"""Pilot 5 — Score-gated selection rules.

The qualitative dump showed the failure structure directly: on RefCOCO val the
CLIP region-text score separates a mask with IoU 0.00 from one with IoU 0.93 by
about 0.01 (e.g. 0.326 vs 0.314), and in 9 of 12 inspected failures the correct
mask is *larger* than the one the score picks, at a slightly lower score.

That shape suggests the score is not useless but is being read at the wrong
resolution: a near-tie should be broken structurally (by mask extent), not by
the third decimal of a cosine. This pilot tests that family of rules, all
training-free and all single-hyper-parameter:

  R1  largest mask with  score >= max - eps * (max - min)      [scale-free gate]
  R2  largest mask with  score >= max - eps                    [absolute gate]
  R3  largest of the top-p fraction by score
  R4  smallest of the top-p fraction by score
  R5  union of all masks above a scale-free gate                [gated composition]
  R6  union of all masks above a scale-free gate, only if each
      member overlaps the argmax mask                           [connected growth]

R1/R2 are the direct test of "break near-ties by extent"; R5/R6 test whether a
*gated* union beats the naive top-k union that Pilots 1 showed to be harmful.
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


def iou(pred, gt):
    union = int(np.logical_or(pred, gt).sum())
    return 1.0 if not union else int(np.logical_and(pred, gt).sum()) / union


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("idea-stage/pilots/out_gated"))
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    config = Config()
    encoder = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                            REPRO / "checkpoints/ViT-B-16.pt",
                            device=args.device, cache_dir=REPRO / "cache/vision")

    EPS_REL = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    EPS_ABS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
    FRACS = [0.05, 0.10, 0.20, 0.30, 0.50]

    acc = {f"R1_largest_rel_eps={e}": [] for e in EPS_REL}
    acc.update({f"R2_largest_abs_eps={e}": [] for e in EPS_ABS})
    acc.update({f"R3_largest_topfrac={f}": [] for f in FRACS})
    acc.update({f"R4_smallest_topfrac={f}": [] for f in FRACS})
    acc.update({f"R5_gated_union_rel_eps={e}": [] for e in EPS_REL})
    acc.update({f"R6_connected_growth_rel_eps={e}": [] for e in EPS_REL})
    acc["R0_argmax_dod"] = []

    t0 = time.perf_counter()
    n = 0
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
        expr = ref["sentences"][0]["sent"]
        t = F.normalize(encoder.text(expr).float().reshape(-1), dim=0)
        c3 = (F.normalize(fused, dim=-1) @ t).numpy()
        area = masks.reshape(len(masks), -1).mean(1)
        n_m = len(masks)

        acc["R0_argmax_dod"].append(iou(masks[int(np.argmax(c3))], gt))

        rng = float(c3.max() - c3.min())
        for e in EPS_REL:
            thr = c3.max() - e * rng
            cand = np.flatnonzero(c3 >= thr)
            acc[f"R1_largest_rel_eps={e}"].append(iou(masks[cand[np.argmax(area[cand])]], gt))
            u = masks[cand[0]].copy()
            for i in cand[1:]:
                np.logical_or(u, masks[i], out=u)
            acc[f"R5_gated_union_rel_eps={e}"].append(iou(u, gt))
            am = int(np.argmax(c3))
            grow = masks[am].copy()
            for i in cand:
                if np.logical_and(masks[i], grow).any():
                    np.logical_or(grow, masks[i], out=grow)
            acc[f"R6_connected_growth_rel_eps={e}"].append(iou(grow, gt))

        for e in EPS_ABS:
            thr = c3.max() - e
            cand = np.flatnonzero(c3 >= thr)
            acc[f"R2_largest_abs_eps={e}"].append(iou(masks[cand[np.argmax(area[cand])]], gt))

        order = np.argsort(-c3)
        for f in FRACS:
            p = max(1, int(round(f * n_m)))
            sub = order[:p]
            acc[f"R3_largest_topfrac={f}"].append(iou(masks[sub[np.argmax(area[sub])]], gt))
            acc[f"R4_smallest_topfrac={f}"].append(iou(masks[sub[np.argmin(area[sub])]], gt))

        n += 1
        if n % 50 == 0:
            print(f"[{n}] {time.perf_counter()-t0:.0f}s", flush=True)

    res = sorted(({"rule": k, "mIoU": float(np.mean(v))} for k, v in acc.items() if v),
                 key=lambda d: -d["mIoU"])
    base = float(np.mean(acc["R0_argmax_dod"]))
    for r in res:
        r["delta_vs_argmax"] = r["mIoU"] - base
    out = dict(n_expressions=n, oracle_single=None, baseline_argmax_dod=base, results=res)
    with (args.output / "summary.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
