#!/usr/bin/env python
"""Pilot 8 — win/loss breakdown and paired significance for the growth operator."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro"); sys.path.insert(0, str(REPRO))
from glod.core import Config, decouple
from glod.data import RefDataset
from glod.io import file_sha256
from glod.vision import VisionEncoder

def iou(p, g):
    u = int(np.logical_or(p, g).sum()); return 1.0 if not u else int(np.logical_and(p, g).sum()) / u

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="refcoco"); ap.add_argument("--split", default="val")
ap.add_argument("--limit", type=int, default=250); ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--all-sentences", action="store_true")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--output", type=Path, default=Path("idea-stage/pilots/out_sig"))
ap.add_argument("--tag", default="run"); ap.add_argument("--score", choices=["c1","c3"], default="c3")
ap.add_argument("--eps", type=float, default=0.3)
ap.add_argument("--mars-gate", type=float, default=0.95)
ap.add_argument("--boot", type=int, default=10000)
args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

ds = RefDataset(REPRO/"data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
cfg = Config()
enc = VisionEncoder(REPRO/"checkpoints/sam_vit_h_4b8939.pth", REPRO/"checkpoints/ViT-B-16.pt",
                    device=args.device, cache_dir=REPRO/"cache/vision")
A, B, C = [], [], []          # argmax, growth(our gate), growth(mars gate)
parts = {"sub": 0, "n_sub": 0}   # growth hurts on part-like targets?
n = 0; t0 = time.perf_counter()
for ref in ds.refs:
    ip = ds.image_path(ref)
    if not ip.exists(): continue
    try:
        im = Image.open(ip).convert("RGB"); gt = ds.ground_truth(ref)
    except Exception: continue
    masks, fg, bg, _ = enc.extract(im, file_sha256(ip))
    if masks.shape[1:] != gt.shape: continue
    fg_t, bg_t = torch.from_numpy(fg).float(), torch.from_numpy(bg).float()
    fused, _, _ = decouple(fg_t, bg_t, masks.mean((1, 2)), cfg)
    sl = [s["sent"] for s in ref["sentences"]] if args.all_sentences else [ref["sentences"][0]["sent"]]
    for sent in sl:
        t = F.normalize(enc.text(sent).float().reshape(-1), dim=0)
        bf = F.normalize(fg_t, dim=-1) if args.score == "c1" else F.normalize(fused, dim=-1)
        s = (bf @ t).numpy(); am = int(np.argmax(s)); rng = float(s.max() - s.min())
        def grow(cand):
            g = masks[am].copy()
            for i in cand:
                if np.logical_and(masks[i], g).any(): np.logical_or(g, masks[i], out=g)
            return g
        ga = iou(masks[am], gt)
        gb = iou(grow(np.flatnonzero(s >= s.max() - args.eps*rng)), gt)
        gc = iou(grow(np.flatnonzero(s >= args.mars_gate*s.max())), gt)
        A.append(ga); B.append(gb); C.append(gc)
        # sub-target stratum: GT smaller than the median proposal area
        med = float(np.median(masks.reshape(len(masks),-1).mean(1)))
        if gt.mean() < med:
            parts["n_sub"] += 1; parts["sub"] += (gb - ga)
        n += 1
    if n % 100 == 0: print(f"[{n}] {time.perf_counter()-t0:.0f}s", flush=True)

A, B, C = np.array(A), np.array(B), np.array(C)
rng_ = np.random.RandomState(0)
d_gb = B - A; d_gc = C - A
def boot_ci(d):
    idx = rng_.randint(0, len(d), size=(args.boot, len(d)))
    means = d[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
def perm_p(d):
    obs = d.mean()
    sign = rng_.choice([-1, 1], size=(args.boot, len(d)))
    null = (d * sign).mean(1)
    return float(np.mean(np.abs(null) >= abs(obs)))
res = dict(tag=args.tag, n=n, eps=args.eps,
           mIoU_argmax=float(A.mean()), mIoU_growth_ourgate=float(B.mean()),
           mIoU_growth_marsgate=float(C.mean()),
           delta_ourgate=float(d_gb.mean()), ci95_ourgate=boot_ci(d_gb), p_ourgate=perm_p(d_gb),
           delta_marsgate=float(d_gc.mean()), ci95_marsgate=boot_ci(d_gc), p_marsgate=perm_p(d_gc),
           win=int((d_gb > 1e-9).sum()), loss=int((d_gb < -1e-9).sum()), tie=int((np.abs(d_gb) <= 1e-9).sum()),
           win_marsgate=int((d_gc > 1e-9).sum()), loss_marsgate=int((d_gc < -1e-9).sum()),
           sub_target_mean_delta=float(parts["sub"]/parts["n_sub"]) if parts["n_sub"] else None,
           n_sub_targets=parts["n_sub"])
json.dump(res, open(args.output/f"summary_{args.tag}.json","w"), indent=2)
print(json.dumps(res, indent=2), flush=True)
