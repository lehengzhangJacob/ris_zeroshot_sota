#!/usr/bin/env python
"""Pilot 7 — Isolate the contribution against the closest published template.

A novelty sweep flagged MARS (arXiv:2504.07942, WACV 2026) Eq. (21)-(22) as
publishing "per-image-relative score gate -> union of the gated masks". If that
template already captures the gain, the growth rule is not a contribution.

This pilot factorises the design on held-out expressions:
  gate  in {max*0.95 (MARS), max - eps*(max-min) (ours)}
  merge in {unconstrained OR (MARS), connected growth from the argmax seed (ours)}
and also reports plain argmax, so the *merge* term and the *gate* term are each
measured independently.
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
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
ap.add_argument("--output", type=Path, default=Path("idea-stage/pilots/out_mars"))
ap.add_argument("--tag", default="run")
ap.add_argument("--score", choices=["c1", "c3"], default="c3")
args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

ds = RefDataset(REPRO/"data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
cfg = Config()
enc = VisionEncoder(REPRO/"checkpoints/sam_vit_h_4b8939.pth", REPRO/"checkpoints/ViT-B-16.pt",
                    device=args.device, cache_dir=REPRO/"cache/vision")
EPS = [0.10, 0.20, 0.30, 0.40, 0.60]
MARSTHR = [0.80, 0.90, 0.95, 0.98]

variants = ["argmax"]
variants += [f"mars_gate{g}_union" for g in MARSTHR]
variants += [f"range_gate{e}_union" for e in EPS]
variants += [f"mars_gate{g}_growth" for g in MARSTHR]
variants += [f"range_gate{e}_growth" for e in EPS]
acc = {v: [] for v in variants}
inter = {v: 0 for v in variants}; un = {v: 0 for v in variants}
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
    sent_list = [s["sent"] for s in ref["sentences"]] if args.all_sentences else [ref["sentences"][0]["sent"]]
    for sent in sent_list:
        t = F.normalize(enc.text(sent).float().reshape(-1), dim=0)
        bf = F.normalize(fg_t, dim=-1) if args.score == "c1" else F.normalize(fused, dim=-1)
        s = (bf @ t).numpy()
        am = int(np.argmax(s)); rng = float(s.max() - s.min())
        preds = {"argmax": masks[am]}
        for g in MARSTHR:
            cand = np.flatnonzero(s >= g * s.max())
            u_ = masks[cand[0]].copy()
            for i in cand[1:]: np.logical_or(u_, masks[i], out=u_)
            preds[f"mars_gate{g}_union"] = u_
            gr = masks[am].copy()
            for i in cand:
                if np.logical_and(masks[i], gr).any(): np.logical_or(gr, masks[i], out=gr)
            preds[f"mars_gate{g}_growth"] = gr
        for e in EPS:
            cand = np.flatnonzero(s >= s.max() - e * rng)
            u_ = masks[cand[0]].copy()
            for i in cand[1:]: np.logical_or(u_, masks[i], out=u_)
            preds[f"range_gate{e}_union"] = u_
            gr = masks[am].copy()
            for i in cand:
                if np.logical_and(masks[i], gr).any(): np.logical_or(gr, masks[i], out=gr)
            preds[f"range_gate{e}_growth"] = gr
        for v in variants:
            val = iou(preds[v], gt); acc[v].append(val)
            inter[v] += int(np.logical_and(preds[v], gt).sum()); un[v] += int(np.logical_or(preds[v], gt).sum())
        n += 1
    if n % 50 == 0: print(f"[{n}] {time.perf_counter()-t0:.0f}s", flush=True)
rows = sorted(({"variant": v, "mIoU": float(np.mean(acc[v])), "oIoU": inter[v]/un[v] if un[v] else None}
               for v in variants), key=lambda d: -d["mIoU"])
base = float(np.mean(acc["argmax"]))
for r in rows: r["delta"] = r["mIoU"] - base
out = dict(tag=args.tag, n_expressions=n, score=args.score, baseline_argmax=base, rows=rows)
json.dump(out, open(args.output/f"summary_{args.tag}.json", "w"), indent=2)
print(json.dumps({"n": n, "baseline": base}, indent=2), flush=True)
for r in rows: print(f"  {r['variant']:24s} mIoU={r['mIoU']:.4f} oIoU={r['oIoU']:.4f} delta={r['delta']:+.4f}", flush=True)
