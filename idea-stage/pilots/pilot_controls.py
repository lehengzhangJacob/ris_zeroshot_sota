#!/usr/bin/env python
"""Pilot 9 — Controls demanded by the area-chair review.

The review's sharpest new objection: connected growth monotonically increases
area, so the gain might be a pure "granularity correction" that a trivial
area-increasing operator reproduces. Controls:

  parent      argmax replaced by its most-overlapping LARGER proposal
  parent_strict  smallest proposal containing >=95% of the argmax pixels
  dilate_k    binary dilation of the argmax mask by k pixels, k in {2,5,10,20}
  largest_gated / gated_union / growth   (reference operators)
and a tolerance sweep on the published MARS gate, plus stratification of the
growth delta by GT-size quantile and by part-expression proxy.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy import ndimage
REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro"); sys.path.insert(0, str(REPRO))
from glod.core import Config, decouple
from glod.data import RefDataset
from glod.io import file_sha256
from glod.vision import VisionEncoder

PART_WORDS = {"hand","hands","arm","arms","leg","legs","head","face","foot","feet","eye","eyes",
              "ear","ears","wheel","wheels","tail","wing","wings","nose","mouth","neck","finger",
              "fingers","paw","paws","hoof","beak","horn","horns","antenna","trunk","handle",
              "lid","door","window","sleeve","collar","tire","tyre","propeller","blade","stem"}

def iou(p, g):
    u = int(np.logical_or(p, g).sum()); return 1.0 if not u else int(np.logical_and(p, g).sum()) / u

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="refcoco"); ap.add_argument("--split", default="val")
ap.add_argument("--limit", type=int, default=250); ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--all-sentences", action="store_true")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--output", type=Path, default=Path("idea-stage/pilots/out_controls"))
ap.add_argument("--tag", default="run"); ap.add_argument("--score", choices=["c1","c3"], default="c3")
args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

ds = RefDataset(REPRO/"data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
cfg = Config()
enc = VisionEncoder(REPRO/"checkpoints/sam_vit_h_4b8939.pth", REPRO/"checkpoints/ViT-B-16.pt",
                    device=args.device, cache_dir=REPRO/"cache/vision")
TAUS = [0.90, 0.925, 0.95, 0.975, 0.99]
DIL = [2, 5, 10, 20]
keys = ["argmax","growth_a095","gatedunion_a095","largest_gated_a095","parent","parent_strict"] \
       + [f"dilate_{k}" for k in DIL] + [f"growth_tau{t}" for t in TAUS]
acc = {k: [] for k in keys}
extra = {"gt_area": [], "is_part": [], "delta_growth": [], "gate_size": [], "n_prop": []}
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
    area = masks.reshape(len(masks), -1).mean(1)
    sl = [s["sent"] for s in ref["sentences"]] if args.all_sentences else [ref["sentences"][0]["sent"]]
    for sent in sl:
        t = F.normalize(enc.text(sent).float().reshape(-1), dim=0)
        bf = F.normalize(fg_t, dim=-1) if args.score == "c1" else F.normalize(fused, dim=-1)
        s = (bf @ t).numpy(); am = int(np.argmax(s))
        def grow(cand):
            g = masks[am].copy()
            for i in cand:
                if np.logical_and(masks[i], g).any(): np.logical_or(g, masks[i], out=g)
            return g
        preds = {"argmax": masks[am]}
        c095 = np.flatnonzero(s >= 0.95*s.max())
        preds["growth_a095"] = grow(c095)
        u_ = masks[c095[0]].copy()
        for i in c095[1:]: np.logical_or(u_, masks[i], out=u_)
        preds["gatedunion_a095"] = u_
        preds["largest_gated_a095"] = masks[c095[np.argmax(area[c095])]]
        for tt in TAUS: preds[f"growth_tau{tt}"] = grow(np.flatnonzero(s >= tt*s.max()))
        # parent: most-overlapping strictly larger proposal
        larger = np.flatnonzero(area > area[am])
        if len(larger):
            ov = np.array([np.logical_and(masks[i], masks[am]).sum() / max(masks[am].sum(),1) for i in larger])
            preds["parent"] = masks[larger[int(np.argmax(ov))]]
            # parent_strict: smallest proposal containing >=95% of argmax pixels
            cont = larger[ov >= 0.95]
            preds["parent_strict"] = masks[cont[np.argmin(area[cont])]] if len(cont) else masks[am]
        else:
            preds["parent"] = masks[am]; preds["parent_strict"] = masks[am]
        for k in DIL:
            preds[f"dilate_{k}"] = ndimage.binary_dilation(masks[am], iterations=k)
        for key in keys:
            acc[key].append(iou(preds[key], gt))
        tok = set(w.strip(".,!?").lower() for w in sent.split())
        extra["gt_area"].append(float(gt.mean())); extra["is_part"].append(bool(tok & PART_WORDS))
        extra["delta_growth"].append(acc["growth_a095"][-1] - acc["argmax"][-1])
        extra["gate_size"].append(len(c095)); extra["n_prop"].append(len(masks))
        n += 1
    if n % 100 == 0: print(f"[{n}] {time.perf_counter()-t0:.0f}s", flush=True)

base = float(np.mean(acc["argmax"]))
gt_area = np.array(extra["gt_area"]); is_part = np.array(extra["is_part"])
d = np.array(extra["delta_growth"])
q = np.quantile(gt_area, [0.25, 0.5, 0.75])
strata = {}
for i,(lo,hi,name) in enumerate([(-1,q[0],"gt_q1_smallest"),(q[0],q[1],"gt_q2"),(q[1],q[2],"gt_q3"),(q[2],2,"gt_q4_largest")]):
    m = (gt_area>lo)&(gt_area<=hi)
    strata[name] = dict(n=int(m.sum()), delta=float(d[m].mean()) if m.sum() else None)
strata["part_word_expr"] = dict(n=int(is_part.sum()), delta=float(d[is_part].mean()) if is_part.sum() else None)
strata["non_part_word_expr"] = dict(n=int((~is_part).sum()), delta=float(d[~is_part].mean()) if (~is_part).sum() else None)
res = dict(tag=args.tag, n=n, score=args.score, baseline_argmax=base,
           operators=sorted([dict(op=k, mIoU=float(np.mean(v)), delta=float(np.mean(v))-base)
                             for k,v in acc.items()], key=lambda x:-x["mIoU"]),
           mean_gate_size=float(np.mean(extra["gate_size"])), mean_n_proposals=float(np.mean(extra["n_prop"])),
           strata=strata)
json.dump(res, open(args.output/f"summary_{args.tag}.json","w"), indent=2)
print(json.dumps({"n":n,"baseline":base,"mean_gate_size":res["mean_gate_size"]}, indent=2), flush=True)
for r in res["operators"]: print(f"  {r['op']:22s} mIoU={r['mIoU']:.4f} delta={r['delta']:+.4f}", flush=True)
print("  strata:", json.dumps(strata, indent=4), flush=True)
