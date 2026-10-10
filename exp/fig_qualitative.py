#!/usr/bin/env python
"""Qualitative figure: cases the selector repairs, and cases it leaves alone.

Picks concrete expressions from the stored records where the released HybridGL
selection is wrong-instance and the selector lands on the oracle, and renders
image | GT | released selection | ours. Contact sheet, deterministic selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glod.data import RefDataset  # noqa: E402
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402
from exp5_analyze_guided import Rec  # noqa: E402
from method import Selector  # noqa: E402

TINT = {"gt": (60, 220, 60), "rel": (230, 70, 70), "ours": (70, 130, 255)}


def overlay(rgb, mask, colour, alpha=0.45):
    out = rgb.astype(np.float32).copy()
    for c in range(3):
        out[..., c] = np.where(mask, out[..., c] * (1 - alpha) + colour[c] * alpha, out[..., c])
    return out.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp4/records_guided_val.jsonl"))
    ap.add_argument("--pairs", type=Path, default=Path("exp/out_exp4/pairs_guided_val.npz"))
    ap.add_argument("--kind", choices=["fixed", "unchanged"], default="fixed")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("exp/out_fig"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    recs = [Rec(json.loads(l)) for l in args.records.open()]
    pairs = np.load(args.pairs)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    rule = Selector()

    if args.kind == "fixed":
        cand = [r for r in recs
                if r.gt[r.final] < 0.1 and r.gt[rule(r)] > 0.6 and r.rel is not None]
        cand.sort(key=lambda r: (r.r["ref_id"], r.r["sent_idx"]))
    else:
        cand = [r for r in recs if rule(r) == r.final and r.gt[r.final] > 0.6]
        cand.sort(key=lambda r: (r.r["ref_id"], r.r["sent_idx"]))
    step = max(1, len(cand) // args.n)
    sel = cand[::step][:args.n]
    print(f"{len(cand)} candidates, rendering {len(sel)}")

    ds = RefDataset(REPRO / "data/refer", "refcoco", "val")
    by_ref = {}
    for r in ds.refs:
        by_ref[int(r["ref_id"])] = r
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                        device=args.device, cache_dir=REPRO / "cache/vision")

    rows = []
    for r in sel:
        ref = by_ref[r.r["ref_id"]]
        ip = ds.image_path(ref)
        im = Image.open(ip).convert("RGB")
        gt = ds.ground_truth(ref)
        masks, _, _, _ = enc.extract(im, file_sha256(ip))
        rgb = np.asarray(im).copy()
        if rgb.shape[:2] != gt.shape:
            rgb = np.asarray(im.resize((gt.shape[1], gt.shape[0]))).copy()
        ours = rule(r)
        tiles = [rgb, overlay(rgb, gt, TINT["gt"]),
                 overlay(rgb, masks[r.final], TINT["rel"]),
                 overlay(rgb, masks[ours], TINT["ours"])]
        strip = np.concatenate(tiles, axis=1)
        img = Image.fromarray(strip)
        cap = (f"[{args.kind}] \"{r.sent[:74]}\"  released IoU={r.gt[r.final]:.2f} "
               f"-> ours IoU={r.gt[ours]:.2f}")
        canvas = Image.new("RGB", (img.width, img.height + 20), (0, 0, 0))
        canvas.paste(img, (0, 20))
        ImageDraw.Draw(canvas).text((4, 4), cap, fill=(255, 255, 255))
        rows.append(canvas)

    W = max(x.width for x in rows)
    sheet = Image.new("RGB", (W, sum(x.height for x in rows)), (0, 0, 0))
    y = 0
    for x in rows:
        sheet.paste(x, (0, y))
        y += x.height
    out = args.output / f"qual_{args.kind}.png"
    sheet.save(out)
    print("saved", out, sheet.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
