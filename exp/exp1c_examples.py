#!/usr/bin/env python
"""Render a montage of wrong-instance failures: image | GT | argmax | oracle.

Used to read the failure mode off the data rather than guess it. Writes one
PNG per selected expression plus a contact sheet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

from glod.core import Config, decouple  # noqa: E402
from glod.data import RefDataset  # noqa: E402
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402

TINT = {"gt": (60, 220, 60), "argmax": (230, 60, 60), "oracle": (70, 130, 255)}


def overlay(rgb, mask, colour, alpha=0.5):
    out = rgb.astype(np.float32).copy()
    for c in range(3):
        out[..., c] = np.where(mask, out[..., c] * (1 - alpha) + colour[c] * alpha, out[..., c])
    return out.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp1/records_taxonomy_allsent_c3.jsonl"))
    ap.add_argument("--cls", default="wrong-instance")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp1/examples"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in args.records.open()]
    sel = [r for r in recs if r["cls"] == args.cls]
    step = max(1, len(sel) // args.n)
    sel = sel[::step][:args.n]

    ds = RefDataset(REPRO / "data/refer", "refcoco", "val", limit=250, seed=101)
    by_ref = {int(r["ref_id"]): r for r in ds.refs}
    cfg = Config()
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                        device=args.device, cache_dir=REPRO / "cache/vision")

    panels = []
    for r in sel:
        ref = by_ref[r["ref_id"]]
        ip = ds.image_path(ref)
        im = Image.open(ip).convert("RGB")
        gt = ds.ground_truth(ref)
        masks, fg, bg, _ = enc.extract(im, file_sha256(ip))
        rgb = np.asarray(im).copy()
        h, w = gt.shape
        if rgb.shape[:2] != (h, w):
            rgb = np.asarray(im.resize((w, h))).copy()
        tiles = [rgb,
                 overlay(rgb, gt, TINT["gt"]),
                 overlay(rgb, masks[r["am"]], TINT["argmax"]),
                 overlay(rgb, masks[r["orc"]], TINT["oracle"])]
        strip = np.concatenate(tiles, axis=1)
        img = Image.fromarray(strip)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, strip.shape[1] - 1, strip.shape[0] - 1], outline=(255, 255, 255))
        panels.append((img, f"[GT={r['iou_am']:.2f}->{r['iou_orc']:.2f} rank={r['rank_orc']}] {r['sent'][:110]}"))

    W = max(p.width for p, _ in panels)
    rows, y = [], 0
    for img, cap in panels:
        canvas = Image.new("RGB", (W, img.height + 22), (0, 0, 0))
        canvas.paste(img, (0, 22))
        ImageDraw.Draw(canvas).text((4, 5), cap, fill=(255, 255, 255))
        rows.append(canvas)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows)), (0, 0, 0))
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    out = args.output / f"sheet_{args.cls}.png"
    sheet.save(out)
    print("saved", out, sheet.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
