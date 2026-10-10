#!/usr/bin/env python
"""EXP-29 — Qualitative grid + taxonomy figure from this run's records.

Selects expressions on which the released pipeline returns a near-disjoint
mask (IoU < 0.1) and the composed operator returns the referent
(IoU > 0.6), and renders image | ground truth | released | operator+growth.
Also renders the oracle-gap taxonomy as a stacked bar from main_*.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec  # noqa: E402
from method import Selector  # noqa: E402
from glod.data import RefDataset  # noqa: E402
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def load_masks(cache_file):
    with np.load(cache_file, allow_pickle=False) as d:
        shape = tuple(int(x) for x in d["mask_shape"])
        return np.unpackbits(d["mask_bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)


def overlay(img, mask, color):
    out = img.copy().astype(np.float32)
    m = mask
    for c in range(3):
        out[..., c] = np.where(m, 0.45 * out[..., c] + 0.55 * color[c], out[..., c])
    return out.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=ROOT / "exp/out_extreme/records_refcoco.jsonl")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    recs = [Rec(json.loads(l)) for l in args.records.open()]
    ds = RefDataset(REPRO / "data/refer", "refcoco", "val")
    global _enc
    _enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                         REPRO / "checkpoints/ViT-B-16.pt", device="cpu",
                         cache_dir=REPRO / "cache/vision")
    ref_by_id = {int(r["ref_id"]): r for r in ds.refs}
    sel = Selector()

    picks = []
    for r in recs:
        i_ours = int(sel(r))
        i_full = int(r.ps[i_ours])
        if r.gt[r.final] < 0.1 and r.gt[i_full] > 0.6 and r.gt[r.final] < r.gt[r.orc] - 0.4:
            picks.append((r, i_ours, i_full))
    # spread over images: one expression per image, prefer horizontal relations
    seen, chosen = set(), []
    picks.sort(key=lambda t: -(t[0].rel is not None and bool(t[0].rel["h"])))
    for r, a, b in picks:
        img_id = ref_by_id[r.r["ref_id"]]["image_id"]
        if img_id in seen:
            continue
        seen.add(img_id)
        chosen.append((r, a, b))
        if len(chosen) >= args.n:
            break
    print(f"{len(picks)} repair candidates, plotting {len(chosen)}")

    fig, axes = plt.subplots(len(chosen), 4, figsize=(4 * 2.6, len(chosen) * 2.0), dpi=200)
    if len(chosen) == 1:
        axes = axes[None]
    cols = ["image", "ground truth", "released guidance", "operator + growth"]
    for row, (r, a, b) in enumerate(chosen):
        ref = ref_by_id[r.r["ref_id"]]
        ip = ds.image_path(ref)
        img = np.asarray(Image.open(ip).convert("RGB"))
        gt = ds.ground_truth(ref)
        masks = _enc.extract(Image.open(ip).convert("RGB"), file_sha256(ip))[0]
        rel_iou = r.gt[r.final]
        full_iou = r.gt[b]
        axes[row, 0].imshow(img)
        axes[row, 1].imshow(overlay(img, gt, (0.1, 0.9, 0.1)))
        axes[row, 2].imshow(overlay(img, masks[r.final], (0.9, 0.1, 0.1)))
        axes[row, 3].imshow(overlay(img, masks[b], (0.1, 0.4, 0.95)))
        axes[row, 2].set_xlabel(f"IoU {rel_iou:.2f}", fontsize=8)
        axes[row, 3].set_xlabel(f"IoU {full_iou:.2f}", fontsize=8)
        axes[row, 0].set_ylabel(f'"{r.sent[:40]}"\n', fontsize=7, rotation=0, ha="right", va="center")
        for c in range(4):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
            if row == 0:
                axes[row, c].set_title(cols[c], fontsize=9)
    fig.tight_layout()
    out = ROOT / "paper/figures/qualitative_fixed.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(ROOT / "paper/figures/qualitative_fixed.pdf", bbox_inches="tight")
    print(f"wrote {out}")

    # ---- taxonomy bar ----
    tags = [("refcoco", "RefCOCO"), ("refcocoplus", "RefCOCO+"), ("refcocog", "RefCOCOg")]
    classes = ["correct", "under-segmentation", "over-segmentation", "wrong-instance", "boundary"]
    colors = ["#93C5FD", "#FCD34D", "#A7F3D0", "#F87171", "#D1D5DB"]
    fig2, ax = plt.subplots(figsize=(5.2, 2.4), dpi=300)
    labels, data = [], []
    for tag, name in tags:
        p = ROOT / f"exp/out_final/main_{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        data.append([d["taxonomy"]["by_class"][c]["gap_share"] for c in classes])
        labels.append(name)
    bottom = np.zeros(len(labels))
    for k, c in enumerate(classes):
        vals = np.array([row[k] for row in data])
        ax.bar(labels, vals, bottom=bottom, color=colors[k], label=c, width=0.6)
        bottom += vals
    ax.set_ylabel("share of oracle-gap mass", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig2.tight_layout()
    fig2.savefig(ROOT / "paper/figures/taxonomy.pdf", bbox_inches="tight")
    print("wrote paper/figures/taxonomy.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
