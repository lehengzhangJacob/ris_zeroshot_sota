#!/usr/bin/env python
"""EXP-45 — qualitative panels for the shipped method.

For a handful of expressions per dataset, render image / ground truth /
released pipeline / ours.  Reads the same records the tables come from plus
the cached VLM boxes, so every panel is a re-analysis of the run's evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_map import cache_file  # noqa: E402
from exp40_final_eval import PUBLISHED, box_overlap, rank01  # noqa: E402
from vlm_boxes import QwenBoxer  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
IMAGES = REPRO / "data/refer/images/train2014"

RECORDS = {
    "refcoco": "exp/out_extreme/records_refcoco.jsonl",
    "refcoco+": "exp/out_extreme/records_refcocoplus.jsonl",
    "refcocog": "exp/out_extreme/records_refcocog.jsonl",
}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=3)
    ap.add_argument("--output", type=Path, default=Path("paper/figures/qualitative.pdf"))
    ap.add_argument("--params", type=Path, default=Path("exp/out_final/optuna_study.json"))
    args = ap.parse_args()

    from glod.data import RefDataset
    params = json.loads(args.params.read_text())["best_params"]
    boxer = QwenBoxer.__new__(QwenBoxer)

    panels = []
    rng = np.random.RandomState(3)
    for ds_name, path in RECORDS.items():
        ds = RefDataset(REPRO / "data/refer", ds_name, "val")
        ref_by_id = {r["ref_id"]: r for r in ds.refs}
        recs = [json.loads(l) for l in open(path)]
        picked = 0
        for r in recs:
            if picked >= args.per_dataset:
                break
            ref = ref_by_id.get(r["ref_id"])
            if ref is None:
                continue
            fn = ds.images[ref["image_id"]]["file_name"]
            try:
                with np.load(cache_file(fn), allow_pickle=False) as z:
                    shape = tuple(int(x) for x in z["mask_shape"])
                    masks = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                                          ).reshape(shape).astype(bool)
            except Exception:  # noqa: BLE001
                continue
            n = len(masks)
            if n != len(r["gt_iou"]):
                continue
            gt = np.asarray(r["gt_iou"], np.float32)
            ph = r["parsed"]["sentence"]
            bp = boxer._path(fn, ph.strip())
            if not bp.is_file():
                continue
            box = json.loads(bp.read_text())["box"]
            if box is None:
                continue
            base = rank01(np.asarray(r["sc"][:n], np.float32)) + params["gem_mix"] * rank01(
                np.asarray(r["score_gem"][:n], np.float32))
            k = min(int(params["k_dec"]), n)
            idx = np.argsort(base)[::-1][:k]
            loc = box_overlap(masks, box, params["mode"])[idx]
            ours = int(idx[int(np.argmax(loc))])
            rel = int(r["final"])
            # keep panels where the method differs from the released pipeline
            if ours == rel or gt[ours] - gt[rel] < 0.25:
                continue
            img = np.asarray(Image.open(IMAGES / fn).convert("RGB"))
            panels.append(dict(ds=ds_name, sent=ph, img=img,
                               gt_boxes=[(box,)],
                               gtmask=None, released=masks[rel], ours=masks[ours],
                               iou_rel=float(gt[rel]), iou_ours=float(gt[ours]),
                               vlm_box=box))
            picked += 1

    if not panels:
        print("no panels selected")
        return 1
    rows = len(panels)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.2 * rows))
    if rows == 1:
        axes = axes[None, :]
    for i, p in enumerate(panels):
        img = p["img"].copy()
        x0, y0, x1, y1 = [int(v) for v in p["vlm_box"]]
        img[y0:y0 + 3, x0:x1] = [255, 200, 0]
        img[y1 - 3:y1, x0:x1] = [255, 200, 0]
        img[y0:y1, x0:x0 + 3] = [255, 200, 0]
        img[y0:y1, x1 - 3:x1] = [255, 200, 0]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{p['ds']}: \"{p['sent'][:44]}\"\n(VLM box in yellow)", fontsize=7)
        axes[i, 1].imshow(p["img"])
        axes[i, 1].set_title("released (IoU %.2f)" % p["iou_rel"], fontsize=8)
        axes[i, 2].imshow(p["img"])
        axes[i, 2].set_title("ours (IoU %.2f)" % p["iou_ours"], fontsize=8)
        axes[i, 3].imshow(p["img"])
        axes[i, 3].set_title("oIoU %.2f -> %.2f" % (p["iou_rel"], p["iou_ours"]), fontsize=8)
        for j, m in ((1, p["released"]), (2, p["ours"]), (3, None)):
            ax = axes[i, j]
            if m is not None:
                ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.55)
            ax.axis("off")
        axes[i, 0].axis("off")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=110, bbox_inches="tight")
    print(f"wrote {args.output} ({len(panels)} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
