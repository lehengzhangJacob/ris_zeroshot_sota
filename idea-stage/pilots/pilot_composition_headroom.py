#!/usr/bin/env python
"""Pilot 1 — Composition headroom in zero-shot RIS proposal selection.

Question: across a frozen SAM proposal pool, how much IoU is lost by the
universal "argmax over ONE proposal" commitment, and how much of that loss is
recovered by allowing the output to be a UNION of a small number of proposals?

This is an ORACLE analysis: it uses the ground-truth mask only to *measure* the
ceiling, never to choose. It therefore bounds what any training-free
compositional selector could achieve, and decides whether the idea is worth
building.

Runs on the cached SAM masks + HybridGL/CLIP features produced by the existing
GLOD-RIS reproduction, so it costs almost no GPU time.
"""
from __future__ import annotations

import argparse
import itertools
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
from glod.vision import VisionEncoder  # noqa: E402


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return inter / union if union else 1.0


def union_iou(masks: np.ndarray, idxs, gt: np.ndarray) -> float:
    acc = masks[idxs[0]].copy()
    for j in idxs[1:]:
        np.logical_or(acc, masks[j], out=acc)
    inter = int(np.logical_and(acc, gt).sum())
    union = int(np.logical_or(acc, gt).sum())
    return inter / union if union else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250, help="references to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("idea-stage/pilots/out_composition"))
    ap.add_argument("--triple-pool", type=int, default=15,
                    help="restrict triple search to the top-N proposals by DOD semantic score")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="0 = all pairs; else cap the pool for the pair search")
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    dataset = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    config = Config()  # GLOD-RIS Eq. (1)-(7) defaults; no GLM guidance used here.

    encoder = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                            REPRO / "checkpoints/ViT-B-16.pt",
                            device=args.device, cache_dir=REPRO / "cache/vision")

    records = []
    t0 = time.perf_counter()
    n_done = 0
    for ref in dataset.refs:
        image_path = dataset.image_path(ref)
        if not image_path.exists():
            if not args.allow_download:
                continue
            try:
                dataset.download_image(ref)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {image_path.name}: {exc}", flush=True)
                continue
        try:
            image = Image.open(image_path).convert("RGB")
            gt = dataset.ground_truth(ref)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {image_path.name}: {exc}", flush=True)
            continue

        image_hash = __import__("glod.io", fromlist=["file_sha256"]).file_sha256(image_path)
        masks, fg, bg, meta = encoder.extract(image, image_hash)

        if masks.shape[1:] != gt.shape:
            print(f"  skip {image_path.name}: mask {masks.shape} vs gt {gt.shape}", flush=True)
            continue

        fg_t, bg_t = torch.from_numpy(fg), torch.from_numpy(bg)
        fused, _, _ = decouple(fg_t, bg_t, masks.mean((1, 2)), config)
        # Use the first sentence of the ref (GLOD/HybridGL evaluate sentence-wise
        # identically for a given ref in practice; first sentence keeps the pilot cheap).
        expression = ref["sentences"][0]["sent"]
        text = encoder.text(expression)
        semantic = (F.normalize(fused.float(), dim=-1) @ F.normalize(text.float().reshape(-1), dim=0)).numpy()

        order = np.argsort(-semantic, kind="stable")
        n = len(masks)

        # --- single-proposal oracle -------------------------------------------
        single_ious = np.array([iou(masks[i], gt) for i in range(n)])
        oracle_single = float(single_ious.max())
        argmax_i = int(order[0])
        argmax_iou = float(single_ious[argmax_i])
        oracle_rank = int(np.where(np.argsort(-single_ious, kind="stable") == argmax_i)[0][0])

        # --- pair oracle -------------------------------------------------------
        pool = range(n) if args.max_pairs <= 0 else order[: args.max_pairs]
        pool = list(pool)
        pair_best, pair_best_idx = oracle_single, (argmax_i,)
        for a, b in itertools.combinations(pool, 2):
            v = union_iou(masks, (a, b), gt)
            if v > pair_best:
                pair_best, pair_best_idx = v, (a, b)

        # --- triple oracle (restricted to the semantic top-N) -------------------
        tpool = [int(i) for i in order[: min(args.triple_pool, n)]]
        triple_best, triple_best_idx = pair_best, pair_best_idx
        for combo in itertools.combinations(tpool, 3):
            v = union_iou(masks, combo, gt)
            if v > triple_best:
                triple_best, triple_best_idx = v, combo

        # --- training-free cheap rules (no new forwards) ------------------------
        union_top = {}
        for j in (2, 3, 5):
            if j <= n:
                union_top[j] = union_iou(masks, [int(i) for i in order[:j]], gt)

        records.append(dict(
            ref_id=int(ref["ref_id"]), image=image_path.name, expression=expression,
            n_masks=n, cache_hit=bool(meta.get("cache_hit")),
            iou_argmax=argmax_iou, oracle_single=oracle_single,
            oracle_rank_of_argmax=oracle_rank,
            oracle_pair=pair_best, pair_idx=list(map(int, pair_best_idx)),
            oracle_triple=triple_best, triple_idx=list(map(int, triple_best_idx)),
            union_top2=union_top.get(2), union_top3=union_top.get(3), union_top5=union_top.get(5),
            gt_area_frac=float(gt.mean()),
        ))
        n_done += 1
        if n_done % 25 == 0:
            el = time.perf_counter() - t0
            print(f"[{n_done}] {el:.0f}s  ({el/n_done:.2f}s/expr)", flush=True)

    if not records:
        print("no records produced", flush=True)
        return 1

    def agg(key):
        vals = [r[key] for r in records if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    def agg_iou(key):
        """Dataset-level (cumulative) oIoU over the recorded subsets."""
        return None

    summary = dict(
        dataset=args.dataset, split=args.split, limit=args.limit, seed=args.seed,
        n_expressions=len(records), n_images=len({r["image"] for r in records}),
        elapsed_s=round(time.perf_counter() - t0, 1),
        mIoU_argmax=agg("iou_argmax"),
        mIoU_oracle_single=agg("oracle_single"),
        mIoU_oracle_pair=agg("oracle_pair"),
        mIoU_oracle_triple=agg("oracle_triple"),
        mIoU_union_top2=agg("union_top2"),
        mIoU_union_top3=agg("union_top3"),
        mIoU_union_top5=agg("union_top5"),
        frac_pair_beats_single=float(np.mean([r["oracle_pair"] > r["oracle_single"] + 1e-9 for r in records])),
        frac_pair_improves_gt2pts=float(np.mean([r["oracle_pair"] > r["oracle_single"] + 0.02 for r in records])),
        mean_gap_pair_minus_single=float(np.mean([r["oracle_pair"] - r["oracle_single"] for r in records])),
        mean_gap_triple_minus_single=float(np.mean([r["oracle_triple"] - r["oracle_single"] for r in records])),
        mean_argmax_rank=float(np.mean([r["oracle_rank_of_argmax"] for r in records])),
    )

    with (args.output / "records.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    with (args.output / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
