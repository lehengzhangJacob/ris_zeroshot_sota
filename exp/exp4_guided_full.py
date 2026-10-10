#!/usr/bin/env python
"""EXP-4 — Full RefCOCO val at the released HybridGL operating point.

For every expression: run the released HybridGL scorer (semantic score with the
0.5 sentence/noun-phrase ensemble, relation guidance, GEM + position guidance)
and store every per-proposal quantity, so the baseline rows, the error taxonomy
and the selector under test are all re-analyses of one pass.

Usage:
    python exp4_guided_full.py --limit 200            # reproduction check
    python exp4_guided_full.py                        # full 10834
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glod.data import RefDataset  # noqa: E402
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402
from hybridgl_guided import GuidedScorer  # noqa: E402


def bboxes(masks):
    """SAM-style XYWH bboxes derived from the masks."""
    out = []
    for m in masks:
        cols = np.flatnonzero(m.any(0))
        rows = np.flatnonzero(m.any(1))
        if not len(cols) or not len(rows):
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        x0, x1 = int(cols[0]), int(cols[-1])
        y0, y1 = int(rows[0]), int(rows[-1])
        out.append([float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp4"))
    ap.add_argument("--tag", default="guided_val")
    ap.add_argument("--points-per-side", type=int, default=8)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--save-scores", action="store_true", default=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    cache_dir = Path(args.cache_dir) if args.cache_dir else REPRO / "cache/vision"
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                        device=args.device, cache_dir=cache_dir,
                        points_per_side=args.points_per_side)
    sc = GuidedScorer(enc, device=args.device)

    refs = ds.refs
    if args.nshards > 1:
        refs = [r for i, r in enumerate(refs) if i % args.nshards == args.shard]
        print(f"shard {args.shard}/{args.nshards}: {len(refs)} refs", flush=True)
    out_path = args.output / f"records_{args.tag}.jsonl"
    n = skipped = 0
    t0 = time.perf_counter()
    inter_acc = np.zeros(3)   # released-pure, released-guided, oracle
    union_acc = np.zeros(3)
    iou_acc = [[], [], []]
    with out_path.open("w") as fh:
        for ref in refs:
            ip = ds.image_path(ref)
            if not ip.exists():
                skipped += 1
                continue
            try:
                im = Image.open(ip).convert("RGB")
                gt = ds.ground_truth(ref)
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            masks, fg, bg, _ = enc.extract(im, file_sha256(ip))
            if masks.shape[1:] != gt.shape:
                skipped += 1
                continue

            n_m = len(masks)
            flat = torch.from_numpy(masks.reshape(n_m, -1)).to(enc.device, torch.float32)
            inter_all = flat @ flat.T
            ar = flat.sum(1)
            union_all = ar[:, None] + ar[None, :] - inter_all
            cont = (inter_all / ar.clamp(min=1)[:, None]).cpu().numpy()
            area_all = (ar / flat.shape[1]).cpu().numpy()
            del flat, inter_all, union_all
            ps_idx = np.arange(n_m)
            for i in range(n_m):
                jj = np.flatnonzero((cont[i] >= 0.95) & (area_all > area_all[i]))
                if len(jj):
                    ps_idx[i] = int(jj[np.argmin(area_all[jj])])
            del cont

            ys, xs = np.mgrid[0:masks.shape[1], 0:masks.shape[2]]
            xs_f, ys_f = xs.reshape(-1).astype(np.float64), ys.reshape(-1).astype(np.float64)
            mflat = masks.reshape(n_m, -1)
            cnt = mflat.sum(1).astype(np.float64)
            cx = (mflat * xs_f).sum(1) / np.maximum(cnt, 1) / masks.shape[2]
            cy = (mflat * ys_f).sum(1) / np.maximum(cnt, 1) / masks.shape[1]
            area = cnt / mflat.shape[1]
            bxs = bboxes(masks)

            inter_gt = np.logical_and(masks, gt).sum((1, 2)).astype(np.float64)
            union_gt = np.logical_or(masks, gt).sum((1, 2)).astype(np.float64)
            gt_iou = np.where(union_gt > 0, inter_gt / np.maximum(union_gt, 1), 1.0)
            orc = int(np.argmax(gt_iou))

            for si, s in enumerate(ref["sentences"]):
                sent = s["sent"]
                try:
                    # HybridGL's `hybrid_features` ARE the G2L masked-region output
                    res = sc.score(im, masks, bxs, fg, sent)
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {ref['ref_id']}:{si} {type(exc).__name__} {exc}", flush=True)
                    continue
                pure, final = res["pure"], res["final"]
                for k, idx in enumerate((pure, final, orc)):
                    inter_acc[k] += inter_gt[idx]
                    union_acc[k] += union_gt[idx]
                    iou_acc[k].append(gt_iou[idx])
                rec = dict(
                    ref_id=int(ref["ref_id"]), sent_idx=si, sent=sent, n=n_m,
                    gt_iou=gt_iou.tolist(), inter_gt=inter_gt.tolist(), union_gt=union_gt.tolist(),
                    area=area.tolist(), cx=cx.tolist(), cy=cy.tolist(), bbox=bxs,
                    ps_idx=ps_idx.tolist(),
                    sc=res["sc"].tolist(), sc_neg=res["sc_neg"].tolist(),
                    sc_noun=res["sc_noun"].tolist(), sc_strip=res["sc_strip"].tolist(),
                    sc_content=res["sc_content"].tolist(),
                    score_gem=res["score_gem"].tolist(),
                    top_idx=res["top_idx"].tolist(), fused_tops=res["fused_tops"].tolist(),
                    pure=pure, final=final, orc=orc,
                    parsed={k: res["parsed"][k] for k in ("sentence", "noun_phrase", "nouns",
                                                          "dirflag", "relaflag")},
                    gt_area=float(gt.mean()),
                )
                fh.write(json.dumps(rec) + "\n")
                n += 1
            if n and n % 100 == 0:
                rate = (time.perf_counter() - t0) / n
                print(f"[{n}] {rate*1000:.0f} ms/expr  pure={np.mean(iou_acc[0]):.4f} "
                      f"guided={np.mean(iou_acc[1]):.4f} oracle={np.mean(iou_acc[2]):.4f}", flush=True)

    m = lambda k: float(np.mean(iou_acc[k]))  # noqa: E731
    o = lambda k: float(inter_acc[k] / union_acc[k])  # noqa: E731
    summary = dict(tag=args.tag, n=n, skipped_refs=skipped, limit=args.limit,
                   elapsed_s=round(time.perf_counter() - t0, 1),
                   pure=dict(mIoU=m(0), oIoU=o(0)),
                   guided=dict(mIoU=m(1), oIoU=o(1)),
                   oracle=dict(mIoU=m(2), oIoU=o(2)))
    with (args.output / f"summary_{args.tag}.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
