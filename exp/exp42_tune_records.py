#!/usr/bin/env python
"""EXP-42 — build records for the held-out tuning split.

Same record schema as exp4_guided_full.py, but the ref list comes from
exp/tuning/<dataset>_refs.json (train refs restricted to on-disk images), so
the official val split is never touched.  Feature extraction hits the
glod.vision cache for these images; only the text-scoring pass runs.
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
    out = []
    for m in masks:
        cols = np.flatnonzero(m.any(0))
        rows = np.flatnonzero(m.any(1))
        if not len(cols) or not len(rows):
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        out.append([float(cols[0]), float(rows[0]),
                    float(cols[-1] - cols[0] + 1), float(rows[-1] - rows[0] + 1)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ds = RefDataset(REPRO / "data/refer", args.dataset, "train")
    by_id = {r["ref_id"]: r for r in ds.refs}
    wanted = json.loads(args.refs.read_text())["ref_ids"]
    if args.limit:
        wanted = wanted[: args.limit]
    refs = [by_id[i] for i in wanted if i in by_id]
    print(f"{args.dataset}: {len(refs)} refs", flush=True)

    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                        device=args.device)
    sc = GuidedScorer(enc, device=args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    t0 = time.perf_counter()
    with args.output.open("w") as fh:
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
            cont = (inter_all / ar.clamp(min=1)[:, None]).cpu().numpy()
            area_all = (ar / flat.shape[1]).cpu().numpy()
            del inter_all
            ps_idx = np.arange(n_m)
            for i in range(n_m):
                jj = np.flatnonzero((cont[i] >= 0.95) & (area_all > area_all[i]))
                if len(jj):
                    ps_idx[i] = int(jj[np.argmin(area_all[jj])])
            del cont
            ys, xs = np.mgrid[0:masks.shape[1], 0:masks.shape[2]]
            mflat = masks.reshape(n_m, -1)
            cnt = mflat.sum(1).astype(np.float64)
            cx = (mflat * xs.reshape(-1).astype(np.float64)).sum(1) / np.maximum(cnt, 1) / masks.shape[2]
            cy = (mflat * ys.reshape(-1).astype(np.float64)).sum(1) / np.maximum(cnt, 1) / masks.shape[1]
            area = cnt / mflat.shape[1]
            bxs = bboxes(masks)
            inter_gt = np.logical_and(masks, gt).sum((1, 2)).astype(np.float64)
            union_gt = np.logical_or(masks, gt).sum((1, 2)).astype(np.float64)
            gt_iou = np.where(union_gt > 0, inter_gt / np.maximum(union_gt, 1), 1.0)
            orc = int(np.argmax(gt_iou))
            for si, s in enumerate(ref["sentences"]):
                try:
                    res = sc.score(im, masks, bxs, fg, s["sent"])
                except Exception:  # noqa: BLE001
                    continue
                rec = dict(ref_id=int(ref["ref_id"]), sent_idx=si, sent=s["sent"], n=n_m,
                           gt_iou=gt_iou.tolist(), inter_gt=inter_gt.tolist(),
                           union_gt=union_gt.tolist(), area=area.tolist(),
                           cx=cx.tolist(), cy=cy.tolist(), bbox=bxs, ps_idx=ps_idx.tolist(),
                           sc=res["sc"].tolist(), sc_neg=res["sc_neg"].tolist(),
                           sc_noun=res["sc_noun"].tolist(), sc_strip=res["sc_strip"].tolist(),
                           sc_content=res["sc_content"].tolist(),
                           score_gem=res["score_gem"].tolist(), noun_scores={
                               k: v.tolist() for k, v in res["noun_scores"].items()},
                           top_idx=res["top_idx"].tolist(), fused_tops=res["fused_tops"].tolist(),
                           pure=int(res["pure"]), final=int(res["final"]), orc=orc,
                           parsed=res["parsed"], gt_area=float(gt.sum() / gt.size))
                fh.write(json.dumps(rec) + "\n")
                n += 1
            if n and n % 300 < 3:
                print(f"  {n} records {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"wrote {args.output}: {n} records ({skipped} refs skipped)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
