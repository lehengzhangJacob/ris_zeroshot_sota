#!/usr/bin/env python
"""EXP-21 — Full gRefCOCO val at the released HybridGL operating point.

Same per-expression record as EXP-4 (every per-proposal quantity the released
scorer computes), but the reference may carry several target annotations or
none. Ground truth is the *union* of the reference's masks (all-False for a
no-target reference), and the record carries the metadata the GRES evaluator
needs: no_target, n_targets, ann_ids, and the vision-cache file so the union
prediction of a multi-mask output can be re-evaluated from the stored masks.

Usage:
    python exp21_grefcoco_full.py --limit 60                 # smoke
    python exp21_grefcoco_full.py                            # full val (14,229)
    python exp21_grefcoco_full.py --shard 0 --nshards 3 --tag grefcoco_sh0
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

from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from hybridgl_guided import GuidedScorer  # noqa: E402
from exp4_guided_full import bboxes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--refs", type=Path, default=None,
                    help="optional json {ref_ids: [...]} for the held-out tuning build")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("exp/out_extreme"))
    ap.add_argument("--tag", default="grefcoco")
    ap.add_argument("--points-per-side", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ds = GreferDataset(REPRO / "data/refer", args.split)
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                        REPRO / "checkpoints/ViT-B-16.pt", device=args.device,
                        cache_dir=REPRO / "cache/vision",
                        points_per_side=args.points_per_side)
    sc = GuidedScorer(enc, device=args.device)

    refs = ds.refs
    if args.refs is not None:
        wanted = json.loads(args.refs.read_text())["ref_ids"]
        by_id = {r["ref_id"]: r for r in refs}
        refs = [by_id[i] for i in wanted if i in by_id]
        print(f"{len(refs)} refs from {args.refs}", flush=True)
    if args.limit:
        refs = refs[:args.limit]
    if args.nshards > 1:
        refs = [r for i, r in enumerate(refs) if i % args.nshards == args.shard]
        print(f"shard {args.shard}/{args.nshards}: {len(refs)} refs", flush=True)

    out_path = args.output / f"records_{args.tag}.jsonl"
    n = skipped = 0
    t0 = time.perf_counter()
    iou_acc = [[], [], []]      # released-pure, released-guided, oracle (vs union GT)
    inter_acc = np.zeros(3)
    union_acc = np.zeros(3)
    n_no_target = 0
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
            masks, fg, bg, meta = enc.extract(im, file_sha256(ip))
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

            no_target = bool(ref["no_target"])
            n_tgt = ds.n_targets(ref)
            n_no_target += int(no_target)

            for si, s in enumerate(ref["sentences"]):
                sent = s["sent"]
                try:
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
                    no_target=no_target, n_targets=n_tgt, ann_ids=ds.target_ann_ids(ref),
                    file_name=ref["file_name"], cache_file=meta.get("cache_file"),
                    gt_iou=gt_iou.tolist(), inter_gt=inter_gt.tolist(), union_gt=union_gt.tolist(),
                    area=area.tolist(), cx=cx.tolist(), cy=cy.tolist(), bbox=bxs,
                    ps_idx=ps_idx.tolist(),
                    sc=res["sc"].tolist(), sc_neg=res["sc_neg"].tolist(),
                    sc_noun=res["sc_noun"].tolist(), sc_strip=res["sc_strip"].tolist(),
                    sc_content=res["sc_content"].tolist(),
                    noun_scores={k: v.tolist() for k, v in res["noun_scores"].items()},
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
    o = lambda k: float(inter_acc[k] / union_acc[k]) if union_acc[k] else 0.0  # noqa: E731
    summary = dict(tag=args.tag, n=n, skipped_refs=skipped, no_target_refs=n_no_target,
                   limit=args.limit, elapsed_s=round(time.perf_counter() - t0, 1),
                   pure=dict(mIoU=m(0), oIoU=o(0)), guided=dict(mIoU=m(1), oIoU=o(1)),
                   oracle=dict(mIoU=m(2), oIoU=o(2)))
    with (args.output / f"summary_{args.tag}.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
