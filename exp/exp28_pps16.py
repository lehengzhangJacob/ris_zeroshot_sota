#!/usr/bin/env python
"""EXP-28 — Pool-density robustness, fresh: pps=8 vs pps=16 on the 716
selection-split expressions.

Re-runs the released scorer with SAM ViT-H at points_per_side=16 (features
cached in cache/vision_pps16), then compares on identical expressions: oracle
mIoU, released mIoU, wrong-instance share, and the operator's delta. Writes
exp/out_extreme/records_pps16.jsonl and exp/out_final/pool_robustness.json.
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
from exp5_analyze_guided import Rec, taxonomy, reltype  # noqa: E402
from method import Selector  # noqa: E402
from hybridgl_guided import GuidedScorer  # noqa: E402
from exp4_guided_full import bboxes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run_records(device="cuda:0"):
    ds = RefDataset(REPRO / "data/refer", "refcoco", "val", limit=250, seed=101)
    cache_dir = REPRO / "cache/vision_pps16"
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                        REPRO / "checkpoints/ViT-B-16.pt", device=device,
                        cache_dir=cache_dir, points_per_side=16)
    sc = GuidedScorer(enc, device=device)
    out_path = ROOT / "exp/out_extreme/records_pps16.jsonl"
    n = 0
    t0 = time.perf_counter()
    with out_path.open("w") as fh:
        for ref in ds.refs:
            ip = ds.image_path(ref)
            if not ip.exists():
                continue
            im = Image.open(ip).convert("RGB")
            gt = ds.ground_truth(ref)
            masks, fg, bg, _ = enc.extract(im, file_sha256(ip))
            if masks.shape[1:] != gt.shape:
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
            for si, s in enumerate(ref["sentences"]):
                res = sc.score(im, masks, bxs, fg, s["sent"])
                rec = dict(ref_id=int(ref["ref_id"]), sent_idx=si, sent=s["sent"], n=n_m,
                           gt_iou=gt_iou.tolist(), inter_gt=inter_gt.tolist(),
                           union_gt=union_gt.tolist(), area=area.tolist(), cx=cx.tolist(),
                           cy=cy.tolist(), bbox=bxs, ps_idx=ps_idx.tolist(),
                           sc=res["sc"].tolist(), sc_neg=res["sc_neg"].tolist(),
                           sc_noun=res["sc_noun"].tolist(), sc_strip=res["sc_strip"].tolist(),
                           sc_content=res["sc_content"].tolist(),
                           score_gem=res["score_gem"].tolist(),
                           top_idx=res["top_idx"].tolist(), fused_tops=res["fused_tops"].tolist(),
                           pure=res["pure"], final=res["final"], orc=int(np.argmax(gt_iou)),
                           gt_area=float(gt.mean()))
                fh.write(json.dumps(rec) + "\n")
                n += 1
            if n and n % 100 == 0:
                print(f"[{n}] {(time.perf_counter()-t0)/n*1000:.0f} ms/expr", flush=True)
    print(f"wrote {out_path} ({n} records)")
    return out_path


def build_pairs(device="cuda:0"):
    ds = RefDataset(REPRO / "data/refer", "refcoco", "val", limit=250, seed=101)
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                        REPRO / "checkpoints/ViT-B-16.pt", device=device,
                        cache_dir=REPRO / "cache/vision_pps16", points_per_side=16)
    out = {}
    for ref in ds.refs:
        ip = ds.image_path(ref)
        if not ip.exists():
            continue
        im = Image.open(ip).convert("RGB")
        masks, _, _, _ = enc.extract(im, file_sha256(ip))
        n = len(masks)
        mf = torch.from_numpy(masks.reshape(n, -1)).to(enc.device, torch.float32)
        inter = mf @ mf.T
        ar = mf.sum(1)
        iou = (inter / (ar[:, None] + ar[None, :] - inter).clamp(min=1)).cpu().numpy().astype(np.float16)
        out[str(int(ref["ref_id"]))] = iou
    path = ROOT / "exp/out_extreme/pairs_pps16.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}")
    return path


def summarise(recs, own, name):
    rt = np.array([reltype(r.rel) for r in recs])
    sp = rt != "none"
    ref = np.array([r.gt[r.final] for r in recs])
    ours = np.array([r.gt[int(own(r))] for r in recs])
    tax = taxonomy(recs, lambda r: r.final)
    if sp.sum() == 0:
        raise ValueError("no spatial expressions")
    return dict(name=name, n=len(recs), proposals_per_image=float(np.mean([len(r.gt) for r in recs])),
                oracle_mIoU=float(np.mean([r.gt[r.orc] for r in recs])),
                released_mIoU=float(ref.mean()), ours_mIoU=float(ours.mean()),
                ours_delta=float((ours - ref).mean()),
                wrong_instance_share=tax["by_class"]["wrong-instance"]["gap_share"],
                under_seg_share=tax["by_class"]["under-segmentation"]["gap_share"],
                delta_spatial=float((ours - ref)[sp].mean()), n_spatial=int(sp.sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()
    rp16 = ROOT / "exp/out_extreme/records_pps16.jsonl"
    pp16 = ROOT / "exp/out_extreme/pairs_pps16.npz"
    if not args.skip_run or not rp16.exists():
        run_records(args.device)
    if not pp16.exists():
        build_pairs(args.device)

    sel = json.loads((ROOT / "exp/select_refs_s101.json").read_text())
    own = Selector()

    pairs16 = np.load(pp16)
    recs16 = [Rec(json.loads(l)) for l in rp16.open()]
    for r in recs16:
        r.pc = pairs16[str(r.r["ref_id"])].astype(np.float32)

    # pps=8 arm: the fresh full-val pass restricted to the same refs
    all8 = [Rec(json.loads(l)) for l in (ROOT / "exp/out_extreme/records_refcoco.jsonl").open()]
    pairs8 = np.load(ROOT / "exp/out_extreme/pairs_refcoco.npz")
    ids16 = {r.r["ref_id"] for r in recs16}
    recs8 = [r for r in all8 if r.r["ref_id"] in ids16 and r.r["ref_id"] in set(sel)]
    for r in recs8:
        r.pc = pairs8[str(r.r["ref_id"])].astype(np.float32)
    recs16 = [r for r in recs16 if r.r["ref_id"] in set(sel)]
    print(f"pps8 arm {len(recs8)} records, pps16 arm {len(recs16)} records")

    a = summarise(recs8, own, "SAM points_per_side=8")
    b = summarise(recs16, own, "SAM points_per_side=16")
    out = {"arm_pps8": a, "arm_pps16": b,
           "note": "identical selection-split expressions in both arms; all rows are "
                   "re-analyses of this run's scoring passes",
           "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ROOT / "exp/out_final/pool_robustness.json").write_text(json.dumps(out, indent=2))
    for arm in (a, b):
        print(f"  {arm['name']}: proposals/img {arm['proposals_per_image']:.1f} "
              f"oracle {arm['oracle_mIoU']:.4f} released {arm['released_mIoU']:.4f} "
              f"ours {arm['ours_mIoU']:.4f} (delta {arm['ours_delta']:+.4f}, "
              f"spatial {arm['delta_spatial']:+.4f}) wrong-inst {arm['wrong_instance_share']:.3f} "
              f"under-seg {arm['under_seg_share']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
