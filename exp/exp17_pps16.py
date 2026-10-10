#!/usr/bin/env python
"""EXP-17 — Does the diagnosis survive a denser proposal pool?

Every claim in the paper is measured with one proposal pool (SAM ViT-H,
points_per_side = 8). The obvious attack is that the diagnosis is a property of
a sparse pool. This script re-runs the full released pipeline with SAM at
points_per_side = 16 on the same 716 selection-split expressions, then compares,
on identical expressions:

  * oracle mIoU            — does the pool actually get better?
  * released mIoU          — does selection get easier or harder?
  * wrong-instance share   — does the diagnosis change?
  * operator delta         — does the repair still work?

The pps=16 pool was built once and cached; this is pooled from three shards.
"""
from __future__ import annotations

import argparse
import json
import sys
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


def build_pairs(cache_dir, ref_ids):
    """Exact pairwise mask IoU per ref, from the pps=16 cache."""
    ds = RefDataset(REPRO / "data/refer", "refcoco", "val", limit=250, seed=101)
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                        device="cuda:0", cache_dir=cache_dir, points_per_side=16)
    out = {}
    for ref in ds.refs:
        rid = str(int(ref["ref_id"]))
        if rid not in ref_ids or rid in out:
            continue
        ip = ds.image_path(ref)
        if not ip.exists():
            continue
        try:
            im = Image.open(ip).convert("RGB")
            masks, _, _, _ = enc.extract(im, file_sha256(ip))
        except Exception:  # noqa: BLE001
            continue
        n = len(masks)
        mf = torch.from_numpy(masks.reshape(n, -1)).to(enc.device, torch.float32)
        inter = mf @ mf.T
        ar = mf.sum(1)
        out[rid] = (inter / (ar[:, None] + ar[None, :] - inter).clamp(min=1)).cpu().numpy().astype(np.float32)
    return out


def load(records, pc_map=None):
    recs = []
    for p in records:
        if not Path(p).exists():
            continue
        for line in Path(p).open():
            r = Rec(json.loads(line))
            if pc_map is not None:
                r.pc = pc_map.get(str(r.r["ref_id"]))
            recs.append(r)
    return recs


def summarise(recs, own, label):
    ok = [r for r in recs if r.pc is not None]
    tax = taxonomy(ok, lambda r: r.final)
    ref = np.array([r.gt[r.final] for r in ok])
    ours = np.array([r.gt[own(r)] for r in ok])
    rt = np.array([reltype(r.rel) for r in ok])
    sp = rt != "none"
    return dict(label=label, n=len(ok),
                oracle_mIoU=tax["oracle_mIoU"], released_mIoU=tax["mIoU"],
                ours_mIoU=float(ours.mean()), delta=float((ours - ref).mean()),
                wrong_instance_share=tax["by_class"]["wrong-instance"]["gap_share"],
                wrong_instance_prev=tax["by_class"]["wrong-instance"]["prevalence"],
                under_seg_share=tax["by_class"]["under-segmentation"]["gap_share"],
                mean_proposals=float(np.mean([r.r["n"] for r in ok])),
                spatial_frac=float(sp.mean()),
                delta_spatial=float((ours - ref)[sp].mean()) if sp.sum() else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("exp/out_final"))
    args = ap.parse_args()
    own = Selector()
    sel = set(json.loads(Path("exp/select_refs_s101.json").read_text()))

    # pps = 8, same 716 expressions
    pairs8 = np.load("exp/out_exp4/pairs_guided_val.npz")
    pc8 = {k: pairs8[k].astype(np.float32) for k in pairs8.files}
    recs8 = load(["exp/out_exp4/records_guided_val.jsonl"], pc8)
    recs8 = [r for r in recs8 if r.r["ref_id"] in sel]

    # pps = 16, three shards
    shard_recs = [f"exp/out_exp4/records_pps16_s101_sh{i}.jsonl" for i in range(3)]
    ids16 = set()
    for p in shard_recs:
        if Path(p).exists():
            for line in Path(p).open():
                ids16.add(str(int(json.loads(line)["ref_id"])))
    print(f"pps16 refs: {len(ids16)}; building pairs ...", flush=True)
    pc16 = build_pairs(REPRO / "cache/vision_pps16", ids16)
    recs16 = load(shard_recs, pc16)
    common = {str(r.r["ref_id"]) for r in recs8} & {str(r.r["ref_id"]) for r in recs16}
    recs8 = [r for r in recs8 if str(r.r["ref_id"]) in common]
    recs16 = [r for r in recs16 if str(r.r["ref_id"]) in common]
    print(f"common refs {len(common)}; pps8 expr {len(recs8)}, pps16 expr {len(recs16)}", flush=True)

    a = summarise(recs8, own, "SAM points_per_side=8")
    b = summarise(recs16, own, "SAM points_per_side=16")
    out = dict(a=a, b=b, common_refs=len(common))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with (args.output / "pool_robustness.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    hdr = ["pool", "expr", "props/img", "oracle", "released", "ours", "Δ ours",
           "wrong-inst share", "wrong-inst prev", "under-seg share", "Δ spatial"]
    print(" | ".join(hdr))
    for v in (a, b):
        print(" | ".join([v["label"], str(v["n"]), f"{v['mean_proposals']:.1f}",
                          f"{v['oracle_mIoU']:.4f}", f"{v['released_mIoU']:.4f}",
                          f"{v['ours_mIoU']:.4f}", f"{v['delta']:+.4f}",
                          f"{v['wrong_instance_share']:.3f}", f"{v['wrong_instance_prev']:.3f}",
                          f"{v['under_seg_share']:.3f}",
                          f"{v['delta_spatial']:+.4f}" if v["delta_spatial"] is not None else "-"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
