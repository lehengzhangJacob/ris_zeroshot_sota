#!/usr/bin/env python
"""EXP-1 — Error taxonomy of the oracle gap.

Decompose Gap = IoU(oracle proposal, GT) - IoU(argmax proposal, GT) into
attributable classes, and report the fraction of the total gap MASS carried by
each class, plus prevalence by GT-area quartile.

Classes (deterministic, applied to the (argmax, oracle) mask pair):
  correct              IoU(argmax, GT) >= 0.5
  under-segmentation   argmax is a strict subset of oracle
  over-segmentation    oracle is a strict subset of argmax
  wrong-instance       IoU(argmax, oracle) < 0.1
  boundary             everything else (overlapping, neither contains the other)

Ambiguity rule (from EXPERIMENT_PLAN §5.2): a pair that is BOTH a strict
containment AND IoU < 0.1 is possible only for tiny masks; containment wins,
because containment is the structural fact. The `ambiguous` bucket is reported
explicitly rather than force-split.

Also records the score-side diagnostics the method design needs: where the
oracle proposal sits in the score ranking, and how close its score is to the
argmax score.

Writes per-expression records so every downstream number is a re-analysis.
"""
from __future__ import annotations

import argparse
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
from glod.io import file_sha256  # noqa: E402
from glod.vision import VisionEncoder  # noqa: E402

CLASSES = ["correct", "under-segmentation", "over-segmentation", "wrong-instance", "boundary"]


def classify(iou_am, iou_orc, sub_am_in_orc, sub_orc_in_am, iou_am_orc):
    if iou_am >= 0.5:
        return "correct"
    if sub_am_in_orc:
        return "under-segmentation"
    if sub_orc_in_am:
        return "over-segmentation"
    if iou_am_orc < 0.1:
        return "wrong-instance"
    return "boundary"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--all-sentences", action="store_true", default=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--score", choices=["c1", "c3"], default="c3")
    ap.add_argument("--output", type=Path,
                    default=Path("exp/out_exp1"))
    ap.add_argument("--tag", default="taxonomy_allsent_c3")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    cfg = Config()
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth",
                        REPRO / "checkpoints/ViT-B-16.pt",
                        device=args.device, cache_dir=REPRO / "cache/vision")

    recs = []
    n = 0
    t0 = time.perf_counter()
    for ref in ds.refs:
        ip = ds.image_path(ref)
        if not ip.exists():
            continue
        try:
            im = Image.open(ip).convert("RGB")
            gt = ds.ground_truth(ref)
        except Exception:  # noqa: BLE001
            continue
        masks, fg, bg, _ = enc.extract(im, file_sha256(ip))
        if masks.shape[1:] != gt.shape:
            continue
        fg_t, bg_t = torch.from_numpy(fg).float(), torch.from_numpy(bg).float()
        fused, _, _ = decouple(fg_t, bg_t, masks.mean((1, 2)), cfg)
        area = masks.reshape(len(masks), -1).mean(1)

        # per-proposal IoU against GT (vectorised)
        inter_gt = np.logical_and(masks, gt).sum((1, 2)).astype(np.float64)
        union_gt = np.logical_or(masks, gt).sum((1, 2)).astype(np.float64)
        ious = np.where(union_gt > 0, inter_gt / np.maximum(union_gt, 1), 1.0)
        orc = int(np.argmax(ious))

        sl = [s["sent"] for s in ref["sentences"]] if args.all_sentences \
            else [ref["sentences"][0]["sent"]]
        for sidx, sent in enumerate(sl):
            t = F.normalize(enc.text(sent).float().reshape(-1), dim=0)
            bf = F.normalize(fg_t, dim=-1) if args.score == "c1" else F.normalize(fused, dim=-1)
            s = (bf @ t).numpy()
            am = int(np.argmax(s))

            iou_am, iou_orc = float(ious[am]), float(ious[orc])
            inter_pair = int(np.logical_and(masks[am], masks[orc]).sum())
            a_am, a_orc = int(masks[am].sum()), int(masks[orc].sum())
            sub_am_in_orc = inter_pair == a_am and a_am < a_orc
            sub_orc_in_am = inter_pair == a_orc and a_orc < a_am
            iou_am_orc = inter_pair / max(a_am + a_orc - inter_pair, 1)
            cls = classify(iou_am, iou_orc, sub_am_in_orc, sub_orc_in_am, iou_am_orc)

            order = np.argsort(-s, kind="stable")
            rank_orc = int(np.flatnonzero(order == orc)[0])
            rank_am = int(np.flatnonzero(order == am)[0])
            rng = float(s.max() - s.min())
            # margin of the oracle's score below the argmax's, in pool-range units
            margin_orc = float((s.max() - s[orc]) / rng) if rng > 0 else 0.0
            # how many proposals sit within the oracle's score (the near-tie set)
            near_tie = int((s >= s[orc]).sum())

            # rank of the argmax when masked to the oracle's connected component is
            # not needed here; record the plain facts.
            tok = set(w.strip(".,!?").lower() for w in sent.split())
            recs.append(dict(
                ref_id=int(ref["ref_id"]), sent_idx=sidx, sent=sent,
                n_proposals=int(len(masks)), gt_area=float(gt.mean()),
                n_gt_px=int(gt.sum()),
                am=am, orc=orc, iou_am=iou_am, iou_orc=iou_orc, gap=iou_orc - iou_am,
                cls=cls, iou_am_orc=float(iou_am_orc),
                sub_am_in_orc=bool(sub_am_in_orc), sub_orc_in_am=bool(sub_orc_in_am),
                area_am=float(area[am]), area_orc=float(area[orc]),
                rank_orc=rank_orc, rank_am=rank_am,
                margin_orc=margin_orc, near_tie=near_tie,
                s_am=float(s[am]), s_orc=float(s[orc]), s_range=rng,
                score_spread_std=float(s.std()),
                is_part=bool(tok & PART_WORDS),
            ))
            n += 1
        if n and n % 200 == 0:
            print(f"[{n}] {time.perf_counter()-t0:.0f}s", flush=True)

    gap = np.array([r["gap"] for r in recs])
    cls = np.array([r["cls"] for r in recs])
    gt_area = np.array([r["gt_area"] for r in recs])
    iou_am = np.array([r["iou_am"] for r in recs])
    iou_orc = np.array([r["iou_orc"] for r in recs])
    total_gap_mass = float(gap.sum())

    q = np.quantile(gt_area, [0.25, 0.5, 0.75])
    qname = ["q1_smallest", "q2", "q3", "q4_largest"]
    qidx = np.digitize(gt_area, q, right=True)

    summary = {
        "tag": args.tag, "n": n, "score": args.score,
        "limit": args.limit, "seed": args.seed, "all_sentences": True,
        "mIoU_argmax": float(iou_am.mean()), "mIoU_oracle": float(iou_orc.mean()),
        "gap_total_mean": float(gap.mean()), "gap_mass_total": total_gap_mass,
        "by_class": {}, "by_class_quartile": {}, "part": {},
    }
    for c in CLASSES:
        m = cls == c
        summary["by_class"][c] = dict(
            n=int(m.sum()), prevalence=float(m.mean()),
            gap_mass=float(gap[m].sum()),
            gap_share=float(gap[m].sum() / total_gap_mass) if total_gap_mass else None,
            mean_gap=float(gap[m].mean()) if m.sum() else None,
            mean_iou_am=float(iou_am[m].mean()) if m.sum() else None,
            mean_iou_orc=float(iou_orc[m].mean()) if m.sum() else None,
        )
    for qi in range(4):
        m = qidx == qi
        summary["by_class_quartile"][qname[qi]] = dict(
            n=int(m.sum()), gt_area_mean=float(gt_area[m].mean()) if m.sum() else None,
            gap_mean=float(gap[m].mean()) if m.sum() else None,
            classes={c: dict(n=int(((cls == c) & m).sum()),
                             gap_mass=float(gap[(cls == c) & m].sum()),
                             share_of_quartile_gap=float(gap[(cls == c) & m].sum() / gap[m].sum())
                             if m.sum() and gap[m].sum() else None)
                      for c in CLASSES},
        )
    is_part = np.array([r["is_part"] for r in recs])
    for name, m in (("part", is_part), ("non_part", ~is_part)):
        summary["part"][name] = dict(
            n=int(m.sum()), gap_mean=float(gap[m].mean()) if m.sum() else None,
            classes={c: int(((cls == c) & m).sum()) for c in CLASSES})

    # score-side diagnostics for the wrong-instance and boundary classes
    for c in CLASSES:
        m = cls == c
        if not m.sum():
            continue
        summary["by_class"][c]["mean_rank_orc"] = float(np.mean([r["rank_orc"] for r in recs if r["cls"] == c]))
        summary["by_class"][c]["mean_margin_orc"] = float(np.mean([r["margin_orc"] for r in recs if r["cls"] == c]))
        summary["by_class"][c]["mean_near_tie"] = float(np.mean([r["near_tie"] for r in recs if r["cls"] == c]))

    with (args.output / f"records_{args.tag}.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    with (args.output / f"summary_{args.tag}.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in ("n", "mIoU_argmax", "mIoU_oracle", "gap_total_mean")}, indent=2))
    for c in CLASSES:
        v = summary["by_class"][c]
        print(f"  {c:20s} n={v['n']:5d} prev={v['prevalence']:.3f} gap_mass_share={v['gap_share']:.3f} "
              f"mean_gap={v['mean_gap']:.4f}" if v["n"] else f"  {c:20s} n=0")
    print("  quartiles:")
    for k, v in summary["by_class_quartile"].items():
        print(f"    {k:12s} n={v['n']:4d} gap={v['gap_mean']:.4f} " +
              " ".join(f"{c[:4]}={v['classes'][c]['share_of_quartile_gap']:.2f}" for c in CLASSES))
    return 0


PART_WORDS = {"hand", "hands", "arm", "arms", "leg", "legs", "head", "face", "foot", "feet", "eye", "eyes",
              "ear", "ears", "wheel", "wheels", "tail", "wing", "wings", "nose", "mouth", "neck", "finger",
              "fingers", "paw", "paws", "hoof", "beak", "horn", "horns", "antenna", "trunk", "handle",
              "lid", "door", "window", "sleeve", "collar", "tire", "tyre", "propeller", "blade", "stem"}

if __name__ == "__main__":
    raise SystemExit(main())
