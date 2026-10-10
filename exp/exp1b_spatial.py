#!/usr/bin/env python
"""EXP-1b — Is the wrong-instance error class resolvable without training?

Wrong-instance carries 82.7 % of the oracle gap mass and 84 % of those
expressions contain a spatial word. The hypothesis tested here: the CLIP score
already isolates a small set of mutually-disjoint near-tied candidates (the
*confusion set*), and the spatial relation in the expression resolves which one
is the referent.

This script measures, for a battery of deterministic resolvers, the mIoU/oIoU
over the same 716-expression sample as EXP-1, plus the per-class breakdown, so
a rule can be selected on evidence rather than intuition.

No training, no GT at decision time: GT is used only to score the output.
"""
from __future__ import annotations

import argparse
import json
import re
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

# ---------------------------------------------------------------- relation parsing
# Deterministic, string-based. The pilot that used CLIP text-embedding similarity
# to detect direction words fired on only 36 % of expressions; a literal matcher
# over the vocabulary RefCOCO actually uses is both wider and auditable.
H_WORDS = {
    "left": -1, "leftmost": -1, "l": -1,
    "right": +1, "rightmost": +1,
}
V_WORDS = {"top": -1, "upper": -1, "above": -1, "topmost": -1, "bottom": +1, "lower": +1, "below": +1}
MID_WORDS = {"middle", "center", "centre", "mid", "central"}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}


def parse_relation(sent):
    """Return a dict describing the spatial relation, or None."""
    words = [w for w in re.findall(r"[a-z0-9]+", sent.lower())]
    rel = dict(h=0, v=0, mid=False, ordinal=None, ordinal_axis=None)
    for i, w in enumerate(words):
        if w in H_WORDS:
            # "the right side" / "on the right" are still horizontal references
            rel["h"] = H_WORDS[w]
        if w in V_WORDS:
            rel["v"] = V_WORDS[w]
        if w in MID_WORDS:
            rel["mid"] = True
        if w in ORDINALS:
            rel["ordinal"] = ORDINALS[w]
            # "second from the left|right|top|bottom" — find the axis word nearby
            for w2 in words[i + 1:i + 4]:
                if w2 in ("left", "right", "l"):
                    rel["ordinal_axis"] = "h"
                    rel["ordinal_dir"] = H_WORDS[w2]
                if w2 in ("top", "bottom", "upper", "lower"):
                    rel["ordinal_axis"] = "v"
                    rel["ordinal_dir"] = V_WORDS[w2]
    if rel["h"] or rel["v"] or rel["mid"] or rel["ordinal"]:
        return rel
    return None


# ---------------------------------------------------------------- confusion set
def confusion_set(masks, scores, k=6, delta=0.25, iou_split=0.30):
    """Greedy: walk the score ranking, keep candidates disjoint from all kept."""
    order = np.argsort(-scores, kind="stable")
    rng = float(scores.max() - scores.min())
    thr = scores.max() - delta * rng if rng > 0 else scores.max()
    kept = []
    for i in order:
        if len(kept) >= k:
            break
        if scores[i] < thr and kept:
            break
        ok = True
        for j in kept:
            inter = int(np.logical_and(masks[i], masks[j]).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(masks[i], masks[j]).sum())
            if inter / max(union, 1) >= iou_split:
                ok = False
                break
        if ok:
            kept.append(int(i))
    return kept


def geom(masks, idx):
    """Centroid (cx, cy) normalized to [0,1] and area fraction."""
    sub = masks[idx]
    h, w = sub.shape[1], sub.shape[2]
    ys, xs = np.nonzero(sub.reshape(len(idx), -1))
    counts = sub.reshape(len(idx), -1).sum(1)
    # per-mask centroid: loop is fine (k <= 6)
    out = []
    for n, i in enumerate(idx):
        m = sub[n]
        yy, xx = np.nonzero(m)
        out.append((float(xx.mean()) / w, float(yy.mean()) / h, float(m.mean())))
    return np.array(out)


def resolve(rel, cand, g):
    """Pick one index from `cand` (list) using relation `rel`; g is [n,3] (cx,cy,area)."""
    cx, cy = g[:, 0], g[:, 1]
    if rel["ordinal"]:
        axis = rel["ordinal_axis"] or "h"
        direction = rel.get("ordinal_dir", -1)
        order = np.argsort(direction * (cx if axis == "h" else cy))
        pos = rel["ordinal"] - 1
        if pos < len(order):
            return cand[int(order[pos])]
        return None
    if rel["h"]:
        return cand[int(np.argmin(rel["h"] * cx))] if rel["h"] < 0 else cand[int(np.argmax(cx))]
    if rel["v"]:
        return cand[int(np.argmin(cy))] if rel["v"] < 0 else cand[int(np.argmax(cy))]
    if rel["mid"]:
        return cand[int(np.argmin(np.abs(cx - 0.5)))]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score", choices=["c1", "c3"], default="c3")
    ap.add_argument("--ks", default="2,3,4,6,8")
    ap.add_argument("--deltas", default="0.10,0.20,0.25,0.40,1.01")
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp1b"))
    ap.add_argument("--tag", default="spatial_s101_c3")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    KS = [int(x) for x in args.ks.split(",")]
    DELTAS = [float(x) for x in args.deltas.split(",")]

    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    cfg = Config()
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                        device=args.device, cache_dir=REPRO / "cache/vision")

    variants = ["argmax", "parent_strict"] + \
               [f"spatial_k{k}_d{d}" for k in KS for d in DELTAS] + \
               ["spatial_oracle_gate"]
    ious = {v: [] for v in variants}
    inter = {v: 0 for v in variants}
    union = {v: 0 for v in variants}
    meta = []  # per-expression record
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
        inter_gt = np.logical_and(masks, gt).sum((1, 2)).astype(np.float64)
        union_gt = np.logical_or(masks, gt).sum((1, 2)).astype(np.float64)
        gt_iou = np.where(union_gt > 0, inter_gt / np.maximum(union_gt, 1), 1.0)

        for sent in [s["sent"] for s in ref["sentences"]]:
            t = F.normalize(enc.text(sent).float().reshape(-1), dim=0)
            bf = F.normalize(fg_t, dim=-1) if args.score == "c1" else F.normalize(fused, dim=-1)
            s = (bf @ t).numpy()
            am = int(np.argmax(s))

            def iou_of(m):
                u = int(np.logical_or(m, gt).sum())
                return 1.0 if not u else int(np.logical_and(m, gt).sum()) / u

            # control: parent_strict containment snap
            larger = np.flatnonzero(area > area[am])
            if len(larger):
                ov = np.array([np.logical_and(masks[i], masks[am]).sum() / max(masks[am].sum(), 1)
                               for i in larger])
                cont = larger[ov >= 0.95]
                ps = int(cont[np.argmin(area[cont])]) if len(cont) else am
            else:
                ps = am

            rel = parse_relation(sent)
            iou_of(gt)  # noqa: B018 - keep helper used
            res = {"argmax": am, "parent_strict": ps}
            cs_cache = {}
            for k in KS:
                for d in DELTAS:
                    cand = confusion_set(masks, s, k=k, delta=d)
                    cs_cache[(k, d)] = cand
                    pick = am
                    if rel is not None and len(cand) >= 2:
                        g = geom(masks, cand)
                        chosen = resolve(rel, cand, g)
                        if chosen is not None:
                            pick = int(chosen)
                    res[f"spatial_k{k}_d{d}"] = pick
            # oracle-gated upper bound: fire only when the confusion set contains
            # the true best proposal (isolates resolver quality from set quality)
            cand = cs_cache[(KS[-1], 1.01)]
            orc = int(np.argmax(gt_iou))
            res["spatial_oracle_gate"] = am
            if rel is not None and len(cand) >= 2 and orc in cand:
                g = geom(masks, cand)
                chosen = resolve(rel, cand, g)
                if chosen is not None:
                    res["spatial_oracle_gate"] = int(chosen)

            for v in variants:
                v_iou = iou_of(masks[res[v]])
                ious[v].append(v_iou)
                inter[v] += int(np.logical_and(masks[res[v]], gt).sum())
                union[v] += int(np.logical_or(masks[res[v]], gt).sum())
            meta.append(dict(ref_id=int(ref["ref_id"]), sent=sent,
                             fired=bool(rel is not None),
                             cs_size=len(cs_cache[(KS[-1], 1.01)]),
                             orc=orc, am=am, rank_orc=int(np.flatnonzero(
                                 np.argsort(-s, kind="stable") == orc)[0]),
                             iou_am=float(gt_iou[am]), iou_orc=float(gt_iou[orc]),
                             rel=(rel or {}), n_prop=int(len(masks))))
            n += 1
        if n and n % 200 == 0:
            print(f"[{n}] {time.perf_counter()-t0:.0f}s", flush=True)

    base = float(np.mean(ious["argmax"]))
    table = [dict(op=v, mIoU=float(np.mean(ious[v])), delta=float(np.mean(ious[v])) - base,
                  oIoU=inter[v] / union[v] if union[v] else None) for v in variants]
    fired = [m for m in meta if m["fired"]]
    summary = dict(tag=args.tag, n=n, score=args.score, seed=args.seed,
                   baseline_argmax=base,
                   oIoU_argmax=inter["argmax"] / union["argmax"],
                   fired=len(fired), fired_frac=len(fired) / max(n, 1),
                   table=sorted(table, key=lambda r: -r["mIoU"]))
    with (args.output / f"summary_{args.tag}.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    with (args.output / f"records_{args.tag}.jsonl").open("w") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    print(json.dumps({k: summary[k] for k in ("n", "baseline_argmax", "fired", "fired_frac")}, indent=2))
    for r in summary["table"][:18]:
        print(f"  {r['op']:20s} mIoU={r['mIoU']:.4f} delta={r['delta']:+.4f} oIoU={r['oIoU']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
