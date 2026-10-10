#!/usr/bin/env python
"""EXP-1d — Collect a reusable per-expression record for wrong-instance repair.

Saves, per expression: the score vector under several prompt decompositions,
per-proposal GT IoU, per-proposal geometry, and — critically — the TRUE pairwise
mask IoU among the top-M proposals under each prompt, so confusion-set
construction can use real mask overlap instead of a bbox proxy.

Prompt variants (all from the same expression, no new model):
  full      the expression verbatim
  nospace   spatial vocabulary removed          ("left green pot" -> "green pot")
  content   nospace minus stopwords / determiners
  noun      "a photo of <last content word>"    (crude category probe)

Views: `fg` = HybridGL G2L masked-region feature, `bg` = complement feature
(the G2L global branch), `fused` = GLOD-RIS DOD fusion of the two.
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

SPATIAL = {"left", "right", "leftmost", "rightmost", "top", "bottom", "upper", "lower",
           "above", "below", "middle", "center", "centre", "mid", "central",
           "front", "back", "behind", "next", "near", "beside", "between", "under",
           "over", "side", "corner", "topmost", "frontmost", "l"}
ORDINAL = {"first", "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th"}
STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "with", "and", "that", "is",
        "are", "this", "these", "those", "there", "it", "its", "his", "her", "their",
        "by", "for", "from", "as", "into", "onto", "up", "down", "out", "off"}
TOP_M = 32


def nospace(sent):
    w = [x for x in re.findall(r"[A-Za-z0-9]+", sent) if x.lower() not in SPATIAL]
    return " ".join(w) if w else sent


def content(sent):
    w = [x for x in re.findall(r"[A-Za-z0-9]+", sent)
         if x.lower() not in SPATIAL and x.lower() not in STOP]
    return " ".join(w) if w else sent


def noun(sent):
    w = [x for x in re.findall(r"[A-Za-z0-9]+", sent)
         if x.lower() not in SPATIAL and x.lower() not in STOP and x.lower() not in ORDINAL]
    return "a photo of " + (w[-1] if w else sent)


VARIANTS = {"full": lambda s: s, "nospace": nospace, "content": content, "noun": noun}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp1d"))
    ap.add_argument("--tag", default="collect_s101")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split, limit=args.limit, seed=args.seed)
    cfg = Config()
    enc = VisionEncoder(REPRO / "checkpoints/sam_vit_h_4b8939.pth", REPRO / "checkpoints/ViT-B-16.pt",
                        device=args.device, cache_dir=REPRO / "cache/vision")

    out_path = args.output / f"records_{args.tag}.jsonl"
    n = 0
    t0 = time.perf_counter()
    with out_path.open("w") as fh:
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
            n_m = len(masks)

            inter_gt = np.logical_and(masks, gt).sum((1, 2)).astype(np.float64)
            union_gt = np.logical_or(masks, gt).sum((1, 2)).astype(np.float64)
            gt_iou = np.where(union_gt > 0, inter_gt / np.maximum(union_gt, 1), 1.0)

            # exact full-pool pairwise IoU on GPU, used for the containment control
            mf = torch.from_numpy(masks.reshape(n_m, -1)).to(enc.device, torch.float32)
            inter_all = (mf @ mf.T)
            ar_all = mf.sum(1)
            union_all = ar_all[:, None] + ar_all[None, :] - inter_all
            iou_all = (inter_all / union_all.clamp(min=1)).cpu().numpy()
            cont_all = (inter_all / ar_all.clamp(min=1)[:, None]).cpu().numpy()  # |i n j| / |i|
            del mf, inter_all, union_all
            area_all = (ar_all / masks.shape[1] / masks.shape[2]).cpu().numpy()
            ps_idx = np.arange(n_m)
            for i in range(n_m):
                jj = np.flatnonzero((cont_all[i] >= 0.95) & (area_all > area_all[i]))
                if len(jj):
                    ps_idx[i] = int(jj[np.argmin(area_all[jj])])
            del cont_all

            flat = masks.reshape(n_m, -1)
            ys, xs = np.mgrid[0:masks.shape[1], 0:masks.shape[2]]
            xs_f, ys_f = xs.reshape(-1), ys.reshape(-1)
            cnt = flat.sum(1).astype(np.float64)
            cx = (flat * xs_f).sum(1) / np.maximum(cnt, 1) / masks.shape[2]
            cy = (flat * ys_f).sum(1) / np.maximum(cnt, 1) / masks.shape[1]
            area = cnt / flat.shape[1]

            fg_n = F.normalize(fg_t, dim=-1)
            bg_n = F.normalize(bg_t, dim=-1)
            fu_n = F.normalize(fused, dim=-1)

            for sent in [s["sent"] for s in ref["sentences"]]:
                scores, tops, pairs = {}, {}, {}
                for name, fn in VARIANTS.items():
                    t = F.normalize(enc.text(fn(sent)).float().reshape(-1), dim=0)
                    scores[name] = dict(
                        fg=(fg_n @ t).numpy().tolist(),
                        fused=(fu_n @ t).numpy().tolist(),
                        bg=(bg_n @ t).numpy().tolist())
                    if name in ("full", "nospace", "content"):
                        top = np.argsort(-np.asarray(scores[name]["fused"]), kind="stable")[:TOP_M]
                        tops[name] = top.tolist()
                        sub = masks[top]
                        inter = np.logical_and(sub[:, None], sub[None, :]).sum((2, 3)).astype(np.float32)
                        ar = sub.reshape(len(top), -1).sum(1).astype(np.float32)
                        union = ar[:, None] + ar[None, :] - inter
                        pairs[name] = (inter / np.maximum(union, 1)).tolist()
                order = np.argsort(-np.asarray(scores["full"]["fused"]), kind="stable")
                orc = int(np.argmax(gt_iou))
                rec = dict(
                    ref_id=int(ref["ref_id"]), sent=sent, n=n_m,
                    prompt_nospace=nospace(sent), prompt_content=content(sent), prompt_noun=noun(sent),
                    gt_iou=gt_iou.tolist(), area=area.tolist(), cx=cx.tolist(), cy=cy.tolist(),
                    inter_gt=inter_gt.tolist(), union_gt=union_gt.tolist(),
                    ps_idx=ps_idx.tolist(),
                    scores=scores, tops=tops, pairs=pairs,
                    orc=orc, rank_orc=int(np.flatnonzero(order == orc)[0]),
                    gt_area=float(gt.mean()), iou_am=float(gt_iou[int(order[0])]),
                )
                fh.write(json.dumps(rec) + "\n")
                n += 1
            if n and n % 200 == 0:
                print(f"[{n}] {time.perf_counter()-t0:.0f}s", flush=True)

    print(json.dumps(dict(n=n, out=str(out_path), elapsed_s=round(time.perf_counter() - t0, 1))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
