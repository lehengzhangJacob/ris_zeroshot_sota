#!/usr/bin/env python
"""EXP-32 — offline scorer/localizer battery on the cached pool.

GLOD-RIS's own ablations say the gain over HybridGL comes from (i) a coarse
spatial prior obtained from an external MLLM (their 3x3 grid -> Gaussian,
worth ~+5 mIoU) and (ii) orthogonal decoupling of the background direction
(~+2 mIoU).  We already own a *dense* spatial prior — the GEM heatmap the
released pipeline computes (`score_gem`) — but the released scorer only ever
uses it among its top-3 candidates.

This probe evaluates, offline and on the full val splits:
  * released argmax (`sc`)                        [reference]
  * DOD: F_fuse = (1-lam) F_fg + lam * F_fg_perp_bg, cos to text
  * GEM localization alone, and joint scores over top-k
    (S_dec * S_loc, S_dec + beta S_loc, category x localization)
Everything is a pure re-analysis of stored features + IoUs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_map import build_map, cache_file  # noqa: E402
from hybridgl_guided import stripped, content_only  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

RECORDS = {
    "refcoco": "exp/out_extreme/records_refcoco.jsonl",
    "refcoco+": "exp/out_extreme/records_refcocoplus.jsonl",
    "refcocog": "exp/out_extreme/records_refcocog.jsonl",
}


class TextEncoder:
    def __init__(self, device="cuda:0"):
        for rel in ("third_party/modified_CLIP", "third_party/segment-anything"):
            p = str(REPRO / "third_party/HybridGL" / rel)
            if p not in sys.path:
                sys.path.insert(0, p)
        import clip
        self.clip = clip
        self.device = device
        self.model, _ = clip.load(str(REPRO / "checkpoints/ViT-B-16.pt"), device=device, jit=False)
        self.model.eval()
        self.cache = {}

    def __call__(self, text):
        if text not in self.cache:
            with torch.inference_mode():
                tok = self.clip.tokenize(text, truncate=True).to(self.device)
                v = self.model.encode_text(tok).float()[0]
                v = v / v.norm()
            self.cache[text] = v.cpu().numpy()
        return self.cache[text]


def load_feats(fn, store):
    if fn in store:
        return store[fn]
    p = cache_file(fn)
    if not p.is_file():
        store[fn] = None
        return None
    with np.load(p, allow_pickle=False) as d:
        fg = d["foreground"].astype(np.float32)
        bg = d["background"].astype(np.float32)
    fg /= np.linalg.norm(fg, axis=1, keepdims=True) + 1e-9
    bg /= np.linalg.norm(bg, axis=1, keepdims=True) + 1e-9
    store[fn] = (fg, bg)
    return fg, bg


def dod_score(fg, bg, t, mu=0.3, tau=0.8, alpha=None):
    """GLOD-RIS Eqs. 1-7 (element-wise, batch form)."""
    cos = (fg * bg).sum(1, keepdims=True)
    resid = fg - cos * bg                       # normalised fg minus its bg component
    lam = np.clip(alpha, mu, tau)[:, None] if alpha is not None else np.full((len(fg), 1), 0.5)
    fuse = (1 - lam) * fg + lam * resid
    return fuse @ t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcocoplus", choices=list(RECORDS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", args.dataset, "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}

    recs = [json.loads(l) for l in open(RECORDS[args.dataset])]
    if args.limit:
        recs = recs[: args.limit]
    print(f"{len(recs)} records ({args.dataset})", flush=True)
    build_map([ds.images[r["image_id"]]["file_name"]
               for r in list(ref_by_id.values())])

    enc = TextEncoder(args.device)
    store = {}
    names = None
    acc = {}
    used = 0

    def add(name, value):
        acc[name] = acc.get(name, 0.0) + value

    for i, r in enumerate(recs):
        ref = ref_by_id[r["ref_id"]]
        fn = ds.images[ref["image_id"]]["file_name"]
        feats = load_feats(fn, store)
        if feats is None:
            continue
        fg, bg = feats
        n = len(fg)
        if n != len(r["gt_iou"]):
            continue
        p = r["parsed"]
        t_sent = enc(p["sentence"])
        t_noun = enc(p["noun_phrase"])
        t_ens = 0.5 * t_sent + 0.5 * t_noun
        t_ens /= np.linalg.norm(t_ens) + 1e-9
        t_strip = enc(stripped(p["sentence"]))
        area = np.asarray(r["area"][:n], dtype=np.float32)
        gem = np.asarray(r["score_gem"][:n], dtype=np.float32)
        # gem is lower-is-better in the released fusion (it enters fused_tops
        # directly); flip sign so higher = better for all our rules
        gem = -gem
        gem_rank = (gem - gem.min()) / (gem.max() - gem.min() + 1e-9)
        gt = np.asarray(r["gt_iou"], dtype=np.float32)

        sc = fg @ t_ens                      # released semantic score
        sc_noun = fg @ t_noun
        sc_strip = fg @ t_strip
        dod = dod_score(fg, bg, t_ens, alpha=area)
        dod_noun = dod_score(fg, bg, t_noun, alpha=area)

        def m(name, scores, topk=None):
            if topk:
                idx = np.argsort(scores)[::-1][:topk]
                j = idx[int(np.argmax(gem[idx]))]
            else:
                j = int(np.argmax(scores))
            add(name, float(gt[j]))

        variants = {
            "argmax_sc": (sc, None),
            "argmax_sc_noun": (sc_noun, None),
            "argmax_sc_strip": (sc_strip, None),
            "argmax_dod": (dod, None),
            "argmax_dod_noun": (dod_noun, None),
            "gem_only": (gem, None),
            "sc_gem_top3": (sc, 3),
            "sc_gem_top5": (sc, 5),
            "sc_gem_top10": (sc, 10),
            "sc_gem_top20": (sc, 20),
            "dod_gem_top5": (dod, 5),
            "dod_gem_top10": (dod, 10),
            "dod_gem_top20": (dod, 20),
        }
        if names is None:
            names = list(variants)
            names += ["released_final", "released_pure", "orc_single"]
            acc = {k: 0.0 for k in names}
        used += 1
        for k, (s, tk) in variants.items():
            m(k, s, tk)
        add("released_final", float(gt[int(r["final"])]))
        add("released_pure", float(gt[int(r["pure"])]))
        add("orc_single", float(gt[int(r["orc"])]))

        # multiplicative joint: normalised semantic x normalised localization
        scn = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
        sn = (sc_noun - sc_noun.min()) / (sc_noun.max() - sc_noun.min() + 1e-9)
        for tag, base in (("sc", scn), ("noun", sn)):
            for k in (5, 10):
                idx = np.argsort(sc)[::-1][:k] if tag == "sc" else np.argsort(sc_noun)[::-1][:k]
                joint = base[idx] * (0.5 + 0.5 * gem_rank[idx])
                add(f"{tag}xgem_top{k}", float(gt[idx[int(np.argmax(joint))]]))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    N = used
    res = {k: round(100 * v / N, 3) for k, v in acc.items()}
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v:7.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(dataset=args.dataset, n=N, mIoU=res), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
