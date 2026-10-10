#!/usr/bin/env python
"""EXP-34 — does a stronger frozen encoder fix the selection bottleneck?

Same offline battery as exp32, but the region-text scorer is SigLIP-so400m
(1152-d, masked-region "local" + blurred-background "global" G2L analogue)
instead of the released CLIP ViT-B/16 G2L features.  Reports mIoU for the
released rows as references, so any gain is read against the same protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_map import cache_file  # noqa: E402
from hybridgl_guided import stripped, content_only  # noqa: E402
from siglip_feats import CACHE as SIG_CACHE, SiglipRegions  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

RECORDS = {
    "refcoco": "exp/out_extreme/records_refcoco.jsonl",
    "refcoco+": "exp/out_extreme/records_refcocoplus.jsonl",
    "refcocog": "exp/out_extreme/records_refcocog.jsonl",
}


def norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco+", choices=list(RECORDS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--text", default="noun", choices=["noun", "sent", "ens", "strip", "content"])
    args = ap.parse_args()

    from glod.data import RefDataset
    from glod.io import file_sha256
    ds = RefDataset(REPRO / "data/refer", args.dataset, "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(RECORDS[args.dataset])]
    if args.limit:
        recs = recs[: args.limit]
    print(f"{len(recs)} records ({args.dataset})", flush=True)

    enc = SiglipRegions(device=args.device)
    tcache = {}

    def text_vecs(p):
        out = {}
        for key, s in (("noun", p["noun_phrase"]), ("sent", p["sentence"]),
                       ("strip", stripped(p["sentence"])), ("content", content_only(p["sentence"]))):
            if s not in tcache:
                tcache[s] = enc.text([s])[0]
            out[key] = tcache[s]
        out["ens"] = norm(0.5 * norm(out["sent"]) + 0.5 * norm(out["noun"]))
        return out

    acc, used, names = {}, 0, None
    for i, r in enumerate(recs):
        ref = ref_by_id[r["ref_id"]]
        fn = ds.images[ref["image_id"]]["file_name"]
        h = file_sha256(REPRO / "data/refer/images/train2014" / fn)
        pth = SIG_CACHE / f"{h}.npz"
        if not pth.is_file():
            continue
        with np.load(pth, allow_pickle=False) as d:
            loc = d["local"].astype(np.float32)
            glob = d["global_"].astype(np.float32)
        n = len(loc)
        if n != len(r["gt_iou"]):
            continue
        gt = np.asarray(r["gt_iou"], dtype=np.float32)
        tv = text_vecs(r["parsed"])
        t = tv[args.text]
        loc_n, glob_n = norm(loc), norm(glob)
        fus = norm(loc_n + glob_n)
        scores = {
            "sig_local": loc_n @ t,
            "sig_global": glob_n @ t,
            "sig_fused": fus @ t,
        }
        # ensemble with the released CLIP score (rank-normalised mean)
        sc_clip = np.asarray(r["sc"][:n], dtype=np.float32)
        rk = lambda v: (np.argsort(np.argsort(v)) / max(1, len(v) - 1))
        scores["sig_x_clip_rank"] = rk(fus @ t) + rk(sc_clip)
        scores["sig_x_sc_noun_rank"] = rk(fus @ t) + rk(np.asarray(r["sc_noun"][:n], np.float32))
        if names is None:
            names = list(scores) + ["released_final", "released_pure", "orc_single"]
            acc = {k: 0.0 for k in names}
        used += 1
        for k, v in scores.items():
            acc[k] += float(gt[int(np.argmax(v))])
        acc["released_final"] += float(gt[int(r["final"])])
        acc["released_pure"] += float(gt[int(r["pure"])])
        acc["orc_single"] += float(gt[int(r["orc"])])
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    res = {k: round(100 * v / used, 3) for k, v in acc.items()}
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v:7.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(dataset=args.dataset, n=used,
                                               text=args.text, mIoU=res), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
