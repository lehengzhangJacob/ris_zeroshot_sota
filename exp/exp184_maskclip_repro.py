#!/usr/bin/env python
"""EXP-184 — MaskCLIP-based zero-shot pipeline, full official val splits.

Released computation (MaskCLIP, CVPR 2023): take the CLIP ViT-B/16 value
branch of the last transformer block (the value projection is used in place of
attention), project it to the 512-d text space with the release's ``proj``,
L2-normalise, and score with the CLIP text embedding.

MaskCLIP is a dense open-vocabulary segmenter and has no instance pool.  For
this diagnosis we give it its own pool from its own value map: three fixed
quantile thresholds (50/70/90th percentile of the per-expression map), 8-
connected components at 224, components below 0.5% of pixels dropped.  The
selected component is the one with highest mean value; the oracle is the
component with highest IoU against ground truth.  A same-pool control also
scores the shared SAM ViT-H proposals with the same dense map, so its
selection can be compared with the other pipelines on identical proposals.

    CUDA_VISIBLE_DEVICES=0 python exp/exp184_maskclip_repro.py --dataset refcoco
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp/literature/repos/clip"))
sys.path.insert(0, "/home/msj_team/Jacob/sys/glod_ris_repro")
from cache_map import cache_file  # noqa: E402
from glod.data import RefDataset  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTF = ROOT / "exp/out_final"
RECORDS = {"refcoco": ROOT / "exp/out_extreme/records_refcoco.jsonl",
           "refcocoplus": ROOT / "exp/out_extreme/records_refcocoplus.jsonl",
           "refcocog": ROOT / "exp/out_extreme/records_refcocog.jsonl"}
CLIP_DIR = Path("/home/msj_team/Jacob/sys/glod_ris_repro/checkpoints")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
S = 224
QS = (0.50, 0.70, 0.90)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(RECORDS), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-sam-control", action="store_true")
    ap.add_argument("--text", choices=["sentence", "noun"], default="noun")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    import clip
    from PIL import Image

    ds_name = {"refcocoplus": "refcoco+"}.get(args.dataset, args.dataset)
    ds = RefDataset(Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer"),
                    ds_name, "val")
    by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(RECORDS[args.dataset])]
    if args.limit:
        recs = recs[: args.limit]
    groups = {}
    for rec in recs:
        ref = by_id.get(rec["ref_id"])
        if ref is None:
            continue
        groups.setdefault(ref["image_id"], []).append(rec)
    print(f"{args.dataset}: {len(recs)} records over {len(groups)} images", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device, download_root=str(CLIP_DIR), jit=False)
    model.float().eval()
    vit = model.visual
    D = vit.transformer.resblocks[0].attn.embed_dim
    proj = vit.proj

    @torch.no_grad()
    def dense_map(img224):
        x = vit.conv1(img224)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = vit.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x).permute(1, 0, 2)  # LND
        for blk in vit.transformer.resblocks[:-1]:
            x = blk(x)
        blk = vit.transformer.resblocks[-1]
        y = blk.ln_1(x)
        qkv = F.linear(y, blk.attn.in_proj_weight, blk.attn.in_proj_bias)
        v = qkv.chunk(3, dim=-1)[2]
        v = F.linear(v, blk.attn.out_proj.weight, blk.attn.out_proj.bias)
        v = v + x
        v = v + blk.mlp(blk.ln_2(v))
        # the release's backbone has final_norm=True: ln_post is the final LN before proj
        feat = vit.ln_post(v[1:, 0]) @ proj  # patch tokens only, [L,512]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        side = int(round(feat.shape[0] ** 0.5))
        return feat.T.reshape(1, feat.shape[1], side, side)

    def decompose(pairs):
        gap = wrong = 0.0
        n = 0
        for a, o in pairs:
            n += 1
            g = max(0.0, o - a)
            gap += g
            if a < 0.5 <= o:
                wrong += g
        return dict(n=n, achieved=float(np.mean([a for a, _ in pairs])),
                    oracle=float(np.mean([o for _, o in pairs])),
                    wrong_instance_share=(wrong / gap if gap > 0 else 0.0),
                    gap_mass=gap)

    own = []
    sam = []
    gt_cache = {}
    I_tot = U_tot = 0.0
    t0 = time.time()
    for gi, (image_id, grecs) in enumerate(sorted(groups.items())):
        ref0 = by_id[grecs[0]["ref_id"]]
        fn = ds.images[image_id]["file_name"]
        img = Image.open(Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer/images/train2014") / fn).convert("RGB")
        W, H = img.size
        img224 = torch.from_numpy(
            ((np.asarray(img.resize((S, S)), np.float32) / 255.0 - MEAN) / STD)
            .transpose(2, 0, 1)).unsqueeze(0).to(device)
        dmap = dense_map(img224)[0].cpu().numpy()  # [512,g,g]
        up = torch.from_numpy(dmap).unsqueeze(0)
        up = F.interpolate(up, size=(S, S), mode="bilinear", align_corners=False)[0].numpy()
        up = up / (np.linalg.norm(up, axis=0, keepdims=True) + 1e-9)
        pool_masks_224 = None
        if not args.no_sam_control:
            with np.load(cache_file(fn), allow_pickle=False) as z:
                shape = tuple(int(x) for x in z["mask_shape"])
                masks = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                                      ).reshape(shape).astype(bool)
            mt = torch.from_numpy(masks[:, ::4, ::4].astype(np.float32)).unsqueeze(1)
            pool_masks_224 = F.interpolate(mt, size=(S, S), mode="nearest")[:, 0].numpy().astype(bool)
        for rec in grecs:
            ref = by_id[rec["ref_id"]]
            if ref["ref_id"] not in gt_cache:
                gt_cache[ref["ref_id"]] = ds.ground_truth(ref)
            gt = gt_cache[ref["ref_id"]]
            sent = (rec["parsed"].get("sentence") or rec.get("sent") or "").strip().lower()
            if args.text == "noun":
                sent = (rec["parsed"].get("noun_phrase") or "").strip().lower() or sent
            with torch.no_grad():
                tf = model.encode_text(clip.tokenize([sent], truncate=True).to(device))
            tf = (tf / tf.norm(dim=-1, keepdim=True))[0].cpu().numpy()
            sim = np.einsum("cxy,c->xy", up, tf)
            # own pool: fixed quantile thresholds + connected components
            comps, means = [], []
            for q in QS:
                thr = np.quantile(sim, q)
                binm = sim >= thr
                lab, nl = ndimage.label(binm, structure=np.ones((3, 3)))
                if nl == 0:
                    continue
                for k in range(1, nl + 1):
                    m = lab == k
                    a = int(m.sum())
                    if a < 0.005 * S * S:
                        continue
                    comps.append(m)
                    means.append(float(sim[m].mean()))
            if comps:
                order = np.argsort(means)[::-1][:12]  # fixed pool rule: top-12 by mean value
                comps = [comps[k] for k in order]
                means = [means[k] for k in order]
                pick = 0
                pred = torch.from_numpy(comps[pick].astype(np.float32)).unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(pred, size=(H, W), mode="nearest")[0, 0].numpy().astype(bool)
                ious = []
                for m in comps:
                    mm = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    mm = F.interpolate(mm, size=(H, W), mode="nearest")[0, 0].numpy().astype(bool)
                    inter = np.logical_and(mm, gt).sum()
                    union = np.logical_or(mm, gt).sum()
                    ious.append(float(inter) / float(union) if union else 0.0)
                a = ious[pick]
                o = max(ious)
                I_tot += float(np.logical_and(pred, gt).sum())
                U_tot += float(np.logical_or(pred, gt).sum())
                own.append((a, o))
            else:
                own.append((0.0, 0.0))
                U_tot += float(gt.sum())
            if pool_masks_224 is not None:
                scores = np.array([sim[m].mean() if m.any() else -1e9 for m in pool_masks_224])
                j = int(np.argmax(scores))
                gtv = np.asarray(rec["gt_iou"], np.float32)
                if len(gtv) == len(pool_masks_224):
                    sam.append((float(gtv[j]), float(gtv.max())))
        del dmap, up, pool_masks_224
        if (gi + 1) % 10 == 0:
            print(f"  {gi+1}/{len(groups)} images, {len(own)} records "
                  f"({time.time()-t0:.0f}s)", flush=True)
    n = len(own)
    m = decompose(own)
    out = dict(dataset=args.dataset, split="val", n=n,
               mIoU=100.0 * m["achieved"], oIoU=100.0 * I_tot / U_tot if U_tot else 0.0,
               oracle_mIoU=100.0 * m["oracle"],
               wrong_instance_share=m["wrong_instance_share"],
               pool=("own dense-map pool: 8-connected components of the MaskCLIP value map at "
                     "fixed quantile thresholds " + "/".join(str(q) for q in QS) +
                     "; components below 0.5% of pixels dropped; top-12 by mean value kept"),
               selection="component with highest mean MaskCLIP value",
               text_query=args.text,
               backbone="CLIP ViT-B/16 (released weights)",
               repo="https://github.com/chongzhou96/MaskCLIP",
               generated="exp/exp184_maskclip_repro.py",
               same_pool_control=(None if args.no_sam_control else
                                  dict(pool="SAM ViT-H pps 8 (shared)", **{
                                      k: (100.0 * v if k in ("achieved", "oracle") else v)
                                      for k, v in decompose(sam).items()})) )
    outp = args.out or (OUTF / f"maskclip_sam_{args.dataset}.json")
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
