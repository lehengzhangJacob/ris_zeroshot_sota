#!/usr/bin/env python
"""EXP-183 — full-val reproduction of the Global-Local (Zero-shot-RIS) selector.

Released pipeline: Seonghoon-Yu/Zero-shot-RIS (CVPR 2023).  Its selection head is
a CLIP ViT-B/32 with (a) token-mask transformers from block 9 for the proposal
("global") feature and (b) masked-image crop features for the "local" feature,
fused as v*global + (1-v)*local.  Text is 0.5*sentence + 0.5*noun phrase.  Score
is cosine; argmax selects one proposal.

The original pool is FreeSOLO R101 (checkpoint link dead; see
globallocal_blocked.json); here the same released selector runs on the SAM
ViT-H pool used by the rest of the diagnosis.  This is the "Global-Local (SAM)"
variant, and the pool substitution is stated in the output.

    CUDA_VISIBLE_DEVICES=0 python exp/exp183_globallocal_repro.py --dataset refcoco
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

ROOT = Path(__file__).resolve().parents[1]
OUTF = ROOT / "exp/out_final"
RECORDS = {"refcoco": ROOT / "exp/out_extreme/records_refcoco.jsonl",
           "refcocoplus": ROOT / "exp/out_extreme/records_refcocoplus.jsonl",
           "refcocog": ROOT / "exp/out_extreme/records_refcocog.jsonl"}
CLIP_DIR = Path("/home/msj_team/Jacob/sys/glod_ris_repro/checkpoints/clip")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(RECORDS), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
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
        fn = ds.images[ref["image_id"]]["file_name"]
        groups.setdefault(fn, []).append(rec)
    print(f"{args.dataset}: {len(recs)} records over {len(groups)} images", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, download_root=str(CLIP_DIR), jit=False)
    model.float().eval()
    vit = model.visual
    v = 0.85 if args.dataset == "refcocog" else 0.95
    r = 0.5

    @torch.no_grad()
    def encode_image_tokens(img224):
        x = vit.conv1(img224)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = vit.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x).permute(1, 0, 2)
        for i, blk in enumerate(vit.transformer.resblocks):
            if i < 9:
                x = blk(x)
        return x  # [50, 1, 768]

    @torch.no_grad()
    def masked_features(img224, masks_np):
        """Exact CLIPViTFM token_masking path (masking_block=9, last_layer=11)."""
        x = encode_image_tokens(img224)  # pre-block-9 tokens, [50,1,768]
        N = len(masks_np)
        D = x.shape[-1]
        size = int(round((x.shape[0] - 1) ** 0.5))
        mt = torch.from_numpy(np.stack(masks_np, 0).astype(np.float32)).to(device)
        mt = F.interpolate(mt[:, None], size=(size, size), mode="bilinear")[:, 0]
        for i in range(9, 12):
            cls = x[:1]
            cur = x.shape[1]
            rest = x[1:].permute(1, 2, 0).reshape(cur, D, size, size)
            rest = rest * mt[:, None]
            nxt = rest.shape[0]
            rest = rest.reshape(nxt, D, size * size).permute(2, 0, 1)
            x = torch.cat([cls.expand(-1, nxt, -1), rest], dim=0)
            x = vit.transformer.resblocks[i](x)
        x = vit.ln_post(x[0].float()) @ vit.proj
        return x  # [N, 512]

    @torch.no_grad()
    def crop_features(crops):
        outs = []
        for c in torch.split(crops, args.batch):
            outs.append(vit(c))
        return torch.cat(outs, 0)

    n_used = 0
    I_tot = U_tot = 0.0
    iou_sum = 0.0
    oracle_sum = 0.0
    gap_wrong = gap_all = 0.0
    t0 = time.time()
    for gi, (fn, grecs) in enumerate(sorted(groups.items())):
        with np.load(cache_file(fn), allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["mask_shape"])
            masks = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                                  ).reshape(shape).astype(bool)
        img = Image.open(Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer/images/train2014") / fn).convert("RGB")
        arr = np.asarray(img, np.float32) / 255.0
        mean_img = arr.reshape(-1, 3).mean(0)
        img224 = torch.from_numpy(
            ((np.asarray(img.resize((224, 224)), np.float32) / 255.0 - MEAN) / STD)
            .transpose(2, 0, 1)).unsqueeze(0).to(device)
        n = len(masks)
        mfeat = masked_features(img224, masks).detach()
        crops = []
        for m in masks:
            ys, xs = np.where(m)
            if len(ys) == 0:
                crops.append(torch.zeros(3, 224, 224)); continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            masked = arr.copy(); masked[~m] = mean_img
            crop = Image.fromarray((masked[y0:y1, x0:x1] * 255).astype(np.uint8)).resize((224, 224))
            crops.append(torch.from_numpy(
                ((np.asarray(crop, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)))
        cfeat = crop_features(torch.stack(crops, 0).to(device)).detach()
        visual = (v * mfeat + (1 - v) * cfeat)
        visual = visual / visual.norm(dim=-1, keepdim=True)
        visual_np = visual.cpu().numpy()
        for rec in grecs:
            gt = np.asarray(rec["gt_iou"], np.float32)
            inter = np.asarray(rec["inter_gt"], np.float64)
            union = np.asarray(rec["union_gt"], np.float64)
            if len(gt) != n:
                continue
            sent = (rec["parsed"].get("sentence") or "").strip().lower()
            noun = (rec["parsed"].get("noun_phrase") or "").strip().lower() or sent
            with torch.no_grad():
                sf = model.encode_text(clip.tokenize([sent], truncate=True).to(device))
                nf = model.encode_text(clip.tokenize([noun], truncate=True).to(device))
            sf = sf / sf.norm(dim=-1, keepdim=True)
            nf = nf / nf.norm(dim=-1, keepdim=True)
            tfeat = (r * sf + (1 - r) * nf)[0].cpu().numpy()
            tfeat = tfeat / (np.linalg.norm(tfeat) + 1e-9)
            pick = int(np.argmax(visual_np @ tfeat))
            a = float(gt[pick]); o = float(gt.max())
            iou_sum += a
            oracle_sum += o
            I_tot += float(inter[pick]); U_tot += float(union[pick])
            g = max(0.0, o - a); gap_all += g
            if a < 0.5 <= o:
                gap_wrong += g
            n_used += 1
        del mfeat, cfeat, visual, masks
        if (gi + 1) % 10 == 0:
            print(f"  {gi+1}/{len(groups)} images, {n_used} records "
                  f"({time.time()-t0:.0f}s)", flush=True)
    n = n_used
    out = dict(dataset=args.dataset, split="val", n=n,
               mIoU=100.0 * iou_sum / max(1, n),
               oIoU=100.0 * I_tot / U_tot if U_tot else 0.0,
               oracle_mIoU=100.0 * oracle_sum / max(1, n),
               wrong_instance_share=(gap_wrong / gap_all if gap_all else 0.0),
               pool="SAM ViT-H (pps 8) pool shared by the diagnosis; the released "
                    "pipeline's own FreeSOLO R101 pool is unavailable (blocked entry)",
               selection=("released Global-Local head: CLIP ViT-B/32 token-masking from block 9 "
                          "(masking_block=9, last_layer=11) for the global feature; masked-image "
                          "crop features for the local feature; visual = v*global + (1-v)*local, "
                          "v=0.95 (0.85 for RefCOCOg); text = 0.5*sentence + 0.5*noun phrase; "
                          "cosine argmax"),
               backbone="CLIP ViT-B/32 (released weights)",
               repo="https://github.com/Seonghoon-Yu/Zero-shot-RIS",
               generated="exp/exp183_globallocal_repro.py")
    outp = args.out or (OUTF / f"globallocal_sam_{args.dataset}.json")
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
