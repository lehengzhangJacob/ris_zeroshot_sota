#!/usr/bin/env python
"""EXP-188 — S+S (Segment, Select, Correct) zero-shot select stage, full val.

Released selection rule: `vp_and_max_sim_clip_choice` with `reverse_blur`
(ssc_ris/select/clip_sim.py), CLIP ViT-B/32, as in the repository's
`--mask-choice reverse_blur` zero-shot path.  The original segment stage uses
GroundingDINO+SAM; here the rule runs on the shared SAM ViT-H automatic pool
(the pool substitution is stated in the protocol).  Full official val splits;
per-expression IoU and pooled oIoU are computed from the same records.

    CUDA_VISIBLE_DEVICES=0 python exp/exp188_splus_repro.py --dataset refcoco
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp/literature/repos/clip"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp/literature/repos/splus"))
sys.path.insert(0, "/home/msj_team/Jacob/sys/glod_ris_repro")
from cache_map import cache_file  # noqa: E402
from glod.data import RefDataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTF = ROOT / "exp/out_final"
RECORDS = {"refcoco": ROOT / "exp/out_extreme/records_refcoco.jsonl",
           "refcocoplus": ROOT / "exp/out_extreme/records_refcocoplus.jsonl",
           "refcocog": ROOT / "exp/out_extreme/records_refcocog.jsonl"}
CLIP_DIR = Path("/home/msj_team/Jacob/sys/glod_ris_repro/checkpoints")


def vp_reverse_blur(original_image, mask, std_dev=50):
    """Released S+S visual prompt (ssc_ris/select/visual_prompting_fns.py)."""
    import cv2
    blur = cv2.GaussianBlur(original_image.copy(), [0, 0], sigmaX=std_dev, sigmaY=std_dev)
    mb = cv2.bitwise_and(blur, blur, mask=(255 * (1 - mask)).astype(np.uint8))
    mo = cv2.bitwise_and(original_image, original_image, mask=(255 * mask).astype(np.uint8))
    return mb + mo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(RECORDS), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    import torch
    import clip
    from PIL import Image
    import cv2
    cv2.setNumThreads(4)
    torch.set_num_threads(4)

    ds_name = {"refcocoplus": "refcoco+"}.get(args.dataset, args.dataset)
    ds = RefDataset(Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer"), ds_name, "val")
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
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(CLIP_DIR), jit=False)
    model.float().eval()
    I_tot = U_tot = 0.0
    own = []
    t0 = time.time()
    for gi, (image_id, grecs) in enumerate(sorted(groups.items())):
        ref = by_id[grecs[0]["ref_id"]]
        fn = ds.images[image_id]["file_name"]
        img = Image.open(Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer/images/train2014") / fn).convert("RGB")
        rgb = np.asarray(img)
        bgr = rgb[:, :, ::-1].copy()
        with np.load(cache_file(fn), allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["mask_shape"])
            masks = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                                  ).reshape(shape).astype(bool)
        import cv2
        bgr224 = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
        blur224 = cv2.GaussianBlur(bgr224.copy(), [0, 0], sigmaX=26, sigmaY=26)
        tensors, idxs = [], []
        for mi, m in enumerate(masks):
            if m.sum() < 500:
                continue
            m224 = cv2.resize(m.astype(np.uint8), (224, 224),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            ann = np.where(m224[..., None], bgr224, blur224).astype(np.uint8)
            tensors.append(preprocess(Image.fromarray(ann[:, :, ::-1])))
            idxs.append(mi)
        if tensors:
            feats = []
            with torch.no_grad():
                for batch in torch.split(torch.stack(tensors), 32):
                    f = model.encode_image(batch.to(device))
                    f = f / f.norm(dim=-1, keepdim=True)
                    feats.append(f.cpu().numpy())
            feats = np.concatenate(feats, 0)
        else:
            feats = None
        for rec in grecs:
            gt = np.asarray(rec["gt_iou"], np.float32)
            if len(gt) != len(masks):
                continue
            sent = (rec["parsed"].get("sentence") or rec.get("sent") or "").strip().lower()
            with torch.no_grad():
                tf = model.encode_text(clip.tokenize([sent], truncate=True).to(device))
            tf = (tf / tf.norm(dim=-1, keepdim=True))[0].cpu().numpy()
            if feats is None:
                own.append((0.0, 0.0))
                continue
            sims = 100.0 * (feats @ tf)
            best = -np.inf; pick = 0
            for k, s in enumerate(sims):  # released >= tie-breaking
                if s >= best:
                    best = s; pick = k
            mi = idxs[pick]
            a = float(gt[mi]); o = float(gt.max())
            I_tot += float(rec["inter_gt"][mi]); U_tot += float(rec["union_gt"][mi])
            own.append((a, o))
        del feats
        if (gi + 1) % 10 == 0:
            print(f"  {gi+1}/{len(groups)} images, {len(own)} records "
                  f"({time.time()-t0:.0f}s)", flush=True)
    n = len(own)
    gap = wrong = 0.0
    for a, o in own:
        g = max(0.0, o - a); gap += g
        if a < 0.5 <= o:
            wrong += g
    out = dict(dataset=args.dataset, split="val", n=n,
               mIoU=100.0 * float(np.mean([a for a, _ in own])),
               oIoU=100.0 * I_tot / U_tot if U_tot else 0.0,
               oracle_mIoU=100.0 * float(np.mean([o for _, o in own])),
               wrong_instance_share=(wrong / gap if gap else 0.0),
               pool="SAM ViT-H pps 8 shared pool (the released stage-1 pool is "
                    "GroundingDINO+SAM; the selection rule is the release's)",
               selection="released `vp_and_max_sim_clip_choice` with reverse_blur, "
                         "CLIP ViT-B/32, std_dev=50, mask area >= 500, released >= tie-breaking",
               repo="https://github.com/fgirbal/segment-select-correct",
               protocol=("official val splits; per-expression mIoU and pooled oIoU under our "
                         "unified evaluator; SAM ViT-H pps 8 pool; published S+S zero-shot "
                         "numbers are oIoU 34.39/36.29/39.29 and mIoU 40.96/41.77/45.36 "
                         "(RefCOCO/+/g val), quoted as published"),
               generated="exp/exp188_splus_repro.py")
    outp = args.out or (OUTF / f"spluss_sam_{args.dataset}.json")
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
