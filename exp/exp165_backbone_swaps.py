#!/usr/bin/env python
"""EXP-165 — backbone-audit swap arms (operator directive 2026-10-16).

Measures the STRICTLY WEAKER side so the paper can claim "fewer/weaker
backbones than the rival, still SOTA":

  box-3b     LLaVA-OneVision-7B -> Qwen2.5-VL-3B-Instruct boxes (arm a; run
             exp164_boxer3b.py first, then `eval-3b`)
  sam-b      SAM ViT-H -> SAM ViT-B proposal pool (ViT-L tried: the HF port
             restructures the prompt encoder and the original-format ViT-L
             checkpoint is not reachable offline), G2L features re-extracted,
             head refit on SAM-B tune rows (arm b; `extract-saml` then
             `eval-saml`)
  clipw      released G2L region-text platform -> plain masked-region CLIP
             features (no G2L fusion, no complement blur; same ViT-B/16
             weights), head refit on tune rows (arm c; `eval-clipw`)
  det-no     gRefCOCO without GroundingDINO (arm d; exp163, separate script)

Every arm is under the deployment-consistent head (refit on the train-carved
tune rows for a changed feature platform/pool; boxes serve the fitted head
directly) on the full official val splits, aligned protocol.

    python exp/exp165_backbone_swaps.py extract-saml --dataset refcoco --shard 0 --nshards 3
    python exp/exp165_backbone_swaps.py eval-saml    --dataset refcoco
    python exp/exp165_backbone_swaps.py eval-clipw   --dataset refcoco
    python exp/exp165_backbone_swaps.py eval-3b      --dataset refcoco
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_map import cache_file, _image_path  # noqa: E402
import exp159_head_rebuild as e159  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))
SAML_CKPT = REPRO / "checkpoints/sam_vit_b_01ec64.pth"
SAML_CACHE = REPRO / "cache/vision_samb"
CLIP_CKPT = REPRO / "checkpoints/ViT-B-16.pt"
PARTS = Path("exp/out_final/backbone_swaps")
IMAGES = REPRO / "data/refer/images/train2014"


# ---- SAM-L cache addressing (mirror cache_map with the ViT-L checkpoint) ---
@functools.lru_cache(maxsize=1)
def _saml_settings():
    from glod.io import digest
    sam = SAML_CKPT
    clip = CLIP_CKPT
    return dict(upstream_commit="f7eb19bb618353076f1da19485c1d9c6236a6f31",
                feature_version=1, preprocessing="official-hybridgl",
                fusion_mode="G2L", masking_block=9, sam_backbone="vit_b",
                points_per_side=8, pred_iou_thresh=0.7,
                stability_score_thresh=0.7, min_mask_region_area=800,
                checkpoints=[(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                             for p in (sam, clip)])


def saml_cache_file(fn):
    from glod.io import digest, file_sha256
    settings = _saml_settings()
    h = file_sha256(_image_path(fn))
    return SAML_CACHE / (digest(dict(image_sha256=h, settings=settings)) + ".npz")


def load_npz_masks_fg(path):
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(int(x) for x in z["mask_shape"])
        masks = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                              ).reshape(shape).astype(bool)
        fg = z["foreground"].astype(np.float32)
    fg /= np.linalg.norm(fg, axis=1, keepdims=True) + 1e-9
    return masks, fg


def gt_arrays(masks, gt):
    inter = np.logical_and(masks, gt).sum((1, 2)).astype(np.float64)
    union = np.logical_or(masks, gt).sum((1, 2)).astype(np.float64)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return iou, inter, union


# ---- plain masked-region CLIP features (arm c) ----------------------------
class PlainEncoder:
    """Masked-region CLIP ViT-B/16, our own masking: no G2L fusion, no
    complement blur.  Same approach as the released local path's input only."""

    def __init__(self, device="cuda:0"):
        for rel in ("third_party/modified_CLIP", "third_party/segment-anything"):
            p = str(REPRO / "third_party/HybridGL" / rel)
            if p not in sys.path:
                sys.path.insert(0, p)
        import clip
        import torch
        import torchvision.transforms as T
        self.clip, self.torch, self.T = clip, torch, T
        self.device = device
        self.model, _ = clip.load(str(CLIP_CKPT), device=device, jit=False)
        self.model.eval()
        self.resize = T.Resize((224, 224), antialias=None)
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.fill = torch.tensor([0.48145466, 0.4578275, 0.40821073]
                                 ).reshape(3, 1, 1)
        self.cache = {}

    def feats(self, fn, masks):
        if fn in self.cache:
            return self.cache[fn]
        from PIL import Image
        rgb = np.asarray(Image.open(_image_path(fn)).convert("RGB")).copy()
        if rgb.shape[:2] != masks.shape[1:]:
            from PIL import Image as I2
            rgb = np.asarray(I2.fromarray(rgb).resize(
                (masks.shape[2], masks.shape[1]))).copy()
        t0 = self.T.ToTensor()(rgb)
        base = self.normalize(t0)
        fill = self.fill.to(base.dtype)
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(masks), 32):
                batch = masks[i:i + 32]
                imgs = []
                for m in batch:
                    mt = self.torch.from_numpy(m.astype(np.float32))
                    imgs.append(self.resize(base * mt[None] + (1 - mt[None]) * fill))
                feats = self.model.encode_image(
                    self.torch.stack(imgs).to(self.device))
                if feats.dim() == 3:          # modified CLIP returns token grids
                    feats = feats[:, 0]       # pooled CLS token
                feats = feats.float()
                feats = feats / feats.norm(dim=1, keepdim=True)
                out.append(feats.cpu().numpy())
        f = np.concatenate(out).astype(np.float32)
        if len(self.cache) > 6:
            self.cache.pop(next(iter(self.cache)))
        self.cache[fn] = f
        return f


# ---- SAM-L extraction ------------------------------------------------------
def _encoder_for_ckpt(sam_path, device, cache_dir, points_per_side=8,
                      min_region_area=800, batch_size=8):
    """VisionEncoder replica for a smaller SAM: glod.vision hardcodes vit_h,
    so the model is built from registry["vit_b"] while the settings (and thus
    the cache key) match exp165's _saml_settings exactly."""
    import torch
    import torchvision.transforms as T
    from glod.vision import VisionEncoder, load_upstream
    enc = VisionEncoder.__new__(VisionEncoder)
    enc.device = torch.device(device)
    enc.batch_size = batch_size
    enc.cache_dir = Path(cache_dir)
    enc.settings = _saml_settings()
    clip, registry, generator, backbone = load_upstream()
    enc.clip = clip
    enc.encoder = backbone.__new__(backbone)
    torch.nn.Module.__init__(enc.encoder)
    enc.encoder.last_layer, enc.encoder.num_heads = 10, 12
    enc.encoder.model, _ = clip.load(str(CLIP_CKPT), device=enc.device, jit=False)
    enc.encoder.eval().requires_grad_(False)
    sam = registry["vit_b"](checkpoint=str(sam_path)).to(enc.device).eval()
    sam.requires_grad_(False)
    enc.generator = generator(sam, points_per_side=points_per_side,
                              pred_iou_thresh=0.7, stability_score_thresh=0.7,
                              crop_n_layers=0, crop_n_points_downscale_factor=1,
                              min_mask_region_area=min_region_area)
    enc.resize = T.Resize((224, 224), antialias=None)
    enc.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    enc.fill = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    return enc


def extract_saml(dataset, split, shard, nshards, device):
    from glod.data import RefDataset
    from glod.io import file_sha256
    ds_split = "train" if split == "tune" else split
    ds = RefDataset(REPRO / "data/refer", e159.DS_NAME[dataset], ds_split)
    fns, seen = [], set()
    for r in ds.refs:
        fn = ds.images[r["image_id"]]["file_name"]
        if fn not in seen:
            seen.add(fn)
            fns.append(fn)
    fns = [f for i, f in enumerate(sorted(fns)) if i % nshards == shard]
    SAML_CACHE.mkdir(parents=True, exist_ok=True)
    todo, missing_imgs = [], 0
    for fn in fns:
        try:
            if not saml_cache_file(fn).is_file():
                todo.append(fn)
        except FileNotFoundError:
            missing_imgs += 1
    print(f"{dataset}/{split}: {len(fns)} images, {len(todo)} to extract, "
          f"{missing_imgs} skipped (image file absent from the repro tree)",
          flush=True)
    if not todo:
        return 0
    enc = _encoder_for_ckpt(SAML_CKPT, device, SAML_CACHE)
    t0 = time.time()
    for k, fn in enumerate(todo):
        from PIL import Image
        ip = _image_path(fn)
        im = Image.open(ip).convert("RGB")
        enc.extract(im, file_sha256(ip))
        if (k + 1) % 25 == 0:
            rate = (k + 1) / (time.time() - t0)
            print(f"  {k+1}/{len(todo)} ({rate:.2f} img/s, eta "
                  f"{(len(todo)-k-1)/max(rate,1e-9)/60:.1f} min)", flush=True)
    print(f"done {len(todo)} in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


# ---- SAM-L evaluation (head refit on SAM-L tune rows) ----------------------
def _fit_saml_heads(dataset, te, limit=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from glod.data import RefDataset
    from vlm_boxes import QwenBoxer
    ds = RefDataset(REPRO / "data/refer", e159.DS_NAME[dataset], "train")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    boxer = QwenBoxer.__new__(QwenBoxer)
    recs = [json.loads(l) for l in open(e159.TUNE[dataset])]
    if limit:
        recs = recs[:limit]
    te.batch([r["parsed"]["sentence"] for r in recs])
    Fs, ys, n_rec = [], [], 0
    for r in recs:
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        p = saml_cache_file(fn)
        if not p.is_file():
            continue
        masks, fg = load_npz_masks_fg(p)
        if len(masks) != len(r["gt_iou"]):
            continue
        gt = ds.ground_truth(ref)
        if gt.shape != masks.shape[1:]:
            continue
        iou, _, _ = gt_arrays(masks, gt)
        bp = boxer._path(fn, r["parsed"]["sentence"].strip())
        box = json.loads(bp.read_text())["box"] if bp.is_file() else None
        our = fg @ te(r["parsed"]["sentence"])
        feats = e159.arm_c_features(masks, fg, r, box, our)
        Fs.append(feats)
        ys.append((iou >= 0.5).astype(np.int8))
        n_rec += 1
    y = np.concatenate(ys)

    def _fit(cols):
        X = np.concatenate([e159.feat_matrix(F, cols) for F in Fs])
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                             learning_rate=0.08, random_state=0)
        clf.fit(X, y)
        return clf

    print(f"  [fit saml] {dataset}: {n_rec} tune records, {len(y)} proposals, "
          f"pos {y.mean():.4f}", flush=True)
    return _fit(e159.ALL_COLS), _fit(e159.GEOM2_COLS)


def eval_saml(dataset, te, models, split="val", records_path=None, limit=0):
    from glod.data import RefDataset
    from vlm_boxes import QwenBoxer
    ds = RefDataset(REPRO / "data/refer", e159.DS_NAME[dataset], split)
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    boxer = QwenBoxer.__new__(QwenBoxer)
    recs = [json.loads(l) for l in
            open(records_path or e159.RECORDS[dataset])]
    if limit:
        recs = recs[:limit]
    te.batch([r["parsed"]["sentence"] for r in recs])
    model_c, model_g2 = models
    arms = ("learned_head_saml", "learned_geom2_saml", "geom2_only_saml")
    acc = {a: dict(iou=0.0, inter=0.0, union=0.0) for a in arms}
    n_used = n_box_none = 0
    t0 = time.time()
    for k, r in enumerate(recs):
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        p = saml_cache_file(fn)
        if not p.is_file():
            continue
        masks, fg = load_npz_masks_fg(p)
        if len(masks) != len(r["gt_iou"]):
            continue
        gt = ds.ground_truth(ref)
        if gt.shape != masks.shape[1:]:
            continue
        iou, inter, union = gt_arrays(masks, gt)
        bp = boxer._path(fn, r["parsed"]["sentence"].strip())
        box = json.loads(bp.read_text())["box"] if bp.is_file() else None
        n_box_none += int(box is None)
        our = fg @ te(r["parsed"]["sentence"])
        feats = e159.arm_c_features(masks, fg, r, box, our)
        if box is None:
            picks = {a: int(np.argmax(model_c.predict_proba(
                e159.feat_matrix(feats, e159.ALL_COLS))[:, 1])) for a in arms}
        else:
            loc = feats["loc"]
            picks = dict(
                learned_head_saml=int(np.argmax(model_c.predict_proba(
                    e159.feat_matrix(feats, e159.ALL_COLS))[:, 1])),
                learned_geom2_saml=int(np.argmax(model_g2.predict_proba(
                    e159.feat_matrix(feats, e159.GEOM2_COLS))[:, 1])),
                geom2_only_saml=int(np.argmax(feats["loc"])),
            )
        for a, j in picks.items():
            acc[a]["iou"] += float(iou[j])
            acc[a]["inter"] += float(inter[j])
            acc[a]["union"] += float(union[j])
        n_used += 1
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
    res = {a: dict(n=n_used, mIoU=round(100.0 * acc[a]["iou"] / n_used, 4),
                   oIoU=round(100.0 * acc[a]["inter"] / acc[a]["union"], 4)
                   if acc[a]["union"] else 0.0) for a in arms}
    return res, n_used, n_box_none


# ---- plain-platform evaluation (arm c) -------------------------------------
def _fit_clipw_head(dataset, te, pe, box_path=None, limit=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from glod.data import RefDataset
    from vlm_boxes import QwenBoxer
    ds = RefDataset(REPRO / "data/refer", e159.DS_NAME[dataset], "train")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    boxer = QwenBoxer.__new__(QwenBoxer)
    if box_path is None:
        box_path = boxer._path
    recs = [json.loads(l) for l in open(e159.TUNE[dataset])]
    if limit:
        recs = recs[:limit]
    te.batch([r["parsed"]["sentence"] for r in recs])
    Fs, ys, n_rec = [], [], 0
    for r in recs:
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        p = cache_file(fn)
        if not p.is_file():
            continue
        masks, fg = load_npz_masks_fg(p)
        if len(masks) != len(r["gt_iou"]):
            continue
        our = pe.feats(fn, masks) @ te(r["parsed"]["sentence"])
        bp = box_path(fn, r["parsed"]["sentence"].strip())
        box = json.loads(bp.read_text())["box"] if bp.is_file() else None
        feats = e159.arm_c_features(masks, fg, r, box, our)
        Fs.append(feats)
        ys.append((np.asarray(r["gt_iou"][:len(masks)]) >= 0.5).astype(np.int8))
        n_rec += 1
    y = np.concatenate(ys)

    def _fit(cols):
        X = np.concatenate([e159.feat_matrix(F, cols) for F in Fs])
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                             learning_rate=0.08, random_state=0)
        clf.fit(X, y)
        return clf

    print(f"  [fit clipw] {dataset}: {n_rec} tune records, {len(y)} proposals, "
          f"pos {y.mean():.4f}", flush=True)
    return _fit(e159.ALL_COLS), _fit(e159.GEOM2_COLS)


def eval_clipw(dataset, te, pe, models, box_path=None, limit=0):
    from glod.data import RefDataset
    from vlm_boxes import QwenBoxer
    ds = RefDataset(REPRO / "data/refer", e159.DS_NAME[dataset], "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    boxer = QwenBoxer.__new__(QwenBoxer)
    if box_path is None:
        box_path = boxer._path
    recs = [json.loads(l) for l in open(e159.RECORDS[dataset])]
    if limit:
        recs = recs[:limit]
    te.batch([r["parsed"]["sentence"] for r in recs])
    model_c, model_g2 = models
    arms = ("learned_head_clipw", "learned_geom2_clipw")
    acc = {a: dict(iou=0.0, inter=0.0, union=0.0) for a in arms}
    n_used = n_box_none = 0
    t0 = time.time()
    for k, r in enumerate(recs):
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        p = cache_file(fn)
        if not p.is_file():
            continue
        masks, fg = load_npz_masks_fg(p)
        if len(masks) != len(r["gt_iou"]):
            continue
        gt = np.asarray(r["gt_iou"][:len(masks)], np.float64)
        inter = np.asarray(r["inter_gt"][:len(masks)], np.float64)
        union = np.asarray(r["union_gt"][:len(masks)], np.float64)
        bp = box_path(fn, r["parsed"]["sentence"].strip())
        box = json.loads(bp.read_text())["box"] if bp.is_file() else None
        n_box_none += int(box is None)
        our = pe.feats(fn, masks) @ te(r["parsed"]["sentence"])
        feats = e159.arm_c_features(masks, fg, r, box, our)
        picks = dict(
            learned_head_clipw=int(np.argmax(model_c.predict_proba(
                e159.feat_matrix(feats, e159.ALL_COLS))[:, 1])),
            learned_geom2_clipw=int(np.argmax(model_g2.predict_proba(
                e159.feat_matrix(feats, e159.GEOM2_COLS))[:, 1])))
        for a, j in picks.items():
            acc[a]["iou"] += float(gt[j])
            acc[a]["inter"] += float(inter[j])
            acc[a]["union"] += float(union[j])
        n_used += 1
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
    res = {a: dict(n=n_used, mIoU=round(100.0 * acc[a]["iou"] / n_used, 4),
                   oIoU=round(100.0 * acc[a]["inter"] / acc[a]["union"], 4)
                   if acc[a]["union"] else 0.0) for a in arms}
    return res, n_used, n_box_none


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract-saml", "eval-saml", "eval-clipw",
                                    "eval-3b"])
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--split", default="val")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)

    if args.cmd == "extract-saml":
        return extract_saml(args.dataset, args.split, args.shard, args.nshards,
                            args.device)
    if args.cmd == "eval-3b":
        from exp164_boxer3b import path3b
        out = PARTS / f"box3b_{args.dataset}.json"
        return e159.evaluate(args.dataset, out_path=out, box_path=path3b)
    te = e159.TextEncoder(args.device)
    if args.cmd == "eval-saml":
        t0 = time.time()
        models = _fit_saml_heads(args.dataset, te, limit=args.limit)
        print(f"  saml heads fitted ({time.time()-t0:.0f}s)", flush=True)
        res, n, none = eval_saml(args.dataset, te, models, limit=args.limit)
        payload = dict(arm="sam_b_pool", dataset=args.dataset, n=n,
                       box_none_records=none, arms=res,
                       generated="exp165_backbone_swaps.py eval-saml")
    else:  # eval-clipw
        pe = PlainEncoder(args.device)
        t0 = time.time()
        models = _fit_clipw_head(args.dataset, te, pe, limit=args.limit)
        print(f"  clipw heads fitted ({time.time()-t0:.0f}s)", flush=True)
        res, n, none = eval_clipw(args.dataset, te, pe, models, limit=args.limit)
        payload = dict(arm="plain_region_text", dataset=args.dataset, n=n,
                       box_none_records=none, arms=res,
                       generated="exp165_backbone_swaps.py eval-clipw")
    out = PARTS / f"{payload['arm']}_{args.dataset}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    for a, v in payload["arms"].items():
        print(f"  {a:24s} mIoU {v['mIoU']:7.3f}  oIoU {v['oIoU']:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
