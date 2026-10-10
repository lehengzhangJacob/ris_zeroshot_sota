"""SigLIP-so400m region features for the cached SAM pool (per image, cached).

Mirrors glod.vision's G2L idea with a stronger frozen encoder:
  local  = image with non-mask pixels filled by the CLIP mean colour
  global = image with the non-mask pixels Gaussian-blurred
Both are encoded at 384^2; the region score uses their mean (normalised).
Cache keyed by image sha256; shardable.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
IMAGES = REPRO / "data/refer/images/train2014"
CACHE = Path("exp/out_siglip")
MODEL_ID = "google/siglip-so400m-patch14-384"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32) * 255.0


class SiglipRegions:
    def __init__(self, device="cuda:0", batch=64):
        import torch
        from transformers import AutoProcessor, AutoModel
        self.torch = torch
        self.device = device
        self.batch = batch
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
        CACHE.mkdir(parents=True, exist_ok=True)

    def _path(self, image_hash):
        return CACHE / f"{image_hash}.npz"

    def text(self, texts):
        t = self.proc(text=list(texts), padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            f = self.model.get_text_features(**t)
        return f.float().cpu().numpy()

    def extract(self, file_name, masks, image_hash):
        p = self._path(image_hash)
        if p.is_file():
            with np.load(p, allow_pickle=False) as d:
                return d["local"], d["global_"]
        import cv2
        from PIL import Image
        rgb = np.asarray(Image.open(IMAGES / file_name).convert("RGB")).copy()
        blurred = cv2.GaussianBlur(rgb.copy(), (15, 15), 0)
        locals_, globals_ = [], []
        for m in masks:
            mm = m[..., None]
            locals_.append((np.where(mm, rgb, MEAN)).astype(np.uint8))
            globals_.append(np.where(mm, rgb, blurred).astype(np.uint8))
        out = {}
        for tag, imgs in (("local", locals_), ("global", globals_)):
            feats = []
            for i in range(0, len(imgs), self.batch):
                chunk = [Image.fromarray(x) for x in imgs[i:i + self.batch]]
                inp = self.proc(images=chunk, return_tensors="pt").to(self.device)
                inp["pixel_values"] = inp["pixel_values"].half()
                with self.torch.inference_mode():
                    f = self.model.get_image_features(**inp)
                feats.append(f.float().cpu().numpy())
            out[tag] = np.concatenate(feats).astype(np.float16)
        np.savez_compressed(p, local=out["local"], global_=out["global"])
        return out["local"], out["global"]

    def load(self, p):
        with np.load(p, allow_pickle=False) as d:
            return d["local"], d["global_"]


def main() -> int:
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(REPRO))
    from glod.data import RefDataset
    from glod.io import digest, file_sha256
    from cache_map import cache_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["refcoco", "refcoco+", "refcocog"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # unique images across the chosen datasets
    images = {}
    for name in args.datasets:
        ds = RefDataset(REPRO / "data/refer", name, "val")
        for r in ds.refs:
            fn = ds.images[r["image_id"]]["file_name"]
            images.setdefault(fn, None)
    fns = sorted(images)
    fns = [fn for i, fn in enumerate(fns) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(fns)} images", flush=True)

    enc = SiglipRegions(device=args.device)
    t0 = time.time()
    skipped = 0
    for i, fn in enumerate(fns):
        h = file_sha256(IMAGES / fn)
        if enc._path(h).is_file():
            skipped += 1
            continue
        with np.load(cache_file(fn), allow_pickle=False) as d:
            shape = tuple(int(x) for x in d["mask_shape"])
            masks = np.unpackbits(d["mask_bits"], count=int(np.prod(shape))
                                  ).reshape(shape).astype(bool)
        enc.extract(fn, masks, h)
        if (i + 1) % 100 == 0:
            done = i + 1 - skipped
            print(f"  {i+1}/{len(fns)} (cached {skipped}) {time.time()-t0:.0f}s "
                  f"{(time.time()-t0)/max(1,done):.1f}s/img", flush=True)
    print(f"done: {len(fns)} images, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
