#!/usr/bin/env python
"""EXP-191 -- released-backbone smoke runs for SigLIP and DINOv2.

These are component runs, not RIS method runs: they verify that the released
checkpoints load and that image/text (SigLIP) or patch (DINOv2) features have
the expected geometry on five RefCOCO val images.  Numeric output is written
to exp/out_final/backbone_smoke_<name>.json.

    python exp/exp191_backbone_smoke.py --backbone siglip
    python exp/exp191_backbone_smoke.py --backbone dinov2
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/out_final"
IMG = Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer/images/train2014")
REC = ROOT / "exp/out_extreme/records_refcoco.jsonl"


def images_and_texts(n=5):
    recs = [json.loads(l) for l in open(REC)][:n]
    ds = json.load(open("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer/refcoco/refs(unc).json")) if False else None
    out = []
    for r in recs:
        s = (r.get("parsed", {}).get("sentence") or r.get("sent") or "").strip()
        out.append((r, s))
    return out


def find_image(sent_rec, idx):
    # Use the image ids recorded in the RefCOCO val manifest via the shared dataset loader.
    import sys
    sys.path.insert(0, "/home/msj_team/Jacob/sys/glod_ris_repro")
    from glod.data import RefDataset
    ds = RefDataset(Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer"), "refcoco", "val")
    by_id = {r["ref_id"]: r for r in ds.refs}
    rows = []
    seen = set()
    for r, s in sent_rec:
        ref = by_id.get(r["ref_id"])
        if ref is None or ref["image_id"] in seen:
            continue
        seen.add(ref["image_id"])
        fn = ds.images[ref["image_id"]]["file_name"]
        rows.append((Image.open(IMG / fn).convert("RGB"), s, r["ref_id"]))
    return rows


def siglip():
    from transformers import AutoModel, AutoProcessor
    name = "google/siglip-so400m-patch14-384"
    t0 = time.time()
    model = AutoModel.from_pretrained(name, device_map=None).eval()
    proc = AutoProcessor.from_pretrained(name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    rows = find_image(images_and_texts(200), 0)[:5]
    imgs = [r[0] for r in rows]; texts = [r[1] for r in rows]
    with torch.no_grad():
        inp = proc(images=imgs, text=texts, padding="max_length", truncation=True, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        out = model(**inp)
        ie = out.image_embeds.float(); te = out.text_embeds.float()
    sims = (ie @ te.T).cpu().numpy()
    diag = np.diag(sims); off = sims[~np.eye(len(rows), dtype=bool)]
    res = {
        "backbone": name, "n": len(rows),
        "embedding_dim": int(ie.shape[-1]),
        "mean_correct_similarity": float(diag.mean()),
        "mean_mismatch_similarity": float(off.mean()),
        "diagonal_minus_mismatch": float(diag.mean() - off.mean()),
        "seconds": round(time.time() - t0, 2),
        "protocol": "five RefCOCO val images each paired with their own sentence; released SigLIP features, diagonal vs off-diagonal cosine similarity; a component smoke run, not a segmentation evaluation",
    }
    (OUT / "backbone_smoke_siglip.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


def dinov2():
    t0 = time.time()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(dev)
    rows = find_image(images_and_texts(200), 0)[:5]
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    feats = []
    with torch.no_grad():
        for img, _, _ in rows:
            arr = np.asarray(img.resize((224, 224))).astype(np.float32) / 255.0
            x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).float()[None].to(dev)
            f = model.forward_features(x)
            feats.append(f["x_norm_patchtokens"].cpu().numpy())
    f = np.concatenate(feats, 0)
    res = {
        "backbone": "dinov2_vits14",
        "n": len(rows),
        "patch_tokens_per_image": int(f.shape[1]),
        "feature_dim": int(f.shape[2]),
        "mean_token_norm": float(np.linalg.norm(f, axis=-1).mean()),
        "token_norm_std": float(np.linalg.norm(f, axis=-1).std()),
        "seconds": round(time.time() - t0, 2),
        "protocol": "five RefCOCO val images resized to 224; released DINOv2 ViT-S/14 patch tokens; geometry/norm check only, a component smoke run, not a segmentation evaluation",
    }
    (OUT / "backbone_smoke_dinov2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["siglip", "dinov2"], required=True)
    a = ap.parse_args()
    {"siglip": siglip, "dinov2": dinov2}[a.backbone]()
