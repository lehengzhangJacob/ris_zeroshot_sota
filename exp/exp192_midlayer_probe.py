#!/usr/bin/env python
"""EXP-192 -- mid-layer vs final-layer CLIP patch features for proposal selection."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp")); sys.path.insert(0, "/home/msj_team/Jacob/sys/glod_ris_repro")
from cache_map import cache_file  # noqa: E402
IMG = Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer/images/train2014")
REC = ROOT / "exp/out_extreme/records_refcoco.jsonl"
CK = "/home/msj_team/Jacob/sys/glod_ris_repro/checkpoints/clip/ViT-B-32.pt"
N = 20

def main():
    import open_clip
    from glod.data import RefDataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=CK, device=dev)
    model.eval(); tok = open_clip.get_tokenizer("ViT-B-32")
    recs = [json.loads(l) for l in open(REC)][:N]
    ds = RefDataset(Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer"), "refcoco", "val")
    by = {r["ref_id"]: r for r in ds.refs}
    rows = []
    for r in recs:
        ref = by.get(r["ref_id"])
        if ref is None: continue
        fn = ds.images[ref["image_id"]]["file_name"]
        with np.load(cache_file(fn)) as z:
            shape = tuple(int(x) for x in z["mask_shape"])
            masks = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)
        sent = (r.get("parsed", {}).get("sentence") or r.get("sent") or "").strip()
        rows.append((fn, sent, masks, np.asarray(r["gt_iou"], np.float32)))
    feats = {}
    def hook(name):
        def f(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            feats[name] = x.detach()
        return f
    blocks = model.visual.transformer.resblocks
    h6 = blocks[6].register_forward_hook(hook("mid")); h11 = blocks[-1].register_forward_hook(hook("final"))
    def pool(t, m):
        # t: [1, N+1, D] (CLS + patches)
        t = t[:, 0]
        if t.shape[0] == 50: t = t[1:]
        t = (t @ model.visual.proj).detach()
        t = t.reshape(7, 7, -1).cpu().numpy()
        mm = cv2.resize(m.astype(np.uint8), (7, 7), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mm.sum() == 0: return None
        v = t[mm].mean(0); return v / (np.linalg.norm(v) + 1e-8)
    def run(which):
        vals = []
        for fn, sent, masks, gt in rows:
            img = preprocess(Image.open(IMG / fn).convert("RGB")).unsqueeze(0).to(dev)
            with torch.no_grad():
                model.encode_image(img)
                tf = model.encode_text(tok([sent]).to(dev))
                tf = (tf / tf.norm(dim=-1, keepdim=True))[0].cpu().numpy()
            p = [pool(feats[which], m) for m in masks]
            scores = [(-1e9 if v is None else float(v @ tf)) for v in p]
            pick = int(np.argmax(scores)); vals.append(float(gt[pick]) if pick < len(gt) else 0.0)
        return 100.0 * float(np.mean(vals))
    out = {"n": len(rows), "protocol": "20 RefCOCO val expressions; cached SAM proposals; open_clip ViT-B/32 patch-token pooling; text = full sentence; argmax cosine; selected IoU averaged",
           "mid_layer_6_mIoU": run("mid"), "final_layer_11_mIoU": run("final"),
           "source": "blindposition2025 mid-layer idea; component probe, not a RIS method reproduction"}
    (ROOT / "exp/out_final/midlayer_probe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))

if __name__ == "__main__": main()
