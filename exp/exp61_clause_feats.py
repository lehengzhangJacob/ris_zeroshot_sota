#!/usr/bin/env python
"""EXP-61 — per-noun (clause) grounding features for the gRefCOCO selector.

A two-object reference ("the batter ... and the umpire ...") names its objects;
the fused sentence score blurs them.  For every proposal we add its similarity
to EACH parsed noun separately:

    f1 max_j  sim(noun_j, proposal)
    f2 second_max_j sim
    f3 mean_j sim
    f4 fraction of nouns with sim in the proposal's per-noun top-10
    f5 rank of the proposal under its argmax noun

Cached per vision-cache file (proposal features are already cached, so this is
a text-encoding + matvec pass).  No labels involved.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
OUT = Path("exp/out_final/clause_feats")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, action="append", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    for rel in ("third_party/modified_CLIP", "third_party/segment-anything"):
        p = str(REPRO / "third_party/HybridGL" / rel)
        if p not in sys.path:
            sys.path.insert(0, p)
    import clip
    model, _ = clip.load(str(REPRO / "checkpoints/ViT-B-16.pt"),
                         device=args.device, jit=False)
    model.eval()

    def embed(text):
        with torch.inference_mode():
            tok = clip.tokenize(text, truncate=True).to(args.device)
            v = model.encode_text(tok).float()
        return v / v.norm(dim=1, keepdim=True)

    feats_cache, text_cache = {}, {}
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = 0
    seen = set()
    for rp in args.records:
        for line in open(rp):
            r = json.loads(line)
            cf = r.get("cache_file")
            if not cf or cf in seen:
                continue
            seen.add(cf)
            dst = OUT / (Path(cf).stem + "_clause.npz")
            if dst.is_file():
                continue
            nouns = [n for n in dict.fromkeys(r["parsed"].get("nouns") or [])][:4]
            if not nouns:
                continue
            if cf not in feats_cache:
                with np.load(cf, allow_pickle=False) as z:
                    f = z["foreground"].astype(np.float32)
                feats_cache[cf] = f / np.maximum(
                    np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
            F = feats_cache[cf]                       # (n, d)
            sims = []
            for noun in nouns:
                key = noun.lower()
                if key not in text_cache:
                    text_cache[key] = embed(noun).cpu().numpy().reshape(-1)
                tv = text_cache[key]
                sims.append(F @ tv)                   # (n,)
            S = np.stack(sims, 1)                     # (n, n_nouns)
            order = np.sort(S, axis=1)[:, ::-1]
            f1 = order[:, 0]
            f2 = order[:, 1] if order.shape[1] > 1 else np.zeros(len(S), np.float32)
            f3 = S.mean(1)
            # fraction of nouns whose per-noun top-10 contains this proposal
            top10 = np.zeros_like(S, dtype=bool)
            for j in range(S.shape[1]):
                idx = np.argsort(S[:, j])[::-1][:10]
                top10[idx, j] = True
            f4 = top10.mean(1).astype(np.float32)
            # rank under the argmax noun
            am = np.argmax(S, axis=1)
            ranks = np.zeros(len(S), np.float32)
            for j in range(S.shape[1]):
                idx = np.argsort(S[:, j])[::-1]
                rk = np.empty(len(S), np.float32)
                rk[idx] = np.arange(len(S), dtype=np.float32)
                sel = am == j
                ranks[sel] = rk[sel]
            f5 = 1.0 - ranks / max(1, len(S) - 1)
            np.savez_compressed(dst, X=np.stack([f1, f2, f3, f4, f5], 1))
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done} files ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {n_done} new clause-feature files ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
