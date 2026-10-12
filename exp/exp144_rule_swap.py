#!/usr/bin/env python
"""EXP-144 — same-pool rule-swap control (operator instruction 2026-10-12).

Runs GLOD-RIS's OWN selection rule on OUR exact candidate pools and released
components:

  * pool        : the SAM ViT-H automatic masks already cached by the released
                  HybridGL pipeline (cache/vision npz, same order as the
                  records' per-candidate arrays);
  * features    : the cached G2L foreground/background CLIP ViT-B/16 features
                  from the same npz;
  * rule        : GLOD-RIS Eqs. (1)-(9) as implemented by the reproduction
                  (glod/core.py decouple + gaussian_map + select_mask): joint =
                  max(DOD region-text similarity, 0) x (3x3 Gaussian
                  localization from MLLM guidance) x max(category similarity, 0),
                  argmax over the semantic top-k.  Constants come from the
                  paper (grid 3, gaussian scale 0.7, mu 0.3, tau 0.8, top-10);
                  nothing is fitted here.
  * guidance    : GLM-4.1V-Thinking-Flash category + 3x3 grid per expression,
                  read from the completed reproduction's cache
                  (/home/msj_team/Jacob/sys/glod_ris_repro/cache/glm).

The rule is held FIXED (identical code path to the reproduction) while the
pool/components are ours; the comparison isolates the rule.  Evaluation is a
deterministic re-analysis of the saved records' GT arrays (inter_gt/union_gt/
gt_iou) under the aligned protocol (mIoU = mean per-expression IoU, oIoU =
pooled intersection/union).

    python exp/exp144_rule_swap.py [--datasets refcoco refcocoplus refcocog]
                                   [--limit N] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

RIS = Path(__file__).resolve().parent.parent
REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(RIS / "exp"))
sys.path.insert(0, str(REPRO))

from cache_map import cache_file, _image_path  # noqa: E402

OUTF = RIS / "exp/out_final"
WORK = RIS / "exp/out_rule_swap"
RECORDS = {"refcoco": RIS / "exp/out_extreme/records_refcoco.jsonl",
           "refcocoplus": RIS / "exp/out_extreme/records_refcocoplus.jsonl",
           "refcocog": RIS / "exp/out_extreme/records_refcocog.jsonl"}
REPRO_DS = {"refcoco": "refcoco", "refcocoplus": "refcoco+", "refcocog": "refcocog"}
PUBLISHED = {"refcoco": dict(mIoU=58.42, oIoU=50.85),
             "refcocoplus": dict(mIoU=54.01, oIoU=45.96),
             "refcocog": dict(mIoU=57.85, oIoU=50.03)}


# ---------------------------------------------------------------- GLM index
def build_glm_index():
    """(image_sha256, expression) -> guidance dict or None (parked)."""
    from glod.glm import PROMPT_VERSION
    cache_dir = REPRO / "cache/glm"
    out = WORK / "glm_index.json.gz"
    if out.is_file():
        with gzip.open(out, "rt") as fh:
            return json.load(fh)
    index = {}
    n_files = n_main = 0
    for f in sorted(cache_dir.glob("*.json")):
        n_files += 1
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        spec = d.get("spec") or {}
        if spec.get("prompt_version") != PROMPT_VERSION:
            continue  # repair-path or older-spec entry
        n_main += 1
        key = f"{spec['image_sha256']}|{spec['expression']}"
        g = d.get("guidance")
        # a recovered/parked pair may coexist; prefer real guidance
        if key not in index or (index[key] is None and g is not None):
            index[key] = g
    WORK.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        json.dump(index, fh)
    print(f"GLM index: {n_files} files, {n_main} main-path, {len(index)} keys",
          flush=True)
    return index


# ------------------------------------------------------------- CLIP text side
class TextEncoder:
    """The released text side: modified_CLIP loaded from the pinned checkpoint."""

    def __init__(self, device="cuda:0", batch=256):
        from glod.vision import load_upstream
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.batch = batch
        clip, _, _, _ = load_upstream()
        self.clip = clip
        self.model, _ = clip.load(str(REPRO / "checkpoints/ViT-B-16.pt"),
                                  device=self.device, jit=False)
        self.model.eval().requires_grad_(False)
        self._cache = {}

    def embed(self, texts):
        """Normalised text embeddings for a list of unique strings."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        with self.torch.inference_mode():
            for i in range(0, len(todo), self.batch):
                chunk = todo[i:i + self.batch]
                tokens = self.clip.tokenize(chunk, truncate=True).to(self.device)
                emb = self.model.encode_text(tokens).float()
                emb = emb / emb.norm(dim=-1, keepdim=True)
                for t, e in zip(chunk, emb):
                    self._cache[t] = e.cpu()  # selection runs on CPU features
        return {t: self._cache[t] for t in texts}


# ------------------------------------------------------------------- runner
def load_masks(fn, store, order, keep=8):
    if fn in store:
        return store[fn]
    with np.load(cache_file(fn), allow_pickle=False) as z:
        shape = tuple(int(x) for x in z["mask_shape"])
        mm = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                           ).reshape(shape).astype(bool)
        fg = z["foreground"]
        bg = z["background"]
    store[fn] = (mm, fg, bg)
    order.append(fn)
    while len(order) > keep:
        store.pop(order.pop(0), None)
    return store[fn]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["refcoco", "refcocoplus", "refcocog"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=OUTF / "rule_swap.json")
    args = ap.parse_args()

    import torch
    from glod.core import Config, decouple, gaussian_map, select_mask
    from glod.data import RefDataset
    from glod.io import file_sha256

    cfg = Config()  # reproduction defaults; every constant from the paper
    index = build_glm_index()
    enc = TextEncoder(args.device)

    doc = dict(
        what=("same-pool rule-swap control: GLOD-RIS's own selection rule on our exact "
              "candidate pools and released components (2026-10-12 operator instruction)"),
        rule=dict(
            description=(
                "joint = max(DOD,0) * localization * max(category,0), argmax over the "
                "semantic top-10. DOD = cos(decoupled fused region feature, expression "
                "text) with Eqs. (1)-(7) decoupling (orthogonal residual on the "
                "background, area-ratio weight clamped to [mu,tau], raw fusion basis); "
                "localization = mean over the mask of the 3x3 cell-centred Gaussian map "
                "of the MLLM grid (scale 0.7, normalised to its max); category = "
                "cos(fused feature, MLLM category text). All-zero grid -> uniform map; "
                "parked guidance -> DOD-only path (no localization/category term)."),
            implementation=("glod/core.py (decouple, gaussian_map, select_mask) imported "
                            "verbatim from the reproduction, so the rule is held fixed "
                            "and only the pool/components differ"),
            constants_from_paper=dict(grid_size=cfg.grid_size, gaussian_scale=cfg.gaussian_scale,
                                      mu=cfg.mu, tau=cfg.tau, top_k=cfg.top_k,
                                      fusion_basis=cfg.fusion_basis,
                                      spatial_reduction=cfg.spatial_reduction,
                                      score_transform=cfg.score_transform),
            text_side=("expression text = released ensemble 0.5*sentence + 0.5*noun phrase "
                       "(the same parsed strings the records' released scores use); "
                       "category text = the MLLM category string, encoded with the same "
                       "modified_CLIP ViT-B/16"),
            note=("nothing is fitted: no constant, threshold or template was selected on "
                  "the val splits; the rule is the reproduction's reading of the paper")),
        guidance_source=dict(cache=str(REPRO / "cache/glm"),
                             model="glm-4.1v-thinking-flash",
                             prompt_version="glod-independent-v1",
                             extraction="category + 3x3 grid per expression, "
                                        "from the completed full-val reproduction"),
        pool_source=dict(records={k: str(v.relative_to(RIS)) for k, v in RECORDS.items()},
                         vision_cache=str(REPRO / "cache/vision"),
                         features=("cached G2L foreground/background CLIP ViT-B/16 features "
                                   "per SAM ViT-H mask (same npz the records were scored on)"),
                         validation=("per-record pool size and per-candidate areas are "
                                     "checked against the records before scoring")),
        protocol=("mIoU = mean per-expression IoU of the selected candidate; oIoU = pooled "
                  "intersection/union; both from the records' saved GT arrays "
                  "(deterministic re-analysis, no model re-run)"),
        datasets={})

    for name in args.datasets:
        tag = name
        print(f"=== {tag}", flush=True)
        ds = RefDataset(REPRO / "data/refer", REPRO_DS[tag], "val")
        ref_by_id = {r["ref_id"]: r for r in ds.refs}
        recs = [json.loads(l) for l in open(RECORDS[tag])]
        if args.limit:
            recs = recs[:args.limit]

        # image sha cache (keyed by file_name)
        sha_cache = {}
        for r in recs:
            ref = ref_by_id.get(r["ref_id"])
            if ref is None:
                continue
            fn = ds.images[ref["image_id"]]["file_name"]
            if fn not in sha_cache:
                sha_cache[fn] = file_sha256(_image_path(fn))

        # expression mapping: repro raw sentence by (ref_id, sent_idx position);
        # the guard below verifies the index alignment after case/punctuation
        # normalisation (the records store the released pipeline's lowercased
        # parse, the reproduction cache is keyed by the raw annotation string)
        sent_map, mismatches = {}, 0
        for rid, ref in ref_by_id.items():
            ordered = sorted(ref["sentences"], key=lambda s: int(s["sent_id"]))
            # (raw annotation text for the cache key, cleaned field for the guard)
            sent_map[rid] = [(s["raw"], s.get("sent", "")) for s in ordered]

        def _norm(s):
            import re

            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()

        # batch-encode the text side
        sent_texts, noun_texts = [], []
        for r in recs:
            entries = sent_map.get(r["ref_id"], [])
            if r["sent_idx"] < len(entries) and _norm(entries[r["sent_idx"]][1]) != _norm(r["sent"]):
                mismatches += 1
            sent_texts.append(r["parsed"]["sentence"])
            noun_texts.append(r["parsed"]["noun_phrase"] or r["parsed"]["sentence"])
        sent_emb = enc.embed(sent_texts)
        noun_emb = enc.embed(noun_texts)

        store, order = {}, []
        acc = dict(n=0, mIoU=0.0, inter=0.0, union=0.0, orc=0.0)
        n_guid = n_parked = n_missing = n_zero = 0
        released = dict(mIoU=0.0, inter=0.0, union=0.0)
        cat_texts = []
        for r in recs:
            ref = ref_by_id.get(r["ref_id"])
            if ref is None:
                continue
            fn = ds.images[ref["image_id"]]["file_name"]
            entries = sent_map.get(r["ref_id"], [])
            expression = entries[r["sent_idx"]][0] if r["sent_idx"] < len(entries) else r["sent"]
            g = index.get(f"{sha_cache[fn]}|{expression}")
            masks, fg, bg = load_masks(fn, store, order)
            n = len(masks)
            if n != r["n"]:
                raise ValueError(f"pool size mismatch for {fn}: npz {n} vs record {r['n']}")
            if not np.allclose(masks.mean((1, 2)), np.asarray(r["area"]), atol=2e-3):
                raise ValueError(f"pool order mismatch for {fn}")
            if g is not None:
                cat_texts.append(g["category"])

        cat_emb = enc.embed(list(dict.fromkeys(cat_texts))) if cat_texts else {}

        for r in recs:
            ref = ref_by_id.get(r["ref_id"])
            if ref is None:
                continue
            fn = ds.images[ref["image_id"]]["file_name"]
            entries = sent_map.get(r["ref_id"], [])
            expression = entries[r["sent_idx"]][0] if r["sent_idx"] < len(entries) else r["sent"]
            key = f"{sha_cache[fn]}|{expression}"
            in_cache = key in index
            g = index.get(key)
            masks, fg, bg = load_masks(fn, store, order)
            gt = np.asarray(r["gt_iou"], np.float32)
            inter_arr = np.asarray(r["inter_gt"], np.float64)
            union_arr = np.asarray(r["union_gt"], np.float64)

            t_sent = r["parsed"]["sentence"]
            t_noun = r["parsed"]["noun_phrase"] or t_sent
            text_t = 0.5 * sent_emb[t_sent] + 0.5 * noun_emb[t_noun]
            if g is None:
                if in_cache:
                    n_parked += 1
                else:
                    n_missing += 1
                cat_t, position = None, None
            else:
                n_guid += 1
                cat_t = cat_emb[g["category"]]
                if any(g["grid"]):
                    position = gaussian_map(masks.shape[1], masks.shape[2], g["grid"],
                                            cfg.gaussian_scale, cfg.hard_grid)
                else:
                    n_zero += 1
                    position = np.ones((masks.shape[1], masks.shape[2]), np.float32)

            fused, _, _ = decouple(torch.from_numpy(fg), torch.from_numpy(bg),
                                   masks.mean((1, 2)), cfg)
            cfg_eff = cfg if g is not None else _replace(cfg, use_category=False,
                                                         use_localization=False)
            j, _ = select_mask(fused, masks, text_t, cat_t, position, cfg_eff)

            acc["n"] += 1
            acc["mIoU"] += float(gt[j])
            acc["inter"] += float(inter_arr[j])
            acc["union"] += float(union_arr[j])
            acc["orc"] += float(gt[int(r["orc"])])
            rel = int(r["final"])
            released["mIoU"] += float(gt[rel])
            released["inter"] += float(inter_arr[rel])
            released["union"] += float(union_arr[rel])

        n = acc["n"]
        entry = dict(
            n=n,
            mIoU=round(100.0 * acc["mIoU"] / n, 4),
            oIoU=round(100.0 * acc["inter"] / acc["union"], 4),
            oracle_mIoU=round(100.0 * acc["orc"] / n, 4),
            released_hybridgl_same_records=dict(
                mIoU=round(100.0 * released["mIoU"] / n, 4),
                oIoU=round(100.0 * released["inter"] / released["union"], 4)),
            published_GLODRIS=PUBLISHED[tag],
            guidance=dict(with_guidance=n_guid, parked_dod_only=n_parked,
                          missing=0 if n == n_guid + n_parked else n - n_guid - n_parked,
                          all_zero_grids=n_zero),
            sent_alignment_mismatches=mismatches,
            expression_mapping=("record sent_idx -> repro split's sentences sorted by sent_id; "
                               "raw (uncased) string used for the cache key"))
        # our box rule on the same records, from the shipped main json
        main = json.loads((OUTF / f"main_{tag}.json").read_text())
        entry["our_box_rule_same_records"] = dict(
            mIoU=main["mIoU"], oIoU=(main.get("oIoU") or {}).get("ours"),
            source=f"exp/out_final/main_{tag}.json (shipped)")
        doc["datasets"][tag] = entry
        print(json.dumps(entry, indent=1), flush=True)

    doc["causal_reading"] = (
        "identical rule and identical guidance, only the pool/components differ: if the "
        "rule underperforms on our pool while our box rule (same records, same pool, same "
        "released features) is far ahead, the separation is attributable to the selection "
        "rule plus guidance usage, not to the candidate pool or the frozen components.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
