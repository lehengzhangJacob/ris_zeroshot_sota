#!/usr/bin/env python
"""EXP-159 — head-rebuild study (operator directive 2026-10-15).

The measured attribution says the shipped RefCOCO-family head is a conjunction:
the released HybridGL scores restrict the pool to their top-k, and our box
prior picks within.  This script re-measures the restriction stage from OUR
OWN mechanisms, as deterministic re-analysis over cached records (cached SAM
pool + cached G2L masked-region features + cached VLM boxes; no vision
re-generation).  Arms (all with the same box-fill pick within their own
candidate set):

  reference    released restriction (top-15 by rank01(sc)+0.4906*rank01(gem))
               + box pick                  -> the shipped head (reproduced)
  attrib rows  released-only top-1; box over the full pool (drop-one deltas)
  A  ours_region_text   top-15 by fg @ t_sentence (full sentence, no released
                        ensemble, no GEM term) + box pick
  A' ours_region_text_key_only   our key top-1, no box
  B  geometric_guards   no text at all: guard = mask center inside the box and
                        mask/box area ratio <= 4 (strict variant [0.1, 2.0]),
                        then box pick within the guard
  C  learned_head_ours  HistGradientBoosting over our own per-proposal terms
                        (our region-text score, box overlap, center-in, area
                        ratio, area, within-row ranks, containment-parent),
                        fitted on the RefCOCO-family held-out TRAIN tune refs
                        only, declared trained

Box-missing records (the VLM answered "none"; ~0.3%): every arm falls back to
the shipped system's own no-box behavior (released top-1), so all arms are
identical there and the deltas isolate the box-restriction stage; the count is
reported per arm.

Writes exp/out_final/head_rebuild_parts/<dataset>.json (per-dataset, per-arm
mIoU + pooled oIoU + n + box stats).

    CUDA_VISIBLE_DEVICES=0 python exp/exp159_head_rebuild.py --dataset refcoco
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# shipped-run convention (see the transfer-wall cycle): 4 BLAS threads keeps
# the tiny per-proposal matvecs from paying a 185-thread OpenBLAS spin storm
# under machine load, and matches the threading the shipped runs used.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_map import build_map, cache_file  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

RECORDS = {
    "refcoco": "exp/out_extreme/records_refcoco.jsonl",
    "refcocoplus": "exp/out_extreme/records_refcocoplus.jsonl",
    "refcocog": "exp/out_extreme/records_refcocog.jsonl",
}
TUNE = {
    "refcoco": "exp/tuning/records_tune_refcoco.jsonl",
    "refcocoplus": "exp/tuning/records_tune_refcoco+.jsonl",
    "refcocog": "exp/tuning/records_tune_refcocog.jsonl",
}
DS_NAME = {"refcoco": "refcoco", "refcocoplus": "refcoco+", "refcocog": "refcocog"}
K_SHORT = 15                 # shipped restriction size (optuna_study.json k_dec)
GEM_MIX = 0.490569059596547   # shipped (optuna_study.json gem_mix)
OUTDIR = Path("exp/out_final/head_rebuild_parts")


def rank01(v):
    v = np.asarray(v, np.float32)
    if len(v) <= 1:
        return np.zeros_like(v)
    return np.argsort(np.argsort(v)).astype(np.float32) / (len(v) - 1)


def box_overlap(masks, box, mode="iou"):
    """exp40 semantics (pixel box -> per-mask overlap)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = masks.shape[1], masks.shape[2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(len(masks), np.float32)
    a_in = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float32)
    box_area = float((y1 - y0) * (x1 - x0))
    tot = masks.sum((1, 2)).astype(np.float32)
    if mode == "cover":
        return a_in / box_area
    if mode == "inside":
        return a_in / np.maximum(tot, 1)
    return a_in / (tot + box_area - a_in + 1e-9)


class TextEncoder:
    """Released CLIP ViT-B/16 text tower (exp32 pattern)."""

    def __init__(self, device="cuda:0"):
        for rel in ("third_party/modified_CLIP", "third_party/segment-anything"):
            p = str(REPRO / "third_party/HybridGL" / rel)
            if p not in sys.path:
                sys.path.insert(0, p)
        import clip
        import torch
        self.clip, self.torch = clip, torch
        self.device = device
        self.model, _ = clip.load(str(REPRO / "checkpoints/ViT-B-16.pt"),
                                  device=device, jit=False)
        self.model.eval()
        self.cache = {}

    def batch(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            with self.torch.inference_mode():
                tok = self.clip.tokenize(chunk, truncate=True).to(self.device)
                v = self.model.encode_text(tok).float()
                v = v / v.norm(dim=1, keepdim=True)
            for t, row in zip(chunk, v.cpu().numpy()):
                self.cache[t] = row
        return self.cache

    def __call__(self, text):
        return self.cache[text]


def load_image(fn, store, order):
    if fn in store:
        return store[fn]
    p = cache_file(fn)
    if not p.is_file():
        store[fn] = None
        return None
    with np.load(p, allow_pickle=False) as z:
        shape = tuple(int(x) for x in z["mask_shape"])
        masks = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                              ).reshape(shape).astype(bool)
        fg = z["foreground"].astype(np.float32)
    fg /= np.linalg.norm(fg, axis=1, keepdims=True) + 1e-9
    store[fn] = (masks, fg)
    order.append(fn)
    while len(order) > 12:
        store.pop(order.pop(0), None)
    return store[fn]


def arm_c_features(masks, fg, r, box, our, loc=None):
    """Our own per-proposal terms (no released scores)."""
    n = len(masks)
    H, W = masks.shape[1], masks.shape[2]
    npix = float(H * W)
    area = np.asarray(r["area"][:n], np.float32)          # normalized
    cx = np.asarray(r["cx"][:n], np.float32)
    cy = np.asarray(r["cy"][:n], np.float32)
    ps = np.asarray(r["ps_idx"][:n], np.int64)
    parent = (ps != np.arange(n)).astype(np.float32)
    feats = dict(our=our, ps_parent=parent, log_area=np.log1p(area * npix))
    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        if loc is None:
            loc = box_overlap(masks, box, "iou")
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        center_in = ((cx * W >= x0) & (cx * W <= x1) &
                     (cy * H >= y0) & (cy * H <= y1)).astype(np.float32)
        box_area = float(max(1, (x1 - x0) * (y1 - y0)))
        ratio = area * npix / box_area
        bcx = 0.5 * (x0 + x1) / W
        bcy = 0.5 * (y0 + y1) / H
        dist = np.sqrt((cx - bcx) ** 2 + (cy - bcy) ** 2).astype(np.float32)
        feats.update(loc=loc, center_in=center_in,
                     log_ratio=np.log1p(ratio), mask_box_ratio=ratio,
                     loc_cover=box_overlap(masks, box, "cover"),
                     dist_center=dist,
                     log_box_area=np.full(n, np.log1p(box_area), np.float32))
    else:
        feats.update(loc=np.zeros(n, np.float32), center_in=np.zeros(n, np.float32),
                     log_ratio=np.zeros(n, np.float32), mask_box_ratio=np.zeros(n, np.float32),
                     loc_cover=np.zeros(n, np.float32), dist_center=np.zeros(n, np.float32),
                     log_box_area=np.zeros(n, np.float32))
    feats["our_rank"] = rank01(feats["our"])
    feats["loc_rank"] = rank01(feats["loc"])
    return feats


ALL_COLS = ["our", "loc", "center_in", "log_ratio", "log_area",
            "our_rank", "loc_rank", "ps_parent"]
GEOM_COLS = ["loc", "center_in", "log_ratio", "log_area", "loc_rank", "ps_parent"]
GEOM2_COLS = GEOM_COLS + ["loc_cover", "dist_center", "log_box_area"]


def feat_matrix(feats, cols=ALL_COLS):
    return np.stack([feats[c] for c in cols], axis=1).astype(np.float32)


ALL_ARMS = ("reference", "attrib_released_only", "attrib_box_full",
            "attrib_box_full_cover", "attrib_box_full_inside",
            "ours_region_text", "ours_key_only",
            "geom_guard", "geom_guard_strict", "learned_head",
            "learned_geom", "learned_geom2")


def picks_for_record(masks, fg, r, box, t_sent, model_c, model_g, model_g2):
    """All arm picks for one record.  Returns dict arm -> pick."""
    n = len(fg)
    sc = np.asarray(r["sc"][:n], np.float32)
    gem = np.asarray(r["score_gem"][:n], np.float32)
    base = rank01(sc) + GEM_MIX * rank01(gem)
    our = fg @ t_sent
    out = {}
    k = min(K_SHORT, n)
    idx = np.argsort(base)[::-1][:k] if k < n else np.arange(n)
    idxA = np.argsort(our)[::-1][:k] if k < n else np.arange(n)
    if box is None:
        # No box (the VLM answered "none"): each arm takes its own box-free
        # analogue so no ours-only arm borrows a released score here.  The
        # reference keeps the shipped system's own no-box behavior (released
        # top-1); the attribution rows have no box-free analogue and share it;
        # the learned heads predict from their box-free features (the box
        # terms are zeroed, exactly as in the tune fit).
        for arm in ("reference", "attrib_released_only", "attrib_box_full",
                    "attrib_box_full_cover", "attrib_box_full_inside"):
            out[arm] = int(idx[0])
        for arm in ("ours_region_text", "ours_key_only", "geom_guard",
                    "geom_guard_strict"):
            out[arm] = int(idxA[0])
        if model_c is not None:
            feats0 = arm_c_features(masks, fg, r, None, our)
            out["learned_head"] = int(np.argmax(
                model_c.predict_proba(feat_matrix(feats0, ALL_COLS))[:, 1]))
        if model_g is not None:
            feats0 = arm_c_features(masks, fg, r, None, our)
            out["learned_geom"] = int(np.argmax(
                model_g.predict_proba(feat_matrix(feats0, GEOM_COLS))[:, 1]))
        if model_g2 is not None:
            feats0 = arm_c_features(masks, fg, r, None, our)
            out["learned_geom2"] = int(np.argmax(
                model_g2.predict_proba(feat_matrix(feats0, GEOM2_COLS))[:, 1]))
        return out, False
    loc = box_overlap(masks, box, "iou")
    out["reference"] = int(idx[int(np.argmax(loc[idx]))])
    out["attrib_released_only"] = int(idx[0])
    out["attrib_box_full"] = int(np.argmax(loc))
    out["attrib_box_full_cover"] = int(np.argmax(box_overlap(masks, box, "cover")))
    out["attrib_box_full_inside"] = int(np.argmax(box_overlap(masks, box, "inside")))
    out["ours_region_text"] = int(idxA[int(np.argmax(loc[idxA]))])
    out["ours_key_only"] = int(idxA[0])
    feats = arm_c_features(masks, fg, r, box, our, loc=loc)
    H, W = masks.shape[1], masks.shape[2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    center_in = feats["center_in"] > 0.5
    ratio = feats["mask_box_ratio"]
    g1 = center_in & (ratio <= 4.0)
    g2 = center_in & (ratio >= 0.1) & (ratio <= 2.0)
    for name, g in (("geom_guard", g1), ("geom_guard_strict", g2)):
        cand = np.flatnonzero(g)
        if len(cand) == 0:
            cand = np.arange(n)
        out[name] = int(cand[int(np.argmax(loc[cand]))])
    if model_c is not None:
        out["learned_head"] = int(np.argmax(
            model_c.predict_proba(feat_matrix(feats, ALL_COLS))[:, 1]))
    if model_g is not None:
        out["learned_geom"] = int(np.argmax(
            model_g.predict_proba(feat_matrix(feats, GEOM_COLS))[:, 1]))
    if model_g2 is not None:
        out["learned_geom2"] = int(np.argmax(
            model_g2.predict_proba(feat_matrix(feats, GEOM2_COLS))[:, 1]))
    return out, True


def iterate_records(dataset, split, ref_by_id, ds, store, order, te, model_c,
                    limit=0, records_path=None):
    path = Path(records_path) if records_path else (
        RECORDS[dataset] if split == "val" else TUNE[dataset])
    recs = [json.loads(l) for l in open(path)]
    if limit:
        recs = recs[:limit]
    te.batch([r["parsed"]["sentence"] for r in recs])
    for r in recs:
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        im = load_image(fn, store, order)
        if im is None:
            continue
        masks, fg = im
        n = len(fg)
        if n != len(r["gt_iou"]):
            continue
        yield r, fn, masks, fg


def fit_learned_head(dataset, te):
    """Fit the learned head on the RefCOCO-family held-out TRAIN tune refs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from glod.data import RefDataset
    from vlm_boxes import QwenBoxer
    ds = RefDataset(REPRO / "data/refer", DS_NAME[dataset], "train")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    store, order = {}, []
    boxer = QwenBoxer.__new__(QwenBoxer)
    Fs, ys = [], []          # per-record feature dicts + labels
    n_rec = 0
    for r, fn, masks, fg in iterate_records(dataset, "tune", ref_by_id, ds,
                                            store, order, te, None):
        bp = boxer._path(fn, r["parsed"]["sentence"].strip())
        box = json.loads(bp.read_text())["box"] if bp.is_file() else None
        our = fg @ te(r["parsed"]["sentence"])
        feats = arm_c_features(masks, fg, r, box, our)
        gt = np.asarray(r["gt_iou"][:len(fg)], np.float32)
        Fs.append(feats)
        ys.append((gt >= 0.5).astype(np.int8))
        n_rec += 1
    y = np.concatenate(ys)

    def _fit(cols):
        X = np.concatenate([feat_matrix(F, cols) for F in Fs])
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                             learning_rate=0.08, random_state=0)
        clf.fit(X, y)
        return clf

    model_c = _fit(ALL_COLS)
    model_g = _fit(GEOM_COLS)
    model_g2 = _fit(GEOM2_COLS)
    print(f"  [fit] {dataset}: {n_rec} tune records, {len(y)} proposals, "
          f"positive rate {y.mean():.4f}", flush=True)
    return model_c, model_g, model_g2


def evaluate(dataset, split="val", limit=0, records_path=None, out_path=None,
             te=None, models=None):
    from glod.data import RefDataset
    from vlm_boxes import QwenBoxer
    ds = RefDataset(REPRO / "data/refer", DS_NAME[dataset], split)
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    build_map([ds.images[r["image_id"]]["file_name"] for r in ds.refs])
    if te is None:
        te = TextEncoder()
    if models is None:
        print(f"[{dataset}] fitting learned heads on tune refs", flush=True)
        t0 = time.time()
        models = fit_learned_head(dataset, te)
        print(f"[{dataset}] heads fitted ({time.time()-t0:.0f}s); val pass", flush=True)
    model_c, model_g, model_g2 = models

    boxer = QwenBoxer.__new__(QwenBoxer)
    store, order = {}, []
    arms = ALL_ARMS
    acc = {a: dict(iou=0.0, inter=0.0, union=0.0) for a in arms}
    n_used = n_box = n_box_none = 0
    t1 = time.time()
    for i, (r, fn, masks, fg) in enumerate(iterate_records(
            dataset, split, ref_by_id, ds, store, order, te, model_c, limit,
            records_path=records_path)):
        bp = boxer._path(fn, r["parsed"]["sentence"].strip())
        box = json.loads(bp.read_text())["box"] if bp.is_file() else None
        n_box += int(bp.is_file())
        n_box_none += int(box is None)
        picks, used_box = picks_for_record(masks, fg, r, box,
                                           te(r["parsed"]["sentence"]), model_c,
                                           model_g, model_g2)
        gt = np.asarray(r["gt_iou"][:len(fg)], np.float64)
        inter = np.asarray(r["inter_gt"][:len(fg)], np.float64)
        union = np.asarray(r["union_gt"][:len(fg)], np.float64)
        for a, j in picks.items():
            acc[a]["iou"] += float(gt[j])
            acc[a]["inter"] += float(inter[j])
            acc[a]["union"] += float(union[j])
        n_used += 1
        if (i + 1) % 500 == 0:
            (OUTDIR / f"{dataset}.progress").write_text(f"{i+1}")
        if (i + 1) % 2000 == 0:
            print(f"  {i+1} records ({time.time()-t1:.0f}s)", flush=True)
    res = {}
    for a in arms:
        v = acc[a]
        res[a] = dict(n=n_used,
                      mIoU=round(100.0 * v["iou"] / n_used, 4),
                      oIoU=round(100.0 * v["inter"] / v["union"], 4) if v["union"] else 0.0)
    payload = dict(
        dataset=dataset, split=split, records=str(records_path) if records_path else None,
        n=n_used,
        box_queried_pct=round(100.0 * n_box / max(1, n_used), 3),
        box_none_records=n_box_none,
        arms=res,
        generated="exp159_head_rebuild.py",
        notes=("deterministic re-analysis over cached records (SAM ViT-H pool + "
               "G2L masked-region features + cached VLM boxes); reference arm "
               "reproduces the shipped head (released top-15 restriction by "
               "rank01(sc)+0.4906*rank01(gem), box-fill pick); ours_* arms use "
               "top-15 by fg @ t_sentence (full sentence, no ensemble, no GEM); "
               "geom_* arms use no text; learned_head (all our terms incl. the "
               "region-text score) and learned_geom (geometric terms only, no "
               "text, no features) are fitted on the RefCOCO-family held-out "
               "TRAIN tune refs only; attrib_box_full_cover/inside are the "
               "full-pool box pick under the other overlap modes; box-missing "
               "records use the shipped no-box fallback (released top-1) in "
               "every arm"),
    )
    out = Path(out_path) if out_path else OUTDIR / f"{dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[{dataset}] wrote {out}")
    for a in arms:
        print(f"  {a:24s} mIoU {res[a]['mIoU']:7.3f}  oIoU {res[a]['oIoU']:7.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcoco", choices=list(RECORDS))
    ap.add_argument("--split", default="val")
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return evaluate(args.dataset, split=args.split, limit=args.limit,
                    records_path=args.records, out_path=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
