"""Set-valued selection for gRefCOCO: cover the VLM box with a union of masks.

gRefCOCO targeted samples are 2-object unions in 99.5% of cases, so a single
proposal caps at ~0.58 cumulative IoU even when chosen by an oracle.  The
selector therefore returns a SET:

  1. candidates come from the same rank-fused shortlist the RefCOCO-family
     pipeline uses (CLIP score + GEM coherence);
  2. the set is grown greedily to cover the VLM box — each step adds the mask
     whose newly covered box area, minus lam * newly added outside-box area,
     is largest;
  3. growth stops when the best gain is no longer positive (or at max_k).

Abstention is a separate branch: when the VLM answers `none`, the prediction
is empty and the sample counts as a correct no-target decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def rank01(v):
    if len(v) <= 1:
        return np.zeros_like(np.asarray(v, np.float32))
    v = np.asarray(v, np.float32)
    return np.argsort(np.argsort(v)).astype(np.float32) / (len(v) - 1)


def load_masks(cache_file):
    with np.load(cache_file, allow_pickle=False) as d:
        shape = tuple(int(x) for x in d["mask_shape"])
        bits = d["mask_bits"]
    return np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool)


def candidates(rec, params):
    """Rank-fused shortlist of proposal indices (the RefCOCO-family base rule)."""
    n = len(rec["gt_iou"])
    src = params.get("dec_source", "sc_x_gem")
    if src == "sc_x_gem":
        base = rank01(rec["sc"][:n]) + params.get("gem_mix", 1.0) * rank01(rec["score_gem"][:n])
    else:
        key = {"sc": "sc", "sc_noun": "sc_noun", "sc_strip": "sc_strip",
               "gem": "score_gem"}[src]
        base = np.asarray(rec[key][:n], np.float32)
    k = min(int(params.get("k_rank", 15)), n)
    return np.argsort(base)[::-1][:k], base


def _clip_box(x0, y0, x1, y1, H, W):
    return (max(0, min(W, int(round(x0)))), max(0, min(H, int(round(y0)))),
            max(0, min(W, int(round(x1)))), max(0, min(H, int(round(y1)))))


def select_set(rec, masks, box, params):
    """Greedy box-cover selection.  Returns (idxs, box_used)."""
    if box is None:
        return [], False
    H, W = masks.shape[1], masks.shape[2]
    x0, y0, x1, y1 = _clip_box(*box, H, W)
    if x1 <= x0 or y1 <= y0:
        return [], False
    cand, base = candidates(rec, params)
    sub = masks[cand][:, y0:y1, x0:x1]
    areas_in = sub.sum((1, 2)).astype(np.float64)
    areas_all = masks[cand].sum((1, 2)).astype(np.float64)
    areas_out = areas_all - areas_in
    keep = areas_all > 0
    if not keep.any():
        return [], False
    lam = float(params.get("lam", 0.5))
    max_k = int(params.get("max_k", 4))
    min_gain = float(params.get("min_gain", 0.0))
    cover = np.zeros((y1 - y0, x1 - x0), bool)
    chosen, cov_all = [], np.zeros((H, W), bool)
    for _ in range(max_k):
        gains = np.full(len(cand), -np.inf)
        for j in range(len(cand)):
            if not keep[j] or j in chosen:
                continue
            newly_in = np.logical_and(sub[j], ~cover).sum()
            newly_out = np.logical_and(masks[cand[j]], ~cov_all).sum() - newly_in
            gains[j] = newly_in - lam * newly_out
        j = int(np.argmax(gains))
        if not np.isfinite(gains[j]) or gains[j] <= min_gain:
            break
        chosen.append(j)
        cover |= sub[j]
        cov_all |= masks[cand[j]]
    return [int(cand[j]) for j in chosen], True


def iou_iu(masks, idxs, gt):
    if not idxs:
        pred = np.zeros(gt.shape, bool)
    else:
        pred = masks[idxs].any(0)
    I = int(np.logical_and(pred, gt).sum())
    U = int(np.logical_or(pred, gt).sum())
    return I, U, (I / U if U else 0.0)


def official_metrics(rows):
    """rows: dicts with gt_nt, abstained, I, U, iou (targeted) — official formulas."""
    total = len(rows)
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0
    for s in rows:
        if s["gt_nt"]:
            if s["abstained"]:
                tp += 1
                acc_iou += 1.0
            else:
                fn += 1
                acc_U += s["U"]      # |pred| with an empty GT
        else:
            if s["abstained"]:
                fp += 1
                acc_U += s["gt_area"]   # empty pred vs GT: union = |GT|
            else:
                tn += 1
                acc_iou += s["iou"]
                acc_I += s["I"]
                acc_U += s["U"]
    return dict(n=total,
                cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                gIoU=100.0 * acc_iou / total if total else 0.0,
                N_acc=100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
                T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0,
                TP=tp, FN=fn, TN=tn, FP=fp)
