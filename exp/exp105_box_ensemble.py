#!/usr/bin/env python
"""EXP-105 — localizer box ensembling over prompt phrasings (box quality axis).

The gIoU bind is now isolated to VLM BOX quality (exp102: required mean
targeted IoU 0.6212, pool picks under GT boxes 0.655, SAM masks under GT
boxes 0.885, but under the actual term boxes 0.477 < pool 0.5335).  The
localizer has only ever been queried with ONE prompt template (limitations
section admits the template was never swept).  Box ensembling across
paraphrases is the standard training-free box-quality lever and is untested.

Phase A: generate boxes for the 262-row sample's 714 (file, term) pairs under
two extra prompts (cached under nounbox_p2 / nounbox_p3 namespaces; the
shipped nounbox cache is untouched).
Phase B: measure per-term box vs best GT object (IoU mean, hit@0.50) for each
prompt, the median/union fusions, and the best-of-3 oracle; then the shipped
termbox-rule construction (learned picker over in-box pool candidates) with
P1 boxes vs fused boxes -> record-level mean targeted IoU m, against the
required 0.6212 and the P1 baseline 0.5335.

    CUDA_VISIBLE_DEVICES=1 python exp/exp105_box_ensemble.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf, load_record  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402
from exp83_two_box_oracle import object_masks  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import CACHE_DIR, IMAGES, MODEL_TAG, PROMPT, parse_box  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2

PROMPT2 = ("Provide the bounding box of the {phrase} in this image. Reply with "
           "ONLY [x0,y0,x1,y1] in pixel coordinates from the top-left corner, "
           "or the single word none if it is not visible.")
PROMPT3 = ("Give the location of the {phrase} as a bounding box. Answer with "
           "only [x0,y0,x1,y1], or none if it is absent.")


def cpath(ns, file_name, term):
    key = hashlib.sha1(
        f"{MODEL_TAG}|{ns}|{file_name}|{term.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def gen_boxes(pairs, per_prompt, device="cuda:0"):
    """pairs: list of (file_name, term). Fills the cache; returns nothing.

    Uses the LlavaBoxer8bit.box() path (NOT a hand-rolled generate): box()
    sets _cur_key so the per-image vision-feature cache engages, and routes
    the cache key through TAG.  A direct proc/model.generate call re-runs the
    vision tower per query (~19 s/query vs ~2 s) and was measured doing so.
    """
    todo = {}
    for name, prompt in per_prompt.items():
        jobs = [(fn, term) for fn, term in pairs if not cpath(name, fn, term).is_file()]
        if jobs:
            todo[name] = jobs
    n_todo = sum(len(v) for v in todo.values())
    print(f"phase A: {n_todo} box queries to run", flush=True)
    if not n_todo:
        return
    from vlm_boxes import LlavaBoxer8bit
    boxer = LlavaBoxer8bit(device=device)
    t0 = time.time()
    done = 0
    for name, jobs in todo.items():
        boxer.TAG = name
        boxer.PROMPT_TEXT = per_prompt[name]
        for fn, term in jobs:
            boxer.box(fn, term)
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                print(f"  {done}/{n_todo} ({el:.0f}s, {60*done/el:.0f}/min)", flush=True)
    print(f"phase A done: {done} in {time.time()-t0:.0f}s", flush=True)


def read_box(ns, fn, term):
    p = cpath(ns, fn, term)
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("box")


def fuse(boxes, mode):
    bs = [b for b in boxes if b is not None]
    if not bs:
        return None
    A = np.asarray(bs, np.float64)
    if mode == "median":
        x0, y0, x1, y1 = np.median(A, 0)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return [float(x0), float(y0), float(x1), float(y1)]
    if mode == "union":
        return [float(A[:, 0].min()), float(A[:, 1].min()),
                float(A[:, 2].max()), float(A[:, 3].max())]
    raise ValueError(mode)


def best_object_iou_box(box, objs):
    """IoU of an (full-res) box vs each GT object's bbox (stride-2)."""
    if box is None:
        return None
    x0, y0, x1, y1 = [v / STRIDE for v in box]
    best = 0.0
    for om in objs:
        ys, xs = np.nonzero(om)
        if not len(ys):
            continue
        ox0, oy0, ox1, oy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        ix = max(0.0, min(x1, ox1) - max(x0, ox0))
        iy = max(0.0, min(y1, oy1) - max(y0, oy0))
        inter = ix * iy
        uni = (x1 - x0) * (y1 - y0) + (ox1 - ox0) * (oy1 - oy0) - inter
        best = max(best, inter / max(1e-9, uni))
    return best


def term_features_box(r, rec, box_of):
    """termbox 8-col features with an injected box source (stride-2 boxes)."""
    masks = r["masks"]
    bs = r["_bs"]
    areas = masks.sum((1, 2)).astype(np.float64)
    terms = rec["terms"]
    out = []
    for ti, term in enumerate(terms):
        box = box_of(term)
        if box is None:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
        x1, y1 = min(masks.shape[2], -(-x1 // STRIDE)), min(masks.shape[1], -(-y1 // STRIDE))
        if x1 <= x0 or y1 <= y0:
            continue
        area_box = float((x1 - x0) * (y1 - y0))
        inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
        bxi = inb / (areas + area_box - inb + 1e-9)
        inbf = inb / np.maximum(areas, 1.0)
        n = len(bs)
        F = np.stack([
            bs, bxi, inbf, areas / masks[0].size,
            np.argsort(np.argsort(bs)) / max(1, n - 1),
            np.full(n, ti, np.float32),
            np.full(n, len(terms), np.float32),
            np.full(n, bxi.max(), np.float32),
        ], 1).astype(np.float32)
        out.append(F)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")

    pairs = [(rec["file_name"], t) for rec in sample for t in rec["terms"]]
    if not args.skip_gen:
        gen_boxes(pairs, {"nounbox_p2": PROMPT2, "nounbox_p3": PROMPT3}, args.device)

    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([x["X"] for x in tr]),
                 np.concatenate([x["y"] for x in tr]))
    for x in tr:
        x["_bs"] = clf_base.predict_proba(x["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    rows = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]
    ref_by_id = {x["ref_id"]: x for x in gd.refs}
    print(f"rows {len(rows)}", flush=True)

    # picker (exp102 construction)
    Xf, yf = [], []
    for rid, rec in recs_by_rid.items():
        r = dedup.get(rid)
        if r is None:
            continue
        iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
        for F in term_features_box(
                r, rec, lambda t, fn=rec["file_name"]: read_box("nounbox", fn, t)):
            Xf.append(F)
            yf.append((iou_gt >= 0.5).astype(np.float32))
    pick_clf = make_clf("logreg", dict(C=g9["C"]))
    pick_clf.fit(np.concatenate(Xf), np.concatenate(yf))

    # ---- phase B ----
    stat = {}
    def bump(k, v, n=1):
        s = stat.setdefault(k, [0.0, 0])
        s[0] += v
        s[1] += n

    for r in rows:
        rec = recs_by_rid[r["rid"]]
        gt = r["gt"]
        objs = object_masks(gd, ref_by_id[r["rid"]]) if r["rid"] in ref_by_id else []
        b1s, b2s, b3s = [], [], []
        for term in rec["terms"]:
            fn = rec["file_name"]
            b1 = read_box("nounbox", fn, term)
            b2 = read_box("nounbox_p2", fn, term)
            b3 = read_box("nounbox_p3", fn, term)
            for tag, b in (("p1", b1), ("p2", b2), ("p3", b3)):
                iou = best_object_iou_box(b, objs)
                if iou is not None:
                    bump(f"iou_{tag}", iou)
                    bump(f"hit_{tag}", 1.0 if iou >= 0.5 else 0.0)
            med = fuse([b1, b2, b3], "median")
            uni = fuse([b1, b2, b3], "union")
            for tag, b in (("med", med), ("uni", uni)):
                iou = best_object_iou_box(b, objs)
                if iou is not None:
                    bump(f"iou_{tag}", iou)
                    bump(f"hit_{tag}", 1.0 if iou >= 0.5 else 0.0)
            # best-of-3 oracle
            cands = [b for b in (b1, b2, b3) if b is not None]
            if cands:
                orc = max(best_object_iou_box(b, objs) for b in cands)
                bump("iou_orc", orc)
                bump("hit_orc", 1.0 if orc >= 0.5 else 0.0)
            b1s.append(b1); b2s.append(b2); b3s.append(b3)

        # rule-level m with P1 boxes vs median boxes vs union boxes
        for tag, boxes in (("p1", b1s), ("med", b2s), ("uni", b3s)):
            boxmap = {}
            for term, b in zip(rec["terms"], boxes):
                boxmap[term] = b
            fs = term_features_box(r, rec, lambda t, bm=boxmap: bm.get(t))
            picks = []
            for F_t in fs:
                pt = pick_clf.predict_proba(F_t)[:, 1]
                order = np.argsort(pt)[::-1]
                picks.append(int(order[0]))
                if g9["topk"] == 2 and len(order) > 1 and pt[order[1]] >= g9["margin"] * pt[order[0]]:
                    picks.append(int(order[1]))
            if not picks:
                continue
            m = r["masks"][picks[0]].copy()
            for i in picks[1:]:
                m |= r["masks"][i]
            I = float(np.logical_and(m, gt).sum())
            U = float(np.logical_or(m, gt).sum())
            bump(f"m_{tag}", I / U if U else 0.0)

    out = dict(
        rows=len(rows),
        per_term_box_iou_mean={k[4:]: stat[k][0] / max(1, stat[k][1])
                               for k in ("iou_p1", "iou_p2", "iou_p3", "iou_med",
                                         "iou_uni", "iou_orc") if k in stat},
        per_term_box_hit50_pct={k[4:]: 100.0 * stat[k][0] / max(1, stat[k][1])
                                for k in ("hit_p1", "hit_p2", "hit_p3", "hit_med",
                                          "hit_uni", "hit_orc") if k in stat},
        rule_mean_targeted_iou={k[2:]: stat[k][0] / max(1, stat[k][1])
                                for k in ("m_p1", "m_med", "m_uni") if k in stat},
        n_terms={k: stat[k][1] for k in stat if k.startswith("iou_")},
        references=dict(p1_baseline_m=0.5335, required_m=0.6212,
                        shipped_box_iou=0.580, shipped_hit50_pct=63.5),
        note=("median/union fusions over prompts 1-3; rule m uses the shipped "
              "picker construction with the fused boxes; held-out sample only"),
        generated="exp105_box_ensemble.py")
    (OUTF / "gref_box_ensemble_sample.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
