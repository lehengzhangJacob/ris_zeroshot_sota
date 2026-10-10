#!/usr/bin/env python
"""EXP-94 — where does the termbox loss live: boxes or picker?

The shipped per-term rule realizes cumulative targeted r 0.5030 on val.  On the
val-matched held-out population, GT per-object boxes + bbox-IoU pick reach
0.7151 (exp83).  Decompose the loss:

  oracle_in_llava_box : per term box (LLaVA), the best pool mask by GT IoU ->
                        union of picks.  Upper bound for ANY picker given these
                        boxes.  If ~0.7, the picker is the gap; if ~0.55, the
                        boxes are.
  oracle_in_gt_box    : same with GT per-object boxes (the construction ceiling).
  shipped_rule        : the frozen shipped termbox rule on the same rows.
  oracle_top2_llava   : allows the best TWO masks per LLaVA box (set structure
                        headroom within boxes).

Held-out rows only; no new extraction; no val.

    python exp/exp94_termbox_ceiling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp83_two_box_oracle import object_masks  # noqa: E402
from exp84_nounbox_sample import npath  # noqa: E402
from exp89_termbox_full_val import term_features  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def main() -> int:
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")
    ref_by_id = {x["ref_id"]: x for x in gd.refs}

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}
    rows = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    print(f"rows {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    acc = dict(li=0.0, lu=0.0, gi=0.0, gu=0.0, si=0.0, su=0.0, t2i=0.0, t2u=0.0)
    for r in rows:
        rec = recs_by_rid[r["rid"]]
        masks, gt = r["masks"], r["gt"]
        iou_gt = (np.logical_and(masks, gt).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(masks, gt).sum((1, 2)), 1.0))
        pick_l, pick_g, pick_t2 = [], [], []
        Fs = term_features(r, rec)
        for ti, term in enumerate(rec["terms"]):
            p = npath(rec["file_name"], term)
            if not p.is_file():
                continue
            box = json.loads(p.read_text()).get("box")
            if box is None:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
            x1, y1 = min(masks.shape[2], -(-x1 // STRIDE)), min(masks.shape[1], -(-y1 // STRIDE))
            if x1 <= x0 or y1 <= y0:
                continue
            cand = masks[:, y0:y1, x0:x1].sum((1, 2))
            inside = np.flatnonzero(cand > 0)
            if not len(inside):
                continue
            order = inside[np.argsort(iou_gt[inside])[::-1]]
            pick_l.append(int(order[0]))
            if len(order) > 1:
                pick_t2.append((int(order[0]), int(order[1])))
        ref = ref_by_id.get(r["rid"])
        if ref is not None:
            for om in object_masks(gd, ref):
                ys, xs = np.nonzero(om)
                if not len(ys):
                    continue
                bb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                cand = masks[:, bb[1]:bb[3], bb[0]:bb[2]].sum((1, 2))
                inside = np.flatnonzero(cand > 0)
                if len(inside):
                    order = inside[np.argsort(iou_gt[inside])[::-1]]
                    pick_g.append(int(order[0]))

        def cum(picks, key):
            if not picks:
                return
            m = masks[picks[0]].copy()
            for p_ in picks[1:]:
                m = np.logical_or(m, masks[p_])
            acc[key + "i"] += float(np.logical_and(m, gt).sum())
            acc[key + "u"] += float(np.logical_or(m, gt).sum())

        cum(pick_l, "l")
        cum(pick_g, "g")
        if pick_t2:
            picked = []
            for a, b in pick_t2:
                picked.extend([a, b])
            cum(list(dict.fromkeys(picked)), "t2")
    # shipped rule needs the fitted pair fallback + picker; reuse exp87-style fits
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km]); ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    Xf, yf = [], []
    for rid, rec in recs_by_rid.items():
        r = dedup.get(rid)
        if r is None:
            continue
        iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
        for F in term_features(r, rec):
            Xf.append(F); yf.append((iou_gt >= 0.5).astype(np.float32))
    pick_clf = make_clf("logreg", dict(C=g9["C"]))
    pick_clf.fit(np.concatenate(Xf), np.concatenate(yf))
    for r in rows:
        rec = recs_by_rid[r["rid"]]
        Fs = term_features(r, rec)
        picks = []
        for F_t in Fs:
            pt = pick_clf.predict_proba(F_t)[:, 1]
            order = np.argsort(pt)[::-1]
            picks.append(int(order[0]))
            if g9["topk"] == 2 and len(order) > 1 and pt[order[1]] >= g9["margin"] * pt[order[0]]:
                picks.append(int(order[1]))
        picks = list(dict.fromkeys(picks))
        if picks:
            m = r["masks"][picks[0]].copy()
            for p_ in picks[1:]:
                m = np.logical_or(m, r["masks"][p_])
            acc["si"] += float(np.logical_and(m, r["gt"]).sum())
            acc["su"] += float(np.logical_or(m, r["gt"]).sum())

    out = {
        "rows": len(rows),
        "oracle_in_llava_box_r": acc["li"] / max(1.0, acc["lu"]),
        "oracle_in_gt_box_r": acc["gi"] / max(1.0, acc["gu"]),
        "oracle_top2_llava_r": acc["t2i"] / max(1.0, acc["t2u"]),
        "shipped_rule_r_heldout_population": acc["si"] / max(1.0, acc["su"]),
        "interpretation": ("oracle_in_llava_box is the picker's ceiling with THESE boxes; "
                           "the gap to it is pick-quality headroom; the gap from it to "
                           "oracle_in_gt_box is box-quality headroom"),
        "generated": "exp94_termbox_ceiling.py",
    }
    (OUTF / "gref_termbox_ceiling.json").write_text(json.dumps(out, indent=2))
    for k in ("oracle_in_llava_box_r", "oracle_in_gt_box_r", "oracle_top2_llava_r",
              "shipped_rule_r_heldout_population"):
        print(f"{k}: {out[k]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
