#!/usr/bin/env python
"""EXP-117 — model change 14: the 300-trial study over the remaining joints.

The shipped rule (change 13) fixed one point in a larger space: score source
(logreg base score), pick box (LLaVA), SAM source (GDINO det0), no 1-term
handling.  exp111 showed the 1-term subpopulation (12.5% of val targeted)
responds to the 2nd detection; exp115 says the box/pick axis carries the
loss; the earlier studies never searched score-source x pick-box x
union-of-boxes x SAM-source JOINTLY, and never included 1-term rows in the
study population at all.

Population: the 262-row val-matched sample (>=2-term) + the 47 held-out
1-term multi-target rows, mixed at val prevalence (0.875/0.125).  Objective:
0.875*m_ge2 + 0.125*m_1term, where each m is the row-mean targeted IoU of the
config's prediction on that group.  Val is untouched; a val pass follows only
if the winner beats the shipped-equivalent mixture by > 0.010 and clears the
required m 0.6212 on the mixture.

    CUDA_VISIBLE_DEVICES=1 python exp/exp117_study_v2.py --trials 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402
from exp103_sam_boxp import downsample  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
from vlm_boxes import IMAGES  # noqa: E402

OUTF = Path("exp/out_final")
MIX = 0.125  # val targeted share of the 1-term subpopulation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/optuna_study_gref14.json"))
    args = ap.parse_args()

    t0 = time.time()
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([x["X"] for x in tr]),
                 np.concatenate([x["y"] for x in tr]))
    for x in tr:
        x["_bs"] = clf_base.predict_proba(x["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    rows2 = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]

    t1 = {}
    for line in open('exp/out_extreme/records_grefcoco_train_tune.jsonl'):
        rec = json.loads(line)
        if rec.get("no_target") or int(rec.get("n_targets") or 0) < 2:
            continue
        ts = terms_of(rec["parsed"])
        if len(ts) == 1:
            t1[rec["ref_id"]] = (rec["file_name"], ts)
    rows1 = [dedup[rid] for rid in t1 if rid in dedup]
    print(f"rows: ge2 {len(rows2)}  1-term {len(rows1)}", flush=True)

    from gdino import Gdino
    g = Gdino(device="cuda:0")
    import torch
    from PIL import Image
    sys.path.insert(0, str(REPRO))
    from glod.vision import load_upstream
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](
        checkpoint=str(REPRO / "checkpoints/sam_vit_h_4b8939.pth")).to("cuda").eval()
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    def sam3(box):
        ms, _, _ = predictor.predict(box=np.asarray(box, np.float32), multimask_output=True)
        return [downsample(m.astype(bool)) for m in ms]

    def terms_for(r):
        rec = recs_by_rid.get(r["rid"])
        if rec is None:
            fn, ts = t1[r["rid"]]
            return fn, ts
        return rec["file_name"], rec["terms"]

    def prep(rows_):
        last = None
        for r in rows_:
            fn, ts = terms_for(r)
            if fn != last:
                predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
                last = fn
            terms = []
            for term in ts:
                boxes, scores = g.detect(fn, term)
                ll = None
                p = lpath(fn, term)
                if p.is_file():
                    ll = json.loads(p.read_text()).get("box")
                det0 = boxes[0] if boxes else None
                det1 = boxes[1] if len(boxes) > 1 else None
                ub = None
                if ll is not None and det0 is not None:
                    ub = [min(ll[0], det0[0]), min(ll[1], det0[1]),
                          max(ll[2], det0[2]), max(ll[3], det0[3])]
                terms.append(dict(
                    ll=ll, det0=det0, det1=det1, ub=ub,
                    sam_ll=sam3(ll) if ll is not None else None,
                    sam_d0=sam3(det0) if det0 is not None else None,
                    sam_d1=sam3(det1) if det1 is not None else None,
                    sam_ub=sam3(ub) if ub is not None else None))
            r["_t3"] = terms
    prep(rows2)
    prep(rows1)
    print(f"precompute done ({time.time()-t0:.0f}s)", flush=True)

    def score_of(r, F, src, alpha, beta):
        base = F[:, 0] if src == "bs" else (
            r["_sc"] if src == "sc" else r["_gem"])
        base = np.asarray(base, np.float32)
        return base * (alpha + beta * F[:, 1])

    def predict(r, cfg):
        gt = r["gt"]
        parts = []
        for tb in r["_t3"]:
            boxes = {"ll": tb["ll"], "d0": tb["det0"], "ub": tb["ub"]}
            picks = []
            for bsrc in cfg["pick_box"]:
                b = boxes.get(bsrc)
                if b is None:
                    continue
                F, _ = box_feats(r["masks"], r["_bs"], b)
                if F is None:
                    continue
                sc = score_of(r, F, cfg["src"], cfg["alpha"], cfg["beta"])
                order = np.argsort(sc)[::-1]
                js = [int(order[0])]
                if cfg["topk"] == 2 and len(order) > 1 and \
                        sc[order[1]] >= cfg["margin"] * sc[order[0]]:
                    js.append(int(order[1]))
                picks.extend(js)
            parts.extend(r["masks"][j] for j in dict.fromkeys(picks))
            for smode in cfg["sam"]:
                ms = tb.get({"ll": "sam_ll", "d0": "sam_d0",
                             "d1": "sam_d1", "ub": "sam_ub"}[smode])
                parts.extend(m for m in (ms or [])
                             if m.sum() / m.size <= cfg["cap"])
        if not parts:
            return None
        m = parts[0].copy()
        for x in parts[1:]:
            m |= x
        I = float(np.logical_and(m, gt).sum())
        U = float(np.logical_or(m, gt).sum())
        return I / U if U else 0.0

    def group_mean(rs, cfg):
        vals = [v for v in (predict(r, cfg) for r in rs) if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    cfg0 = dict(src="bs", alpha=1.811, beta=0.234, topk=2, margin=0.631, cap=0.5)
    cells = {}
    for pb in ("ll", "ll+d0"):
        for sam in ("d0", "d0+d1"):
            cfg = dict(cfg0, pick_box=pb.split("+"), sam=sam.split("+"))
            m2 = group_mean(rows2, cfg)
            m1 = group_mean(rows1, cfg)
            mix = (1 - MIX) * m2 + MIX * m1
            cells[f"{pb}__{sam}"] = dict(m_ge2=m2, m_1term=m1, mix=mix)
            print(f"{pb:6s} x {sam:6s}  ge2 {m2:.4f}  1term {m1:.4f}  mix {mix:.4f}", flush=True)
    a = cells["ll__d0"]["mix"]
    b = cells["ll+d0__d0"]["mix"]
    c = cells["ll__d0+d1"]["mix"]
    d_ = cells["ll+d0__d0+d1"]["mix"]
    out = dict(cells=cells,
               main_box=b - a, main_sam=c - a,
               interaction=(d_ - b) - (c - a),
               note=("factorial around the exp117 winner on the mixed held-out population; "
                     "interaction = (both - box-only) - (sam-only - neither): >0 means the "
                     "levers compound; ~max(main effects) means additive"),
               generated="exp120_factorial.py")
    (OUTF / "gref_factorial.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
