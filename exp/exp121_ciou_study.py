#!/usr/bin/env python
"""EXP-121 — cIoU-targeted study (model change 15): 300 trials on the r objective.

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

    CUDA_VISIBLE_DEVICES=1 python exp/exp121_ciou_study.py --trials 300
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
                    default=Path("exp/out_final/optuna_study_gref15.json"))
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
        parts = predict_parts(r, cfg)
        if parts is None:
            return None
        gt = r["gt"]
        m = parts[0].copy()
        for x in parts[1:]:
            m |= x
        I = float(np.logical_and(m, gt).sum())
        U = float(np.logical_or(m, gt).sum())
        return I / U if U else 0.0

    def predict_parts(r, cfg):
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
        return parts

    def group_mean(rs, cfg):
        vals = [v for v in (predict(r, cfg) for r in rs) if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    def group_r(rs, cfg):
        tI = tU = 0.0
        for r in rs:
            parts = predict_parts(r, cfg)
            if parts is None:
                continue
            gt = r["gt"]
            m = parts[0].copy()
            for x in parts[1:]:
                m |= x
            tI += float(np.logical_and(m, gt).sum())
            tU += float(np.logical_or(m, gt).sum())
        return tI / max(1.0, tU)

    cfg0 = dict(src="bs", alpha=1.2205, beta=0.6172, topk=2, margin=0.7294,
                pick_box=["ll"], sam=["d0"], cap=0.25)
    m2_0, m1_0 = group_mean(rows2, cfg0), group_mean(rows1, cfg0)
    mix0 = (1 - MIX) * m2_0 + MIX * m1_0
    print(f"shipped-equiv: ge2 {m2_0:.4f}  1-term {m1_0:.4f}  mix {mix0:.4f}",
          flush=True)

    def objective(trial):
        cfg = dict(
            src=trial.suggest_categorical("src", ["bs", "sc", "gem"]),
            alpha=trial.suggest_float("alpha", 0.0, 2.0),
            beta=trial.suggest_float("beta", 0.0, 1.5),
            topk=trial.suggest_int("topk", 1, 2),
            margin=trial.suggest_float("margin", 0.5, 1.0),
            cap=trial.suggest_categorical("cap", [0.15, 0.25, 0.5]),
        )
        pb = trial.suggest_categorical("pick_box", ["ll", "d0", "ll+d0", "ll+ub"])
        cfg["pick_box"] = pb.split("+")
        sam_sel = trial.suggest_categorical(
            "sam", ["d0", "d0+d1", "d0+ub", "ub", "off"])
        cfg["sam"] = [] if sam_sel == "off" else sam_sel.split("+")
        r2, r1 = group_r(rows2, cfg), group_r(rows1, cfg)
        rmix = (1 - MIX) * r2 + MIX * r1
        m2, m1 = group_mean(rows2, cfg), group_mean(rows1, cfg)
        mmix = (1 - MIX) * m2 + MIX * m1
        trial.set_user_attr("r_mix", rmix)
        trial.set_user_attr("m_mix", mmix)
        # cIoU-r objective with a gIoU-safety floor (m must stay above the
        # 0.6212 requirement that the v14 config cleared)
        penalty = 0.0 if mmix >= 0.6212 else -1.0
        return rmix + penalty

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=args.seed))
    st.enqueue_trial(dict(src="bs", alpha=1.811, beta=0.234, topk=2,
                          margin=0.631, cap=0.5, pick_box="ll+d0", sam="d0+d1"))
    st.enqueue_trial(dict(src="bs", alpha=1.811, beta=0.234, topk=1,
                          margin=0.631, cap=0.25, pick_box="ll+d0", sam="d0+d1"))
    st.enqueue_trial(dict(src="bs", alpha=1.811, beta=0.234, topk=2,
                          margin=0.631, cap=0.25, pick_box="d0", sam="d0"))
    st.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = st.best_trial
    r2_0, r1_0 = group_r(rows2, cfg0), group_r(rows1, cfg0)
    out = dict(n_trials=args.trials, best_params=best.params,
               r_mix=best.user_attrs["r_mix"], m_mix=best.user_attrs["m_mix"],
               v14_equivalent=dict(r_mix=(1 - MIX) * r2_0 + MIX * r1_0,
                                    m_mix=(1 - MIX) * m2_0 + MIX * m1_0),
               required_m=0.6212,
               population=("262 val-matched >=2-term rows + 47 held-out 1-term rows, "
                           "mixed at val prevalence 0.875/0.125"),
               note=("objective = mixture cumulative r (cIoU side) with an m >= 0.6212 "
                     "safety floor; val pass only if r improves and val gIoU stays above "
                     "its floor"),
               generated="exp121_ciou_study.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    print(f"study done ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
