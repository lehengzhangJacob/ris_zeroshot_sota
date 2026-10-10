#!/usr/bin/env python
"""EXP-110 — abstention operating-point study for the new GDINO+SAM rule.

The frozen GDINO+SAM rule scores gIoU 73.95 on full val (cIoU 47.14) against
the 74.34 floor — 0.39 short — with the SHIPPED abstention (any-probe-no
ensemble + pair-gate p_sel) that was tuned for the old, weaker mask rule.
For the new rule the operating point is re-studied on held-out train rows at
val prevalence (no target records are touched):

  abstain iff (pair-gate max_p < p_abs) or (>= k of the 3 existence probes say no)
  (use_ens off reproduces gate-only abstention)

Rows: the 262 val-matched targeted rows (new-rule masks computed here, SAM
from the top-1 GDINO box) + train no-target rows from the tune records, each
weighted w_nt = (0.6256/0.3744)(n_t/n_nt).  Objective: calib weighted gIoU
with a no-regression penalty on weighted cIoU vs the shipped 40.96.  Fit/calib
split by ref (seed 0, 0.3).  Val untouched; one frozen val pass follows.

    CUDA_VISIBLE_DEVICES=1 python exp/exp110_abstain_study.py --trials 300
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
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp103_sam_boxp import downsample  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from vlm_boxes import IMAGES  # noqa: E402

OUTF = Path("exp/out_final")
NT_FRAC = 0.6256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/optuna_study_gref13.json"))
    args = ap.parse_args()

    study = json.loads((OUTF / "optuna_study_gref12.json").read_text())
    cfg = dict(study["best_params"])
    cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]

    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))

    # gate scores + pair prediction area for every tune row (targeted + no-target)
    for r in tr:
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        r["max_p"] = float(pp.max())
        k = int(np.argmax(pp))
        pred = candidate_mask(r["masks"], ids[k])
        r["pred_area"] = float(pred.sum())
        r["pred_I"] = float(np.logical_and(pred, r["gt"]).sum()) if r["gt"] is not None else 0.0
        r["pred_U"] = float(np.logical_or(pred, r["gt"]).sum()) if r["gt"] is not None else r["pred_area"]
    print(f"gate scores for {len(tr)} tune rows ({time.time()-t0:.0f}s)", flush=True)

    # ---- targeted held-out rows with the NEW rule's masks ----
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    dedup = {r["rid"]: r for r in tr}
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    rows = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]

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

    last = None
    for k, r in enumerate(rows):
        rec = recs_by_rid[r["rid"]]
        fn = rec["file_name"]
        if fn != last:
            predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
            last = fn
        parts = []
        for term in rec["terms"]:
            boxes, scores = g.detect(fn, term)
            b = None
            p = lpath(fn, term)
            if p.is_file():
                b = json.loads(p.read_text()).get("box")
            if b is None and boxes:
                b = boxes[0]
            if b is not None:
                F, _ = box_feats(r["masks"], r["_bs"], b)
                if F is not None:
                    sc = F[:, 0] * (cfg["alpha"] + cfg["beta"] * F[:, 1])
                    order = np.argsort(sc)[::-1]
                    js = [int(order[0])]
                    if cfg["topk"] == 2 and len(order) > 1 and \
                            sc[order[1]] >= cfg["margin"] * sc[order[0]]:
                        js.append(int(order[1]))
                    parts.extend(r["masks"][j] for j in js)
            if boxes:
                ms, _, _ = predictor.predict(
                    box=np.asarray(boxes[0], np.float32), multimask_output=True)
                for m in ms:
                    m = downsample(m.astype(bool))
                    if m.sum() / m.size <= cfg["sam_area_cap"]:
                        parts.append(m)
        if parts:
            pred = parts[0].copy()
            for x in parts[1:]:
                pred |= x
            r["new_I"] = float(np.logical_and(pred, r["gt"]).sum())
            r["new_U"] = float(np.logical_or(pred, r["gt"]).sum())
        else:
            r["new_I"] = r["pred_I"]
            r["new_U"] = r["pred_U"]
    print(f"new-rule masks for {len(rows)} targeted rows ({time.time()-t0:.0f}s)", flush=True)

    nt_rows = [r for r in tr if r["no_target"]]
    t_rows = rows
    print(f"targeted {len(t_rows)}  no-target {len(nt_rows)}", flush=True)

    rng = np.random.RandomState(args.seed)
    refs = sorted({r["rid"] for r in t_rows} | {r["rid"] for r in nt_rows})
    rng.shuffle(refs)
    n_fit = int((1 - args.calib_frac) * len(refs))
    fit_refs = set(refs[:n_fit])
    t_calib = [r for r in t_rows if r["rid"] not in fit_refs]
    nt_calib = [r for r in nt_rows if r["rid"] not in fit_refs]
    print(f"calib targeted {len(t_calib)} no-target {len(nt_calib)}", flush=True)

    def votes_no(r, k):
        return sum(1 for v in r["ens"] if v) >= k

    def weighted(rs_t, rs_nt, k_abs, p_abs, use_ens):
        n_t = len(rs_t)
        n_nt = len(rs_nt)
        if not n_t or not n_nt:
            return 0.0, 0.0
        w_nt = (NT_FRAC / (1 - NT_FRAC)) * (n_t / n_nt)
        wiou = wI = wU = wsum = 0.0
        for r in rs_t:
            abst = (r["max_p"] < p_abs) or (use_ens and votes_no(r, k_abs))
            wsum += 1.0
            if abst:
                continue
            wiou += r["new_I"] / r["new_U"] if r["new_U"] else 0.0
            wI += r["new_I"]
            wU += r["new_U"]
        for r in rs_nt:
            abst = (r["max_p"] < p_abs) or (use_ens and votes_no(r, k_abs))
            wsum += w_nt
            if abst:
                wiou += w_nt
            else:
                wU += w_nt * r["pred_area"]
        gio = wiou / wsum if wsum else 0.0
        cio = wI / wU if wU else 0.0
        return gio * 100.0, cio * 100.0


    gio, cio = weighted(t_rows, nt_rows, 1, float(g8["p_sel"]), True)
    print(f"held-out (all rows) weighted: gIoU_w {gio:.2f} cIoU_w {cio:.2f}", flush=True)
    # N_acc / T_acc at the shipped operating point on this population
    # abstained = (max_p < p_abs) or any-probe-no (same condition as the pass)
    def abst(r):
        return (r["max_p"] < float(g8["p_sel"])) or any(r["ens"])
    tp = sum(1 for r in nt_rows if abst(r))
    fn_ = len(nt_rows) - tp
    tn = sum(1 for r in t_rows if not abst(r))
    fp = len(t_rows) - tn
    out = dict(
        model=dict(kind="GDINO+SAM termbox union (change 13)", study_params=cfg,
                   abstention=dict(k_abs=1, p_abs=float(g8["p_sel"]), use_ens=True),
                   source_study="optuna_study_gref12.json"),
        metrics=dict(n=len(t_rows) + len(nt_rows), cIoU=None, gIoU=None,
                     N_acc=100.0 * tp / max(1, tp + fn_), T_acc=100.0 * tn / max(1, tn + fp),
                     TP=tp, FN=fn_, TN=tn, FP=fp,
                     cIoU_val_prevalence=cio, gIoU_val_prevalence=gio),
        population=("held-out train: val-matched multi-target sample rows + all no-target "
                    "tune rows at val prevalence weight"),
        generated="exp113_gdino_heldout.py")
    (OUTF / "gref_gdino_heldout.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["metrics"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
