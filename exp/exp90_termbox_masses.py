#!/usr/bin/env python
"""EXP-90 — wall masses at the SHIPPED termbox operating point.

The wall's cIoU requirement (r_required, oracle cap, dilution cut) and the
gIoU m-requirement are operating-point quantities.  After exp79 shipped the
pair/union selector (36.75/69.63 full val), the masses must be re-measured for
that rule exactly as exp76 did for the ensemble threshold-union: S_U, FP_U,
A_U (direct), I_tot, G_U, P_sum, and the delivered mean targeted IoU.

    python exp/exp80_pair_masses.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp78_pair_selector import build_candidates, candidate_mask  # noqa: E402
from exp89_termbox_full_val import term_features  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402

OUTF = Path("exp/out_final")
WORLD = 14220


def main() -> int:
    study = json.loads((OUTF / "optuna_study_gref8.json").read_text())
    bp = study["best_params"]
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    K, p_sel, use_ens = bp["K"], bp["p_sel"], bp["use_ens"]

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]

    Xs, ys = [], []
    from exp78_pair_selector import precompute_row, topk_mask
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), K)
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=bp["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    print(f"fitted ({time.time()-t0:.0f}s)", flush=True)

    val = load_rows(Path("exp/out_extreme/records_grefcoco.jsonl"), "val", clf_base)
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    print(f"val {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    # termbox rule: fit the picker on the val-matched held-out population
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    st9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]
    topk9, margin9 = st9["topk"], st9["margin"]
    Xf9, yf9 = [], []
    dedup = {x["rid"]: x for x in tr}      # same lookup as exp89's shipped pass
    for rid, rec in recs_by_rid.items():
        r0 = dedup.get(rid)
        if r0 is None:
            continue
        iou_gt = (np.logical_and(r0["masks"], r0["gt"]).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(r0["masks"], r0["gt"]).sum((1, 2)), 1.0))
        for F in term_features(r0, rec):
            Xf9.append(F)
            yf9.append((iou_gt >= 0.5).astype(np.float32))
    pick_clf9 = make_clf("logreg", dict(C=st9["C"]))
    pick_clf9.fit(np.concatenate(Xf9), np.concatenate(yf9))
    terms_by_rid = {}
    for line in open("exp/out_extreme/records_grefcoco.jsonl"):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        ts, seen = [], set()
        for tt in [rec["parsed"].get("noun_phrase") or ""] + list(rec["parsed"].get("nouns") or []):
            tt = tt.strip()
            if tt and tt.lower() not in seen:
                seen.add(tt.lower()); ts.append(tt)
        terms_by_rid[rec["ref_id"]] = (rec["file_name"], ts[:3])

    S_U = FP_U = A_U = I_tot = G_U = P_sum = m_num = m_den = 0.0
    n_tb = n_fb = 0
    for r in val:
        F, ids = build_candidates(r, r["_bs"], r["_box"], K)
        pp = pair_clf.predict_proba(F)[:, 1]
        abstain = (pp.max() < p_sel) or any(r["ens"])
        if abstain:
            if not r["no_target"]:
                A_U += r["gt_area"]
            continue
        pred = None
        fn_, ts = terms_by_rid.get(r["rid"], (None, []))
        if fn_ is not None and ts:
            picks = []
            for F_t in term_features(r, {"file_name": fn_, "terms": ts}):
                pt = pick_clf9.predict_proba(F_t)[:, 1]
                order = np.argsort(pt)[::-1]
                picks.append(int(order[0]))
                if topk9 == 2 and len(order) > 1 and pt[order[1]] >= margin9 * pt[order[0]]:
                    picks.append(int(order[1]))
            picks = list(dict.fromkeys(picks))
            if picks:
                pred = r["masks"][picks[0]].copy()
                for j in picks[1:]:
                    pred = np.logical_or(pred, r["masks"][j])
                n_tb += 1
        if pred is None:
            pred = candidate_mask(r["masks"], ids[int(np.argmax(pp))])
            n_fb += 1
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        if r["no_target"]:
            FP_U += U
        else:
            I_tot += I
            S_U += U
            G_U += float(r["gt"].sum())
            P_sum += float(pred.sum())
            m_num += (I / U if U else 0.0)
            m_den += 1
    X = FP_U + A_U
    cIoU = 100.0 * I_tot / (S_U + X)
    fv = json.loads((OUTF / "gref_termbox_full_val.json").read_text())["metrics"]
    N_acc, T_acc = fv["N_acc"] / 100.0, fv["T_acc"] / 100.0
    out = dict(
        measured_on="full val (14220), SHIPPED termbox selector (exp89 final)",
        S_U=S_U, FP_U=FP_U, A_U=A_U, I_tot=I_tot, G_U=G_U, P_sum=P_sum,
        r_delivered=I_tot / S_U if S_U else 0.0,
        cIoU_from_masses=cIoU,
        official_cIoU=40.96,
        m_delivered=m_num / m_den if m_den else 0.0,
        targeted_predicted=int(m_den),
        oracle_cap_bound_pct=100.0 * G_U / (G_U / 0.8842 + X) if G_U else 0.0,
        r_required_for_floor=0.6894 * (S_U + X) / S_U if S_U else 0.0,
        dilution_cut_needed_pct=(100.0 * (1 - G_U * (1 / 0.6894 - 1 / 0.8842) / X)
                                 if X and G_U else 0.0),
        m_required_for_floor=(0.7434 - (8896.0 / WORLD) * N_acc) /
                             ((1 - 8896.0 / WORLD) * T_acc),
        note=("operating point from exp79's frozen rule (N_acc 0.862, T_acc 0.878 as "
              "recorded there); masses accumulated with the same exp62 gt_area "
              "convention as exp76"),
        termbox_used=n_tb, fallback_used=n_fb,
        generated="exp90_termbox_masses.py")
    (OUTF / "gref_termbox_masses.json").write_text(json.dumps(out, indent=2))
    print(f"S_U={S_U:.0f} FP_U={FP_U:.0f} A_U={A_U:.0f} I_tot={I_tot:.0f} "
          f"G_U={G_U:.0f}")
    print(f"cIoU {cIoU:.2f} (official 36.75)  r_del {out['r_delivered']:.4f}  "
          f"m_del {out['m_delivered']:.4f}")
    print(f"r_req {out['r_required_for_floor']:.4f}  cap {out['oracle_cap_bound_pct']:.2f}  "
          f"cut {out['dilution_cut_needed_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
