#!/usr/bin/env python
"""EXP-74 — attribute the pair variant's val-prevalence win: features or abstention?

exp73's study compared variant optima at RAW calib prevalence and picked
`base`; at VAL prevalence the `pair` optimum reads higher (44.66/79.08 vs
41.63/67.45).  But the two optima also chose different rule regimes (pair:
use_ens=True, N_acc 93.8; base: use_ens=False, N_acc 72.7), so the comparison
conflates the feature change with the abstention regime — and the direct
signal diagnostic (pair-winner vs GT-best agreement 29.8%, chance 25%) says
the pair columns are near-useless.

This script runs the attribution grid on the identical fit/calib split:
feature set {base X, XA} x rule params {base's, pair's} (+ the use_ens flip),
every cell evaluated on calib at VAL prevalence — the frame the official
metric lives in and the one exp73's docstring pre-registered.  If
base-X with pair's rule params matches XA with pair's params, the pair
features add nothing and the gate fails; if XA strictly dominates at fixed
params, the features carry signal.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp73_pair_study import (REPRO, TOPK, build_pair_feats, evaluate_ens,  # noqa: E402
                              mark_rule, weighted, NT_FRAC)
from exp62_clause_selector import load_record, make_clf, score_rows  # noqa: E402
from exp67_ensemble_eval import probe_no  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g7 = json.loads((OUTF / "optuna_study_gref7.json").read_text())
    base_p = g7["variants"]["base"]["params"]
    pair_p = g7["variants"]["pair"]["params"]
    print(f"base params {base_p}")
    print(f"pair params {pair_p}")

    t0 = time.time()
    gd = GreferDataset(REPRO / "data/refer", "train")
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    rows = []
    for line in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        rec = json.loads(line)
        ph = (rec["parsed"].get("sentence") or "").strip()
        if not ph:
            continue
        r = load_record(rec, gd, by_id, mc, gc)
        if r is None:
            continue
        r["ens"] = [probe_no(t, rec["file_name"], ph)
                    for t in ("exists", "exists2", "exists3")]
        if any(v is None for v in r["ens"]):
            continue
        r["file_name"] = rec["file_name"]
        r["phrase"] = ph
        rows.append(r)
    print(f"rows {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    # identical split as exp73 (seed 0, calib_frac 0.3, sorted rids)
    rng = np.random.RandomState(0)
    refs = sorted({r["rid"] for r in rows})
    rng.shuffle(refs)
    n_fit = int((1 - 0.3) * len(refs))
    fit_refs = set(refs[:n_fit])
    for r in rows:
        r["split"] = "fit" if r["rid"] in fit_refs else "calib"
    fit_rows = [r for r in rows if r["split"] == "fit"]
    calib_rows = [r for r in rows if r["split"] == "calib"]

    clf5 = make_clf("logreg", dict(C=st5["C"]))
    clf5.fit(np.concatenate([r["X"] for r in fit_rows]),
             np.concatenate([r["y"] for r in fit_rows]))
    score_rows(rows, clf5)
    for r in rows:
        top = [int(j) for j in np.argsort(r["scores"])[::-1][:TOPK]]
        r["PF"] = build_pair_feats(r, r, r["phrase"], top)
        r["XA"] = np.concatenate([r["X"], r["PF"]], 1)
    print(f"features ready ({time.time()-t0:.0f}s)", flush=True)

    def fit_scores(C, key):
        clf = make_clf("logreg", dict(C=C))
        clf.fit(np.concatenate([r[key] for r in fit_rows]),
                np.concatenate([r["y"] for r in fit_rows]))
        for r in calib_rows:
            p = clf.predict_proba(r[key])[:, 1]
            r["scores"] = p
            r["max_p"] = float(p.max())

    cells = {}
    for feat, key in (("base", "X"), ("pair", "XA")):
        for pname, bp in (("base_p", base_p), ("pair_p", pair_p)):
            for ue in (False, True):
                fit_scores(bp["C"], key)
                mark_rule(calib_rows, bp["p_inc"], bp["p_abs"], ue)
                m = weighted(calib_rows)
                cells[(feat, pname, ue)] = m
                print(f"  feat={feat:4s} params={pname} use_ens={int(ue)}: "
                      f"cIoU {m['cIoU']:6.2f} gIoU {m['gIoU']:6.2f} "
                      f"N {m['N_acc']:5.1f} T {m['T_acc']:5.1f}", flush=True)

    xa_p = cells[("pair", "pair_p", True)]
    xb_p = cells[("base", "pair_p", True)]
    delta = xa_p["gIoU"] - xb_p["gIoU"]
    verdict = ("FEATURES ADD NOTHING (attribution shows the pair win is the "
               "abstention regime)" if delta < 0.5 else
               "FEATURES CARRY SIGNAL (XA beats base at identical params)")
    out = dict(cells={f"{k[0]}|{k[1]}|ens{int(k[2])}": v for k, v in cells.items()},
               attribution_delta_gIoU_fixed_params=round(delta, 3),
               diag_agreement=g7.get("pair_winner_vs_gtbest_agreement_pct"),
               verdict=verdict,
               note=("all cells at val prevalence on the exp73 calib split; the "
                     "fixed-params contrast (pair_p, use_ens=True) is the only one "
                     "that isolates the feature change from the rule regime"),
               generated="exp74_pair_attribution.py")
    (OUTF / "gref_pair_attribution.json").write_text(json.dumps(out, indent=2))
    print(f"{verdict}; delta {delta:+.2f} gIoU; wrote gref_pair_attribution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
