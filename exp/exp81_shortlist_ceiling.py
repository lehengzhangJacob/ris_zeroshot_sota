#!/usr/bin/env python
"""EXP-81 — deployable-shortlist oracle ceiling for the pair selector.

exp77's 0.825 best-pair figure used an ORACLE ranking (top-12 by GT IoU).
The shipped pair selector can only draw candidates from the top-K of the
frozen base classifier — so its achievable ceiling is the best-pair oracle
WITHIN that deployable shortlist.  This measures, on held-out train targeted
rows, the cumulative best-single / best-pair / greedy-union within the
top-K-by-base-score shortlist for K = 6, 12, 24, 50, and the full pool.

If best-pair@K6-base ~= best-pair@oracle-ranking, the shortlist is fine and
the pair classifier's scoring is the bottleneck; if it is far below, the base
scorer's shortlist is the binding constraint and the next model change must
improve candidate RECALL, not re-ranking.

    python exp/exp81_shortlist_ceiling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp62_clause_selector import load_record, make_clf, score_rows  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    st = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    t0 = time.time()
    gd = GreferDataset(REPRO / "data/refer", "train")
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    rows = []
    for line in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        rec = json.loads(line)
        r = load_record(rec, gd, by_id, mc, gc)
        if r is not None and not r["no_target"]:
            rows.append(r)
    clf = make_clf("logreg", dict(C=st["C"]))
    clf.fit(np.concatenate([r["X"] for r in rows]),
            np.concatenate([r["y"] for r in rows]))
    score_rows(rows, clf)
    print(f"targeted rows {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    KS = [6, 12, 24, 50, 0]           # 0 = full pool
    acc = {K: dict(sI=0.0, sU=0.0, pI=0.0, pU=0.0, tI=0.0, tU=0.0, qI=0.0, qU=0.0)
           for K in KS}
    for r in rows:
        gt = r["gt"]
        masks = r["masks"]
        order = np.argsort(r["scores"])[::-1]
        for K in KS:
            top = list(order if K == 0 else order[:min(K, len(order))])
            # best single in shortlist
            best_s = (-1.0, None)
            for i in top:
                I = float(np.logical_and(masks[i], gt).sum())
                U = float(np.logical_or(masks[i], gt).sum())
                v = I / max(U, 1.0)
                if v > best_s[0]:
                    best_s = (v, i)
            i1 = best_s[1]
            I1 = float(np.logical_and(masks[i1], gt).sum())
            U1 = float(np.logical_or(masks[i1], gt).sum())
            acc[K]["sI"] += I1
            acc[K]["sU"] += U1
            # best pair, best triple, best 4-union (exhaustive on top-10)
            t10 = top[:10]
            best_p = (-1.0, None)
            best_t = (-1.0, None)
            best_q = (-1.0, None)
            import itertools
            for combo in itertools.combinations(t10, 2):
                m = np.logical_or(masks[combo[0]], masks[combo[1]])
                v = float(np.logical_and(m, gt).sum()) / max(1.0, float(np.logical_or(m, gt).sum()))
                if v > best_p[0]:
                    best_p = (v, combo)
            for combo in itertools.combinations(t10, 3):
                m = np.logical_or(np.logical_or(masks[combo[0]], masks[combo[1]]), masks[combo[2]])
                v = float(np.logical_and(m, gt).sum()) / max(1.0, float(np.logical_or(m, gt).sum()))
                if v > best_t[0]:
                    best_t = (v, combo)
            for combo in itertools.combinations(t10, 4):
                m = masks[combo[0]].copy()
                for c in combo[1:]:
                    m = np.logical_or(m, masks[c])
                v = float(np.logical_and(m, gt).sum()) / max(1.0, float(np.logical_or(m, gt).sum()))
                if v > best_q[0]:
                    best_q = (v, combo)
            for key, sel in (("p", best_p), ("t", best_t), ("q", best_q)):
                if sel[1] is not None:
                    m = masks[sel[1][0]]
                    for c in sel[1][1:]:
                        m = np.logical_or(m, masks[c])
                else:
                    m = masks[i1]
                acc[K][key + "I"] += float(np.logical_and(m, gt).sum())
                acc[K][key + "U"] += float(np.logical_or(m, gt).sum())
    out = {}
    for K in KS:
        a = acc[K]
        out[f"K{K if K else 'full'}"] = {
            "best_single_cumulative": round(a["sI"] / max(1.0, a["sU"]), 4),
            "best_pair_cumulative": round(a["pI"] / max(1.0, a["pU"]), 4),
            "best_triple_cumulative": round(a["tI"] / max(1.0, a["tU"]), 4),
            "best_quad_cumulative": round(a["qI"] / max(1.0, a["qU"]), 4),
        }
        print(f"K={K if K else 'full':>4}: single {a['sI']/max(1,a['sU']):.4f}  "
              f"pair {a['pI']/max(1,a['pU']):.4f}  triple {a['tI']/max(1,a['tU']):.4f}  "
              f"quad {a['qI']/max(1,a['qU']):.4f}", flush=True)
    art = dict(measured_on="held-out train targeted rows, deployable top-K by base score "
                           "(oracle choice within the shortlist)",
               ladder=out,
               oracle_ranked_reference={"best_single": 0.6957, "best_pair": 0.8250,
                                        "note": "exp77, top-12 by GT IoU (non-deployable)"},
               interpretation=("compare best_pair at K6 vs the oracle-ranked 0.825: if close, "
                               "re-ranking is the bottleneck; if far, candidate recall is"),
               generated="exp81_shortlist_ceiling.py")
    (OUTF / "gref_shortlist_ceiling.json").write_text(json.dumps(art, indent=2))
    print("wrote gref_shortlist_ceiling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
