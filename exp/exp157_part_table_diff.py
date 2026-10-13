#!/usr/bin/env python
"""EXP-157 — locate the single extra part in the gRefCOCO train part-table
rebuild (10,352 parts now vs 10,351 in the 16:58-18:32 shipped runs).

Replicates exp138's train part-table prologue (load_rows -> clf_base -> pair_clf
-> per-row gate -> build_parts), then diffs per-row part counts and per-part
feature rows against the cached exp/out_transfer/train_parts.npz.

    CUDA_VISIBLE_DEVICES=2 python exp/exp157_part_table_diff.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RIS / "exp"))
sys.path.insert(0, "/home/msj_team/Jacob/sys/glod_ris_repro")

from exp138_learned_union_selector import (  # noqa: E402
    TRAIN, build_candidates, build_parts, load_rows, make_clf, precompute_row,
    terms_of, topk_mask,
)


def main() -> int:
    st5 = json.loads((RIS / "exp/out_final/optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((RIS / "exp/out_final/optuna_study_gref8.json").read_text())["best_params"]
    cfg = dict(json.loads((RIS / "exp/out_final/optuna_study_gref14.json").read_text())["best_params"])
    cfg["sam_area_cap"] = float(cfg.pop("cap", cfg.get("sam_area_cap", 0.5)))
    cfg.setdefault("gd_thresh", 0.25)
    p_sel = float(g8["p_sel"])

    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(TRAIN, "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]), np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km]); ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))

    terms_by_rid = {}
    for line in open(TRAIN):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        ts = terms_of(rec["parsed"])
        if ts:
            terms_by_rid[rec["ref_id"]] = (rec["file_name"], ts)

    new = {}   # ri -> (X rows, meta)
    for ri, r in enumerate(tr):
        if r["no_target"]:
            continue
        fn_ts = terms_by_rid.get(r["rid"])
        if not fn_ts:
            continue
        pp = pair_clf.predict_proba(build_candidates(r, r["_bs"], r["_box"], g8["K"])[0])[:, 1]
        if (pp.max() < p_sel) or any(r["ens"]):
            continue
        built = build_parts(r, fn_ts[0], fn_ts[1], cfg, train_side=True)
        if built is None:
            continue
        masks, X, meta = built
        new[ri] = (np.asarray(X, np.float32), meta, fn_ts[0])

    old_npz = np.load(RIS / "exp/out_transfer/train_parts.npz")
    old = {}
    for idx, ri in enumerate(old_npz["row_ids"]):
        old.setdefault(int(ri), []).append(old_npz["X"][idx])

    print(f"new rows {len(new)} parts {sum(len(v[0]) for v in new.values())}")
    print(f"old rows {len(old)} parts {sum(len(v) for v in old.values())}")
    ri_all = sorted(set(new) | set(old))
    ndiff = 0
    for ri in ri_all:
        n_new = len(new[ri][0]) if ri in new else 0
        n_old = len(old.get(ri, []))
        if n_new != n_old:
            ndiff += 1
            print(f"\nROW ri={ri} rid={tr[ri]['rid']} fn={new.get(ri, (None,None,None))[2]} "
                  f"n_new={n_new} n_old={n_old}")
            if ri in new:
                for k, row in enumerate(new[ri][0]):
                    print(f"  new[{k}] pool={row[0]:.0f} sam={row[1]:.0f} area={row[2]:.4f} "
                          f"f_ll={row[3]:.4f} f_d0={row[4]:.4f} bs={row[5]:.4f} gem={row[6]:.4f} "
                          f"rank={row[7]:.0f} bi/di={row[8]:.0f} mi={row[9]:.0f} ti={row[10]:.0f} "
                          f"meta={new[ri][1][k]}")
            for k, row in enumerate(old.get(ri, [])):
                print(f"  old[{k}] pool={row[0]:.0f} sam={row[1]:.0f} area={row[2]:.4f} "
                      f"f_ll={row[3]:.4f} f_d0={row[4]:.4f} bs={row[5]:.4f} gem={row[6]:.4f} "
                      f"rank={row[7]:.0f} bi/di={row[8]:.0f} mi={row[9]:.0f} ti={row[10]:.0f}")
    # same-count rows: compare X values
    val_diff = 0
    for ri in ri_all:
        if ri not in new or ri not in old:
            continue
        a, b = new[ri][0], np.asarray(old[ri], np.float32)
        if a.shape == b.shape and not np.allclose(a, b):
            val_diff += 1
            print(f"\nROW ri={ri} rid={tr[ri]['rid']} same shape, values differ:")
            d = np.abs(a - b).max(0)
            print("  max|dX| per col:", np.round(d, 4).tolist())
    print(f"\nrows with count diff: {ndiff}; rows with value diff only: {val_diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
