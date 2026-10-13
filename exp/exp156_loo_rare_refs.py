#!/usr/bin/env python
"""EXP-156 — replicate the exp138 gRefCOCO recipe accounting and prepare/run the
rare-ref leave-one-out (review must-fix, 2026-10-13).

Modes
-----
account   Replicate exp138's split and weighting exactly (load_rows ref list,
          RandomState(0) shuffle, 70/30) and report the realized accounting:
          per-stratum counts, base, per-stratum weights, fit-part mass shares,
          rare refs present in the part table, their fit/held-out assignment.
          Verifies against the exp138 log's "fitted on 7443 / held-out 2908".
loo       For every rare ref (n_targets >= 3) that is in the FIT set, rerun the
          shipped recipe config with that ref removed (--drop-rids), held-out
          tau reselection and one val pass per refit.  Pre-registered: these are
          DIAGNOSTICS ONLY — the shipped row does not change regardless of the
          outcomes (this statement is recorded in the output).

    python exp/exp156_loo_rare_refs.py --mode account
    python exp/exp156_loo_rare_refs.py --mode loo          # ~1-2 min per refit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RIS / "exp"))
sys.path.insert(0, "/home/msj_team/Jacob/sys/glod_ris_repro")

OUTF = RIS / "exp/out_final"
WORK = RIS / "exp/out_transfer"

from exp138_learned_union_selector import TRAIN, load_rows  # noqa: E402

DROP_ARG_PRESENT = None  # set in main() after checking exp138 supports --drop-rids


def exp138_refs_and_split():
    """Replicate exp138's ref list and 70/30 split (RandomState(0))."""
    rows = load_rows(TRAIN, "train", None)
    refs = sorted({r["rid"] for r in rows})
    rng = np.random.RandomState(0)
    refs_sh = list(refs)
    rng.shuffle(refs_sh)
    n_fit = int(0.7 * len(refs_sh))
    fit_refs = set(refs_sh[:n_fit])
    return rows, refs, fit_refs


def composition():
    """exp138's per-ref n_targets composition (records file, last-wins)."""
    nt = {}
    for line in open(TRAIN):
        rec = json.loads(line)
        nt[int(rec["ref_id"])] = len([a for a in (rec.get("ann_ids") or [])
                                      if isinstance(a, int) and a >= 0])
    comp = Counter(v for v in nt.values() if v > 0)
    return nt, comp


def run_account() -> int:
    rows, refs, fit_refs = exp138_refs_and_split()
    print(f"load_rows rows: {len(rows)}  distinct refs: {len(refs)}  "
          f"fit refs: {len(fit_refs)}  held-out refs: {len(refs) - len(fit_refs)}")

    trows = json.loads((WORK / "train_rows.json").read_text())
    trnpz = np.load(WORK / "train_parts.npz")
    row_ids = trnpz["row_ids"]
    rid_of_row = {k: v["rid"] for k, v in trows.items()}
    fit_mask = np.array([rid_of_row[str(int(ri))] in fit_refs for ri in row_ids])
    print(f"parts: fit {int(fit_mask.sum())}  held-out {int((~fit_mask).sum())}"
          f"   (exp138 log anchor: 7443 / 2908)")

    nt, comp = composition()
    tot_keep = sum(comp.values())
    n_keep = len(comp)
    base = tot_keep / n_keep
    w_of = {k: base / comp[k] for k in comp}
    print("comp:", dict(sorted(comp.items())), f"tot_keep {tot_keep} n_keep {n_keep}")
    print("base:", round(base, 4))
    print("per-stratum weight:", {k: round(v, 4) for k, v in sorted(w_of.items())})

    k_part = np.array([nt.get(rid_of_row[str(int(ri))], 0) for ri in row_ids])
    w_part = np.array([w_of.get(k, 1.0) for k in k_part])
    mass_fit = float(w_part[fit_mask].sum())
    print(f"fit mass (sum of part weights): {mass_fit:.2f}")
    for k in sorted(set(k_part[fit_mask].tolist())):
        sel = fit_mask & (k_part == k)
        print(f"  k={k}: fit parts {int(sel.sum()):5d}  "
              f"mass {100 * w_part[sel].sum() / mass_fit:6.2f}%  "
              f"weight {w_of[k]:.4f}")
    rare = fit_mask & (k_part >= 3)
    print(f"RARE only-in-fit (k>=3): parts {int(rare.sum())}  "
          f"mass {100 * w_part[rare].sum() / mass_fit:.2f}%")
    all_mass = float(w_part.sum())
    print(f"RARE full table: parts {int((k_part >= 3).sum())}  "
          f"mass {100 * w_part[k_part >= 3].sum() / all_mass:.2f}%")

    # rare refs: in records, in table, fit/held-out
    rare_recs = {r: nt[r] for r in nt if nt[r] >= 3}
    tab_refs = {v["rid"] for v in trows.values()}
    print("rare refs in records:", dict(sorted(rare_recs.items())))
    for tag, sel in (("fit", lambda r: r in fit_refs),
                     ("held-out", lambda r: r not in fit_refs)):
        rs = sorted(r for r in rare_recs if r in tab_refs and sel(r))
        print(f"rare refs WITH parts in {tag}:",
              [(r, rare_recs[r], f"k={rare_recs[r]}") for r in rs])

    out = dict(
        load_rows_rows=len(rows), distinct_refs=len(refs),
        fit_refs=len(fit_refs), held_out_refs=len(refs) - len(fit_refs),
        fit_parts=int(fit_mask.sum()), held_out_parts=int((~fit_mask).sum()),
        comp={str(k): v for k, v in sorted(comp.items())},
        tot_keep=tot_keep, n_keep=n_keep, base=round(base, 6),
        weight_per_stratum={str(k): round(v, 6) for k, v in sorted(w_of.items())},
        fit_mass=round(mass_fit, 2),
        mass_share_by_stratum={str(k): round(
            100 * float(w_part[fit_mask & (k_part == k)].sum()) / mass_fit, 2)
            for k in sorted(set(k_part[fit_mask].tolist()))},
        rare_fit_parts=int(rare.sum()),
        rare_fit_mass_share=round(100 * float(w_part[rare].sum()) / mass_fit, 2),
        rare_refs_with_parts_fit=[
            dict(rid=r, k=rare_recs[r])
            for r in sorted(r for r in rare_recs if r in tab_refs and r in fit_refs)],
        rare_refs_with_parts_held_out=[
            dict(rid=r, k=rare_recs[r])
            for r in sorted(r for r in rare_recs
                            if r in tab_refs and r not in fit_refs)],
        rare_refs_no_parts=[dict(rid=r, k=rare_recs[r])
                            for r in sorted(r for r in rare_recs if r not in tab_refs)],
    )
    (OUTF / "transfer_loo_accounting.json").write_text(json.dumps(out, indent=2))
    print("written:", OUTF / "transfer_loo_accounting.json")
    return 0


# leave-one-rare-ref-out design (2026-10-13 review must-fix).  The weight update
# is uniform over the train-side ref composition, so dropping a rare ref changes
# (a) the fit set, (b) the composition counts and base.  Six runs isolate both:
#   k3 comp-only  (drop a held-out k3 ref)     -> weights with comp[3]=3, fit unchanged
#   k3 mass+comp  (drop the fit k3 ref)        -> same weights, fit loses its parts
#   k4 comp-only  / k4 mass+comp  (same logic, k4)
#   k5 mass+base  (drop the only k5 ref)       -> n_keep 7->6, base 400/7->399/6
#   k6 base-only  (drop the no-parts k6 ref)   -> same base perturbation, fit unchanged
# Paired differences: (k3 fit) - (k3 comp-only) isolates the fit-mass effect of
# ref 31716; (k4 fit) - (k4 comp-only) isolates 45269; (k5) - (k6) isolates 42328
# under the shared base perturbation.  A no-drop control run validates repro.
LOO_RUNS = [
    ("control", "", "no drop (reproduction control)"),
    ("k3_comp_only", "64040", "held-out k3 ref leaves the composition"),
    ("k3_fit", "31716", "fit k3 ref removed: mass + composition"),
    ("k4_comp_only", "81979", "held-out k4 ref leaves the composition"),
    ("k4_fit", "45269", "fit k4 ref removed: mass + composition"),
    ("k5_fit", "42328", "fit k5 ref removed: mass + base (n_keep 7->6)"),
    ("k6_base_only", "24536", "no-parts k6 ref removed: base perturbation alone"),
]


def run_loo(parallel: int = 3) -> int:
    print("pre-registered: diagnostics only; the shipped row does not change "
          "regardless of outcomes", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    import os
    import itertools
    gpus = ["0", "1", "3"]
    counter = itertools.count()  # next() is atomic under the GIL

    def one(item):
        label, rid, note = item
        out = OUTF / (f"gref_transfer_loo_{label}.json")
        if out.is_file():
            print(f"[{label}] cached", flush=True)
            return label
        cmd = [sys.executable, "exp/exp138_learned_union_selector.py",
               "--rank-features", "--sample-weight-composition",
               "--tau-stratum", "two", "--apply-val", "--out", str(out)]
        if rid:
            cmd += ["--drop-rids", rid]
        env = dict(os.environ)
        slot = gpus[next(counter) % len(gpus)]
        env["CUDA_VISIBLE_DEVICES"] = slot
        print(f"[{label}] running (gpu {slot}): {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=str(RIS), env=env)
        print(f"[{label}] rc={r.returncode}", flush=True)
        return label

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(one, LOO_RUNS))

    results = []
    for label, rid, note in LOO_RUNS:
        out = OUTF / (f"gref_transfer_loo_{label}.json")
        if not out.is_file():
            print(f"[{label}] missing output", flush=True)
            continue
        d = json.loads(out.read_text())
        vm = d.get("val_metrics") or {}
        results.append(dict(label=label, dropped_rid=rid or None, note=note,
                            held_out_cIoU=(d.get("held_out_at_tau") or {}).get("cIoU"),
                            val_cIoU=vm.get("cIoU"), val_gIoU=vm.get("gIoU"),
                            tau=d.get("tau_selected")))
        print(f"[{label}] held-out {results[-1]['held_out_cIoU']} "
              f"val {results[-1]['val_cIoU']} / {results[-1]['val_gIoU']}", flush=True)

    by = {r["label"]: r for r in results}
    def _d(a, b):
        ra, rb = by.get(a), by.get(b)
        if not ra or not rb or ra["val_cIoU"] is None or rb["val_cIoU"] is None:
            return None
        return round(ra["val_cIoU"] - rb["val_cIoU"], 4)
    summary = dict(
        pre_registered="leave-one-rare-ref-out diagnostics; the shipped row does NOT "
                       "change regardless of outcomes (no re-shipping)",
        design=("dropping a rare ref changes the fit set and the composition/base; the "
                "paired runs separate the two effects (see LOO_RUNS docstring)"),
        results=results,
        effects=dict(
            fit_k3_mass_effect=_d("k3_fit", "k3_comp_only"),
            fit_k4_mass_effect=_d("k4_fit", "k4_comp_only"),
            fit_k5_mass_over_base_effect=_d("k5_fit", "k6_base_only"),
            comp_k3_effect=_d("k3_comp_only", "control"),
            comp_k4_effect=_d("k4_comp_only", "control"),
            base_perturbation_effect=_d("k6_base_only", "control"),
        ),
        shipped_val_cIoU=52.7061, tau_stratum_only_val_cIoU=52.148,
        reading=("each effect is the val-cIoU delta of the paired runs; the mass effects "
                 "measure how much of the weight increment rests on the individual rare "
                 "fit refs, the composition/base effects measure the normalization "
                 "sensitivity to single refs leaving the train population"),
    )
    (OUTF / "transfer_loo_rare.json").write_text(json.dumps(summary, indent=2))
    print("written:", OUTF / "transfer_loo_rare.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["account", "loo"])
    args = ap.parse_args()
    return run_account() if args.mode == "account" else run_loo()


if __name__ == "__main__":
    raise SystemExit(main())
