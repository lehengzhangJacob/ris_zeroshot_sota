#!/usr/bin/env python
"""EXP-39 — Optuna study for the box-prior selection rule, on the HELD-OUT split.

Tuning split: seeded RefCOCO-family TRAIN samples (exp/tuning/*), never the
official val/test splits used for the SOTA claim.

Objective: min over the three datasets of mIoU — the pipeline must not lose any
dataset.  Search space: candidate-set source and size, box-overlap mode, how
the localization and semantic scores combine, and whether the containment
growth repair is applied.

Writes exp/out_final/optuna_study.json with n_trials and the best parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DATASETS = ("refcoco", "refcoco+", "refcocog")


def rank01(v):
    if len(v) <= 1:
        return np.zeros_like(v)
    order = np.argsort(np.argsort(v)).astype(np.float32)
    return order / (len(v) - 1)


class Split:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.off = z["offsets"]
        self.n = len(self.off) - 1
        self.d = {k: z[k] for k in z.files if k != "offsets"}
        self.z = z

    def rec(self, i):
        a, b = self.off[i], self.off[i + 1]
        return {k: v[a:b] for k, v in self.d.items()}

    def mIoU(self, params):
        total = 0.0
        for i in range(self.n):
            r = self.rec(i)
            total += float(r["gt_iou"][self.select(r, params)])
        return 100.0 * total / self.n

    @staticmethod
    def select(r, p):
        n = len(r["gt_iou"])
        if p["dec_source"] == "sc_x_gem":
            base = rank01(r["sc"]) + p["gem_mix"] * rank01(r["gem"])
        else:
            base = r[p["dec_source"]]
        k = min(p["k_dec"], n)
        idx = np.argsort(base)[::-1][:k] if k < n else np.arange(n)
        pre = "locq_" if p.get("loc_source") == "qwen2" else "loc_"
        loc = r[f"{pre}{p['mode']}"][idx]
        if p["combine"] == "loc_only":
            pick = int(np.argmax(loc))
        elif p["combine"] == "add3":
            # localization + semantic + released coherence (GEM), all rank-based
            rg = rank01(r["gem"][idx])
            pick = int(np.argmax(rank01(loc) + p["beta"] * rank01(base[idx])
                                 + p["gem_mix"] * rg))
        else:
            rl, rb = rank01(loc), rank01(base[idx])
            if p["combine"] == "mul":
                pick = int(np.argmax((rl + p["eps"]) * (rb + p["eps"]) ** p["beta"]))
            else:  # add
                pick = int(np.argmax(rl + p["beta"] * rb))
        j = int(idx[pick])
        if p["grow"] and r["has_box"][0] > 0.5:
            j = int(r["ps_idx"][j])
        return j


def main() -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=Path, default=Path("exp/tuning"))
    ap.add_argument("--tag", default="noun")
    ap.add_argument("--trials", type=int, default=320)
    ap.add_argument("--output", type=Path, default=Path("exp/out_final/optuna_study.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = {}
    for ds in DATASETS:
        p = args.pre / f"pre_{ds}_{args.tag}.npz"
        if not p.is_file():
            print(f"missing {p}", flush=True)
            return 1
        splits[ds] = Split(p)
        print(f"{ds}: {splits[ds].n} records", flush=True)

    def objective(trial):
        p = dict(
            dec_source=trial.suggest_categorical("dec_source", ["sc", "sc_noun", "sc_strip", "gem", "sc_x_gem"]),
            k_dec=trial.suggest_categorical("k_dec", [1, 3, 5, 8, 10, 15, 25, 10 ** 6]),
            mode=trial.suggest_categorical("mode", ["cover", "inside", "iou"]),
            combine=trial.suggest_categorical("combine", ["loc_only", "mul", "add", "add3"]),
            loc_source=trial.suggest_categorical("loc_source", ["llava", "qwen2"]),
            beta=trial.suggest_float("beta", 0.0, 3.0),
            eps=trial.suggest_float("eps", 0.0, 0.2),
            gem_mix=trial.suggest_float("gem_mix", 0.0, 2.0),
            grow=trial.suggest_categorical("grow", [False, True]),
        )
        vals = []
        for ds, sp in splits.items():
            v = sp.mIoU(p)
            trial.set_user_attr(f"mIoU_{ds}", v)
            vals.append(v)
        return min(vals)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = study.best_trial
    report = dict(
        n_trials=len(study.trials),
        best_value=best.value,
        best_params=best.params,
        best_per_dataset={k: v for k, v in best.user_attrs.items()},
        tuning_split=dict(source="RefCOCO-family TRAIN samples (seeded)", tag=args.tag,
                          records={ds: splits[ds].n for ds in splits}),
        top10=[dict(value=t.value, params=t.params) for t in
               sorted(study.trials, key=lambda t: -(t.value or -1))[:10]],
        generated="exp39_optuna.py",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("n_trials", "best_value", "best_params")}, indent=2))
    print("per dataset:", {k: round(v, 3) for k, v in best.user_attrs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
