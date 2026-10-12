#!/usr/bin/env python
"""EXP-148 — assemble exp/out_final/learned_prior.json from the augmented run.

Reads the exp138 augmented-training output (val metrics on the full official
gRefCOCO val split) and the shipped prompt-prior record, and writes the gated
artifact with the trained-component spec, the per-dataset full-split metrics,
and the staged-target comparison (T1 > prompt-prior 51.32 cIoU; T2 towards the
supervised reference 68.94 cIoU / 74.34 gIoU, labelled, never a floor here).
"""
from __future__ import annotations

import json
from pathlib import Path

OUTF = Path("exp/out_final")


def main() -> int:
    src = json.loads((OUTF / "gref_learned_prior_aug3.json").read_text())
    shipped = json.loads((OUTF / "main_grefcoco.json").read_text())["shipped"]
    t1 = float(shipped["cIoU"])
    vm = src["val_metrics"]
    aug = src.get("augment") or {}
    n_iter, depth = 250, 3
    node_budget = n_iter * (2 ** (depth + 1) - 1)

    doc = dict(
        what=("learned localization prior over the gRefCOCO candidate pool, trained on "
              "gRefCOCO TRAIN with degradation augmentation (2026-10-12 Punch-2 directive); "
              "fixes the measured train->val part-quality shift that broke the naive "
              "selector (part-IoU 0.413 train vs 0.355 val)"),
        trained_component=dict(
            name="part-acceptance prior (HistGradientBoostingRegressor over the exp138 "
                 "12-dim per-part feature vector)",
            params=node_budget,
            params_note=("upper bound on fitted node values: 250 boosting iterations of "
                         "depth-3 trees; the feature vector is 12-dimensional and the part "
                         "pool is unchanged from the shipped rule"),
            training_data=(f"{src['train_parts']} parts from the gRefCOCO TRAIN-side "
                           "population (78,930-ref train split; held-out carve-out = 30% "
                           "of train refs used for threshold tuning only)"),
            supervision=("per-part IoU against the union of the reference's ground-truth "
                         "masks; no-target refs are excluded from part training and keep "
                         "the shipped existence/abstention machinery untouched"),
            epochs=(f"one fit, {n_iter} boosting iterations (max_depth {depth}, lr 0.08); "
                    f"{1 + int(aug.get('copies', 0))}x training rows via degradation "
                    f"augmentation (box-evidence scale u~U({aug.get('aug_lo', 0.6)},1) with "
                    "coherent label scaling)"),
            threshold_policy=("acceptance threshold tuned on the held-out train carve-out "
                              f"only (policy {src['policy']}, tau={src['tau_selected']}); "
                              "official val evaluated once per fixed configuration"),
            declared_training="yes - this is a TRAINED component; the training-free "
                              "prompt-prior variant remains the shipped row of record and "
                              "is reported alongside"),
        datasets=dict(grefcoco=dict(
            n=vm["n"], cIoU=round(vm["cIoU"], 4), gIoU=round(vm["gIoU"], 4),
            N_acc=round(vm.get("N_acc", 0.0), 3), T_acc=round(vm.get("T_acc", 0.0), 3),
            protocol=("official gRefCOCO val evaluation on the complete split (n=14,229 "
                      "sentences), no-target treatment identical to the shipped row"),
            prompt_prior_shipped=dict(cIoU=t1, gIoU=float(shipped["gIoU"]),
                                      source="exp/out_final/main_grefcoco.json"),
            supervised_reference=dict(cIoU=68.94, gIoU=74.34, method="InstAlign",
                                      note="labelled reference row, never a floor")),
        ),
        evaluation=dict(
            source="exp/out_final/gref_learned_prior_aug3.json",
            held_out_at_tau=src["held_out_at_tau"],
            anchors_held_out=src["anchors_held_out"],
            note=("held-out numbers are the train-carved carve-out; the val numbers are a "
                  "single frozen application of the tuned configuration")),
    )
    ds = doc["datasets"]["grefcoco"]
    ds["targets"] = dict(
        T1=dict(definition=f"exceed the prompt-prior record ({t1} cIoU)",
                met=bool(ds["cIoU"] > t1), delta=round(ds["cIoU"] - t1, 4)),
        T2=dict(definition="close the gap towards the supervised reference (68.94 cIoU / "
                           "74.34 gIoU)",
                cIoU_gap_remaining=round(68.94 - ds["cIoU"], 4),
                gIoU_gap_remaining=round(74.34 - ds["gIoU"], 4)),
    )
    (OUTF / "learned_prior.json").write_text(json.dumps(doc, indent=2))
    print("wrote", OUTF / "learned_prior.json")
    print(f"val: n={ds['n']} cIoU={ds['cIoU']} gIoU={ds['gIoU']} | shipped {t1}/"
          f"{shipped['gIoU']} | T1 {'MET' if ds['targets']['T1']['met'] else 'MISSED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
