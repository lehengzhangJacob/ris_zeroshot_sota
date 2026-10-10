#!/usr/bin/env python
"""EXP-76 — measure A_U directly (closes the circularity in the cIoU identity).

The review of exp71 flagged that A_U (GT union mass of abstained targeted
samples) was RECOVERED as I_tot/cIoU - S_U - FP_U, which makes the
"I_tot/(S_U+FP_U+A_U) reproduces official cIoU" check circular by
construction.  This pass accumulates A_U directly from the records under the
shipped ensemble rule (sum of the same gt_area quantity exp62's evaluate uses
for abstained targeted samples), so the identity becomes a genuine test.

    python exp/exp76_measure_au.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO, build_rows  # noqa: E402
from exp62_clause_selector import load_record, make_clf  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    st = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    params = dict(C=st["C"])
    t0 = time.time()
    gd_tr = GreferDataset(REPRO / "data/refer", "train")
    by_tr = {r["ref_id"]: r for r in gd_tr.refs}
    feats = []
    for line in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        r = load_record(json.loads(line), gd_tr, by_tr, {}, {})
        if r is not None:
            feats.append(r)
    clf = make_clf("logreg", params)
    clf.fit(np.concatenate([r["X"] for r in feats]),
            np.concatenate([r["y"] for r in feats]))
    print(f"classifier fit ({time.time()-t0:.0f}s)", flush=True)

    rows = build_rows(Path("exp/out_extreme/records_grefcoco.jsonl"), "val", clf, params)
    print(f"val rows {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    S_U = FP_U = A_U = I_tot = 0.0
    G_U = P_sum = 0.0                  # GT mass / predicted-mask mass, predicted-targeted
    tgt_abstained = 0
    for r in rows:
        abstain = any(r["ens"])
        if not abstain and not np.flatnonzero(r["scores"] >= st["p_inc"]).size:
            abstain = True
        if abstain:
            if r["no_target"]:
                continue
            tgt_abstained += 1
            A_U += r["gt_area"]            # the exact quantity exp62 uses
            continue
        sel = np.flatnonzero(r["scores"] >= st["p_inc"])
        pred = r["masks"][sel].any(0)
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        if r["no_target"]:
            FP_U += U
        else:
            I_tot += I
            S_U += U
            G_U += float(r["gt"].sum())
            P_sum += float(pred.sum())
    cIoU = 100.0 * I_tot / (S_U + FP_U + A_U)
    # oracle-selector cap bound (review-prompted): with r_orc = I_orc/U_orc held at the
    # pilot's 0.8842, cIoU_orc = I_orc/(I_orc/0.8842 + X) is maximised at I_orc = G_U
    # (no selector can intersect more than the GT mass on this partition):
    X = FP_U + A_U
    cap_upper = 100.0 * G_U / (G_U / 0.8842 + X) if G_U else 0.0
    # dilution X' needed for the cap to reach the floor: G_U/(G_U/0.8842 + X') = 0.6894
    X_max = G_U * (1 / 0.6894 - 1 / 0.8842) if G_U else 0.0
    out = dict(
        measured_on="full val (14220), shipped ensemble rule, exp62 selector",
        S_U=S_U, FP_U=FP_U, A_U=A_U, I_tot=I_tot, G_U=G_U, P_sum=P_sum,
        targeted_abstained=tgt_abstained,
        cIoU_from_masses=cIoU,
        official_cIoU=34.9419,
        r_delivered=I_tot / S_U,
        oracle_cap_bound_pct=cap_upper,
        dilution_X_max_for_oracle_floor=X_max,
        dilution_cut_needed_pct=100.0 * (1 - X_max / X) if X else 0.0,
        note=("A_U measured directly (sum of the exp62 abstained-targeted gt_area), "
              "not recovered from cIoU; the identity cIoU = I_tot/(S_U+FP_U+A_U) is now "
              "a genuine cross-check against the official 34.9419. G_U = GT union mass "
              "over the PREDICTED-targeted partition (the ceiling on any selector's "
              "intersection there); oracle_cap_bound uses the reviewer-prompted bound "
              "cIoU_orc <= G_U/(G_U/0.8842 + X), not the earlier (incorrect) "
              "r_orc*S_U/(S_U+X) which exceeded the available GT mass"),
        generated="exp76_measure_au.py")
    (OUTF / "gref_au_direct.json").write_text(json.dumps(out, indent=2))
    print(f"S_U={S_U:.0f} FP_U={FP_U:.0f} A_U={A_U:.0f} I_tot={I_tot:.0f} "
          f"G_U={G_U:.0f} P_sum={P_sum:.0f}")
    print(f"cIoU from masses = {cIoU:.4f} (official 34.9419); "
          f"abstained targeted = {tgt_abstained}")
    print(f"oracle cap BOUND = {cap_upper:.2f} (was 65.81 with the invalid formula); "
          f"X_max={X_max:.0f} vs X={X:.0f} -> dilution cut needed "
          f"{100*(1-X_max/X):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
