#!/usr/bin/env python
"""EXP-71 — promote the 3-probe ensemble rule to shipped, from artifacts.

The ensemble (abstain iff any of exists/exists2/exists3 answers no, plus the
no-selection rule) won the held-out train comparison by gIoU (70.00 vs 69.57,
exp69) and improves full-val gIoU 68.37 -> 69.77 at -0.15 cIoU.  This script
rewrites main_grefcoco.json and gref_wall.json so every downstream macro reads
the new operating point.  All numbers are computed from:

  gref_ensemble_roc.json        (exp68: full val frontier)
  gref_rule_showdown.json       (exp69: same-rows train pick)
  gref_ensemble_aggregates.json (exp70: union masses -> exact cIoU requirement)

The cIoU-side requirement is re-derived here because the previous
`realization_needed` field mixed an r-value (delivered) with an
oracle-fraction (needed); this script writes an unambiguous replacement:

  r        = sum I / sum U over PREDICTED targeted samples
  r*       = floor * (S_U + FP_U + A_U) / S_U   (masses held at the measured
             operating point; A_U = GT union mass of abstained targeted)
  oracle_cap = r_oracle * S_U / (S_U + FP_U + A_U)   (cIoU a perfect selector
             would reach on the current pool at the current abstention)
"""
from __future__ import annotations

import json
from pathlib import Path

OUTF = Path("exp/out_final")
CIOU_FLOOR, GIOU_FLOOR = 0.6894, 0.7434


def main() -> int:
    roc = json.loads((OUTF / "gref_ensemble_roc.json").read_text())
    show = json.loads((OUTF / "gref_rule_showdown.json").read_text())
    agg = json.loads((OUTF / "gref_ensemble_aggregates.json").read_text())

    ens_val = agg["rules"]["shipped_ensemble"]["val"]
    ship_val = agg["rules"]["shipped_single_probe"]["val"]
    S_U, FP_U = ens_val["S_U"], ens_val["FP_U"]
    r_del = ens_val["r_delivered"]
    cIoU_val = show["winner_val"]["cIoU"] / 100.0
    I_tot = r_del * S_U
    A_U = I_tot / cIoU_val - S_U - FP_U              # abstained-targeted GT mass
    r_star = CIOU_FLOOR * (S_U + FP_U + A_U) / S_U
    r_oracle_cum = 0.8842                             # deployable fused pool, exp50
    oracle_cap = r_oracle_cum * S_U / (S_U + FP_U + A_U)
    # fraction of (FP_U+A_U) that must disappear for the oracle selector to clear
    # cIoU floor: with f = floor/oracle, need S_U/(S_U+X) >= f  =>  X <= S_U(1-f)/f
    _f = CIOU_FLOOR / r_oracle_cum
    _X_max = S_U * (1 - _f) / _f
    dilution_cut_needed = 1 - _X_max / (FP_U + A_U)

    # ---------------- main_grefcoco.json ----------------
    mp = OUTF / "main_grefcoco.json"
    main = json.loads(mp.read_text())
    old_ship = main.get("shipped", {})
    hist = old_ship.get("history_single_probe") or {
        k: old_ship.get(k) for k in ("cIoU", "gIoU")}
    main["shipped"] = {
        "cIoU": round(100 * cIoU_val, 4),
        "gIoU": round(show["winner_val"]["gIoU"], 4),
        "N_acc": round(show["winner_val"]["N_acc"], 2),
        "T_acc": round(show["winner_val"]["T_acc"], 2),
        "n": 14220,
        "rule": ("three-probe existence ensemble (exists/exists2/exists3, abstain when "
                 "any answers no), plus no-selection abstention and the frozen exp62 "
                 "selector; rule chosen on held-out gRefCOCO train by gIoU in exp67/exp69, "
                 "single full-val pass"),
        "history_single_probe": hist,
        "best_published": old_ship.get("best_published"),
    }
    tm = main.get("tuned_model")
    if tm:
        tm["rows"] = {"cIoU": round(100 * cIoU_val, 4), "gIoU": show["winner_val"]["gIoU"],
                      "N_acc": show["winner_val"]["N_acc"], "T_acc": show["winner_val"]["T_acc"]}
        if not isinstance(tm.get("provenance"), dict):
            tm["provenance"] = {"note": tm.get("provenance")}
        tm["provenance"]["existence_rule"] = (
            "3-probe ensemble, held-out-selected (exp67/exp68/exp69/exp70); history: "
            "single-probe shipped row 35.0884/68.3729")
    mp.write_text(json.dumps(main, indent=2))
    print(f"updated {mp}: shipped {main['shipped']['cIoU']}/{main['shipped']['gIoU']}")

    # ---------------- gref_wall.json ----------------
    wp = OUTF / "gref_wall.json"
    w = json.loads(wp.read_text())
    ex = w["exact_selector_requirement"]
    ex["full_val_shipped"] = {
        "cIoU": round(100 * cIoU_val, 4), "gIoU": show["winner_val"]["gIoU"],
        "n": 14220, "N_acc": show["winner_val"]["N_acc"], "T_acc": show["winner_val"]["T_acc"],
        "note": ("authoritative shipped numbers over the complete val split (exp64 path, "
                 "reproduced by exp69 on identical rows); ensemble rule per exp67-71"),
    }
    ex["history_single_probe_full_val"] = {"cIoU": 35.0884, "gIoU": 68.3729,
                                           "N_acc": 86.3, "T_acc": 79.15}
    ex["m_requirement_full_val"] = {
        "operating_point": {"N_acc": ens_val["N_acc"] / 100.0, "T_acc": ens_val["T_acc"] / 100.0},
        "mean_targeted_IoU_required": round(ens_val["m_required_for_floor"] / 100.0, 4),
        "mean_targeted_IoU_delivered": round(ens_val["m_delivered"] / 100.0, 4),
        "formula": "m = (0.7434*14220 - N_acc*8896) / (T_acc*5324)",
        "history_single_probe": {
            "operating_point": {"N_acc": ship_val["N_acc"] / 100.0,
                                "T_acc": ship_val["T_acc"] / 100.0},
            "mean_targeted_IoU_required": ship_val["m_required_for_floor"] / 100.0,
            "mean_targeted_IoU_delivered": ship_val["m_delivered"] / 100.0},
    }
    ex["cIoU_requirement_full_val"] = {
        "definition": ("masses measured on the full val at the shipped operating point; "
                       "r = sum I / sum U over PREDICTED targeted samples; "
                       "A_U = GT union mass of ABSTAINED targeted samples; "
                       "S_U = union mass of predicted targeted; FP_U = union mass of "
                       "predicted no-target"),
        "S_U": S_U, "FP_U": FP_U, "A_U": round(A_U, 1),
        "r_delivered": r_del,
        "r_required_for_floor": round(r_star, 4),
        "multiple_needed": round(r_star / r_del, 2),
        "oracle_cumulative": r_oracle_cum,
        "oracle_fraction_delivered": round(r_del / r_oracle_cum, 4),
        "oracle_fraction_required": round(r_star / r_oracle_cum, 4),
        "oracle_cap_at_operating_point_cIoU": round(100 * oracle_cap, 2),
        "verdict": (f"the cIoU floor needs r={r_star:.3f} = {100*r_star/r_oracle_cum:.0f}% of the "
                    f"deployable-shortlist oracle ({multiple_needed_str(r_star, r_del)}x the "
                    f"delivered r={r_del:.3f}); even a PERFECT selector at this abstraction mass "
                    f"reaches only cIoU {100*oracle_cap:.1f} < {100*CIOU_FLOOR:.1f}, so cIoU "
                    f"additionally requires ~{100*dilution_cut_needed:.0f}% of the "
                    f"(FP_U+A_U)={FP_U+A_U:.0f} dilution mass to disappear"),
        "supersedes": ("the earlier realization block mixed an r-value (delivered 0.470) with "
                       "oracle-fraction values (0.781/0.686/0.513); this block is unambiguously "
                       "r-denominated and derived from measured masses"),
        "provenance": "exp70 aggregates + exp69 winner; identity checked against official cIoU",
    }
    a3 = w["decomposition"]["axis3_abstention"]
    a3["shipped_combination_full_val_ensemble"] = {
        "N_acc": show["winner_val"]["N_acc"] / 100.0,
        "T_acc": show["winner_val"]["T_acc"] / 100.0,
        "note": "3-probe ensemble rule (shipped); exp67/exp68/exp69"}
    a3["shipped_combination_full_val"] = a3["shipped_combination_full_val_ensemble"]
    a3["history_single_probe"] = {"N_acc": 86.30, "T_acc": 79.15,
                                  "note": "single-probe rule, before exp71 promotion"}
    gd = ex["gIou_gap_decomposition"]
    gd["roc_robustness_ensemble"] = [
        {"point": k, "N_acc": round(v["val"]["N_acc"] / 100.0, 4),
         "T_acc": round(v["val"]["T_acc"] / 100.0, 4),
         "m_required": round(v["val"]["m_required_for_floor"] / 100.0, 4) if v["val"]["m_required_for_floor"] else None,
         "frac_of_k50_oracle_mean": (round(v["val"]["m_required_for_floor"] / 87.52, 4)
                                     if v["val"]["m_required_for_floor"] else None)}
        for k, v in roc["rules"].items()]
    gd["roc_robustness_ensemble_note"] = (
        "full val frontier of the 3-probe ensemble (exp68); m_required is the mean targeted "
        "IoU the gIoU floor demands at that ROC point with the shipped selector's mask quality; "
        "delivered m is 0.485 at every point measured; the best realization ever measured "
        "(perfect GT box, k=15) is 0.686")
    obj = ex.get("reviewer_objection_transfer_bound_gIou")
    if obj is not None:
        obj["gate_result_val"] = {
            "rule_ens_only": roc["rules"]["ens_vote_ge1"]["val"],
            "rule_ens_empty(WINNER)": roc["rules"]["ens_vote_ge1_and_empty"]["val"],
            "frontier_best_val_gIoU": roc["best_val_gIoU"],
            "required_roc_point": roc["required_roc_point"],
            "verdict": roc["verdict"],
        }
        obj["status"] = ("ANSWERED: the distribution-robust ensemble was built, gated on "
                         "held-out train, and measured on the full val frontier; it improves "
                         "the shipped gIoU to 69.77 but NO operating point reaches 74.34 - "
                         "the wall is selector-bound, not existence-bound")
        obj["ship_change"] = ("shipped rule promoted to the ensemble (held-out gIoU winner): "
                              "gIoU 68.37 -> 69.77, cIoU 35.09 -> 34.94; see exp71")
    w["ensembles"] = {
        "artifacts": ["exp/out_final/gref_ensemble_roc.json",
                      "exp/out_final/gref_rule_showdown.json",
                      "exp/out_final/gref_ensemble_aggregates.json"],
        "scripts": ["exp/exp67_ensemble_eval.py", "exp/exp68_ensemble_roc.py",
                    "exp/exp69_rule_showdown.py", "exp/exp70_ensemble_aggregates.py",
                    "exp/exp71_update_ship.py"],
        "probes": ["exp/out_vlm_boxes (exists)", "+ |exists2", "+ |exists3"],
    }
    wp.write_text(json.dumps(w, indent=2))
    print(f"updated {wp}")
    print(f"  r_delivered={r_del:.4f} r_required={r_star:.4f} "
          f"x{r_star/r_del:.2f} oracle_cap={100*oracle_cap:.1f} "
          f"multiple_of_oracle={100*r_star/r_oracle_cum:.0f}%")
    return 0


def multiple_needed_str(r_star, r_del):
    return f"{r_star/r_del:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
