#!/usr/bin/env python
"""EXP-162 — land the component-provenance promotion (operator directive 2026-10-15).

The head-rebuild study (exp159) measured an ours-only arm with parity or
better on all three full official RefCOCO-family val splits
(+2.03/+1.69/+0.43 mIoU over the released-score head, which it reproduces
bit-exactly): the learned selector head.  Per the directive this arm is
promoted into the shipped head.  This script lands the cascade:

  * main_refcoco/refcocoplus/refcocog.json — shipped = learned head (declared
    trained), rows/oIoU gain a learned_head entry, the training-free
    realization stays as the reference;
  * sota_table.json — the box-prior row is relabelled the training-free
    reference and the learned head is added as the shipped row;
  * test_columns.json — test columns re-evaluated under the learned head
    (exp161); the training-free test numbers are kept as fields;
  * protocol_audit.json — additive shipped_learned_head block per dataset.

    python exp/exp162_write_promoted_head.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUTF = Path("exp/out_final")
PARTS = OUTF / "head_rebuild_parts"
DATASETS = ("refcoco", "refcocoplus", "refcocog")
MAIN = {ds: OUTF / f"main_{ds}.json" for ds in DATASETS}

HEAD_RULE = (
    "learned selector head (DECLARED TRAINED): HistGradientBoostingClassifier "
    "(max_depth 3, 250 iterations, lr 0.08, seed 0) over eight of-ours "
    "per-proposal terms (full-sentence region-text score on the cached "
    "masked-region features; box-overlap IoU; center-in-box; log mask/box "
    "area ratio; log mask area; within-record ranks of the region-text score "
    "and of box overlap; containment-parent flag); fitted on the "
    "RefCOCO-family held-out TRAIN tune refs only (1,137 / 1,120 / 765 "
    "records); official val and test are evaluation-only; the head uses no "
    "released HybridGL scores -- the restriction stage is rebuilt from our "
    "own terms (exp/out_final/head_rebuild.json, "
    "exp/out_final/component_provenance.json)")
TF_RULE = ("released-score top-15 restriction (rank01(sc) + 0.4906*rank01(gem)) "
           "+ box-fill pick -- the training-free realization, reproduced "
           "bit-exactly as the exp159 reference arm")


def load_parts():
    return {ds: json.loads((PARTS / f"{ds}.json").read_text()) for ds in DATASETS}


def main() -> int:
    parts = load_parts()
    learned, ref = {}, {}
    for ds in DATASETS:
        a = parts[ds]["arms"]
        learned[ds] = a["learned_head"]
        ref[ds] = a["reference"]

    # ---- 1. main jsons -----------------------------------------------------
    for ds in DATASETS:
        m = json.loads(MAIN[ds].read_text())
        m["rows"]["learned_head"] = learned[ds]["mIoU"]
        m.setdefault("oIoU", {})["learned_head"] = learned[ds]["oIoU"]
        m["mIoU_training_free_reference"] = ref[ds]["mIoU"]
        m["oIoU_training_free_reference"] = ref[ds]["oIoU"]
        m["mIoU"] = learned[ds]["mIoU"]
        m["shipped"] = dict(
            mIoU=learned[ds]["mIoU"], oIoU=learned[ds]["oIoU"],
            best_published=(m.get("shipped") or {}).get("best_published"),
            rule=HEAD_RULE,
            training_free_reference=dict(mIoU=ref[ds]["mIoU"], oIoU=ref[ds]["oIoU"],
                                         rule=TF_RULE),
            arm_selection=("head fitted on the held-out TRAIN tune refs; official "
                           "val applied once per frozen configuration "
                           "(exp159_head_rebuild.py)"),
            evidence="exp/out_final/head_rebuild.json",
            provenance=("component-provenance promotion (operator directive "
                        "2026-10-15); see exp/out_final/component_provenance.json"))
        m["head_promotion_note"] = (
            "2026-10-15: the shipped head is the learned selector (declared "
            "trained). The previous shipped head (released-score restriction + "
            "box, training-free) is reproduced bit-exactly and retained as the "
            "training-free reference in rows/oIoU/shipped.training_free_reference.")
        MAIN[ds].write_text(json.dumps(m, indent=2))
        print(f"main {ds}: shipped learned {learned[ds]['mIoU']} "
              f"(training-free ref {ref[ds]['mIoU']})")

    # ---- 2. sota_table -----------------------------------------------------
    st = json.loads((OUTF / "sota_table.json").read_text())
    methods = st["methods"]
    # idempotence: drop any learned-head row from an earlier run before inserting
    methods[:] = [m for m in methods
                  if not m.get("name", "").startswith("Ours: learned selector head")]
    idx = None
    for i, meth in enumerate(methods):
        if meth.get("name", "").startswith("Ours: dense VLM box prior"):
            idx = i
            break
    assert idx is not None, "box-prior row not found in sota_table.json"
    methods[idx]["name"] = "Ours: dense VLM box prior (training-free reference)"
    methods[idx]["setting"] = ("zero-shot / training-free (frozen SAM+CLIP+VLM); "
                              "reference row -- the shipped head is the learned "
                              "selector row below")
    row = dict(
        name="Ours: learned selector head on RefCOCO family (shipped; declared trained)",
        venue="this run", measured_by_us=True,
        evidence="exp/out_final/head_rebuild.json",
        setting=("declared trained: head fitted on RefCOCO-family held-out TRAIN "
                 "tune rows only; frozen SAM + VLM + cached masked-region features; "
                 "official val/test evaluation-only"),
        oIoU_source=("aligned pooled re-evaluation (exp159_head_rebuild.py; "
                     "protocol_audit.json)"),
        **{ds: dict(mIoU=learned[ds]["mIoU"], oIoU=learned[ds]["oIoU"])
           for ds in DATASETS})
    methods.insert(idx + 1, row)
    (OUTF / "sota_table.json").write_text(json.dumps(st, indent=2))
    print(f"sota_table: inserted learned-head shipped row after index {idx}")

    # ---- 3. test_columns ---------------------------------------------------
    tc_path = OUTF / "test_columns.json"
    learned_tc_path = OUTF / "test_columns_learned.json"
    if learned_tc_path.is_file():
        ltc = json.loads(learned_tc_path.read_text())["splits"]
        tc = json.loads(tc_path.read_text())
        changed = []
        for key, entry in tc["splits"].items():
            r = ltc.get(key)
            if r is None:
                continue
            entry["training_free_mIoU"] = entry["mIoU"]
            entry["training_free_oIoU"] = entry["oIoU"]
            entry["head"] = "learned selector head (shipped; declared trained)"
            entry["mIoU"] = r["arms"]["learned_head"]["mIoU"]
            entry["oIoU"] = r["arms"]["learned_head"]["oIoU"]
            entry["delta_mIoU"] = round(entry["mIoU"] - entry["published"]["mIoU"], 4)
            entry["delta_oIoU"] = round(entry["oIoU"] - entry["published"]["oIoU"], 4)
            entry["learned_head_evidence"] = f"exp/out_test_columns/learned_{key}.json"
            changed.append(key)
        wins = sum(1 for e in tc["splits"].values() if e["delta_mIoU"] > 0)
        wins_o = sum(1 for e in tc["splits"].values() if e["delta_oIoU"] > 0)
        wins_tf = sum(1 for e in tc["splits"].values() if e.get("training_free_mIoU", 0) > e["published"]["mIoU"])
        wins_tf_o = sum(1 for e in tc["splits"].values() if e.get("training_free_oIoU", 0) > e["published"]["oIoU"])
        n = len(tc["splits"])
        tc["pareto_ledger"] = (
            f"Scoped precisely: the shipped method (learned selector head, declared "
            f"trained; 2026-10-15 component-provenance promotion) strictly exceeds "
            f"the published GLOD-RIS row on {wins}/{n} test-column mIoU values and "
            f"{wins_o}/{n} test-column oIoU values; the training-free realization "
            f"exceeds {wins_tf}/{n} and {wins_tf_o}/{n} respectively and remains "
            f"reported as the zero-shot reference row. Published numbers are "
            f"transcribed from GLOD-RIS Table I test columns (oIoU blocks verified "
            f"against the known val oIoU before transcription); our columns are "
            f"aligned re-evaluations (exp161/exp159, protocol_audit.json). GLOD-RIS "
            f"does not report gRefCOCO, so our gRefCOCO results are compared there "
            f"against the same-class floors (PSALM ECCV 2024 Table 12: 42.0 cIoU / "
            f"43.3 gIoU; SELF1E-8B CVPR 2026 arXiv:2603.19026: 44.4 cIoU) and the "
            f"run's own prior record, per the 2026-10-08 operator standard. Beyond "
            f"the mIoU/oIoU table, the ledger carries reproducibility (their "
            f"published numbers do not reproduce on our independent implementation: "
            f"-25.7/-23.1/-20.6 mIoU vs the val columns, glodris_repro.json) and "
            f"comparison protocol (GLOD-RIS releases no code; our faithful "
            f"reimplementation of the described method does not reach the published "
            f"numbers, so every comparison quotes their rows as published and "
            f"cross-harness) and deployability (our path is fully local). One axis is not dominated "
            f"and stays declared: the gRefCOCO path adds a second frozen detector "
            f"next to the VLM prior.")
        tc["head_promotion_note"] = (
            "2026-10-15: test columns re-evaluated under the promoted learned head "
            "(exp161, deterministic re-analysis of the same cached records); the "
            "training-free numbers are retained as training_free_mIoU/oIoU fields.")
        tc_path.write_text(json.dumps(tc, indent=2))
        print(f"test_columns: updated {len(changed)} splits "
              f"(wins learned {wins}/{wins_o}, training-free {wins_tf}/{wins_tf_o})")
    else:
        print("WARNING: test_columns_learned.json missing; test columns untouched")

    # ---- 4. protocol_audit -------------------------------------------------
    pa = json.loads((OUTF / "protocol_audit.json").read_text())
    for ds in DATASETS:
        e = pa["datasets"][ds]
        e["shipped_learned_head"] = dict(
            mIoU=learned[ds]["mIoU"], oIoU=learned[ds]["oIoU"],
            note=("2026-10-15 component-provenance promotion: the shipped head is "
                  "the learned selector; the re_evaluation block above documents "
                  "the training-free realization, reproduced bit-exactly as the "
                  "exp159 reference arm"))
    pa["head_promotion_note"] = (
        "2026-10-15: the shipped numbers on the three RefCOCO-family datasets are "
        "the learned-head columns (shipped_learned_head); every number remains under "
        "the same audited protocol (mIoU = mean per-expression IoU; oIoU = global "
        "pooled), re-measured by exp159/exp161 as deterministic re-analysis of the "
        "same saved records.")
    (OUTF / "protocol_audit.json").write_text(json.dumps(pa, indent=2))
    print("protocol_audit: shipped_learned_head added for three datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
