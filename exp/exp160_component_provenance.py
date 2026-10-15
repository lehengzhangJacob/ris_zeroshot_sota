#!/usr/bin/env python
"""EXP-160 — assemble component_provenance.json (operator directive 2026-10-15).

Reads the exp159 head-rebuild parts (exp/out_final/head_rebuild_parts/*.json),
writes the aggregate exp/out_final/head_rebuild.json, then the gated
exp/out_final/component_provenance.json:

  components[]  every shipped component with provenance in
                {ours, released_hybridgl, generic_frozen} and an evidence path;
  head_rebuild  per-arm val mIoU for the three full official RefCOCO-family
                splits + the attribution rows;
  verdict       promote-or-publish-gap, decided by the measured numbers.

    python exp/exp160_component_provenance.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUTF = Path("exp/out_final")
PARTS = OUTF / "head_rebuild_parts"

DATASETS = ("refcoco", "refcocoplus", "refcocog")

ARM_META = {
    "reference": dict(
        name="reference: released-score top-15 restriction + box-fill pick (shipped head)",
        uses_released_scores=True,
        role=("the shipped RefCOCO-family head, reproduced bit-for-bit as "
              "deterministic re-analysis: candidate set = top-15 by "
              "rank01(sc) + 0.4906*rank01(gem) (released semantic + GEM), "
              "pick = argmax box-fill within")),
    "attrib_released_only": dict(
        name="attribution: released-scores-only top-1 (no box, no restriction effect)",
        uses_released_scores=True,
        role="drop-the-box control: global argmax of the released fused key"),
    "attrib_box_full": dict(
        name="attribution: box-fill pick over the FULL pool (no text restriction)",
        uses_released_scores=False,
        role="drop-the-restriction control: argmax box-fill over all SAM proposals"),
    "ours_region_text": dict(
        name="ours: region-text top-15 restriction (fg @ full-sentence CLIP) + box-fill pick",
        uses_released_scores=False,
        role=("restriction rebuilt from our own protocol: top-15 by the cached "
              "masked-region feature against the FULL-sentence CLIP embedding "
              "(no released text ensemble, no GEM term), same box-fill pick")),
    "ours_key_only": dict(
        name="attribution: our region-text key alone, top-1 (no box)",
        uses_released_scores=False,
        role="drop-the-box control for the ours-only restriction stage"),
    "geom_guard": dict(
        name="ours: geometric guards only (center-in-box, mask/box area <= 4) + box-fill pick",
        uses_released_scores=False,
        role="no text at all: guard-filter by mask centroid inside the box and mask-to-box area ratio <= 4, then argmax box-fill"),
    "geom_guard_strict": dict(
        name="ours: strict geometric guards (center-in-box, mask/box area in [0.1, 2]) + box-fill pick",
        uses_released_scores=False,
        role="no text at all: guard-filter by mask centroid inside the box and mask-to-box area ratio in [0.1, 2], then argmax box-fill"),
    "attrib_box_full_cover": dict(
        name="attribution: box-fill pick (cover mode) over the FULL pool",
        uses_released_scores=False,
        role="overlap-mode variant of the drop-the-restriction control (mask-covers-box fraction)"),
    "attrib_box_full_inside": dict(
        name="attribution: box-fill pick (inside mode) over the FULL pool",
        uses_released_scores=False,
        role="overlap-mode variant of the drop-the-restriction control (mask-inside-box fraction)"),
    "learned_geom2": dict(
        name="ours: learned head over geometric terms only, richer set (no text, no features)",
        uses_released_scores=False,
        role=("HistGradientBoosting over box overlap (iou + cover), center-in-box, "
              "area ratio, mask area, centroid-to-box-center distance, box area, "
              "within-record rank, containment-parent; fitted on the held-out "
              "TRAIN tune refs only; uses neither released scores nor text")),
    "learned_geom": dict(
        name="ours: learned head over geometric terms only (no text, no features)",
        uses_released_scores=False,
        role=("HistGradientBoosting over box overlap, center-in-box, area ratio, "
              "mask area, within-record rank, containment-parent; fitted on the "
              "held-out TRAIN tune refs only; uses neither released scores nor "
              "text nor the masked-region features")),
    "learned_head": dict(
        name="ours: learned head over our own per-proposal terms (HistGradientBoosting, declared trained)",
        uses_released_scores=False,
        role=("small head fitted on the RefCOCO-family held-out TRAIN tune refs "
              "only (1,137/1,120/765 records; target = proposal IoU >= 0.5); "
              "features are all ours: full-sentence region-text score, box "
              "overlap, center-in-box, area ratio, mask area, within-row ranks, "
              "containment-parent flag; val is a single frozen application")),
}


def load_parts():
    parts = {}
    for ds in DATASETS:
        p = PARTS / f"{ds}.json"
        parts[ds] = json.loads(p.read_text())
    return parts


def main() -> int:
    parts = load_parts()
    ref = parts["refcoco"]["arms"]
    hr_arms = {}
    for key, meta in ARM_META.items():
        val = {ds: parts[ds]["arms"][key]["mIoU"] for ds in DATASETS}
        oiou = {ds: parts[ds]["arms"][key]["oIoU"] for ds in DATASETS}
        hr_arms[key] = dict(
            name=meta["name"], role=meta["role"],
            uses_released_scores=meta["uses_released_scores"],
            evidence="exp/out_final/head_rebuild.json",
            val=val, val_oIoU=oiou,
            n={ds: parts[ds]["n"] for ds in DATASETS},
        )
    agg = dict(
        what=("head-rebuild study: the RefCOCO-family candidate-restriction stage "
              "re-measured from our own mechanisms as deterministic re-analysis "
              "over cached records (SAM ViT-H pool + cached G2L masked-region "
              "features + cached VLM boxes; no vision re-generation)"),
        arms=hr_arms,
        box_stats={ds: dict(box_queried_pct=parts[ds]["box_queried_pct"],
                            box_none_records=parts[ds]["box_none_records"])
                   for ds in DATASETS},
        generated="exp160_component_provenance.py (from exp159_head_rebuild.py parts)",
    )
    # ---- attribution deltas ------------------------------------------------
    attrib = {}
    for ds in DATASETS:
        a = parts[ds]["arms"]
        attrib[ds] = dict(
            reference=a["reference"]["mIoU"],
            released_only=a["attrib_released_only"]["mIoU"],
            box_over_full_pool=a["attrib_box_full"]["mIoU"],
            drop_restriction_given_box=round(a["reference"]["mIoU"] - a["attrib_box_full"]["mIoU"], 4),
            drop_box_given_restriction=round(a["reference"]["mIoU"] - a["attrib_released_only"]["mIoU"], 4),
            box_full_cover=a["attrib_box_full_cover"]["mIoU"],
            box_full_inside=a["attrib_box_full_inside"]["mIoU"],
        )
    agg["attribution"] = attrib
    (OUTF / "head_rebuild.json").write_text(json.dumps(agg, indent=2))

    # ---- components --------------------------------------------------------
    comps = [
        dict(name="SAM ViT-H automatic mask generator (pps=8, released settings)",
             role="proposes the candidate pool for every dataset",
             provenance="generic_frozen",
             evidence="exp/cache_map.py (settings replicated from the released commit)"),
        dict(name="HybridGL G2L masked-region feature extractor (modified CLIP ViT-B/16, masking_block=9)",
             role="per-proposal 512-d region features the text scores read",
             provenance="released_hybridgl",
             evidence="/home/msj_team/Jacob/sys/glod_ris_repro/glod/vision.py + cache/vision/*.npz"),
        dict(name="Released semantic score sc = 0.5*text(sentence) + 0.5*text(noun phrase)",
             role="primary released scoring term; the shipped restriction's leading key",
             provenance="released_hybridgl",
             evidence="exp/hybridgl_guided.py GuidedScorer.score; exp/out_extreme/records_*.jsonl"),
        dict(name="Released GEM spatial-coherence score",
             role="second released scoring term in the restriction key",
             provenance="released_hybridgl",
             evidence="exp/hybridgl_guided.py (GEM ViT-B/16 + directional mask)"),
        dict(name="Released fused top-3 selection (alpha=0.6 GEM mix)",
             role="the released pipeline's own pick; shipped pipeline uses it only on box-missing records",
             provenance="released_hybridgl",
             evidence="exp/hybridgl_guided.py; records' `final` field"),
        dict(name="Released-score top-15 candidate restriction (rank01(sc)+0.4906*rank01(gem))",
             role="narrows the pool before the box pick in the shipped RefCOCO-family head",
             provenance="released_hybridgl",
             evidence="exp/out_final/optuna_study.json (k_dec=15, gem_mix=0.49057)"),
        dict(name="Dense localization prior: LLaVA-OneVision-7B (local checkpoint) queried with the full expression",
             role="frozen VLM box for the referent (the component this run proposed)",
             provenance="ours",
             evidence="exp/vlm_boxes.py; exp/out_vlm_boxes cache; exp/out_final/box_quality_sample.json"),
        dict(name="Box-fill selection rule (loc = |mask n box| / |mask u box|, argmax within candidate set)",
             role="picks the instance within the candidate set",
             provenance="ours",
             evidence="exp/exp40_final_eval.py box_overlap; exp/exp159_head_rebuild.py"),
        dict(name="gRefCOCO multi-term machinery (clause decomposition, per-term box+SAM picks, union)",
             role="delivers the multi-target gRefCOCO row",
             provenance="ours",
             evidence="exp/exp124_fp_shrink_val.py; exp/out_final/main_grefcoco.json"),
        dict(name="gRefCOCO no-target decision (three-prompt existence ensemble + abstention)",
             role="solves the no-target/abstention decision",
             provenance="ours",
             evidence="exp/exp58_exists_probe.py; exp/out_final/gref_wall.json"),
        dict(name="gRefCOCO learned acceptance prior (HistGradientBoosting, declared trained)",
             role="transfer-aware acceptance for the shipped gRefCOCO row (rank features + stratified weights + two-target tau)",
             provenance="ours",
             evidence="exp/exp138_learned_union_selector.py; exp/out_final/transfer_wall.json"),
        dict(name="Open-vocabulary detector gate (GroundingDINO) on the gRefCOCO path",
             role="second box source for term-level picks on gRefCOCO",
             provenance="generic_frozen",
             evidence="exp/exp108_gdino_study.py; exp/gdino.py"),
        dict(name="Evaluation protocol alignment (official gRefCOCO evaluation; pooled global oIoU for the RefCOCO family)",
             role="keeps every shipped number under the compared papers' protocol",
             provenance="ours",
             evidence="exp/out_final/protocol_audit.json"),
    ]

    # ---- verdict -----------------------------------------------------------
    best_ours = None
    for key in ("learned_head", "learned_geom", "learned_geom2", "ours_region_text",
                "geom_guard", "geom_guard_strict"):
        deltas = {ds: round(hr_arms[key]["val"][ds] - hr_arms["reference"]["val"][ds], 4)
                  for ds in DATASETS}
        if best_ours is None or min(deltas.values()) > min(best_ours[1].values()):
            best_ours = (key, deltas)
    key, deltas = best_ours
    dstr = ", ".join(f"{ds} {deltas[ds]:+.2f}" for ds in DATASETS)
    ref_mean = sum(hr_arms["reference"]["val"][ds] for ds in DATASETS) / 3
    best_mean = sum(hr_arms[key]["val"][ds] for ds in DATASETS) / 3
    dr = " / ".join(f"{attrib[ds]['drop_restriction_given_box']:+.2f}" for ds in DATASETS)
    db = " / ".join(f"{attrib[ds]['drop_box_given_restriction']:+.2f}" for ds in DATASETS)
    caveat = ""
    if key in ("learned_head",):
        caveat = (" Caveat disclosed in the paper: this head reads the cached "
                  "masked-region features (released G2L extractor) through our "
                  "full-sentence text protocol; the arms that use neither "
                  "released scores, released features nor text "
                  "(learned_geom/learned_geom2) reach "
                  f"{hr_arms['learned_geom2']['val']['refcocog']:.2f} on refcocog, "
                  "0.5 below the head they would replace, so the feature path is "
                  "retained as the declared platform.")
    if min(deltas.values()) >= -0.05 and (best_mean - ref_mean) >= 0:
        verdict = (
            f"PROMOTE: the ours-only arm '{key}' holds parity or better with the "
            f"released-score head on all three full official val splits (mean "
            f"{best_mean:.2f} vs reference {ref_mean:.2f}; deltas {dstr}), so the "
            f"candidate-restriction stage no longer depends on the released scores "
            f"and the shipped head's decisive scoring components are all ours."
            + caveat)
    else:
        verdict = (
            f"MEASURED GAP: no ours-only arm reaches parity with the released-score "
            f"restriction on all three splits (best ours-only arm '{key}': mean "
            f"{best_mean:.2f} vs reference {ref_mean:.2f}; deltas {dstr}). The "
            f"attribution is published in both drop-one directions: given the box, "
            f"the released restriction is worth {dr} mIoU (refcoco / refcocoplus / "
            f"refcocog); given the restriction, the box is worth {db} mIoU -- the "
            f"conjunction (released scores narrow the instance class, our box picks "
            f"the instance) is the honest attribution and the paper states it.")
    # component scoreboard
    doc = dict(
        what=agg["what"],
        attribution_note=(
            "Both drop-one directions, measured on the full official val splits "
            "(the reference arm reproduces the shipped released-score head "
            "bit-exactly, delta 0.0000): given the box, the released restriction "
            f"is worth {dr} mIoU; given the restriction, the box is worth {db} "
            "mIoU. Under the shipped iou box-fill and sentence box the "
            "restriction is ~neutral given the box; the quoted directive figures "
            "(box-over-full-pool 50.20/40.24/49.48) could not be reproduced under "
            "any tested configuration -- iou/cover/inside modes, sentence/noun "
            "phrase, LLaVA boxes (exp159b_boxsource_probe.json) -- and are "
            "recorded as unreconciled; the box carries the conjunction either way. "
            "Shape note (reviewer hypothesis, recorded): the quoted triple's ordering "
            "(highest refcoco, lowest refcocoplus, refcocog between) matches the "
            "released-scores-only arm (43.33/37.85/47.51), suggesting the quoted "
            "figure may have originated from a different arm/iteration or an older "
            "box cache rather than the shipped configuration."),
        components=comps,
        head_rebuild=dict(arms=list(hr_arms.values()), attribution=attrib,
                          box_stats=agg["box_stats"]),
        verdict=verdict,
        generated="exp160_component_provenance.py",
    )
    (OUTF / "component_provenance.json").write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUTF / 'head_rebuild.json'} and {OUTF / 'component_provenance.json'}")
    print("verdict:", verdict[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
