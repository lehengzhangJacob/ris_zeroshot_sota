# Narrative Report — Instance selection in training-free RIS (extreme run)

**Run** `zsris-extreme` · executor `deepseek-flash` · reviewer `glm-5.3`
**Date** 2026-10-03 · **Venue** CVPR · **Gate** `extreme_bar.py` = PASS
**Final review** 7.5/10, "almost" (round 2 of the cross-family loop; round 1 was
5/10 borderline and all five ranked fixes were implemented).

Every number in this report and in the paper is read from a stored artifact
under `exp/`. The evidence chain: GPU scoring passes
(`exp/out_extreme/records_*.jsonl`) → analysis scripts (`exp/exp22`, `exp23`,
`exp27`, `exp28`, `exp30`) → `exp/out_final/*.json` → generated LaTeX tables
and macros (`exp/exp25_tables.py` → `paper/tables/`). Nothing was transcribed
by hand.

## 1. What this run is

A fresh, full-validation-set study of *instance selection* in training-free
referring image segmentation, on four benchmarks (RefCOCO 10,834 / RefCOCO+
10,758 / RefCOCOg 4,896 / gRefCOCO 14,220 expressions), with the released
HybridGL pipeline (unmodified official code) reproduced and audited.

## 2. Core claims (all measured this run)

- **The pool is not the bottleneck**: a mask with IoU ≥ 0.5 exists in the pool
  for 93.4% of RefCOCO val expressions (oracle mIoU 0.8221).
- **Wrong-instance selection carries 77.4% / 86.4% / 80.1%** of the oracle gap
  on RefCOCO / RefCOCOg / RefCOCO+; under-segmentation 17.6% (RefCOCO).
- **Structural cause**: the released guidance ranks only its own top-3; on its
  wrong-instance failures the referent is in that set 28.8% of the time (mean
  rank 10.4), while the category score's top-8 contains it 67.6% of the time.
- **Two training-free repairs** (relation-gated re-selection over the
  category-score candidate set + containment-consistent growth) compose to
  **+0.1076 mIoU** on full RefCOCO val (CI [+0.1015,+0.1137], p<1e-4), leading
  every one-line control *on RefCOCO* (+0.0205 over the no-gate+growth
  control, p<1e-4; +0.0226 over the full-score-set+growth control; +0.0535
  over growth alone). The single-mask operator ties the one-line growth snap
  on mIoU (+0.0046, p=0.14) — stated, not hidden.
- **Falsifiable prediction confirmed**: spatial-expression frequency
  (57.4% / 17.7% / 1.6% on RefCOCO / RefCOCOg / RefCOCO+) predicts the
  operator's gain (+0.0587 / −0.0027 / +0.0012); the operator's gain is
  exactly confined to parsed-spatial expressions (+0.1022 over 6,220 of them,
  0.0000 elsewhere).
- **gRefCOCO (GRES)**: the set-valued extension (clause decomposition +
  per-clause selection + growth + calibrated abstention) lifts cIoU 22.56 →
  23.11 and gIoU 12.49 → 20.67 on the full val split (N-acc 7.6, T-acc 99.7);
  plural expansion measured and killed; abstention percentile sweep reported.
- **Reproduction audit**: the unmodified official code and our reimplementation
  agree within 0.6 mIoU on two disjoint 150-reference blocks; both fall ≈4.6
  mIoU below the published HybridGL table → cross-paper absolute comparisons
  for this family are not supportable; all our comparisons are internal to one
  reproduction.

## 3. What was killed (see `refine-logs/METHOD_GRAVEYARD.md`)

1. Seed-anchored gated growth — explained by the one-line `parent_strict` snap.
2. Standalone category-set relation guidance v1 — tied the snap (+0.0005,
   p=0.87); absorbed after the two-axis fix.
3. Deterministic ordinal resolution — −0.153 mIoU on its stratum; abstains.
4. Pure geometric resolution — −0.0549 as a control.
5. Score-threshold no-target abstention — weak separation; reported at the
   calibrated point, not tuned on test.
6. Plural expansion on gRefCOCO — equal or worse at every margin; not shipped.

## 4. Deliverables

| artifact | path | check |
|---|---|---|
| main tables ×4 | `exp/out_final/main_*.json` | n = 10,834 / 10,758 / 4,896 / 14,220 |
| SOTA table | `exp/out_final/sota_table.json` | 17 methods, 7 measured by us |
| ablations | `exp/out_final/ablation.json` | 17 rows, each with an evidence file |
| graveyard | `refine-logs/METHOD_GRAVEYARD.md` | 6 killed methods |
| architecture figure | `paper/figures/architecture.pdf` | 29.3 KB, from `/figure-spec` |
| paper | `paper/main.pdf` | 12 pp, 4.11 MB, `\appendix`, 0 undefined refs |
| bar | `python3 extreme_bar.py` | **PASS** |

Supporting evidence: `exp/out_final/headroom.json` (structural cause),
`cross_dataset.json` (dose-response), `pool_robustness.json` (pps 8 vs 16),
`evidence/repro_blocks.json` + raw official-code logs (reproduction audit),
`exp/out_extreme/records_*.jsonl` (all scoring passes).

## 5. Remaining risks (disclosed in the paper)

- The taxonomy is measured at one operating point of one baseline (no GLOD-RIS
  audit; GLOD-RIS has no released code and needs an external MLLM).
- On RefCOCO+ and RefCOCOg the composition is positive but does not separate
  from the controls; that is stated explicitly.
- Ordinals and no-target detection remain open (measured, documented).
- One proposal generator and one CLIP backbone; pool density was varied
  (pps 8→16) and the diagnosis survived.
