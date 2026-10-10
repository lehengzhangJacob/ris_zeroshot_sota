# Final Proposal

**Run**: `zsris-20261001` · executor `deepseek-flash` · reviewer `glm-5.3`
**Date**: 2026-10-02
**Status**: analysis-scoped contribution, reviewer verdict **PROCEED** (conditional)

---

## 1. Problem Anchor (frozen — do not drift)

Training-free / zero-shot referring image segmentation is universally implemented as
*propose-then-select*: a frozen SAM emits a candidate pool, a frozen CLIP scores each
candidate against the expression, and the pipeline returns `argmax`. Every published
method in the local library plus the two downloaded surveys does exactly this, and every
one of them tunes *global constant* fusion weights.

**Anchored question**: when a training-free RIS pipeline fails, is the failure in the
proposal pool or in the selector, and is the selector's loss fixable by re-weighting the
cues it already has?

**Anchored answer (measured)**: the pool is fine, the selector is the problem, and
re-weighting cannot fix it — because the region-text score is nearly flat with respect to
true IoU. This is the contribution.

## 2. Method Thesis

There is no method. The deliverable is a **diagnostic** with five measured findings and a
set of negative results that close off entire families of fixes. The one method-shaped
result that emerged (seed-anchored gated growth) was killed by its own control: a one-line
containment snap reproduces 98 % of its gain, and the gain is concentrated entirely in
large-GT expressions, zero on small ones, and negative on part expressions.

Stating this plainly, with the control that killed it, is more useful to the field than
the method would have been — and it is what the reviewer accepted.

## 3. Baseline Selection (with justification)

**Chosen baseline: HybridGL (CVPR 2025), official code `github.com/fhgyuanshen/HybridGL`.**

Rationale:

1. **It is the only strong zero-shot RIS method in this comparison set with complete,
   official, runnable code.** Verified against every alternative:
   - GLOD-RIS (the reference paper): repo `Wujj440/GLOD-RIS` is a **README placeholder,
     0 `.py` files**. The six tarballs in `sys/RIS/*.tar.gz` are **LaTeX paper sources
     only, 0 `.py` files** (CaR, HybridGL, IteRPrimE, LGD, RefDiff, TAS) — the brief's
     "code archives" are not code.
   - BMS: repo is a placeholder (README + 1 PDF). TAS, LGD, CaR, MCCE-REC: no official code.
   - Ref-Diff (82 `.py`) and IteRPrimE (69 `.py`) do have code, but Ref-Diff needs Stable
     Diffusion v1.5 plus a bespoke `vitb16_stego_head.ckpt`, and IteRPrimE reports **mIoU
     only** for the RefCOCO series (no oIoU) with a self-contradictory λ (0.8 in the
     implementation section vs 0.2 in its own ablation).
2. **It is GLOD-RIS's direct ancestor.** GLOD-RIS's visual feature extractor *is*
   HybridGL's `ϕGL_image` (G2L, ViT-B/16), and GLOD-RIS's "Keyword-Gradient" ablation
   baseline *is* HybridGL's spatial module. Gains and diagnostics transfer.
3. **It runs fully offline** — SAM ViT-H + CLIP ViT-B/16 + GEM, no external API. GLOD-RIS
   requires a ZhipuAI GLM-4.1V call per expression, and the local reproduction shows that
   dependency failing in production (aborted at 223/10,834 expressions with
   `GLM invalid response: Grid must contain exactly n*n binary values`).
4. **Its own ablation supplies the honest testbed.** HybridGL reports RefCOCO val
   "L alone" 24.65 oIoU / 32.64 mIoU, "G2L" 31.71 / 40.27, full model 41.81 / 49.48. The
   unguided scorer used here (mIoU 0.3501 / oIoU 0.2708) sits between the L-alone and G2L
   rows, so the testbed is a faithful unguided operating point rather than a broken pipeline.

**Why not GLOD-RIS as the baseline**: no official code, an external API dependency that
provably fails, and a 3×3 grid whose granularity ablation (19.35 / 50.85 / 25.13 for
2×2 / 3×3 / 4×4) is a prompt-format cliff rather than a granularity trade-off. It remains
the reference point the diagnostic is motivated by, not the substrate it runs on.

## 4. Dominant Contribution

**A pool-level diagnostic of selection failure in training-free zero-shot RIS**, with
five findings and six measured negative results. The two findings with no prior art found:

- **Flatness**: over a SAM proposal pool, CLIP region-text cosine separates an IoU-0.00
  mask from an IoU-0.93 mask by ~0.01; Spearman(score, IoU) = 0.22; the best mask sits at
  score-rank 4.7.
- **Bounded composition**: the best-union-of-2 oracle exceeds the best-single-proposal
  oracle by only +2.5 mIoU, which structurally bounds every "merge/grow/compose
  proposals" family — including the published MARS gate+union template, which scores
  below plain argmax on RIS (-0.011).

## 5. Claims and Evidence

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | The proposal pool is not the bottleneck | single-proposal oracle mIoU 0.8285 ; **scorer-independent** | measured |
| C2 | The standard selector discards most of the achievable IoU | argmax 0.3501 = 42.3 % of oracle, at the unguided operating point | measured |
| C3 | The region-text score is near-flat w.r.t. IoU | ~0.01 cosine gap between IoU 0.00 and 0.93; Spearman 0.22; rank-of-best 4.7 | measured |
| C4 | No re-weighting of existing cues can close the gap | global linear sweep over 7 cues caps at 0.3798 = 46 % of oracle | measured |
| C5 | The composition/union family is structurally bounded | union-of-2 oracle 0.8511 vs single 0.8265 → +2.5 mIoU | measured |
| C6 | The published cross-regime transfer fails | MARS gate+union 0.3387 < argmax 0.3501 | measured |
| C7 | A one-line containment snap explains the apparent method gain | `parent_strict` +0.0654 vs growth +0.0670 (98 %) | measured |

## 6. Must-Not-Claim (from the reviewer, recorded verbatim)

> "that the 42 %-oracle-recovery / flat-selection pathology characterizes training-free RIS
> at the SOTA operating point — every flatness, ceiling, and headroom number is measured on
> the guidance-stripped scorer (0.3501 vs 0.4948) and must be scoped as 'at the unguided
> operating point matched to HybridGL's L-alone ablation', never generalized to guided
> pipelines or stated as a field-wide result."

Additional prohibitions:

- Do **not** claim a novel merge/growth algorithm. The operator is the connected component
  of the argmax seed in the overlap graph of gated masks.
- Do **not** present 0.3501 → 0.4116 as a headline result. It is a granularity correction.
- Do **not** claim scorer-agnosticism: only two scorers were tested (plain G2L and DOD),
  both CLIP ViT-B/16.
- Do **not** claim the per-expression re-weighting ceiling (0.5215) as achievable — it is
  the maximum over 972 discrete settings and is inflated.

## 7. Compute Envelope

Total pilot spend: **~2.5 GPU-hours** across 4× RTX 4090 (nine pilots, all on one card at
a time, GPU 1 avoided as another process held it). Remaining plan: ~1 GPU-day for EXP-2,
~2 for EXP-3. Well inside the 8 GPU-hour pilot budget for what was run; the proposed
plan is deliberately hours-scale, not days-scale.
