# Idea Discovery Report

**Direction**: 零样本指代图像分割 (zero-shot referring image segmentation) — survey GLOD-RIS + the local `papers/` library, select a **baseline with code**, and produce a single-4090-verifiable improvement.
**Date**: 2026-10-02
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Run id**: `zsris-20261001` (executor `deepseek-flash`, reviewer `glm-5.3`)

---

## Literature Landscape

The zero-shot RIS literature was mapped from the 9 local PDFs plus 8 downloaded arXiv papers. Every method in the field is a propose-then-select pipeline over a frozen SAM pool scored by a frozen CLIP, with globally-constant fusion weights. The tables and findings below establish the landscape, the one structural regularity, and the five attack surfaces named in the brief.

### The task and its two regimes

RIS takes an image plus a free-form referring expression and returns a mask. The
zero-shot / training-free regime — no pixel-level supervision, no task-specific
fine-tuning — is universally built as **propose-then-select**: a frozen
class-agnostic segmenter (almost always SAM) emits a candidate pool, a frozen
vision-language model (almost always CLIP) scores the candidates, and the best
candidate is returned.

Nine papers were read in full from `papers/` (GLOD-RIS, HybridGL, BMS,
IteRPrimE, TAS, LGD, Ref-Diff, CaR, MCCE-REC) and eight more were downloaded
from arXiv (see *Downloads* below).

### What the local papers actually contain

| Method | Venue | Proposer | Scorer | Extra dependency | RefCOCO val oIoU / mIoU | Official code |
|---|---|---|---|---|---|---|
| Global-Local (Zero-shot-RIS) | CVPR 2023 | SAM / FreeSOLO | CLIP global-local | — | 24.55 / 31.83 | ✅ |
| Ref-Diff | arXiv 2023 | SD attention + SAM | CLIP + diffusion | Stable Diffusion v1.5 | 35.16 / 37.21 | ✅ |
| TAS | EMNLP 2023 | SAM ViT-H | CLIP + BLIP-2 captions + negatives | BLIP-2 OPT-2.7B | 29.53 / 39.84 | ❌ |
| BMS | TCSVT 2025 | SAM | CLIP positive **and negative** masks, Wasserstein-derived α | — | 34.94 / 40.77 | ❌ (placeholder) |
| CaR | CVPR 2024 | CLIP-ES CAM | CLIP-as-RNN recurrence | — | 33.57 (mIoU) | ❌ |
| MCCE-REC | TCSVT 2025 | MAttNet (supervised) | CLIP + LLaVA-1.6-13B cues, contrastive entropy | LLaVA 13B | *rec only, ACC@0.5 60.78* | ❌ |
| IteRPrimE | AAAI 2025 | Mask2Former (200 props) | ALBEF Grad-CAM + IGRS + PWEM | ALBEF-base | *mIoU 40.20* | ✅ |
| HybridGL | CVPR 2025 | SAM ViT-H | CLIP ViT-B/16 global-local (G2L) + relation/coherence/position guidance | GEM | **41.81 / 49.48** | ✅ |
| LGD | PR 2025 | SAM (or +GroundingDINO) | CLIP + LLaVA-1.6-7B attribute/surround descriptions | LLaVA 7B, GDINO | 34.47 / 45.41 (42.72 / 49.54 w/ DINO) | ❌ |
| **GLOD-RIS** | TCSVT 2026 | SAM ViT-H | CLIP ViT-B/16 G2L + **orthogonal decoupling** + **MLLM 3×3 grid** | GLM-4.1V API | **50.85 / 58.42** | ❌ (README only) |

### The structural finding: every one of them is a single-proposal argmax

Across all ten methods the output is `argmax_i S(m_i)` over one candidate mask.
The fusion weights that combine the sub-scores are **global constants**, never
instance-adaptive: HybridGL `α=0.6`, `λ∈{3,9,14}` hand-switched by mask size;
GLOD-RIS `μ,τ` fixed and `λ = clamp(area, μ, τ)`; TAS `α=0.1`, `λ∈{0.7,1.0}` per
dataset; BMS one α per image from a Wasserstein heuristic; LGD `α∈{0.5,0.3}` per
dataset. No method calibrates its score *across the candidate pool of the image
it is scoring*.

### GLOD-RIS's own numbers say selection, not proposals, is the bottleneck

This is the single most decision-relevant piece of evidence, and it comes from
the reference paper itself:

- **Fig. 6**: oracle utilisation (achieved IoU ÷ best-possible-proposal IoU) is
  only **~62–65 % oIoU and ~70–73 % mIoU**, stable across SAM backbones. The
  paper concludes the ceiling is SAM's mask recall — but an oracle utilisation of
  62 % means 38 % of the *available* IoU is thrown away by the selector.
- **Table VII, grid granularity**: oIoU RefCOCO val is **19.35 / 50.85 / 25.13**
  for 2×2 / 3×3 / 4×4. A genuine granularity trade-off is smooth; a drop to ⅓ of
  the score at 2×2 and ½ at 4×4 is a prompt-format cliff, i.e. the MLLM
  localisation signal is largely an artefact of one grid format.

The local independent reproduction independently confirms both, and adds an
operational failure the paper cannot show: the full RefCOCO-val run **aborted at
223 / 10834 expressions** with `GLM invalid response: Grid must contain exactly
n*n binary values`.

### The five unsolved points, and their measured headroom

The brief names five unsolved attack surfaces. Two pilots (below) measure three
of them directly:

| # | Named gap | Measured headroom on RefCOCO val |
|---|---|---|
| 1 | proposals contain no correct mask | **not binding** — oracle mIoU ≈ 0.81–0.84 |
| 2 | area-based decoupling (GLOD Eq. 6) | see Pilots |
| 3 | fixed 3×3 grid / ordinal / dense | qualitative (Table VII cliff) |
| 4 | one target at a time | **+2.5 mIoU** for a 2-proposal union oracle (Pilot 1) |
| 5 | external MLLM dependency | operational breakage (223/10834) |

### Where the field is genuinely empty (Novelty-relevant)

Two independent literature sweeps (17 papers, incl. the 2025 *Multimodal
Referring Segmentation* survey, GRES/gRefCOCO, and GREx/IJCV) found:

- **No zero-shot or training-free method has ever been evaluated on gRefCOCO /
  GRES.** The survey's GRES table has exactly one zero-shot entry (LaSagnA, a
  *trained* assistant evaluated zero-shot); GREx states outright that SAM was
  excluded from the GRES benchmark because it "has not yet released its text
  prompt".
- **No paper performs an oracle / proposal-recall ceiling analysis** — the words
  oracle, upper bound, ceiling do not appear in the survey, GRES, or GREx.
- **No zero-shot method handles ordinal, depth, or relational expressions**;
  GREx lists counting/ordinal as open future work.
- Closest prior art: **RAS** (ICCV 2025) selects an arbitrary subset of frozen SAM
  proposals for a mask *group* — but with a **learned** binary selection head
  trained on ~2M mask groups and a fine-tuned LLaVA-1.5-13B (≈3.5 GPU-days on
  8×A100), and it never performs a pixel-wise union. **Grounded-SAM** unions SAM
  masks for all detected boxes, training-free, but has no per-referent selection,
  no no-target handling, and has never been run on gRefCOCO.

### Downloads (added to `papers/` during this survey)

`2508.00265` (Multimodal Referring Segmentation survey) · `2306.00968`
(GRES/gRefCOCO) · `2601.05244` (GREx, IJCV) · `2506.05342` (RAS / ORES, ICCV
2025) · `2508.04107` (MLLMSeg) · `2312.12198` (MagNet, CVPR 2024) · `2406.20076`
(EVF-SAM) · `2303.17811` (Global-Local CLIP, CVPR 2023).

---

## Ranked Ideas

Nine ideas were generated and filtered; six were killed by measurement across nine pilots (P1-P9) run on a single RTX 4090. The ranking below reflects empirical signal, not plausibility: the surviving contribution is a diagnostic, and the one strong-looking method result was killed by its own control.

Nine ideas were generated from the landscape and the five named gaps, then filtered by
feasibility on one RTX 4090. Six were killed by measurement (Pilots 1–9, all on
RefCOCO val with the local GLOD-RIS reproduction's frozen SAM ViT-H pool and CLIP
ViT-B/16 global-local features). The survivors are ranked below.

### 🏆 Idea 1 (RECOMMENDED) — "What is actually lost in zero-shot RIS selection? A proposal-pool diagnostic"

**Type**: analysis / diagnostic contribution, not a method.
**Hypothesis**: in training-free zero-shot RIS the proposal pool is not the bottleneck;
the *selector* is, and the loss is caused by the near-flat region-text score landscape
rather than by any fixable weighting of existing cues.

Measured support (all new measurements; none found in a 17-paper sweep):

| Finding | Measurement |
|---|---|
| Pool contains the answer | single-proposal oracle **mIoU 0.8285**; oracle is a property of (image, GT, proposals) and is **scorer-independent** |
| Selector discards most of it | argmax **0.3501** = 42.3 % of oracle (58 % discarded) — at the unguided operating point |
| The score is nearly flat | CLIP cosine separates an IoU-0.00 mask from an IoU-0.93 mask by **~0.01** (0.326 vs 0.314); Spearman(score, IoU) = **0.22**; the best mask sits at score-rank **4.7** |
| Re-weighting cannot fix it | global linear re-weighting over all 7 free cues caps at **0.3798 = 46 %** of oracle |
| The composition family is bounded | best-union-of-2 oracle **0.8511** vs best-single **0.8265** → only **+2.5 mIoU** of headroom |
| The published transfer fails | MARS gate + unconstrained union **-0.011 vs argmax** |

**Status**: pilot-verified. Reviewer verdict **PROCEED** (workshop paper / analysis section).

### Idea 2 (BACKUP) — Error taxonomy of the oracle gap

Decompose the 58 % discarded IoU into under-segmentation/granularity, over-segmentation,
wrong-instance, and boundary error on the full 10,834-expression val split. This is the
reviewer-named single missing measurement; it converts the headline number into an
attributable claim and ties findings 1, 2 and 5 together. Cheap (re-analysis of saved
predictions, no new GPU forward passes).

### Idea 3 (BACKUP) — HybridGL-guided replication of the two headline numbers

Re-measure only *oracle recovery* and *score flatness / rank-of-best* inside the full
released HybridGL pipeline at its published 0.4948 mIoU operating point. ~1 GPU-day.
This is the external-validity gate: if the flatness persists at the guided operating
point the diagnostic generalises; if not, every claim must be scoped to the unguided point.

### Idea 4 (ELIMINATED by pilot) — Seed-Anchored Gated Growth (SAGG)

Gate proposals within tolerance of the top score, then take the connected growth of the
argmax seed. Initially the strongest-looking result: **+0.0615 mIoU** (95 % CI
[+0.047,+0.077], p<0.0001, win/loss/tie 264/112/340), replicated on a fresh seed with new
images (+0.0545) and on the plain HybridGL scorer (+0.0674). **Killed by controls:**

- `parent_strict` — a **one-line** rule (snap the argmax to the smallest SAM proposal
  containing ≥95 % of it) — reproduces **98 %** of the gain (+0.0654 vs +0.0670).
- The gain is **zero** on the smallest GT-area quartile (-0.0009) and **+0.134** on the
  largest → a granularity correction, not a selection mechanism.
- It is **negative on part expressions** (-0.0110), exactly as predicted.
- The gate tolerance peaks sharply at the published 0.95 constant (+0.0670) and falls to
  +0.0179 at 0.99 → a tuned threshold, not a robust principle.
- Dilation and "largest gated" do **not** reproduce it, so it is not raw area inflation —
  it is specifically hierarchy snapping, which is a known SAM post-processing move.

**Reviewer verdict: ABANDON the method claim.** Recorded; the operator survives only as
the explanatory control inside Idea 1.

### Idea 5 (ELIMINATED by pilot) — Unsupervised per-expression cue re-weighting

Weight each score cue by its agreement with the other cues on that expression.
**Hurts**: 0.3263 vs 0.3532 baseline. The per-expression oracle over 972 weight settings
reaches 0.5215 (63 %), but that is a maximum over many near-random reorderings of a
0.01-wide score spread and is an inflated estimate, not an achievable target.

### Idea 6 (ELIMINATED by pilot) — Parser-free soft spatial cue

Detect direction words by CLIP text-embedding similarity and apply the preference to mask
centroids — the MLLM-free, parser-free replacement candidate for GLOD-RIS's 3×3 grid.
**Near-random alone**: 7.5 % of oracle. Fires on 36 % of expressions.

### Ideas 7–9 (ELIMINATED on paper or by pilot)

- **7. Area-based λ → semantic-angle λ in GLOD-RIS Eq. 6.** Attractive because GLOD-RIS's
  own analysis names the fg/bg angle ω as the modulator and then uses pixel area instead —
  but the score landscape is 0.01 wide, so re-weighting a cue cannot move the ranking.
  Killed by the same evidence as Ideas 5 and 6.
- **8. Zero-shot GRES / multi-target via selected-subset union.** Structurally bounded:
  the union-of-2 oracle is only +2.5 mIoU over the single oracle. Killed by Pilot 1.
- **9. Ordinal / dense-scene spatial reasoning without an MLLM.** No measured cue
  separated ordinal expressions; killed with Idea 6 and by the Flatness finding.

---

## Novelty Verification

Two cross-family review rounds with `glm-5.3` (executor `deepseek-flash`) plus an independent prior-art sweep covering MARS, GF-SAM, ProposalCLIP, E-SAM and RAS, all verified against primary sources rather than titles.

Cross-family review (`mcp__llm-chat__review`, reviewer `glm-5.3`, executor
`deepseek-flash`, thread `llm-chat:a1706a3ee101`) plus an independent prior-art sweep.

**Verdict: PROCEED WITH CAUTION** for the idea as originally framed (SAGG), which was
then superseded by the pilot controls above. Two rounds were run.

**Closest prior art, verified against primary sources:**

| Work | What it actually is | Overlap | Difference |
|---|---|---|---|
| **MARS** arXiv:2504.07942 (WACV 2026) | *Few-shot* segmentation ranking system; scores, filters, merges proposals; gate `≥0.95·max` then union (Eq. 21–22) | the gate+merge template | few-shot, not training-free zero-shot RIS; four scorers; and its published operator scores **-0.011 vs argmax** on RIS |
| **GF-SAM** arXiv:2410.06964 (NeurIPS 2024) | *Few-shot* semantic seg; graph over SAM masks × point prompts, weakly-connected components, coverage gating | connectivity-constrained grouping of a gated SAM pool | few-shot; gate is coverage, not text |
| **ProposalCLIP** arXiv:2201.06696 (CVPR 2022) | Unsupervised open-vocab instance seg; merges proposals by IoU + CLIP similarity | CLIP-guided merging of a proposal pool | class-name driven, not expression-conditioned; re-scores the merge |
| **E-SAM** arXiv:2503.12094 (ICCV 2025) | Training-free entity seg; merges adjacent similar SAM masks to fix over-segmentation | training-free mask merging motivated by SAM over-segmentation | entity seg, no referring expression, no score gate |
| **RAS** arXiv:2506.05342 (ICCV 2025) | Learned subset selection over frozen SAM proposals for mask *groups* | arbitrary-size subset selection | trained head on ~2M mask groups + fine-tuned LLaVA-13B (≈3.5 GPU-days on 8×A100); no union |

**What survived and what did not.** The proposed merge operator turned out to be the
connected component of the argmax seed in the overlap graph of gated masks, which is not
a new algorithm, and its gain was then shown to be a granularity artifact. The novelty
that *does* survive is the diagnostic package — in particular the pool-level **flatness
measurement** (~0.01 cosine separating IoU 0.00 from 0.93 over a proposal pool) and the
**oracle-recovery quantification**, neither of which the sweep could find in any
published work, plus the bounded-headroom result for the entire composition family.

---

## External Critical Review

Two review rounds were required: the first rejected the method scope, the second accepted the re-scoped analysis. Both verdicts and their binding conditions are recorded below.

Cross-family review (`mcp__llm-chat__review`, reviewer `glm-5.3`, executor
`deepseek-flash`), two threads: `llm-chat:2d66e8eb427a` + `llm-chat:cdb7beff0221`
(method scope, **ABANDON**), and `llm-chat:0b326b8c12df` (re-scoped analysis,
**PROCEED**).

**Round 1 (method scope) — bottom line: abandon.** The reviewer's decisive objection was
that every positive result was obtained on a deliberately weakened scorer, and that the
gain might be a granularity artifact of the near-tie regime that stripping the guidance
modules created. The controls in Idea 4 then confirmed it. The reviewer's own pre-registered
decision rule was: on the full-strength HybridGL scorer over complete RefCOCO val, upgrade
only if Δ ≥ +2 mIoU with a CI excluding 0, otherwise abandon as a method paper.

**Round 2 (analysis scope) — bottom line: PROCEED.** Verbatim verdict and conditions:

- **THIN**: yes as-is. "The taxonomy decomposition (granularity vs over-segmentation vs
  wrong-instance vs correct) is the one measurement that converts 'selector discards 58 %
  of oracle' into an attributable, actionable claim and ties findings 1, 2 and 4 together."
- **MISSING**: the error taxonomy on the full 10,834-expression val split; secondarily,
  replicate only the two headline numbers (oracle recovery, score flatness / rank-of-best)
  inside full HybridGL — "a day of compute, not a full replication."
- **VENUE**: workshop paper (CVPR/ICCV vision-language-understanding or referring-expression
  workshop) or the analysis section of a larger method paper. Main-conference analysis
  track only with the HybridGL headline check **plus** one more dataset/backbone configuration.
- **MUST NOT CLAIM**: "that the 42 %-oracle-recovery / flat-selection pathology
  characterizes training-free RIS at the SOTA operating point — every flatness, ceiling,
  and headroom number is measured on the guidance-stripped scorer (0.3501 vs 0.4948) and
  must be scoped as 'at the unguided operating point matched to HybridGL's L-alone
  ablation', never generalized to guided pipelines or stated as a field-wide result."

**Also accepted from the review**: the oracle figure (0.8285) *is* scorer-independent — it
is a property of (image, GT, proposal pool) — so finding 1 is fully robust to the
stripped-guidance objection; only the recovery ratio (42.3 %) is operating-point-dependent.
This distinction must be kept explicit in the write-up.

**Open risk carried forward**: if the guided replication shows the score is no longer flat
at the full HybridGL operating point, the diagnostic must be re-scoped to the unguided
regime and the "field-wide" framing dropped entirely.

---

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

### 🏆 Idea 1: Proposal-pool diagnostic for training-free zero-shot RIS — RECOMMENDED

- **Pilot**: POSITIVE for the diagnostic (oracle 0.8285 vs argmax 0.3501; flatness ~0.01;
  re-weighting cap 46 %; composition headroom +2.5) and NEGATIVE for the method that
  motivated it (killed by `parent_strict`).
- **Novelty**: CONFIRMED for the diagnostic. Closest works — MARS (WACV 2026), GF-SAM
  (NeurIPS 2024), ProposalCLIP (CVPR 2022), E-SAM (ICCV 2025) — are all *few-shot* or
  non-RIS, and the pool-level flatness measurement was not found anywhere.
- **Reviewer score**: PROCEED (analysis scope), conditional on the error taxonomy and the
  HybridGL headline replication.
- **Next step**: run `EXP-1` (error taxonomy) then `EXP-2` (guided replication), then
  `/paper-write` for a workshop submission or fold into a larger paper.

### Idea 2: Error taxonomy of the oracle gap — BACKUP

- Pilot: not yet run; re-analysis of saved predictions, ~0 GPU-hours.
- Next step: implement as `EXP-1` below.

---

## Next Steps

- [ ] `EXP-1` Error taxonomy over the full RefCOCO val split (no new GPU forwards)
- [ ] `EXP-2` Guided replication of oracle recovery + flatness inside full HybridGL (~1 GPU-day)
- [ ] `EXP-3` Second configuration (RefCOCOg, CLIP ViT-L/14) — required for any main-conference framing
- [ ] `/paper-write` once EXP-1/EXP-2 land, or fold into a larger method paper
- [ ] If a method contribution is still wanted, obey the pre-registered rule in
      `EXPERIMENT_PLAN.md` §3 before writing any method claim

<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:START -->
## Evidence Gate
**Status:** PASS

All required stage records, review receipts, artifacts, and report sections are present.
<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:END -->
