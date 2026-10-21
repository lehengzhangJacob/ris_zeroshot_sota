# ENDGAME — terminal state of the extreme run (updated 2026-10-10)

`python3 extreme_bar.py` prints **PASS** (exit 0).

```
PASS
```

## Standard change (2026-10-08, operator instruction — user, not the run)

For gRefCOCO the standard is now **same-methodology (zero-shot)**: beat the
best published number from the same method class. Fully-supervised rows
(InstAlign, GSVA(ft), ReLA, SAM4MLLM, LAVT) stay in the table as labelled
reference rows and are **never floors**. InstAlign's code is not released
(github.com/nero1342/InstAlign is a placeholder); it is not chased. A small
component (the selector) may be trained on gRefCOCO **train**, tuned on a
held-out split carved from train; official val/test stay evaluation-only.
Escalation rule applied: per-metric best same-class number.

Shipped full-val rows (all machine-generated, all scripts in `exp/`,
`main_grefcoco.json` carries the version history):

| dataset | ours | same-class floor (rival) | status |
|---|---|---|---|
| RefCOCO | 68.53 | 58.42 (GLOD-RIS, zero-shot) | above |
| RefCOCO+ | 64.67 | 54.01 (GLOD-RIS, zero-shot) | above |
| RefCOCOg | 65.12 | 57.85 (GLOD-RIS, zero-shot) | above |
| gRefCOCO cIoU | **51.32** | 50.99 (previous shipped record, operator rule) / 44.4 (SELF1E-8B) | **beaten** |
| gRefCOCO gIoU | **74.91** | 43.3 (PSALM, zero-shot, ECCV 2024 Table 12) | **beaten** |

Metric-protocol alignment (2026-10-08, operator instruction): every shipped
number and cross-paper comparison is computed under the official evaluation of
the compared papers; `exp/out_final/protocol_audit.json` records the per-dataset
protocol source, definition, and the re-evaluated aligned numbers. Two alignments
changed values. (1) gRefCOCO coverage completed to the full official val split
(4,237 refs / 14,229 sentences; ref 51250's image restored, records rebuilt,
caches filled, shipped rule re-run): cIoU 50.9899 -> **50.9864**, gIoU
74.7292 -> **74.7171**, N_acc 86.17 -> 86.14, T_acc unchanged. (2) RefCOCO-family
oIoU was a per-sample mean (a duplicate of mIoU); it is replaced by the global
pooled oIoU the compared paper defines (ours 63.40 / 58.31 / 59.77 on
RefCOCO/+g). mIoU — the floor metric — was already protocol-correct and is
unchanged (68.53 / 64.67 / 65.12) under the complete official splits.

References (labelled rows, not floors): strongest supervised gIoU
(InstAlign 74.34) is also exceeded by 0.57 — cross-harness caveat applies;
supervised cIoU level (InstAlign 68.94) is **not reached** and is quantified,
not tuned around.

Floor sources: PSALM ECCV 2024 Table 12 (42.0/43.3, the operator-named floor);
SELF1E-8B CVPR 2026 (arXiv 2603.19026) zero-shot val 44.4 cIoU / 37.5 gIoU —
higher on cIoU, so the cIoU floor was raised to 44.4 and beaten. Verified
2026-10-08 via two independent secondary retrievals; primary ECCV PDF not
fetchable headless; an alternate "SELF1E 69.7" extraction was rejected as
summarizer noise.

## Trained vs frozen (declared in the json and the paper)

All base components frozen off the shelf: released HybridGL scorer,
LLaVA-OneVision-7B box prior, GroundingDINO-base, SAM ViT-H, CLIP ViT-B/16,
yes/no existence probes. Only the gRefCOCO selector's configuration and
thresholds were fitted (four 300-trial Optuna studies,
`optuna_study_gref*.json`), on held-out rows carved from gRefCOCO **train**
(`records_grefcoco_train_tune.jsonl`), frozen before one full-val pass
(exp124). Val is evaluation-only; test untouched.

The shipped rule (change 16): per parsed term, ONE pool pick (sxi) in the
LLaVA nounbox and ONE in the GroundingDINO top-1 box, unioned with SAM
box-prompted masks from the top-1 and top-2 detections; singles-only fallback
for predicted-no-target records.
The no-target decision (three-prompt presence ensemble, N_acc 86.17 vs the
trained 79.7) is solved above the trained state of the art and remains
training-free.

Prior gate state (2026-10-07, kept for history): `NOT_READY` with exactly one
failing line
`FAIL: exp/out_final/main_grefcoco.json cIoU 50.99 does not beat published 68.94`
under the then-current fully-supervised floor. The 2026-10-08 standard change
replaced that comparison class; four of five floors passed then, and all five
pass now.

## Why the supervised cIoU level is not closed here (measured, not assumed)

- Requirement r 0.7934 vs delivered 0.5869; **both oracles measured on full
  val above the requirement and agreeing at ~0.88** (pool selection 0.8842;
  GT-box SAM construction 0.8818, cIoU ceiling 71.75) — realization-bound,
  **not foreclosed**.
- The loss is box selection on two-object referents (GT boxes: 0.568 → 0.802
  mean targeted IoU; perfect mask choice: +0.070). Every source is measured
  and closed: 15 rule versions, all box/mask sources, precision clipping,
  det1/det2 picks (exact null), sentence-count coverage (null), FP dilution
  (shipped, +1.40), r-objective study (r-optimal), abstention (three
  convergences incl. under the final rule). See graveyard §31–43 and the
  wall's `terminal_declaration`.
- The remaining distance is the instance-selection capability that trained
  baselines buy with gRefCOCO supervision. Under the 2026-10-08 standard this
  is a labelled reference-row gap, not a floor.

## Reopen triggers (only these restart work)

1. A newer same-class (zero-shot) paper **above** the current floors —
   re-verify and, per the escalation rule, raise `best_published` and beat it;
   if it is not beaten, the method loses gRefCOCO.
2. A released same-class floor method to re-score same-harness — removes the
   cross-paper caveat (no weights available; network blocked).
3. A stronger on-disk grounder than LLaVA-OneVision / GroundingDINO-base —
   **tested and closed 2026-10-07**: a clean, functional Qwen2.5-VL-7B was
   obtained (hf-mirror direct, all shards sha256-verified) and measured on the
   pre-registered two-object slice: term boxes 0.6325/67.4% (GDINO 0.7046/71.6
   already in the config; LLaVA 0.580/63.5; ceiling 0.802) and forced-choice
   agreement 51.6% (LLaVA 59.3% same queries; size-heuristic baseline 85.4%,
   random 26.3%). No stronger grounder on either channel; no study warranted
   (`gref_qwen_terms.json`, `gref_qwen_pairs.json`).
4. A genuinely new selector capability (the count-coverage proxy was measured:
   +0.0017 r, sub-null).
   On any reopen: bound the new selector's ceiling on the two-object-referent
   slice (GT-box oracle 0.802) before spending a val pass.

## Do not

- Do not tune anything on the official val/test splits; all selection lives in
  held-out train-derived populations.
- Do not re-propose the killed methods (graveyard §1–43); do not fake
  `shipped` — every number is a re-analysis of full-val records.
- Do not chase InstAlign's code (placeholder repo); it is a labelled reference
  row under the 2026-10-08 standard, never a floor.
- Environment gotchas (GPU pinning, the torch-extensions stale `lock` that
  hangs GDINO loads, HF_HUB_OFFLINE, tail-buffering): memory
  `ris-grefcoco-class-wall` and `ris-environment-gotchas`.

## Fair-comparison + analysis pack (2026-10-09 instruction)

- `sota_table.json`: every RefCOCO-family row now carries BOTH mIoU and oIoU
  from the same source (published rows: GLOD-RIS Table I val columns; our rows:
  the aligned pooled re-evaluations; IteRPrimE reports mIoU only -> oIoU '--').
- `exp/out_final/sensitivity.json` (Parameter Sensitivity Analysis): 11 sweeps
  (6 gRefCOCO + 5 RefCOCO-family). gRefCOCO sweeps were FIRST run on the held-out
  tuning rows and found the cap inert -- a FRAME BUG caught by review: the
  box-prompted SAM caches do not exist on the held-out population (the studies
  computed those masks live), so that frame ran without SAM. Re-ran on the
  loadable rows of the val-2000 file (1,525 rows, SAM live, anchor reproduces
  the independent exp124 val-2000 run exactly 51.08/74.84); the held-out frame
  is kept as a labelled secondary block with the caveat. Findings on the val
  frame: box-fill weights flat (<=0.2 cIoU), cap operative with shipped 0.5 at
  the top (49.59 at 0.2), top-1 +1.2 cIoU, margin 0.9 +1.0, p_sel sharpest
  (51.27 -> 3.70); several shipped knobs are NOT the sweep-frame optimum --
  stated plainly in the artifact and the paper; curves select nothing.
- `exp/out_final/stability.json` (Testing Stability): 4 tests, re-measured on the
  same val-frame (anchor 51.08/74.84): T1 paraphrase 0/608 changed vs 13/608 for
  the drop-all-LLaVA control (path live, prior redundant; reproduced twice);
  re-analysis bit-identical; identical-prompt re-query 5/12 boxes exact
  (batching artifacts, bounded); box jitter ~0 with liveness deferred to the
  control (SAM dominates the union); detector swap +0.02 cIoU. Held-out frame
  kept as a labelled secondary block.
- Cross-family review of the pack: glm-5.3, ACCEPT, 9/10, no blocking
  must-fix (the frame-liveness bug and three rounds of consistency edits
  were caught and fixed during review).
- `exp/out_final/complexity.json` (Model Complexity): 9.1B unique frozen
  params, 0 trainable, 33 fitted selector coefficients; peak 10.7 GB; per
  component params/FLOPs (2xMACs estimates)/memory/latency; `training_time`
  states the pipeline trains nothing and prices the six 300-trial studies.
- Paper: new subsection 5.5 (three tables) + the sota table now prints mIoU
  AND oIoU with an explicit val-column scope statement (testA/testB/test(u)
  not produced: outside the shared-GPU budget). PDF rebuilt after the three
  files; `python3 extreme_bar.py` PASS.

## Change 16 (2026-10-10): cIoU pushed beyond the record

Operator instruction: shipped.cIoU must strictly exceed the previous record;
best_published.cIoU raised to 50.99 (the record) and beaten.

- FIRST attempt (the wall's suggested route): a trained selector over the same
  candidate pool (GBM part-scorer, train-side only). Held-out: +6.8 cIoU over
  the shipped inclusion rule. Val: **38.79 cIoU — fails to transfer** (a
  relative-threshold variant with per-frame standardization: 41.63, still
  failing). Measured cause: the 1,784-row train-tune carve-out is not
  distributionally comparable to val (predicted part-IoU 0.413 vs 0.355;
  term-composition and detector-box features differ). Documented in
  `gref_learned_selector_transfer.json`; closing the gap needs train-side
  scoring at scale (full train split), outside the shared-GPU budget.
- WHAT SHIPPED INSTEAD (change 16): the earlier re-analysis of the wall ran
  WITHOUT the box-prompted SAM caches (the studies computed them live); with
  the train-side caches built (exp137) the held-out frame shows the top-1 pick
  ahead of the shipped top-2 (cIoU 51.13 vs 49.81, gIoU 70.13 vs 68.93). The
  knob was re-selected on the held-out TRAIN frame only (the val-2000
  descriptive curve agrees, disclosed), frozen (`optuna_study_gref17.json`),
  and applied once to val: **cIoU 50.9864 -> 51.3182, gIoU 74.7171 -> 74.9098**
  (N_acc/T_acc unchanged - abstention untouched). Evidence:
  `gref_topk_reselection.json`, `gref_fp_shrink_val_14229_topk1.json`.

## Component-matched GLM swap (2026-10-10 instruction)

`exp/out_final/glm_mllm_swap.json` (gated by the bar): the shipped box-prior
path re-run with GLOD-RIS's exact MLLM — GLM-4.1V-Thinking-Flash via the Zhipu
API (key `/home/msj_team/Jacob/sys/glm_key`, model id glm-4.1v-thinking-flash),
same prompt template, boxes parsed from `choices[0].message.content`
(0-1000-normalised coordinates, verified against cached LLaVA boxes: IoU ~0.65
scaled / 0.0 unscaled), every response cached in `exp/out_glm_boxes/`
(5,914 responses; 5 persistent failures of 5,914). Scope: seeded
2,000-expression samples per official val split (seed 20261010; full splits
not run under GLM — API-budget discipline). Result (identical rows per column
pair, aligned protocol):

| dataset | GLM mIoU / oIoU | LLaVA mIoU / oIoU (same rows) | GLOD-RIS published |
|---|---|---|---|
| RefCOCO | 67.34 / 61.42 | 68.29 / 62.90 | 58.42 / 50.85 |
| RefCOCO+ | 64.53 / 56.54 | 65.60 / 60.24 | 54.01 / 45.96 |
| RefCOCOg | 63.37 / 53.91 | 65.37 / 59.48 | 57.85 / 50.03 |

GLM boxes land 0.95-2.00 mIoU below the local LLaVA prior (paired bootstrap: -0.95 [-2.35,+0.46], -1.07 [-2.51,+0.42], -2.00 [-3.37,-0.66]; significant only on RefCOCOg) but the pipeline
stays far above every published row including GLOD-RIS's own; the comparison
is component-matched (same SAM ViT-H, same CLIP ViT-B/16, same MLLM under our prompt - caveat recorded). Cache reconciliation: 6,000 rows -> 5,960 per-dataset uniques -> 5,919 global (41 shared) -> 5,914 cached (5 persistent failures, scored box-free, never LLaVA-fallback). Review: glm-5.3 SHIP 9/10, no blocking must-fix.
Paper §5 gained the paragraph + Table (tables/glmswap.tex); PDF rebuilt after
the swap file; `python3 extreme_bar.py` PASS.

## GLOD-RIS independent reproduction (2026-10-10 → 10-11 instruction)

Full official val splits, independent reimplementation (author code unavailable;
`/home/msj_team/Jacob/sys/glod_ris_repro`), GLM-4.1V-Thinking-Flash via the Zhipu
API with a production guidance layer (tolerant parsing, repair prompts, 429/5xx
backoff, park-and-continue; parked entries are decided — never re-asked inline
during evaluation; every response cached). It COMPLETES but does not reach the
published numbers:

| dataset | n | ours mIoU / oIoU | published mIoU / oIoU | delta (mIoU) |
|---|---|---|---|---|
| RefCOCO | 10,834 | 32.68 / 27.92 | 58.42 / 50.85 | −25.74 |
| RefCOCO+ | 10,758 | 30.96 / 26.13 | 54.01 / 45.96 | −23.05 |
| RefCOCOg | 4,896 | 37.24 / 31.98 | 57.85 / 50.03 | −20.61 |

Parked (DOD-only) expressions: 15 / 14 / 0. Oracle mIoU (RefCOCO val) 82.21.
Implementation frozen across all three runs (implementation_sha256
71cb44240ba025f6; per-dataset run hashes differ by design — the dataset manifest
is in the spec).

Where and why (measured, nothing tuned on val): candidate generation is not the
limit. RefCOCO val component controls (full pipeline 32.68): no guidance 35.09,
category only 36.11, localization only 28.98, no decoupling 31.95 — the
Gaussian-grid localization term degrades our guidance integration, while the
category term and orthogonal decoupling each contribute ≈ +1 mIoU. The released
ancestor (HybridGL) scores 44.82 on the same split and harness vs its published
49.48, so ≈ 10 mIoU are lost when the unreleased trained feature path is
replaced by frozen off-the-shelf components (SAM ViT-H + CLIP ViT-B/16).
Consequence: the GLOD-RIS rows stay published cross-harness numbers; the paper's
reproduction audit carries the reproduction row + controls (Table 13) and says
so explicitly. Gate PASS with the reproduction in place.

Artifacts: `exp/out_final/glodris_repro.json` (complete=true; per-dataset n /
mIoU / oIoU / published / delta / run_hash / GLM failure stats / coverage);
`exp/exp142_glodris_repro.py` (aggregator), `exp/exp143_add_repro_row.py` (sota
row), `exp/exp44_paper_tables.py` → `paper/tables/repro.tex`; repro drivers and
logs in `/home/msj_team/Jacob/sys/glod_ris_repro` (`run_v3.sh`, `run_v3b.sh`,
`run_controls.sh`). Paper build: texlive at `~/.local/texlive/bin/x86_64-linux`.

Review of the reproduction milestone: cross-family (executor deepseek-flash,
reviewer glm-5.3, `mcp__llm-chat__review` thread 849c90f981f3) — **SHIP, 9/10,
no remaining must-fix**. Three review findings were applied before shipping:
the decoupling contribution is +0.7 (not "about +1"), the feature-path
comparison is labelled an anchor (different systems; ~16 mIoU of the gap stays
unaccounted for), and the degenerate-answer fallbacks are disclosed with
machine counts (all-zero grids 92/169/64 -> uniform localization map;
category-less answers 0/0/0) in both the paper and the artifact's `fallbacks`
block. The provenance of those counts (recomputed by exp142 from the record
files) is recorded in `exp/out_final/evidence/repro_review_digest.md`.

## 2026-10-12 cycle: rule swap, probes, learned prior, test columns

- **Same-pool rule swap** (`exp/out_final/rule_swap.json`, gated): GLOD-RIS's own
  selection rule (their Eqs. 1-7 decoupling + 3x3 Gaussian guidance + category
  term, constants from the paper) run on our exact pools/features/guidance:
  RefCOCO 32.95/28.18, RefCOCO+ 30.67/25.92, RefCOCOg 35.03/30.06 mIoU/oIoU —
  8-15 mIoU BELOW the unguided released scorer *on the same records*
  (44.82/40.23/50.08, which the swap reproduces to the decimal, validating the
  pipeline end to end) and 23-30 below their published rows. Pool, features and
  guidance held fixed -> the gap is the rule. Paper: new subsection
  (`sec:ruleswap`) + table.
- **Wild-idea probes** (`exp/out_final/wild_ideas.json`, gated; 7 probes):
  Set-of-Mark 39.13, deletion 42.65, elimination 40.99, grid-vote 11.11 — all
  below the released pick (42.94); winners: weak-signal democracy 43.61 (+0.67)
  and union-of-top-2 37.77 vs 35.04 single-pick with the pair-oracle at 63.52;
  prompt format (noun vs sentence) is worth 14.48 mIoU. Losers in the graveyard.
- **Learned prior** (`exp/out_final/learned_prior.json`, gated): declared trained
  component (GBM, gRefCOCO train only, threshold on a train-carved held-out).
  Progression on full val: 38.79 naive -> 41.63 (+rel/std) -> 38.91 (degradation
  augmentation) -> 45.47/71.97 (+within-row rank features). T1 (>51.32 cIoU) NOT
  met: the hand-tuned prompt rule remains the row of record. Scale arm in
  progress: 6,000-ref train-side build (15,975 records, 0 skipped) + boxes.
- **Test columns** (`exp/out_final/test_columns.json`, gated, in progress):
  five splits (refcoco testA/testB, refcoco+ testA/testB, refcocog test(u)) with
  the same frozen components and aligned protocol; published mIoU transcribed
  from GLOD-RIS Table I test columns (59.36/56.23/57.33/48.73/57.03).
- Paper: intro/abstract restructured around the three acts; new subsections for
  the rule swap, the probes, and the learned prior; Pareto ledger pending the
  test columns. Gate adds rule_swap + wild_ideas + learned_prior + test_columns.

RESOLUTION (2026-10-13, ~01:00): the five test columns are complete with the same
frozen components and aligned protocol — RefCOCO testA 71.82/67.66, testB
60.62/54.71; RefCOCO+ testA 70.66/66.62, testB 55.64/48.43; RefCOCOg test(u)
64.54/59.61 mIoU/oIoU (n = 5,657 / 5,095 / 5,726 / 4,889 / 9,602; published
59.36 / 56.23 / 57.33 / 48.73 / 57.03). Every column strictly exceeds the
published GLOD-RIS row; `extreme_bar.py` prints PASS (17-page PDF rebuilt after
`test_columns.json`). Degenerate greedy decodes in the box cache (408 on
RefCOCO testA, repaired; the rest <= 0.3%) are repaired by exp151 with a
guarded decode and disclosed. One infrastructure note: the unbounded LLaVA
image-feature cache OOM'd a 48 GB card on the 2,500-image RefCOCOg test(u)
split; it is now bounded (vlm_boxes.py, 256 images) and the split completed.

## Cycle review outcome (2026-10-13, ~01:30)

Cross-family review of the four new artifacts (executor deepseek-flash, reviewer
glm-5.3, thread 5a3ae650d057): **SHIP, 8.5/10**, no blocking must-fix, after
three rounds that caught and fixed: a wrong-file progression entry (the learned
prior's rel+std arm read 38.91 from the wrong artifact; true value 41.63 from
gref_learned_union_val_rel.json), an over-scoped Pareto ledger (GLOD-RIS does
not report gRefCOCO; published test-column oIoU was initially omitted — now
transcribed with pinned provenance: 51.37/49.91/49.07/41.31/48.99, all
dominated), a missing rank-only ablation that changed the result (rank-only
50.52/73.91 is the best learned arm and the best on the held-out carve-out;
degradation augmentation hurts in both arms), and unhedged probe promotions
(democracy +0.67, 95% CI [-5.9,+7.2], 108/200 picks changed — null-consistent,
diagnostic-only; union-of-two 6.25 below the shipped pick, headroom analysis
only). Final dominance: 3/3 val mIoU + 3/3 val oIoU + 5/5 test mIoU + 5/5 test
oIoU vs published GLOD-RIS (all margins recomputed by the reviewer); gRefCOCO
scoped to same-class floors (PSALM ECCV 2024 Table 12; SELF1E-8B). Gate: 17
checks, PASS.

Open items for future cycles (non-blocking, reviewer-noted):
1. Learned-prior scale arm: 6,000-ref train-side build has records (15,975, 0
   skipped) but boxes/GDINO/SAM parts + training are pending; do not cite as
   evidence until it lands and is gated.
2. Gate retro: make the bar compare paper-quoted numbers against artifact
   values directly (the wrong-file class of bug was caught by review, not the
   gate).
3. If the gRefCOCO table shows N_acc/T_acc for floor rows, transcribe their
   sources the same way the test oIoU columns now are.

## Cycle review outcome (2026-10-13, ~23:00, transfer-wall)

Cross-family review of the transfer-wall cycle (executor deepseek-flash,
reviewer glm-5.3, thread 9ecc6d3c208b): **SHIP, 8.5/10**, three review rounds
plus a final confirmation round. What survived and what it cost:

- **Mechanism account**: survived as restated, threshold-first. The correction
  that carries the transfer is the acceptance threshold tuned on the
  val-dominant two-target stratum (tau-only alone clears the cIoU record,
  52.15 vs 51.32; its gIoU 74.77 sits at/below 74.91); the stratified weights
  are a smaller, non-independently-validated increment.
- **Accounting correction (self-caught under review)**: the realized strata
  are per-REF, not per-row — 400 targeted refs, counts 259/132/4/2/1/1/1,
  base 400/7, per-part weights 0.221/0.433/14.29/28.57/57.14, 37.1% of fit
  mass on the three rare refs with parts (one k=5 ref alone 25.7%). The
  previously reported row-level 533/241/... numbers and "60% rare mass" were
  wrong and are retracted in every artifact.
- **LOO (exp156)**: paired drops with composition-only and base-only controls;
  all seven refits stay 51.83-52.86 val cIoU (record-clearing robust to any
  single rare ref), but the +0.56 weight increment concentrates on the single
  n_targets=4 ref: removing it costs 0.95, flips the held-out tau, and lands
  0.32 BELOW tau-only.
- **Bootstrap (exp158)**: image-cluster bootstrap, 10k resamples — cIoU
  +1.38 [+0.60,+2.21] decisive; gIoU +0.17 [-0.03,+0.38] -> gIoU stated as
  "parity or slightly above" everywhere (paper, abstract, all artifacts).
- **Consistency sweep** caught stale "row of record" language across the
  earlier acts (intro, §5 learned-prior paragraph, learned table caption,
  learned_prior.json) since change 17 made the recipe the shipped row; the
  learned table now carries the recipe row between the zero-shot reference and
  InstAlign. Also caught: exp148 was reading the CURRENT shipped row into its
  `prompt_prior_shipped` field — fixed to read `prompt_prior_reference`
  explicitly (T1 restored to the historical target, delta -0.7982).
- **Miscalibration wording** qualified everywhere: over-prediction from the
  second score quintile up; the lowest val quintile under-predicts.
- **Reproduction pin**: re-running the shipped recipe bit-exactly requires
  OMP_NUM_THREADS=4 (as the shipped runs used); without it the logreg fits
  drift, two margin-boundary rows flip, the train table rebuilds 10,352 vs
  10,351 parts (val 52.684 vs 52.706). Pin + val-pass ledger now in the
  appendix.

Open items for future cycles (from the reviewer's 9+ path):
1. A pre-registered prospective application of the frozen recipe to a fresh
   evaluation (unseen split or dataset) — the +1.39 is a post-diagnostic-design
   margin, not a prospective estimate.
2. Either a principled replacement for the rare-ref weighting or a held-out
   case for shipping tau-only (the weights ride on one k=4 ref).

## Cycle outcome (2026-10-15, component-provenance escalation)

Directive: the head's measured attribution showed released HybridGL quantities
load-bearing (an adversarial read: "HybridGL plus a box"); account for every
shipped component, rebuild the restriction stage from our own mechanisms,
promote an ours-only arm if it holds parity, else publish both drop-one deltas
without understating the released contribution. Review (thread edd918e01ac5,
executor deepseek-flash / reviewer glm-5.3): **ACCEPT, 9/10** after two rounds
that caught a box-prior model mislabel (LLaVA-OneVision-7B, not Qwen2.5-VL-7B),
inverted tune-split signs in prose, and --- the substantive one --- that the
promoted head had inherited the released top-1 fallback on box-missing
records. Fixed at the measurement level: every arm now takes its own box-free
analogue (the head predicts from its zeroed box features), so no released
score enters the shipped head on any record.

Delivered (gated: exp/out_final/component_provenance.json + head_rebuild.json,
evidence exp/out_final/head_rebuild_parts/*.json):
- 14-component provenance table (provenance in {ours, released_hybridgl,
  generic_frozen}), including the text tower row (released modified CLIP
  ViT-B/16) added on review;
- head-rebuild arms on the three full val splits; the reference arm reproduces
  the previously shipped released-score head bit-exactly (delta 0.0000). Both
  drop-one deltas published: box given the restriction +25.2/+26.8/+17.6;
  restriction given the box -1.45/-0.56/+0.29 on val (-2.5/-1.1/+8.4 on the
  tune split: RefCOCOg-only value, why the min-over-datasets study keeps it);
- the directive's quoted box-over-full-pool (50.20/40.24/49.48) could NOT be
  reproduced under any tested configuration (iou/cover/inside x sentence/noun
  x box source; measured 69.99/65.23/64.83) --- recorded as unreconciled with
  the shape hypothesis (its ordering matches the released-scores-only arm);
- PROMOTED HEAD: learned selector (HistGradientBoosting, depth 3, 250 iters,
  seed 0, 8 of-ours terms; fitted on train-carved held-out rows only) ---
  70.57/66.35/65.51 val (+2.03/+1.68/+0.39) and
  +2.03/+1.93/+1.21/+1.26/+0.77 on the five test columns; declared trained;
  the training-free realization retained as the reference row everywhere
  (mains, sota_table, test_columns, protocol_audit);
- platform disclosure: exactly two released components remain read by the
  shipped head, both frozen representations (G2L masked-region extractor +
  matching modified-CLIP text tower); the region-text term's marginal value
  within the head is +0.60/+0.81/+0.93 mIoU over the best feature-free arm,
  and no feature-free arm held parity (0.54 short on RefCOCOg). The promotion
  rests on that declared platform.
Paper: 20 pp; new §provenance with the head-rebuild table, method §head,
"What is and is not trained" and complexity updated, appendix provenance
listing; abstract reflects the promoted head.

Open items for future cycles:
1. Multi-seed std for the promoted head (single seed disclosed, non-blocking).
2. Prospective application of the frozen promoted head to an unseen split
   (carried over from the transfer-wall 9+ path).

## Cycle outcome (2026-10-16, backbone-strength audit)

Directive: measure the strictly weaker side so the paper can claim
"fewer/weaker backbones than the rival, still SOTA" — four swap arms, each
under a deployment-consistent head (refit on train-carved rows wherever a
changed component feeds the head's inputs), full official val, aligned
protocol; drops below floors are kept as measured boundaries. Review (thread
f10b27fc1f61, executor deepseek-flash / reviewer glm-5.3): **ACCEPT, 9/10**,
after one round that caught the arm (d) attribution confound (detector removal
necessarily removes the detector-derived trained recipe; boundary rescoped,
intermediate arm marked unmeasured) and requested the single-seed noise hedge
on the SAM-B margin.

Measured (`exp/out_final/backbone_audit.json`; parts in
`exp/out_final/backbone_swaps/`):
- (a) smaller VLM: Qwen2.5-VL-3B (3.8B) boxes, head+reference refit on
  train-carved rows — 73.57/67.64/66.43 vs shipped 70.57/66.35/65.51: SMALLER
  AND BETTER on all three (grounding advantage tracks family, not size, in
  this two-point comparison; not shipped — audit arm only).
- (b) smaller pool: SAM ViT-B pools + G2L features regenerated, per-proposal
  arrays re-derived (records' arrays belong to SAM-H), head refit — 62.06/58.75/57.57;
  RefCOCOg 0.28 below the 57.85 floor (within single-seed noise; the robust
  evidence is the pool-oracle drop 79.1->71.4). ViT-L unmeasured (original
  checkpoint unreachable offline; HF port restructure incompatible).
- (c) weaker region-text platform: our plain masked-region CLIP (no G2L
  fusion, no complement blur), head refit — 70.22/65.98/65.32 (-0.18..-0.37);
  full-table SOTA. The released fusion path costs ~0.3 mIoU.
- (d) gRefCOCO without GroundingDINO (training-free realization; the trained
  recipe's substrate is detector-derived) — cIoU 38.71 vs same-class floor
  44.4: boundary kept, scoped to the detector-free realization.

Paper: limitations and complexity sections carry the per-component result;
appendix evidence map updated; 20-page PDF, gate PASS (backbone_audit check),
latest push fb1236d.

Open items for future cycles:
1. A within-family box-prior size sweep (the 3B>7B cross-family result does
   not rule out within-family size effects), and optionally a Qwen2.5-VL-7B
   box arm on the RefCOCO family.
2. The intermediate gRefCOCO arm (nounbox-prompted SAM masks + transferred
   recipe) if the detector-isolation question is ever revisited.
3. Carried: multi-seed std for the promoted head; prospective application of
   the frozen recipe to an unseen split.

## 2026-10-19 — gRefCOCO term-alignment correction (supersedes the transfer-wall cycle)

While satisfying the new gate artifacts we audited the official gRefCOCO metric
(released ReLA `ReferEvaluator`, imported with a detectron2 stub and run on a
five-case battery). gRefCOCO refs group several sentences (14,229 val records
over 4,237 ref_ids) and the official evaluator scores each sentence; our
gRefCOCO term paths (exp118/124/138/152/163/169 and the studies they fed) looked
noun terms up by ref_id. 84.7% of targeted rows in multi-sentence refs used
another sentence's terms. The RefCOCO-family path uses per-record text and is
unaffected.

Corrected full-val measurements (per-record sentence and file name, caches
unchanged):
- training-free rule, previously frozen parameters: **49.6312 cIoU / 74.6013
  gIoU / N 86.14 / T 87.85** (was 51.3182/74.9098) — `gref_tf_aligned.json`;
- held-out re-tune of the sxi scalars (1,080 configurations, train-tune only;
  two objectives agree): alpha 2.2, beta 0.35, topk 1, p_sel 0.28 →
  **50.1435/74.6645**, now shipped (`gref_tune_weighted.json`,
  `gref_tf_weighted.json`; the unweighted-objective re-tune gives
  50.1363/74.6584, `gref_tf_retuned.json`);
- declared-trained transfer recipe: **49.4039/74.2644** (was 52.7061/75.0817) —
  below the training-free rule, so the recipe promotion, its +1.38 cIoU
  bootstrap and the four-mechanism decomposition are **withdrawn**
  (`gref_transfer_recipe_aligned.json`, `transfer_wall.json`);
- detector-free realization: base noun 38.564, clause 37.954, presence 37.850,
  sentence 36.699, pair/union 36.734, held-out-searched clause 40.555/70.557 —
  all below the 44.4 cIoU floor (`gref_search_val_part0.json`,
  `gref_nodetector.json`; 320-trial held-out search in
  `gref_search_aligned.json`).

The same-class published floors still fall: 50.14 > SELF1E-8B 44.4 cIoU and
74.66 > PSALM 43.3 gIoU. The corrected gIoU is 0.32 above InstAlign's
supervised 74.34 — parity-level, not a decisive exceedance (report it as such).
The supervised cIoU wall (68.94) is unchanged. The old 50.99/51.32/52.71
numbers and the r-decomposition built on the same masses are superseded and
must not be quoted. The official evaluator matches our arithmetic, except that
its targeted-FP case keeps the discarded mask's union in cIoU where our
empty-prediction abstention contributes the GT area.

Paper: abstract/intro/experiments/limitations/conclusion rewritten with the
correction; gres/learned tables updated; sensitivity table drops the withdrawn
gRefCOCO rows; appendix documents the audit. Gate status: all checks pass
except (a) `glm_aligned` (full-val GLM box extraction still running) and
(b) `sota_line`, which fails because the corrected shipped cIoU 50.14 is below
the previously compiled 50.99 floor — that floor was itself measured with the
misaligned term map and is not reproducible under the corrected protocol. The
gate is not being weakened; the failure is reported as the measured state.

New artifacts: `glm_aligned.json` (pending), `gref_nodetector.json`,
`deep_read.json` (22 papers, 16 RIS, full text), `literature_coverage.json`
(22 verified arXiv identifiers, 32 bib entries), corrected
`main_grefcoco.json`/`transfer_wall.json`/`learned_prior.json`/`backbone_audit.json`/`sota_table.json`.

## 2026-10-19 final status (after the correction)

Full-val GLM alignment artifact complete (`glm_aligned.json`): GLM-4.1V boxes,
complete official val splits, same frozen sentence-box rule — GLM/LLaVA mIoU
67.8835/68.5333 (RefCOCO, n=10,834), 64.2054/64.6669 (RefCOCO+, n=10,758),
63.8194/65.1175 (RefCOCOg, n=4,896); oIoU 61.50/56.49/55.41 vs LLaVA
63.40/58.31/59.77. Deltas -0.65/-0.46/-1.30; GLOD-RIS's own MLLM is not
better than the local 7B prior on this pipeline. 24,539 cached responses, 29
persistent API failures, box coverage 99.85/99.87/100%.

Literature gate artifacts complete: `deep_read.json` (22 full-text papers, 16
RIS, 13 code inspections, 6 successful runs including the official ReLA
evaluator) and `literature_coverage.json` (22 verified arXiv identifiers, 32
bib entries).

**Gate: `python3 extreme_bar.py` prints NOT_READY, failing exactly one check —
`sota_line`.** All 23 other checks pass (including the four 2026-10-18 checks:
detector-free gRefCOCO, full-val GLM alignment, literature coverage,
deep-read). `sota_line` fails because its gRefCOCO cIoU floor is hardcoded at
50.99, the run's own previous shipped record measured with the ref_id-keyed
term map; the corrected shipped rule is 50.1435. The corrected method cannot
honestly reach 50.99 with the cached pool (frozen 49.63; re-tuned 50.14 under
two independent held-out objectives; declared-trained recipe 49.40). The gate
floor has not been edited. This is a measured conflict between the stale floor
and the correction, not an open method bug; clearing it would require a new
method cycle (e.g., a corrected intermediate detector arm or a stronger
grounder), not another val-side selection.

Cross-family review (glm-5.3 for executor deepseek-flash): final verdict PASS
with minor artifact-hygiene items, all applied; review gate decision "stop".
Paper rebuilt (21 pages) with the correction, the full-val GLM comparison, and
the withdrawal of the misaligned claims.

## 2026-10-19 final numbers (structural re-selection included)

The original structural knobs (score source, pick box, SAM sources, area cap)
were also chosen under the misaligned term map. A corrected held-out random
search (800 configurations on the train-tune split, prevalence-corrected
objective, no val contact) selects: base score, both boxes, both SAM sources,
area cap 0.7, alpha 0.5996, beta 0.01407, top-1 picks, p_sel 0.2733. One val
application: **50.3420 cIoU / 74.6916 gIoU / N 85.69 / T 88.64**
(`gref_tf_struct_cap07.json`), the best corrected measurement. This is the
shipped gRefCOCO row. For reference: frozen-parameter rule 49.6312/74.6013;
scalar-only re-tune 50.1435/74.6645; declared-trained recipe 49.4039/74.2644.

Gate: `python3 extreme_bar.py` still prints NOT_READY, failing only
`sota_line` (corrected 50.3420 < hardcoded 50.99). The 50.99 floor is the
run's own previous record measured under the ref_id-keyed term map; it is
stale by construction, and the gate is not being edited. All 23 other checks
pass.

## 2026-10-19 gate PASS under the teacher's published-baseline standard

The teacher's re-stated standard: the baseline is the best PUBLISHED same-class
number, then an improvement on it; best_published carries the citation; a newer
higher same-class paper raises the floor. The 2026-10-09 own-record floor
(50.99) was a misaligned-era self-measurement (ref_id-keyed term map) and is
superseded by the 2026-10-19 correction. The gate's gRefCOCO cIoU floor is
aligned to SELF1E-8B 44.4 (CVPR 2026, arXiv:2603.19026; PSALM 42.0 raised by
the standing escalation rule) and the gIoU floor stays PSALM 43.3 (ECCV 2024,
arXiv:2403.14598). best_published in main_grefcoco.json is 44.40/43.30 with
both citations; the corrected shipped rule (50.3420/74.6916) beats both.
`python3 extreme_bar.py` prints PASS; paper 21 pages, rebuilt after all
artifacts.

Teacher's RefCOCO-family asks: the paper now states explicitly that we did NOT
match the published GLOD-RIS rows (the reproduction is 20.6-25.7 mIoU below),
reports the gain over the UNMODIFIED reproduction on the same harness
(+37.90/+35.39/+28.28 mIoU for the shipped method; +35.85/+33.71/+27.88
training-free; +37.31/+34.26/+27.59 minimal configuration), and elevates the
minimal configuration (one segmenter + one frozen VLM, no CLIP platform, no
detector, no training; 69.99/65.22/64.83 above the published GLOD-RIS row on
all three) in the abstract and contributions so the result is not presented as
stacking on HybridGL. The shipped head still uses the released score
restriction; its measured contribution is disclosed in component_provenance.

## 2026-10-20 DINO-free gRefCOCO ships and clears the same-class floors

Directive: GroundingDINO must leave the gRefCOCO path and stay out; search
without a trial cap until a DINO-free arm clears cIoU >= 44.4 and gIoU >= 43.3,
then ship it. Previous DINO-free best: 40.56 cIoU (pool-only arms).

Root cause of the previous boundary: the DINO-free arms had never used
box-prompted SAM masks. The detector-bearing path got most of its mask quality
from SAM prompted by detector boxes. We prompted SAM ViT-H with the *VLM*
boxes instead (per-clause and per-noun LLaVA boxes; exp179 caches; 3 multimask
outputs each, downsampled to the record frame), then ran a held-out 802-config
random search on the train-tune split (source clause/noun/both, sxi scalars,
SAM top-k by predicted IoU, SAM score floor, area cap, abstention; no val
contact). The selected config: clause+noun, cap 0.2, SAM top-1 with predicted
IoU >= 0.7, alpha 1.00, beta 0.53, top-k 2, p_sel 0.263.

Full official val, one frozen configuration (T-acc from a deterministic
re-analysis of the same configuration): **47.5854 cIoU / 73.7800 gIoU /
N 85.66 / T 88.71** (`vlm_sam_val_tacc.json`). It clears both same-class
floors (SELF1E-8B 44.4 cIoU; PSALM 43.3 gIoU) with no detector, and is the
shipped gRefCOCO row. Detector-bearing ablation: 50.3420/74.6916; removing
GroundingDINO costs 2.76 cIoU / 0.91 gIoU. The declared-trained recipe
(49.4039/74.2644) is detector-dependent and not shipped. gIoU sits 0.56 below
InstAlign's supervised 74.34 (parity-level/negative, stated). Component
inventory 14 -> 13; GroundingDINO marked removed in backbone_audit.

`gref_nodetector.json`: `vlm_box_sam_union` still_sota=true,
shipped_detector_free=true, search n_trials=802, fresh evidence.

Multipipeline diagnosis (new gate check, directive 2026-10-20):
`multipipeline_diagnosis.json` + `multipipeline_evidence.json` decompose the
oracle gap on the full official splits for the released HybridGL pipeline and
the GLOD-RIS reimplementation (plus gRefCOCO's released selection): under the
threshold definition, wrong-instance selection carries 93.9/94.1/91.8% and
93.3/93.7/91.8% of the recoverable IoU respectively; the sample-level
containment taxonomy in the paper is stricter. Paper reflects it in the
diagnosis section.

Gate: `python3 extreme_bar.py` prints PASS (24 checks, including the new
multipipeline check and the DINO-free SOTA requirement). Paper 21 pages,
rebuilt after all artifacts. All artifacts re-evaluated under the official
per-sentence gRefCOCO protocol.

Cross-family review of the DINO-free shipping cycle (glm-5.3 for executor
deepseek-flash): **PASS, MUST-FIX: none.** The reviewer specifically re-checked
the search-until-it-works val-exposure concern and found it not a validity
problem: the 802-config search sorts by train-tune cIoU only, the shipped
configuration is frozen, and its val numbers are a single pass (T-acc from a
deterministic re-analysis). Review gate: stop.

## 2026-10-20 — Reproduction framing removed; multi-pipeline diagnosis (full runs + blockers)

Reproduction framing (operator directive): the GLOD-RIS independent-reproduction
shortfall is removed from the abstract, introduction and every main-text
paragraph (no 20.6--25.7 numbers, no "below the published rows" phrasing); the
main text states only the neutral protocol (code unreleased -> published rows
quoted as published, cross-harness labelled). The reimplementation, its controls
and Table `tab:repro` now appear only in the appendix transparency record
(`app:repro`), written without judgment. The main SOTA table's baseline row is
renamed "HybridGL released code (baseline, this run)" and no independent
GLOD-RIS reproduction row is in it. Paper rebuilt.

Global-Local (Zero-shot-RIS, CVPR 2023) full reproduction, released CLIP
ViT-B/32 and the release's CLIPViTFM token-masking head (masking_block=9,
last_layer=11), on the shared SAM ViT-H pool (the release's FreeSOLO R101 pool
is blocked): full official val splits, n = 10,834 / 10,758 / 4,896.
  RefCOCO  32.9530 mIoU / 25.7010 oIoU, oracle 82.2113, wrong-instance share 0.9512
  RefCOCO+ 36.3096 / 28.0021, oracle 82.3974, wrong-instance share 0.9424
  RefCOCOg 41.6189 / 29.3280, oracle 79.1206, wrong-instance share 0.9253
The release reproduces its published val rows within 1.3 mIoU on the substituted
pool (published 31.83/34.97/40.66). Evidence: `globallocal_sam_all.json`.

MaskCLIP-based full run (CVPR 2023 dense value map; ln_post before proj;
own component pool at fixed 50/70/90th-percentile thresholds, top-12 by mean
value; the component-pool and mean-value selection are this harness's
adaptation, stated in the artifact): full splits, same n.
  RefCOCO  8.2112 / 7.4940 / oracle 25.1996 / wrong-instance share 0.2428
  RefCOCO+ 10.5480 / 9.5426 / oracle 28.0812 / 0.2734
  RefCOCOg 14.3046 / 12.4215 / oracle 31.7946 / 0.3161
Same-pool control (the same dense map on the SAM pool): 8.15/9.83/11.57 mIoU,
wrong-instance share 0.9667/0.9634/0.9524. The own-pool result is pool-limited;
on a good pool the same weak selector is wrong-instance dominated. Evidence:
`maskclip_sam_all.json`.

Blocked, with repo/reason/evidence (not sampled):
  TAS (https://github.com/suoych/TAS): the entry point requires a per-sentence
    `overall_caption_path/...json` the repo neither ships nor generates; README
    asks the user to prepare BLIP-2 captions first. Evidence blocked_tas.json.
  Ref-Diff (https://github.com/kodenii/Ref-Diff): test.py loads a trained
    `best_model.pth`; config resume is empty, pretrain/ has only a 2.9 MB stego
    head, and no released segmenter checkpoint exists. Evidence blocked_refdiff.json.
  Global-Local exact FreeSOLO pool: weight URL cloudstor... returns status 000;
    repo vendors only a FreeSOLO config and requires detectron2; no FreeSOLO
    code. Evidence blocked_globallocal_freesolo.json.

`exp/out_final/multipipeline_diagnosis.json` has 4 full pipelines (HybridGL
released 44.82/40.23/50.08; GLOD-RIS reimplementation 32.68/30.96/37.24;
Global-Local 32.95/36.31/41.62; MaskCLIP 8.21/10.55/14.30) plus the three
blocked entries, each with repo, full-split n, oracle, achieved and
wrong-instance share; each pipeline's evidence is a single fresh aggregate file.
Paper diagnosis section now carries Table 1 (one row per pipeline, achieved /
oracle and wrong-instance shares) plus the blocked-pipeline paragraph; the
blocked methods are cited (TAS, Ref-Diff, Global-Local, MaskCLIP).

Gate: `python3 extreme_bar.py` prints PASS after the paper rebuild (all checks,
including multipipeline_diagnosis with the new schema and PDF freshness).
Cross-family review blocked externally: mcp__llm-chat__review returned HTTP 429
(code 1113, "余额不足或无可用资源包"); a minimal glm-5.3 probe returned the same
429. Recorded in `exp/out_final/review_attempt.json`; not self-reviewed with the
executor family.

Protocol alignment per reproduction (2026-10-20, follow-up): every pipeline in
`multipipeline_diagnosis.json` now carries a `protocol` string and a
`published` block (per-dataset mIoU/oIoU plus source; MaskCLIP marked
`comparable: false` because its row is a harness adaptation). The diagnosis
`definition` states that all achieved/oracle values are single-protocol (our
unified official evaluator) and published numbers are reported separately,
cross-harness. The paper Table 1 caption states the same. The independent
GLOD-RIS reproduction row was removed from `sota_table.json` (the paper table
already had only the released HybridGL baseline row). Paper rebuilt; gate PASS;
pushed 5cea41e.

## 2026-10-20 (continued) — Intensive traversal, falsification, third full pipeline

Deep-read dossier expanded to 36 full-text papers (24 RIS, 12 adjacent), each
with a local source_file under exp/literature/papers/, verified arXiv/DOI id,
>=2 reported numbers, >=2 anchors, a verbatim quote and a code{} block. 13 new
full texts were downloaded and read (LGD, IteRPrimE, S+S, RAS, Venice-H1,
PinPoint, Prompt Sensitivity, MARS, E-SAM, ProposalCLIP, GF-SAM, T2S,
Semantic-SAM, plus HybridGL). New bib entries were added with verified titles/
authors. Successful repo runs now 9 (adds HybridGL official code and the
Global-Local/S+S reproductions).

Falsification mandate: 3 attacks recorded in multipipeline_diagnosis.json:
1) MaskCLIP own component pool (oracle 25.2/28.1/31.8, wrong share
   24.3/27.3/31.6%) REFUTES the diagnosis as universal: this pipeline is
   pool-limited.
2) Global-Local global-local encoder (wrong shares 95.1/94.2/92.5%) does NOT
   repair selection.
3) The same MaskCLIP dense map on the shared SAM pool is wrong-instance
   dominated (96.7/96.3/95.2%): the claim survives in its conditional form.

Third new full pipeline: S+S (Segment, Select, Correct) released zero-shot
select stage (`vp_and_max_sim_clip_choice`, reverse_blur, CLIP ViT-B/32,
area>=500), full official val on the shared SAM ViT-H pool (published stage-1
pool is expression-conditioned GroundingDINO+SAM, cross-harness stated):
  RefCOCO  29.4591 mIoU / 22.1236 oIoU, oracle 82.2113, wrong share 0.9538
  RefCOCO+ 35.3453 / 27.7031, oracle 82.3974, wrong share 0.9469
  RefCOCOg 38.9233 / 30.0590, oracle 79.1206, wrong share 0.9316
Published S+S zero-shot: mIoU 40.96/41.77/45.36, oIoU 34.39/36.29/39.29;
the achieved gap is the pool substitution, stated in the protocol block.
Evidence: exp/out_final/spluss_sam_all.json. The diagnosis now carries 5 full
pipelines + 3 documented blockers and the 3 falsification rows; paper Table 1
and the diagnosis text include S+S and the attack.

Gate: `python3 extreme_bar.py` = PASS after the paper rebuild. Review still
blocked externally (Zhipu 429 code 1113, balance); no self-review.

Review retry (2026-10-21): Zhipu access restored. Cross-family review completed
via mcp__llm-chat__review (thread 9a8e89251e7e; reviewer glm-5.3, executor
deepseek-flash). The first pass found a must-fix: duplicate SOTA rows with
stale change-16/17 provenance notes, plus the shipped.config tune metrics
being unlabelled and a stale "supervised gIoU 74.34 exceeded (74.72)" sentence
in protocol_audit. Fixes: deduplicated sota_table.json and corrected the note
to the actual DINO-free rule; added metrics_scope to main_grefcoco shipped
config; appended the current 47.5854/73.7800 row and the 0.56-below-74.34
boundary to protocol_audit, marking 74.72 superseded. Paper rebuilt, gate PASS.
Follow-up verdict: PASS, MUST-FIX: none. Recorded in review_attempt.json.

## 2026-10-21 — six-pipeline diagnosis, four gate-required new full entries, review PASS

Gate 26 now requires deep_read papers with appreciation/critique/gap and a
fresh critique_evidence path per paper, and >=4 NEW full pipelines. Added:
- exp/out_final/deep_read_critique_evidence.json: per-paper critique evidence
  for all 36 papers (source_file, critique, gap, verbatim quote, anchors).
- S+S released rectangle-prompt selection variant, full val on the SAM ViT-H
  pool (the rectangle option is documented in the released
  vp_and_max_sim_clip_choice): RefCOCO 8.9100 mIoU / 6.6632 oIoU, oracle
  82.2113, wrong-instance 0.9673; RefCOCO+ 10.1959 / 7.5794, oracle 82.3974,
  wrong-instance 0.9653; RefCOCOg 10.5672 / 7.3101, oracle 79.1206,
  wrong-instance 0.9533. Evidence spluss_rectangle_sam_all.json. This is a
  selection-prompt variant, not an independent method; the paper labels it as
  such and the published S+S numbers are marked comparable:false.
- The diagnosis is now 6 full entries (HybridGL, GLOD-RIS reimpl,
  Global-Local SAM, MaskCLIP own pool, S+S reverse_blur, S+S rectangle) +
  3 blockers + 3 falsification rows; evidence split:
  maskclip_sam_all.json (own pool) vs maskclip_sam_samepool_all.json (control).
- sota_table.json caption corrected: the independent GLOD-RIS reimplementation
  is not a row in the main table (operator removal); it lives in the appendix
  transparency record and multipipeline_diagnosis.json.

Updated paper: Table 1 now has 6 rows; the diagnosis text reports the S+S
rectangle prompt sensitivity (8.91/10.20/10.57 mIoU, wrong-instance shares
96.7/96.5/95.3%) as a released selection-prompt variant.

Review (2026-10-21, current package): cross-family review PASS, thread
96d0f0798668 (glm-5.3 / deepseek-flash). Must-fixes found and fixed: evidence
split for the MaskCLIP own-pool vs same-pool runs; rectangle published block
comparable:false; verdict/what wording qualification; stale GLOD-RIS caption
claim removed. Follow-up verdict PASS, none. Recorded in review_attempt.json.
Gate `python3 extreme_bar.py` = PASS after the rebuild.

## 2026-10-21 — continuous literature traversal (63 papers), synthesis, adopted GEM probe, review PASS

Raised the dossier from 36 to 63 full-text papers (40 RIS/same-class), with
appreciation, critique, gap and a fresh critique_evidence path on every entry;
22 source repositories inspected and 15 successful runs (full-split method
runs: HybridGL, S+S reverse, Global-Local, LLaVA-OneVision box rule,
Qwen2.5-VL, plain CLIP, MaskCLIP, GEM-only; component/metric runs: SAM ViT-B
pool, GroundingDINO cache, ReLA single- and multi-target batteries, FG-CLIP
via RefChess, SigLIP and DINOv2 smoke runs). Each paper carries both a
strengths statement (best idea and measured contribution) and an
evidence-backed critique; the new entries include earlygrounding2026,
blindposition2025, MaskRIS, Pseudo-RIS, HARIS, target-coherence, UniRES++,
RESAnything, GREx, ResAgent, Qwen3-VL-Seg, VespaSeg, PhysMLLMs, ELBO-T2IAlign,
EVF-SAM, Text4Seg, plug-in feedback, OVS-DINO, SCI-CLIP, MAVISEG, DSS, SatOV,
MLLMSeg, GEM, FG-CLIP, SigLIP and DINOv2.

Findings adopted into measurements:
- GEM-only selection on all three full val splits (exp189_gem_selector.py,
  gem_selector.json): 44.81/32.81/42.20 mIoU vs released 44.82/40.23/50.08.
  Coherence alone matches within 0.01 mIoU on RefCOCO but loses 7.4/7.9 on
  RefCOCO+/g, so the semantic term is load-bearing where attribute and
  compositional language dominate. The paper reports this as a scoped ablation;
  the shipped method is unchanged.
- External cross-checks: earlygrounding2026's independent table reports
  Global-Local ViT-B at 33.12/35.29/40.08 mIoU vs our 32.95/36.31/41.62
  (actual max difference 1.54, now stated as 0.2-1.5 points / within 1.6 mIoU),
  and a SAM-proposal MaskCLIP at 25.62/27.49/30.13 vs our dense-map adaptation
  8.21/10.55/14.30. The paper now labels the MaskCLIP row as an adaptation,
  not that method's strongest released route.
- RefChess released MCTS (FG-CLIP base + GEM, 120 RefCOCO val expressions,
  16 simulations) completed at 54.13 mIoU / 52.60 oIoU / acc@0.5 62.5 in 3m32s;
  a full 10,834-expression split would take ~5.3h per dataset, so it is
  recorded as an evidence-backed blocker and an FG-CLIP component run, not a
  full pipeline.

Artifacts: exp/out_final/literature_synthesis.json (17 ideas, 7 gaps, 7
venues, 8 years, 6 queries; one adopted-this-cycle measured idea);
exp/out_final/deep_read.json; exp/out_final/deep_read_critique_evidence.json;
exp/out_final/code_blockers.json (17 public-code papers not run, each with an
honest scope or artifact blocker); exp/out_final/backbone_smoke_siglip.json and
backbone_smoke_dinov2.json.

Corrections made in response to review: removed the "GLOD-RIS (reimpl.)" row
and its quoted wrong-instance shares from the main diagnosis Table 1 (the
independent reproduction is appendix-only and remains in
multipipeline_diagnosis.json); removed the "Published" column from the
same-pool rule-swap table; corrected the Global-Local agreement range; changed
the gRefCOCO learned acceptance prior in literature_synthesis from adopted to
rejected (measured but not shipped under the DINO-free requirement) and
labelled it "not shipped" in the appendix provenance.

Review (2026-10-21, current package): cross-family review PASS, thread
7e14b02315e0 (glm-5.3 / deepseek-flash), must-fix none after the four fixes
above; recorded in review_attempt.json. `python3 extreme_bar.py` = PASS.
