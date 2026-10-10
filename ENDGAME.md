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
