# Review digest — 2026-10-12/13 cycle (four new gated artifacts + paper restructure)

Machine artifacts to check this digest against: `exp/out_final/rule_swap.json`,
`exp/out_final/wild_ideas.json`, `exp/out_final/learned_prior.json`,
`exp/out_final/test_columns.json`; paper sections `05_experiments.tex`
(§Same-pool rule swap, §Counterintuitive probes, §Learned localization prior,
§Full-table completion), `01_intro.tex`, `main.tex` (abstract); tables
`ruleswap.tex`, `learned.tex`, `testcols.tex`; `refine-logs/METHOD_GRAVEYARD.md`;
`ENDGAME.md` (2026-10-12 cycle + resolution).

## 1. Same-pool rule swap (`rule_swap.json`; instruction of 2026-10-12)

GLOD-RIS's own selection rule (Eqs. 1-7 decoupling + 3x3 Gaussian guidance term
+ category term, constants from the paper; implementation = the reproduction's
`glod/core.py` verbatim) run on our exact records, SAM ViT-H pools and cached
G2L CLIP features, with GLM-4.1V guidance read from the reproduction's full-val
cache (coverage 10,819+15, 10,744+14, 4,896+0; all-zero grids 92/169/64 handled
identically). Val mIoU/oIoU: RefCOCO 32.95/28.18, RefCOCO+ 30.67/25.92,
RefCOCOg 35.03/30.06. The released ancestor on the same records reproduces its
shipped row to the decimal (44.82/38.16, 40.23/33.59, 50.08/41.45), validating
the swap end to end. Causal reading stated in the paper: with pool, features,
guidance and evaluation fixed, the gap is the rule as transplanted (including
its interaction with our feature geometry) and its guidance usage, not the
candidate pool or the frozen components.

## 2. Counterintuitive-mechanism probes (`wild_ideas.json`; >= 6 probes)

Fixed 2,000-row RefCOCO val sample, released pick baseline 42.94 mIoU.
(a) Set-of-Mark index selection 39.13 (parsed-only 39.53; 2/200 unusable --
the two fallback rows back-solve to IoU ~0 on their released picks too, so the
fallback behaved as intended and changed nothing);
(b) counterfactual deletion 42.65; (c) negative/elimination 40.99; (d)
weak-signal democracy 43.61 (+0.67; 95% CI [-5.9,+7.2] paired on n=200, 108/200
picks changed -- null-consistent, diagnostic-only, not promoted); (e)
boolean mask arithmetic: fixed unions top-2 37.77 vs single 35.04, pair-oracle
63.52 (construction headroom, simple policies capture little); (f) prompt-format
sensitivity 14.48 mIoU (noun 53.84 vs sentence 68.32; paraphrases change nothing);
(g) grid-vote aggregation 11.11. Losers recorded in the graveyard with numbers.

## 3. Learned localization prior (`learned_prior.json`; Punch-2 directive)

Declared trained component (HistGradientBoostingRegressor, 12-dim per-part
features identical to exp138; fitted on 70% of gRefCOCO TRAIN references,
threshold tuned on the held-out 30% carve-out; val a single frozen application
per arm). Arms measured on full val (n=14,229): 38.79 naive (absolute threshold)
-> 41.63 (+relative threshold & standardization) -> 38.91 (+degradation
augmentation on top: it HURTS) -> 50.52/73.91 (+within-row rank features, no
augmentation; best learned arm, also best on the held-out carve-out 56.09 vs
55.08, so arm selection is held-out-based) -> 45.47 (rank + augmentation:
augmentation hurts again). T1 (exceed the prompt-prior 51.32/74.91) NOT met, by
0.8 cIoU; the paper says so plainly, keeps the hand-tuned prompt rule as the row
of record, and reports the learned prior as a measured transfer study. Scale
arm: 6,000-ref train-side build in progress (15,975 records, 0 skipped).

## 4. Full-table completion (`test_columns.json`; Pareto ledger)

Full official test splits, same frozen components, aligned protocol; published
mIoU AND oIoU transcribed from GLOD-RIS Table I test columns (the oIoU blocks
verified against the audited val oIoU 50.85/45.96/50.03 before transcription).
ours mIoU/oIoU vs published mIoU/oIoU: RefCOCO testA 71.82/67.66 vs 59.36/51.37;
testB 60.62/54.71 vs 56.23/49.91; RefCOCO+ testA 70.66/66.62 vs 57.33/49.07;
testB 55.64/48.43 vs 48.73/41.31; RefCOCOg test(u) 64.54/59.61 vs 57.03/48.99.
Every column strictly exceeds the published row (mIoU margins +12.46/+4.39/
+13.33/+6.91/+7.51; oIoU +16.29/+4.80/+17.55/+7.12/+10.62). Box-cache
degenerate greedy decodes (408 on testA) repaired via exp151 (guarded decode,
disclosed); the unbounded LLaVA feature cache was bounded after an OOM on the
2,500-image test(u) split.

## Review questions

1. Do the four artifacts' numbers match what the paper text and tables claim?
2. Is the rule-swap causal reading sound given the exact reproduction of the
   released-ancestor rows on the same records?
3. Is the learned-prior framing honest (T1 missed; trained vs frozen declared)?
4. Is the Pareto ledger complete and non-overreaching? Must-fix list, verdict,
   score.
