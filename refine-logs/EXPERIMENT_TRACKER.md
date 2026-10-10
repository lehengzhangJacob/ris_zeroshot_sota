# Experiment Tracker

Run `zsris-20261001` · executor `deepseek-flash` · reviewer `glm-5.3`
All artifacts under `exp/`. Consolidated evidence: `exp/out_final/`, `NARRATIVE_REPORT.md`.

## Status

| ID | Experiment | Cost | Status | Artifact |
|---|---|---|---|---|
| EXP-1 | Error taxonomy of the oracle gap | 0 GPU | **done** | `exp/out_exp5/taxonomy_refcoco_guided.json`, `exp/out_exp1/` |
| EXP-1b/1d/1e | Unguided-point spatial selector search + true-IoU controls | 0 GPU | **done** | `exp/out_exp1b/`, `exp/out_exp1d/`, `exp/out_exp1e/` |
| EXP-2 | Full RefCOCO val, released HybridGL scorer (10,834 expr.) | 0.9 GPU-h | **done** | `exp/out_exp4/records_guided_val.jsonl` |
| EXP-6 | Ablations (vocabulary tiers, k, δ, relation strata) | 0 GPU | **done** | `exp/out_exp6/ablations_s101.json` |
| EXP-7 | Candidate-set headroom diagnostics | 0 GPU | **done** | `exp/out_exp7/headroom_refcoco_guided.json` |
| EXP-8..10 | Method rebuild on the diagnosis | 0 GPU | **done** | `exp/out_exp8.log`, `exp9_guided_v3.py`, `exp10_nosnap.py` |
| EXP-11/12 | Final tables, controls, sensitivity, held-out | 0 GPU | **done** | `exp/out_final/final_refcoco.{json,md}` |
| EXP-13 | Scope predictor (RefCOCO vs RefCOCOg) | 0 GPU | **done** | `exp/out_final/scope.json` |
| EXP-3′ | Second dataset: RefCOCOg val, released scorer | 0.5 GPU-h | **done** | `exp/out_final/final_refcocog.{json,md}` |
| EXP-4 | Pre-registered method gate | — | **PASS** | see below |

Total spend ≈ 1.6 GPU-hours on 4× RTX 4090 (one job per card; GPU 1 avoided while
another user's process held it; no process of another user was touched).

## Pre-registered gate (EXP-4), full HybridGL scorer + full RefCOCO val

| criterion | required | measured | verdict |
|---|---|---|---|
| Δ mIoU over argmax | ≥ +2.0 | **+15.10** | pass |
| paired bootstrap 95 % CI excludes 0 | yes | [+0.0493, +0.0601] vs released; p < 0.0001 | pass |
| oIoU not degraded | yes | 0.4431 vs released 0.3816 | pass |
| part expressions | Δ ≥ −1.0 mIoU | **+4.39** | pass |
| `parent_strict` present as a row | yes | 0.5024 / 0.4072, Δ +0.0541 | pass |

The method (`exp/method.py`) also beats the `parent_strict` baseline on both metrics
(mIoU 0.5029 vs 0.5024, oIoU 0.4431 vs 0.4072) while targeting the error class that
holds 77 % of the oracle gap.

## Development history (why the method looks like this)

1. First attempt — a purely geometric resolver over a score-tied confusion set —
   worked at the unguided scorer (+0.1331 mIoU on the 716 sample) but **failed the
   gate** at the released scorer (+0.0093). Recorded, not shipped.
2. EXP-7 located the cause: the released guidance only ever chooses among
   `topk(score_clip, 3)`, and on wrong-instance failures the referent is in that set
   30 % of the time (mean rank 9.1), but in the top-8 of the *category* score 70 % of
   the time.
3. The method was rebuilt around that finding: enlarge the candidate set to the
   category score's top-8, gate by the relation, and rank with the pipeline's own
   coherence score. That passes the gate.
4. An ordinal resolver was implemented and measured; it lost 0.153 mIoU on its
   stratum, so the operator abstains on ordinals. Recorded as a limitation.

## Controls (all on full RefCOCO val)

| control | Δ mIoU | purpose |
|---|---|---|
| geometric extremum of the candidate set | −0.0549 | not "pick the leftmost mask" |
| rank by category score instead of coherence | −0.0024 | semantic re-ranking alone does nothing |
| rank by full semantic score | −0.0117 | idem |
| mirrored relation | −0.1813 | relation semantics are real |
| whole pool instead of top-8 category set | +0.0484 | category restriction is a refinement |
| coherence ranking with no relation gate | +0.0430 | side gate adds +0.012 |

## Open risks carried forward

1. **Scope boundary**: neutral on RefCOCOg (Δ −0.0034, CI touching 0), where spatial
   expressions are 3× rarer. Predicted by the diagnostic; not tuned away.
2. **Reproduction gap**: released point reproduces at 0.4482/0.3816 vs published
   49.48/41.81 on RefCOCO val; 0.5008/0.4145 vs 51.25/42.47 on RefCOCOg. Internal
   comparisons unaffected; absolute rows not comparable to the published table.
3. **Ordinal expressions abstain** (n = 444).
4. **One proposal generator, one CLIP backbone.**
5. The unguided-point selector result uses a 716-expression sample, not the full split.
