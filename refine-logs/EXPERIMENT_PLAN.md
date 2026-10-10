# Experiment Plan

**Run**: `zsris-20261001` · **Budget**: all 4× RTX 4090, shared with the RVOS job and an existing process on GPU 1. Check `nvidia-smi` and use cards that have free memory. Do not kill other processes. Total planned ≈ 3 GPU-days.

---

## 0. Why this plan looks like this

A method claim was proposed, piloted, and killed by its own control (see
`IDEA_REPORT.md` §Ranked Ideas, Idea 4). The reviewer's binding objection was that every
positive number was measured on a guidance-stripped scorer, and its single missing
measurement was an error taxonomy of the oracle gap. So this plan is ordered
**information-per-GPU-hour**, and EXP-1 costs no GPU forwards at all.

---

## 1. Must-run experiment blocks

### EXP-1 — Error taxonomy of the oracle gap  *(≈0 GPU-hours, gates everything else)*

**Question**: decompose the 0.8285 − 0.3501 = 0.4785 mIoU gap into attributable error classes.

**Data**: RefCOCO val, full split (10,834 expressions), or the existing 716-expression
held-out sample as a fast first pass.

**Method**: for each expression, compare the argmax proposal and the oracle proposal to GT
and classify:
- `correct` (argmax IoU ≥ 0.5)
- `under-segmentation` — oracle is a strict superset of argmax (argmax ⊆ oracle), the case
  the containment snap fixes
- `over-segmentation` — oracle is a strict subset of argmax
- `wrong-instance` — argmax and oracle are near-disjoint (IoU < 0.1)
- `boundary` — everything else

**Report**: the fraction of the total gap mass (Σ IoU_oracle − IoU_argmax) attributable to
each class, plus prevalence by GT-area quartile.

**Success criterion**: a clean split. Expected from the controls: `under-segmentation`
dominates the large-GT quartiles and vanishes in Q1, `wrong-instance` dominates the
residual. Whatever it shows, it becomes the paper's central table.

**Reads**: `PILOT_EVIDENCE.json`, `out_controls/summary_controls_allsent_c3.json`.

### EXP-2 — Guided replication of the two headline numbers  *(≈1 GPU-day, external validity gate)*

**Question**: do `oracle recovery` and `score flatness / rank-of-best` survive at the
**full HybridGL operating point** (published 41.81 oIoU / 49.48 mIoU), where the guidance
modules (relation, GEM coherence, position) are active?

**Method**: run the released HybridGL code on RefCOCO val with guidance intact; log, per
expression, the full per-proposal score vector, the oracle IoU, the rank of the best
proposal, and the score gap between the top-1 and the best-IoU proposal. Note that the
**oracle itself is scorer-independent** — it only depends on (image, GT, proposal pool) —
so the comparison isolates the selector.

**Pre-registered decision rule** (from the reviewer, adopted verbatim):

| Outcome | Action |
|---|---|
| flatness persists (top-1 vs best-IoU gap still ≪ genuine separation, rank-of-best still ≫ 1) | diagnostic generalises; write as a field-level analysis with the guided point as the main table |
| flatness vanishes (guidance restores a well-separated score) | **re-scope every claim** to the unguided operating point; the paper becomes "why the unguided selector fails", not a field-wide statement |
| oracle recovery ≥ 0.9 at the guided point | drop the framing entirely; the guidance modules already solve what the diagnostic describes |

**Must not**: present the stripped-scorer numbers as the headline (reviewer instruction).

### EXP-3 — Second configuration  *(≈2 GPU-days, required only for a main-conference framing)*

RefCOCOg (UMD) with CLIP ViT-L/14 (or SigLIP), same protocol. Purpose: show the flatness
measurement is a property of region-text scoring rather than of one checkpoint. The
reviewer made this a precondition for any main-conference analysis track.

### EXP-4 — Pre-registered method gate  *(only if a method claim is still wanted)*

**Do not run this hoping to revive SAGG.** It exists so that any future method claim is
pre-registered rather than post-hoc:

- Run on the **full HybridGL scorer**, complete RefCOCO val (10,834 expressions).
- First reproduce full HybridGL within ~1 mIoU of its published 49.48; if the reproduction
  is off by more, fix the pipeline before interpreting anything.
- Go/no-go: **Δ ≥ +2 mIoU over full-HybridGL argmax, paired bootstrap CI excluding 0,
  oIoU not degraded, and the part-expression stratum Δ ≥ −1 mIoU.** Anything less →
  no method paper.
- The `parent_strict` containment snap must be reported as a baseline in the same table.

---

## 2. Controls that must appear in any write-up

Every one of these is a real risk of the analysis being an artifact. All are already run.

| Control | Purpose | Result |
|---|---|---|
| `parent_strict` containment snap | is the gain just SAM-hierarchy snapping? | +0.0654 vs growth +0.0670 → **yes** |
| `parent` (most-overlapping larger) | is it "go bigger"? | −0.1531 → no, containment specifically |
| `dilate_{2,5,10,20}` | raw area inflation? | +0.006 / −0.001 / −0.022 / −0.057 → no |
| `largest_gated` | "pick the big one in the gate"? | +0.0051 → no |
| `gated_union` (MARS as published) | does the published operator work here? | −0.0114 → no |
| gate tolerance sweep 0.90→0.99 | hyper-parameter robustness | sharp interior peak at 0.95; +0.018 at 0.99 → fragile |
| GT-area quartile stratification | is the gain size-dependent? | Q1 −0.0009 … Q4 +0.1340 → **entirely size-dependent** |
| part-expression stratification | does it hurt parts? | −0.0110 on part expressions → **yes** |
| paired bootstrap (10k) + permutation | is the gain significant? | +0.0615, CI [+0.047,+0.077], p<0.0001 → significant but see above |

## 3. Data and metrics

- **Data**: RefCOCO / RefCOCO+ / RefCOCOg (already downloaded: annotations + on-demand
  COCO train2014 images under `glod_ris_repro/data/refer`). **No gRefCOCO locally** — it
  would need a separate download and is out of scope for the analysis paper.
- **Metrics**: mIoU (per-expression mean, the primary) and oIoU (dataset-level, reported
  alongside). Report both always: the controls show mIoU and oIoU can move differently,
  and that difference is itself diagnostic.
- **Always report `oracle_mIoU`** for the same pool. It is the only scorer-independent
  reference point and it is what makes the analysis comparable across papers.

## 4. Statistical protocol

Paired bootstrap (10,000 resamples) over expressions for every reported delta, plus a
sign-flip permutation test for the p-value, plus win/loss/tie counts. Never report a
delta without its CI. **Any hyper-parameter must be selected on one sample and reported
on a disjoint one** — this plan's numbers already obey that (selection on seed-101
first-sentence, reporting on held-out all-sentences and on fresh seed 202).

## 5. Expected failure points (stated in advance)

1. **The flatness may not replicate at the guided operating point** (EXP-2). This is the
   single largest risk; the mitigation is re-scoping, not re-analysis.
2. **The taxonomy may not be cleanly separable** — `wrong-instance` and `under-segmentation`
   will overlap on part-whole ambiguity. Mitigation: make the classifier deterministic and
   report the ambiguous bucket explicitly rather than forcing a split.
3. **n is small for part expressions** (26). Do not draw a strong conclusion from that
   stratum; report it as suggestive and note the count.
4. **One proposal generator, one CLIP backbone, one dataset.** Any field-level claim is
   unsupported until EXP-3.

## 6. Run order

```
EXP-1 (0 GPU-h)  ->  EXP-2 (1 GPU-day)  ->  [write up]
                                    \-> EXP-3 (2 GPU-days) only if a main-conf. framing is wanted
EXP-4            ->  only if a method claim is revived; obey the pre-registered rule
```
