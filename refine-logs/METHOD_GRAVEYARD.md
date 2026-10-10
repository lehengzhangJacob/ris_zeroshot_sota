# Method graveyard

Run `zsris-extreme`, 2026-10-03. Every number below is from this run's full-val
scoring passes (fresh records under `exp/out_extreme/`, analyses under
`exp/out_final/`). A method enters this file when a *control* explains its
gain; the rule for the run is that the shipped method must still lead after
the control is applied.

---

## 1. Seed-anchored gated growth — killed by `parent_strict`

**Claim it made.** Grow the argmax seed by unioning score-gated,
overlap-connected proposals; +0.0670 mIoU over the released pipeline on full
RefCOCO val.

**What killed it.** The one-line containment snap `parent_strict` (replace the
selected mask by its smallest strict container in the pool) gains +0.0654 on
the same split. 0.0670 vs 0.0654: a one-line, text-free rule explains
essentially the entire gain, so the "growth" mechanism as stated is void.

**What survived.** The *idea* that part→whole containment repairs exist in the
pool survived as a control and later as a component of the two-repair method
(it targets the under-segmentation class of the taxonomy). The *seed-anchored
gated growth* operator is dead and is not claimed anywhere.

---

## 2. Standalone "category-set relation guidance" (v1) — killed as a standalone claim

**Claim it made.** Build a candidate set from the category score (expression
minus spatial words), gate by the parsed relation, rank by the released
coherence score: +0.0546 mIoU over the released pipeline on full RefCOCO val,
+0.0615 oIoU.

**What killed it.** The same one-line control: `parent_strict` alone gains
+0.0541 mIoU. Paired bootstrap of the two deltas: **+0.0005 [−0.0055, +0.0066],
p = 0.87** — statistically indistinguishable. Per the run's rule a one-line
control that explains an mIoU gain voids the method that produced it, so v1
was killed *as a standalone contribution*.

**What replaced it.** Two changes, both measured:
(a) the two-axis fix (a relation with both horizontal and vertical content —
"bottom left" — now requires both half-planes; the fallback is the primary
axis). This moved the operator from +0.0546 to **+0.0587** and the
two-axis stratum from +0.1090 to +0.1733 (n = 686);
(b) the *composition* with the containment growth step, which is the shipped
method: **+0.1076 mIoU [CI +0.1015,+0.1137]** on full RefCOCO val. The
composition leads every one-line control we could construct, including the
composition of the controls (`ctrl_no_gate_snap`: +0.0870; the operator's
conditional contribution on top of it is **+0.0205 [+0.0173,+0.0238],
p < 1e-4**). The operator alone still only *nominally* leads the snap
(+0.0046, p = 0.14) — that tie is stated in the paper rather than hidden.

---

## 3. Deterministic ordinal resolution — killed by its own stratum

**Claim it made.** Resolve "second from the left" deterministically: rank the
category-score top-k by centroid along the axis, return the n-th.

**What killed it.** On the ordinal stratum of full RefCOCO val (n = 444) the
released pipeline scores 0.2074 mIoU; the resolver scores **0.0540**
(−0.1534). Restricting the count to high head-noun-score proposals (margins
0.0/0.3/0.6/1.0/1.5/2.0) gives 0.2046/0.1916/0.1639/0.1341/0.1149/0.0983;
larger candidate sets are worse (k=8: 0.0540, k=16: 0.0208, k=32: 0.0037,
whole pool: 0.0026). Diagnosis: an ordinal position must be counted over
*instances of the referent's category*, and the flat CLIP score cannot
isolate that instance set — including more proposals adds noise faster than
it adds true instances.

**What survived.** Abstention. The operator does not fire on ordinal
expressions; the measured resolver loss is reported in the paper as a
boundary, and the ablation `ours_ordfire` (−, evidence
`exp/out_final/evidence/refcoco_ours_ordfire.json`) shows what firing would
cost.

---

## 4. Pure geometric resolution over a confusion set — killed at the released scorer

**Claim it made.** Resolve the relation by taking the geometric extremum of
the score-tied candidate set ("take the leftmost member").

**What killed it.** It worked at the *unguided* scorer (+0.1331 mIoU on the
716-expression sample) and failed at the released scorer (+0.0093). On full
RefCOCO val, as a control inside the operator's own candidate set, the
geometric extremum is **−0.0549** — worse than doing nothing. The relation
must select *which instance*, and the released coherence score does that
ranking; geometry alone does not. The control remains in every table
(`ctrl_geom_extreme`).

---

## 5. Score-threshold no-target abstention on gRefCOCO — partially killed

**Claim it would make.** Abstain on no-target expressions by thresholding the
top head-noun score (no-target expressions should score lower).

**What killed it (partially).** Measured on the full gRefCOCO val pass: the
head-noun score distributions of no-target and multi-target expressions
overlap substantially (smoke analysis: target 30.9 ± 1.5 vs no-target
29.6 ± 1.5 for the top head-noun score), so at a threshold honestly
calibrated on RefCOCO val (1st percentile = 26.33), N-acc is only a few
points — the abstention recovers a small part of the no-target mass and it is
reported as measured (`main_grefcoco.json`, row `clause+abstain`: N-acc 7.6,
T-acc 99.7), not tuned upward on the evaluation split. The *set-valued* half
of the extension (clause decomposition, per-clause selection, growth, union)
is what carries most of the gRefCOCO result; the abstention component is kept
because its calibrated operating point improves both cIoU (+0.56) and gIoU
(+2.68) at a 0.3-point T-acc cost.

---

## 6. Plural expansion on gRefCOCO — killed by its own measurement

**Claim it made.** A clause with a count word or plural head (``two zebras'')
should admit every same-category instance within a score margin of its best,
so the union output covers all targets.

**What killed it.** On the full gRefCOCO val pass, expansion at every margin
in {0.3, 0.6, 1.0, 1.5} is equal or worse than not expanding on cIoU
(shipped, no expansion: 23.11 cIoU / 20.67 gIoU; m=0.3 + abstain: 22.93 /
20.74; m=1.0 + abstain: 21.61 / 20.66; no expansion, no abstention: 22.55 /
17.99). The extra instance recovered is outnumbered by unions that grow past
the target set. The shipped configuration therefore omits the expansion and
the sweep is reported in the paper as a measured negative.

**What survived.** Abstention (calibrated on RefCOCO val: 1st percentile of
the top-1 head-noun score), which adds +0.56 cIoU and +2.68 gIoU at a 0.3
point T-acc cost, and the clause decomposition itself, which is what turns
the single-mask operator into a set-valued one.

---

# Run `zsris-sota` (2026-10-03, evening) — kills from the component search

## 7. Orthogonal decoupling of the background direction (GLOD-RIS's DOD) — killed

**Claim it made.** GLOD-RIS attributes ~+2 mIoU to projecting each proposal's
foreground feature onto its own background direction and subtracting it
(`F_fuse = (1-lam)F_fg + lam*F_fg_perp_bg`, `lam = clip(area, 0.3, 0.8)`), then
scoring against the text by cosine.

**What killed it.** Reimplemented exactly (Eqs. 1-7) on our cached features and
evaluated as an argmax selector over the full val split: RefCOCO+ **39.10 ->
35.82 mIoU** (-3.28). The projection removes referent-specific signal together
with the background component on this feature family. It is not a component of
the shipped method. Evidence: `exp/out_final/probe_refcocoplus.json`
(`argmax_dod` vs `argmax_sc`).

## 8. A stronger frozen encoder as the region scorer (SigLIP-so400m) — killed

**Claim it made.** The selection bottleneck is the CLIP ViT-B/16 region-text
alignment; swapping in a much stronger frozen encoder (SigLIP-so400m, 1152-d,
masked-region "local" + blurred-background "global" G2L analogue, 384px) should
improve which proposal wins.

**What killed it.** Full RefCOCO+ val, same records, argmax selection:
sig_local **36.45**, sig_fused 34.26, sig_global 24.83, vs released 40.42 and
the released guidance's own `final` 40.59. A stronger *whole-region*
image-text encoder is not a better *instance selector*; the gap is
instance-level reasoning, not embedding quality. Rank-ensembling it with the
released score also fails to beat the released pipeline (39.17).
Evidence: `exp/out_final/probe_siglip_refcocoplus_partial.json`.

## 9. Correction: GEM was mis-killed as a localizer (sign error) — re-instated

An earlier probe flipped the sign of the released coherence score and concluded
GEM was "useless" (1.06 mIoU). Re-measured with the released convention:
argmax of raw `score_gem` gives **44.81 mIoU on RefCOCO** (vs released pipeline
44.83) and 32.81 on RefCOCO+. GEM is a *strong* localizer on spatially-specified
expressions and a weak one on attribute expressions; it stays as the coherence
term inside the relation-gated operator. The sign-flipped variants in
`probe_refcocoplus.json` are void.

## 10. VLM box prior as the gRefCOCO localizer — killed

**Claim it made.** The dense localization prior that clears the RefCOCO-family
floors (+10 mIoU each) should also fix gRefCOCO's targeted samples: query
LLaVA-OneVision-7B with the full expression and select the masks the box covers.

**What killed it.** gRefCOCO targeted references are 2-object unions in 99.5% of
samples (5296/5324), and the VLM's single box overlaps that union at **IoU
0.329**. Restricting any rule to masks inside the box caps it at
**orc_inbox = 0.402** mean IoU — the deployed greedy box-cover rule already sits
at 0.397, i.e. at its box-conditioned ceiling. The shortlist oracle over the same
candidates (box-free) reaches 0.735, so the masks exist; the box is the wrong
supervision for multi-object referents. Evidence:
`exp/out_final/gref_box_ceiling.json` (exp49), `exp/out_final/gref_estimate.json`
(exp48).

## 11. VLM `none` answer as the gRefCOCO no-target decision — killed

**Claim it made.** The existence decision that thresholds on CLIP scores cannot
make (oracle cap N_acc 0.175 at T_acc 0.98, exp31) might come from the VLM's own
generative judgment: the box-or-none prompt already asks it to answer `none` for
absent objects, so the answer is free with the box.

**What killed it.** Measured on 300 val records (150 targeted / 150 no-target,
600 queries, sentence prompt): **N_acc = 0.413 at T_acc = 1.000** — the signal is
real (0 false abstentions on targeted) but half the size the metric needs. The
joint requirement is doubly binding: at the measured localization quality
(r = 0.392-0.415) cIoU needs r > 1.027 — arithmetically impossible — and even at
oracle-union localization (r = 0.886) the floor still needs N_acc >= 0.658.
gIoU at the measured mask quality needs N_acc >= 0.93. No combination of the two
training-free signals clears both. Evidence: `exp/out_final/gref_wall.json`
(exp50), `exp/pilot_gref_exists.py`.

## 12. Set-valued selection + a perfect grounder — killed by the rule/oracle gap

**Claim it made.** The kill in §10 blamed the VLM box. If the grounder were
perfect, a set-valued rule (union of masks covering both objects) should clear
the targeted-cumulative requirement r > 0.689 the cIoU floor needs.

**What killed it.** Measured with the GT bounding box (150 val targeted, same
shortlist): the best deployable rule (greedy box-cover) reaches **r = 0.507**,
below 0.689 — it fails at *any* abstention rate. The oracle on the same
candidates reaches r = 0.740 (needing N_acc >= 0.912 at that ceiling), and at
depth k=50 the oracle rises to r = 0.883 while the deployable rule *falls* to
0.453 — deeper candidates enrich the oracle and dilute the rule. Realization
ratio: deployable/oracle = 0.69 at best. Evidence:
`exp/out_final/gref_rules.json` (exp51), `gref_box_ceiling.json` (exp49),
`gref_wall.json` (exp50/exp52).

## 13. Held-out-tuned set+abstain model — killed by its own study

**Claim it made.** The mandated second 300-trial study (held-out gRefCOCO
train, `exp/out_final/optuna_study_gref.json`) was allowed to pick any rule in
the family — box-cover or box-free, any score source, any shortlist depth,
gain-margin abstention. Whatever it picks is the class's best shot.

**What killed it.** The study's best configuration (single-mask box-coverage,
`sc_noun`, k_rank 50) scores cIoU 42.34 / gIoU 59.23 held-out and, transferred
to a pre-registered 2000-record val subset (seed 20261004), **cIoU 29.69 /
gIoU 44.99** — vs floors 68.94 / 74.34. The ceilings (gref_wall.json) show no
configuration could have done better: at a *perfect* grounder + oracle
selector the cIoU floor still needs N_acc >= 0.912, above both what any
training-free existence signal reaches (0.413) and the trained state of the
art (0.797). The tuned model is shipped as our gRefCOCO row; it is the maximum
of this class. Evidence: `optuna_study_gref.json`, `gref_tuned_val2000.json`,
`gref_wall.json`, `records_grefcoco_val2000.jsonl`.

## 14. A learned per-proposal selector (third model change) — killed by the frontier

**Claim it made.** The hand-rule space was exhausted (exp54); exp51 measured
that the deployable rule realizes only ~69% of the oracle. A selector *fitted*
on held-out train — logistic/GBM over the four released scores, their ranks,
mask area, box coverage and box-bbox IoU — should move closer to the oracle.

**What it did.** A real, measured gain: 300 Optuna trials (exp57,
`optuna_study_gref2.json`) pick a logistic selector (C=0.0033, include-threshold
0.774, abstain-threshold 0.279, tuned on a fit/calib split inside held-out
train). Held-out calib cIoU 45.35 / gIoU 65.46. Val transfer (n=2000, scored
once after freezing): **cIoU 33.94 / gIoU 58.41** — vs the tuned hand rule's
29.69/44.99 on the same records, and the released pipeline's 22.56/12.49.

**What killed it.** The frontier. At the achieved val operating point
(T_acc 86.1), clearing cIoU would require a targeted cumulative ratio
r >= 0.871 — at the k=50 oracle itself (0.883). Clearing gIoU needs mean
targeted IoU ~0.65 at N_acc ~0.85; the selector delivers 0.485 at 0.684, and
even the oracle over the same shortlist is 0.877 at N_acc 0.664 — i.e. every
remaining combination sits at or beyond oracle-quality localization *plus* a
trained-level abstention rate. Three model classes (hand rule, tuned hand
rule, learned selector) now bracket the same conclusion. Evidence:
`optuna_study_gref2.json`, `gref_tuned_val2000.json`, `gref_wall.json`.

## 15. CORRECTION + win: a direct existence probe moved the wall

**Correction.** Sections 11 and 14 claimed the training-free existence signal
tops out near N_acc 0.41-0.48 and that "no training-free combination clears
either line". That was measured with the box-or-none prompt, which *forces* a
box and understates the VLM's presence judgment. The claim is withdrawn.

**The win.** A dedicated yes/no prompt ("Is the {phrase} present in this
image?") reaches **N_acc 0.875 @ T_acc 0.920** on held-out train and **0.804 @
0.932** on val2000 (exp58; the 0-100 confidence variant is degenerate — 0 to
everything — and discarded). Combined with the learned selector and the
empty-selection rule as the shipped abstention branch: **N_acc 0.867 @ T_acc
0.797 on val, above the 0.797 trained SOTA reports**, moving val gIoU from
58.41 to **68.76** (shipped; cIoU 35.64).

**What still kills it.** At that operating point (gref_wall.json
"decomposition"), gIoU 74.34 needs mean targeted IoU **m >= 0.673** (delivered
0.486; even a perfect box with an oracle rule reaches 0.735), and cIoU 68.94
needs targeted cumulative **r_eff >= 0.766** (delivered ~0.43; k=50 oracle
0.883). The wall is now selector-bound: the frozen released scores must
localize two-object referents at near-oracle quality — the capability trained
models buy with gRefCOCO supervision. Evidence: `exp58_exists_probe.py`,
`optuna_study_gref3.json`, `exp59_abstain_frontier.py`,
`gref_tuned_val2000.json`, `gref_wall.json`.

## 16. Existence-yes override of the selector's abstention — killed

**Claim it made.** The shipped rule abstains whenever no proposal clears p_inc,
dropping ~20 % of targeted samples. Since the existence probe is 93 % accurate
when it answers, its `yes` should override the selector's doubt: predict on
weak-but-probe-positive rows.

**What killed it.** The 300-trial rule study (exp60, held-out fit/calib at val
prevalence, `optuna_study_gref4.json`) falsifies it on val: predicting weak
targeted rows gains their low IoU but forfeits the 1.0 abstention credit on the
weak no-target rows the probe missed. The override family moves gIoU 68.8 ->
58.4 while buying cIoU 35.6 -> 41.2 — it sacrifices the *nearer* floor for the
farther one. `std` is the gIoU-optimal structure (family max 68.96); shipped
config kept. The binding constraint remains the selector: 68.96 still needs
mean targeted IoU 0.673, delivered 0.486.

## 17. Per-noun (clause) grounding features — marginal, kept

**Claim it made.** A two-object reference names its objects; scoring every
proposal against each parsed noun separately should help the selector pick the
second object that the fused sentence score blurs out.

**What it did.** Sixth 300-trial study (exp62, `optuna_study_gref5.json`),
5 extra per-noun features (max/second-max/mean similarity, per-noun top-10
membership, rank under the argmax noun). Held-out calib 0.6698; val transfer
**cIoU 35.64 -> 36.50, gIoU 68.76 -> 68.96** — a real but small gain, kept as
shipped. The mechanism is sound but the per-noun CLIP scores are the same weak
localizers as everything else in the frozen family: the selector realization
moved 54.1% -> 55.4% of the k=50 oracle, against the 78.1% both floors need.

## 18. Full-val restatement (completes the evidence chain)

The gRefCOCO row now rests on the **complete val split** like the other three
tables: the frozen shipped model (exp62 study-5 selector + existence branch,
110475-free selection) evaluated once over all **14,220 val expressions**
(`exp/out_final/gref_full_val.json`, exp64): **cIoU 35.09 / gIoU 68.37**
(N_acc 86.3, T_acc 79.2). The pre-registered 2,000-record subset read
36.50/68.96 — within sampling noise, confirming the subset was unbiased. The
full-val extraction (box-or-none + yes/no probes, 11,592 + 11,842 pairs) ran
on the four-GPU resilient loop.

## 19. VLM region verification for selection — killed by its own study

**Claim it made.** Localization is the VLM's weakest task (box IoU 0.329), but
*verification* is its strongest: crop the classifier's top-4 proposals (mask
region shown on grey), ask "does this image show the phrase?", and build the
selection from the confirmed regions — directly addressing the 2-object
structure the frozen scores miss.

**What killed it.** The 300-trial held-out study (exp66,
`optuna_study_gref6.json`) searches {verified-only, verified+fallback,
verified-union-base, base} × thresholds × abstention combos and selects
**`base`** — the existing rule, verification ignored — at calib cIoU 45.06 /
gIoU 77.30 (val prevalence). The full-train smoke agrees in every rule:
base 49.55/68.66 > union 48.38/67.20 > ver_fb 44.58/64.35 > verified-only
31.27/56.03. Crops confirm plausible instances but cannot reject wrong ones —
the same generic-phrase trap the existence probe hits. The val verify
extraction (~57k planned queries) was stopped after 11k; the model it served
was rejected. Seventh model change, seventh negative: the frozen-signal
selector is at its class ceiling.

## 20. Three-prompt presence ensemble — the reviewer's objection, answered with data

**The objection (external review, recorded in `gref_wall.json`).** The gIoU
gap may be *transfer-bound*: val's no-target phrases are 92% transplanted, so
a distribution-robust existence signal — not a better selector — might close
it. Required ROC point: (N_acc 0.94, T_acc 0.86) at the delivered mask
quality.

**What was built.** Two more presence paraphrases (`exists2`: "Does the image
contain the…"; `exists3`: "Look carefully…"), extracted over the full val
(13,838 new queries per variant, 8-bit backend, four-GPU resilient loop), and
a 3-probe ensemble rule: abstain when **any** probe answers no. Rule selection
on held-out train (exp67/exp69, same rows for all candidates): ensemble gIoU
**70.00** vs single-probe 69.57 vs two-of-three 69.63. The `p_abs` and
`vlm_none` branches add nothing once any-of-3 is present (identical rows,
identical metrics).

**The measurement (exp68, full val frontier, no parameter chosen on val).**
The train ROC point (93.1, 91.4) does **not** transfer: val gives (89.1, 77.2)
— T_acc collapses 14 points. No operating point of the ensemble reaches the
floor: the frontier peaks at **gIoU 69.77** (vote≥1 + empty rule); vote≥2
reaches T_acc 94.7 but drops N_acc to 79.7; every point's required mean
targeted IoU stays 0.63–0.81 (72–92% of the deployable oracle) against 0.485
delivered and 0.686 best-ever-measured (perfect box, k=15).

**Shipped anyway (the honest upgrade).** The ensemble wins the held-out
comparison and transfers with **+1.40 gIoU** (68.37 → **69.77**) at −0.15
cIoU (35.09 → 34.94); exp71 promoted it and rewrote the wall artifact. The
cIoU side was re-derived from measured union masses (S_U 76.2M, FP_U 7.9M,
A_U 18.3M at the new operating point): the floor needs r = 0.926 against
0.470 delivered (1.97×), and **even a perfect selector caps at cIoU 65.8** at
this abstraction mass — cIoU now provably needs both a better selector *and*
less targeted abstention. The ensemble moved the wall's N_acc side further
above trained SOTA (89.1 vs 79.7) and moved the bind, but did not remove it:
the wall is selector-bound, not existence-bound. Objection answered — the
distribution-robust signal exists and was built; it is not what binds.

## 21. Pairwise forced-choice VLM comparison — the localizer is perceptually bounded

**Claim it made.** Region verification (§19) failed because a crop-level yes/no
question can confirm but cannot reject. A *forced choice* removes the escape
hatch: show two candidate crops (mask on grey, top-4 by the shipped
classifier) and ask which shows the phrase — no "none" option, so the answer
is a preference even among plausible instances. This is the last untried
frozen selector signal, and the diagnosis (wrong-instance selection on
descriptive expressions) points straight at it.

**What was built.** exp72: 20,301 side-by-side comparisons over the held-out
train sample (1,398 records x 6 unordered pairs x both presentation orders,
LLaVA-OV 7B 8-bit, cache namespace `pair`). exp73: 300-trial held-out study,
base (17 exp62 features) vs +pair (preference mean, win count, has_pair);
exp74: attribution grid at val prevalence.

**What killed it — three independent measurements.**
1. *Direct signal test*: the pairwise Copeland winner among the top-4 matches
   the GT-best proposal **29.8%** of targeted rows — chance is 25%. The VLM
   cannot discriminate instances even with the rejection option removed.
2. *Gate*: base 0.6826 > pair 0.6778 on the raw-calib objective (the exp62
   protocol, and the pre-registered +0.005 margin is not met in either frame).
3. *Attribution (exp74)*: at val prevalence the pair variant looked better
   (44.66/79.08 vs 41.63/67.45) — but base with the *same rule params* and the
   abstention regime flipped reaches 44.84/79.09. The gain was `use_ens=True`,
   not the features. At identical params the feature delta is +0.89 gIoU
   (within calib noise, n=423) and it changes sign across param sets.

**The boundary is now perceptual, measured three ways**: the box prior
overlaps a 2-object union at IoU 0.329; crop yes/no verification cannot
reject (§19); forced-choice preference is at chance. No selection machinery
over this localizer's signals can approach the oracle realization both floors
demand. Eighth negative model change; the frozen-signal class is exhausted.
Artifacts: `exp/out_final/optuna_study_gref7.json`,
`gref_pair_attribution.json`, `exp/out_pairwise_logs/`.

## 22. Stronger localizer (model change 9) — foreclosed by arithmetic before it started

**The idea.** The wall is perceptual: LLaVA-OneVision's boxes overlap 2-object
unions at IoU 0.329, and it fails both verification formats. The paper's own
limitation names "a stronger grounder" as the fix; the RefCOCO line of this
run uses Qwen2.5-VL boxes elsewhere.

**Why it cannot pass, with numbers.** At the shipped operating point the gIoU
floor demands mean targeted IoU **0.6432**. The perfect-box + current-rule
realization measured in this run is **0.547** — 9.6 points SHORT — and any
real localizer lands between the delivered 0.485 and that perfect-box 0.547.
A stronger VLM narrows the gap and cannot close it: under a *perfect* box the
selector rule is itself short of the floor. (The cIoU side is even more
foreclosed: ≥105% of the deployable oracle at the measured abstraction mass.)
The Qwen2.5-VL download was attempted anyway; the HF cache holds 5
`.incomplete` shards and the proxy dies with SSL EOF on resume. Stopped —
the arithmetic above makes the extraction pointless for the floor, though it
remains the most promising *narrowing* lever if the paper ever wants one.
Recorded in `gref_wall.json: exact_selector_requirement.stronger_localizer_foreclosure`.

## 23. Learned pair/union selector — the first POSITIVE model change (9th)

**The mechanism the wall had missed.** exp77 measured, over the selector's own
pool on held-out train: perfect single pick 0.696 cumulative targeted IoU,
perfect pair 0.825, greedy union 0.872 — while the shipped threshold-union
delivered 0.561. The per-proposal selector cannot express "these two masks
together are the referent"; its threshold-union of many positives adds area
that dilutes IoU below even single-mask picking. exp62's per-proposal labels
also can't see unions (a one-of-two-object mask scores IoU~0.5 vs the union,
straddling the label boundary).

**The model.** Candidates = top-6 singles + 15 pairs of the frozen exp62
classifier; a second logistic model scores each candidate on union features
(base probabilities, union area, pair overlap, box coverage of the union,
released scores); selection = best candidate above p_sel, ensemble abstention
unchanged. 300-trial Optuna study on the held-out fit/calib split, objective
min(cIoU/68.94, gIoU/74.34) at val prevalence: **best 0.6764 vs shipped-
threshold-union reference 0.6408 (margin +0.0356)**; calib cIoU 46.63 (vs
44.18), T_acc 90.3 (vs 83.8). Frozen, one full-val pass (exp79):

**FULL VAL: cIoU 36.75 / gIoU 69.63** (N 86.2, T 87.8) vs the ensemble's
34.94/69.77 — cIoU **+1.81**, T_acc **+10.6**, gIoU −0.14; the pre-registered
min-ratio objective improves 0.5068 → 0.5331. Shipped (exp80 re-measured the
wall masses at the new operating point, identity closing exactly): r_required
collapses 0.9263 → **0.8473, BELOW the pool oracle's 0.8842** — the cIoU floor
is no longer foreclosed. The pool oracle at this operating point would reach
cIoU **71.50 > 68.94** with a 19.4% dilution surplus. cIoU is now bounded
purely by selector realization (delivered r 0.4516, needs 1.88×). gIoU stays
foreclosed (required m 62.09 vs perfect-box-rule 54.7). The wall's shape
changed: what remains is "how close can a training-free selector come to the
pool oracle", and the answer moved for the first time in nine model changes.

## 24. Compositional union candidates (model change 10) — measured, not pursued

**The idea.** exp80 showed cIoU needs targeted cumulative r = 0.8473; exp81 then
measured the deployable union oracles: at the shipped K=6 shortlist the
exhaustive best union of 2 is 0.7250 and **triples and quads are WORSE**
(0.7099 / 0.6670 — extra shortlist masks are distractors, not complements);
at full pool, exhaustive best-pair is 0.8362 and exp77's greedy subset-growth
oracle reaches 0.8723. So the requirement (0.8473) exceeds the pair oracle
(0.8362) and sits 2.5 points below the greedy-subset oracle.

**Why it was not built into a study.** The path that could reach 0.8473
requires (a) candidate reach at the near-full-pool scale — the exact opposite
of what the 300-trial study chose (K=6 beat K=8/10/12 on the objective,
because the classifier cannot exploit wider candidate sets) — and (b) >97%
realization of the greedy oracle from deployable signals, against the 62%
(pair/K6) to 69% (perfect box, same-k) realizations measured across ten model
changes. Both requirements are 30+ points beyond anything measured. Recorded
in `gref_shortlist_ceiling.json: ciou_requirement_context` with the capped
note that the K>=12 union columns were a top-10 search cap, not measurements.

## 25. Scaling the pair selector's fit set (model change 10, alternative route) — flat curve

**The idea.** The shipped pair classifier is fitted on 1,398 held-out train
records; the gRefCOCO train split is ~14x larger, and using more *train* refs
(never val/test) is within the established protocol. If realization were still
data-limited, that extraction (box + released scores over ~20k train records,
hours on 4 GPUs) would be the one untried lever.

**What killed it — before the extraction.** exp82's learning curve, frozen
shipped rule, evaluated on the 423-row val-prevalence calib: fit on 51 refs
-> targeted r 0.5519; 128 -> 0.5365; 257 -> 0.5584; 386 -> 0.5626; 515 ->
0.5538. Non-monotone, slope +0.023 r per 100% of 1,398 refs (noise at this
calib size), cIoU flat at 45.3-47.1. The pair classifier SATURATES AT ~50
refs: its binding constraint is feature information (base probabilities, box
coverage from the IoU-0.329 localizer, overlap geometry), not sample size.
`gref_fit_curve.json`. The extraction was not launched.

**Where this leaves the cIoU gap** (all measured, one place): delivered
targeted r 0.4516 full-val; requirements/ceilings: r_req 0.8473 at current
masses, ~0.78-0.85 as masses tighten; pool-pair oracle 0.8362, greedy-subset
oracle 0.8723; deployable realizations across ten model changes 54-69%;
fit-size: flat. Every measured route to the 1.88x realization jump is now
either taken, foreclosed, or measured flat.

## 26. Route closure — every measured path to the gRefCOCO floors, in one place

At the shipped pair/union selector (36.75/69.63 full val), the floors are
cIoU 68.94 / gIoU 74.34. Every route measured in this run, with its status:

1. **Existence axis** — SOLVED (3-probe ensemble, N_acc 89.1 val > trained
   79.7). Not the bind (§20).
2. **Selection quality** — TEN model changes (rules, learned selector, clause
   features, existence probes, region verification, forced-choice pairwise,
   pair/union). The pair/union change was the first positive (+1.81 cIoU
   shipped). Deployable realization across all ten: 54-69% of the relevant
   oracle. The cIoU floor needs 1.88x the delivered targeted cumulative
   (0.8473 vs 0.4516) at current masses; the gIoU floor needs m 62.1 vs
   delivered 47.8.
3. **Fit-set scaling** — FLAT (exp82: pair classifier saturates at ~50 refs;
   targeted r 0.537-0.563 across 51->515 refs). More train annotations are
   not the lever. (§25)
4. **Compositional (triple+) candidates** — oracle-margin foreclosed: the
   requirement 0.8473 exceeds the full-pool best-PAIR oracle 0.8362; the
   greedy-subset oracle 0.8723 leaves ~3% slack (97% realization demanded vs
   62-69% ever measured); the study chose K=6 because larger K hurt the
   deployable objective. (§24, exp81)
5. **Stronger / additional localizer** — three times blocked: perfect-box
   arithmetic leaves the threshold-union rule 9.6 mean-IoU points short of
   the gIoU requirement; Qwen2.5-VL download dies on proxy SSL EOF (5
   .incomplete shards sit in the HF cache); LLaVA-Video-7B-Qwen2 is complete
   on disk but its LlavaQwenForCausalLM architecture cannot load under
   transformers 4.49 without lmms-lab's absent fork (`import llava` missing).
6. **Abstention re-shaping** — the required ROC point (0.905, 0.99) at the
   delivered m is not producible by any combination of the two abstention
   signals (ens-only: 89.1/77.2; p_sel-gated pair rule: 86.2/87.8; the
   coupling is signal quality, not rule form).

**Terminal state**: the cIoU side is realization-bound with a measured 1.88x
gap and no lever; the gIoU side is locally foreclosed (perfect-box + current
rule short of the requirement). The paper states this boundary with every
number above reproduced from artifacts; the shipped row improved monotonically
across the last three model changes (35.09/68.37 -> 34.94/69.77 -> 36.75/69.63
cIoU/gIoU). If a future run lifts the localizer (the sole unblocked lever), the
in-box oracle ceilings (wall: orc_inbox 0.4016 at k15 vs in-GT-box 0.7345)
bound the expected gain.

## 27. Per-term box selection — SECOND POSITIVE model change (10th)

**The structural discovery behind it.** gRefCOCO *val* targeted records are
**100% multi-target** (n_targets>=2 for all 5,324) while the held-out train
sample is only 33% multi — the tuning population never matched the evaluation
distribution, and the union IS the task on val. Separately, the localizer is
much better at single-object boxes (per-noun box vs its object: IoU 0.580,
63.5% select>=0.50) than at union boxes (IoU 0.329).

**The construction.** One box per parsed term (noun_phrase + nouns, max 3) ->
a learned per-term picker (logreg on 8 features incl. base score x bbox-IoU,
top-2 with margin 0.815) selects a mask inside each box -> union of picks.
exp83 measured the ceiling from GT alone: GT per-object boxes + the bbox-IoU
pick reach r 0.7151 vs the then-shipped 0.5538 on the same rows (two-box
oracle 0.8145). exp87's 300-trial study on the val-matched population:
calib r 0.5506 vs shipped 0.4268 (+0.124). Frozen, one full-val pass (exp89):

**FULL VAL: cIoU 40.96 / gIoU 71.11** (N 86.2, T 87.8; termbox on 4,666
predicted records, fallback 1,241) vs the pair rule's 36.75/69.63 — **cIoU
+4.21, gIoU +1.48**, the largest single-change gain in the run. Masses at the
new operating point (exp90, identity closes exactly): r_delivered 0.4516 ->
**0.5030**, m_delivered 0.4779 -> **0.5228**; cIoU requirement 0.8467, still
below the pool oracle 0.8842 (ceiling 71.50 > floor).

**Defect found and fixed en route.** The 8-bit processor's padded batches
emit immediate EOS for every row shorter than the longest in the batch (minimal
repro: only the longest row decodes); the first val extraction produced 74%
empty raws. Re-extracted single-query (empty-aware resume); record coverage
37% -> 99.85%. Coverage-scaled val: 37% -> 37.80, 75.7% -> 39.25, 99% ->
40.96 cIoU.

## 28. Hybrid candidate family (model change 11) — NEGATIVE, gate artifact exposed

**The idea.** exp94 showed the termbox rule realizes 86% of its own in-box
oracle (0.5059 vs 0.5855 LLaVA-box; 0.6873 GT-box) and that the termbox
construction's ceiling (0.687) is BELOW the cIoU requirement (0.8467), while
pair/union candidates reach a 0.836 full-pool pair oracle. Union of the two
candidate families, one learned selector over both feature blocks.

**What happened — in two acts.**
1. exp95 (300 trials, gate searched): best calib r 0.5948 vs termbox 0.5529
   (+0.042) — a "win". The val pass then COLLAPSED: cIoU 22.27 / gIoU 67.48,
   T_acc 40.9 (was 87.8) — the p_inc selection gate (0.631) does not transfer:
   on val the GBM's probabilities sit below the threshold, and the gate
   abstains on half the targeted records. `gref_hybrid_full_val.json`.
2. exp97 (300 trials, gate removed — selection only, shipped abstention): best
   calib r **0.4018 vs termbox 0.5529 (−0.151)** and vs pair 0.5177 (−0.116).
   Forced to rank every row, the hybrid GBM is WORSE than both shipped
   selectors. The v1 "win" was entirely the gate's selective predicting, not
   the hybrid ranking; the gate itself is a val-transfer failure.

**Conclusion.** The hybrid ranking does not beat the termbox selector, and the
only configuration where it appeared to (gated) fails the deployment
distribution. Tenth and eleventh model changes: #10 positive and shipped
(per-term box, 40.96/71.11), #11 negative and recorded. The shipped row is
unchanged and re-verified. The one remaining live lever per exp94 is BOX
quality (+0.10 r headroom vs +0.08 pick headroom, of which the picker already
realizes 86%) — blocked on localizer availability (Qwen download SSL-blocked;
LLaVA-Video needs the absent lmms-lab fork).

## 29. Qwen2.5-VL localizer route — definitively blocked (hash-verified)

The one remaining live lever after exp94 (box-quality headroom +0.10 r vs
pick-quality +0.08, of which the picker already realizes 86%) was a stronger
localizer. Qwen2.5-VL-7B's HF cache held 5 `.incomplete` shards; a download
resumed (the proxy briefly recovered) and reported every shard "resume from
N/N" (complete bytes); the CLI then hung finalizing. Assembling the model
manually from the blobs (safetensors headers matched the index's weight_map
with 100% key overlap per shard) produced a model that loads and generates
fluent prose but HALLUCINATES the image content (two different descriptions
for the same image) and degenerates on any coordinate prompt.

**Definitive check**: HF blob filenames ARE the sha256 of their content.
`sha256sum` against the blob names: **all five FAIL**. The downloaded bytes
are corrupt — the proxy mangled ~14GB across every attempt (the previous
session's shards were the same files). The manual assembly is garbage
(marked `CORRUPT_DO_NOT_USE.txt`); the CLI finalize hung and was killed.
LLaVA-Video-7B-Qwen2 (the other on-disk option) remains unloadable without
lmms-lab's absent fork. The stronger-localizer route requires network this
environment does not provide reliably; recorded, closed, and hash-verified.

**This iteration's model-change ledger**: #10 per-term box (positive, shipped
40.96/71.11); #11 hybrid candidates (negative, measured twice: gated win that
collapses on val to 22.27, and ungated calib r 0.4018 vs 0.5529); stronger
localizer (blocked, hash-verified corruption).

## 30. LLaVA-Video as the stronger localizer — measured weaker (route closed)

The remaining lever (exp94: box-quality headroom +0.10 r) needed a localizer
better than LLaVA-OneVision at per-object boxes. Qwen2.5-VL: hash-verified
corrupt downloads (§29). LLaVA-Video-7B-Qwen2: weights complete on disk,
custom architecture absent — so the architecture was HAND-ROLLED from the
checkpoint with native transformers classes (SigLIP-so400m vision tower 421
tensors + mlp2x_gelu projector + Qwen2-7B, `exp99_llavavideo.py`), passing the
same coherence test that caught the corrupt assembly (two prompts, consistent
image-specific content; plausible distinct boxes for laptop/guitar).

**Measured result (exp100/101, val-matched sample, 600 term picks):**
per-term box vs object: mean IoU **0.319** and select>=0.50 **33.0%** vs
LLaVA-OneVision's 0.580 / 63.5%. Deployed proxies: bbox-IoU pick r 0.247
(OV 0.416), sxi 0.431 (OV 0.519), both below the shipped rule's 0.482.
Single-view 384 without anyres may cost some quality, but the gap is large
and robust; LLaVA-Video is video-tuned and weaker at image grounding.

**Conclusion.** No available localizer beats the incumbent; the +0.10
box-quality headroom is unreachable in this environment (corrupt Qwen,
weaker LLaVA-Video, missing fork for nothing better). The shipped termbox row
(40.96/71.11) stands, with its realization already at 86% of what the
incumbent's boxes permit.

## 31. SAM box-prompted masks (model change 13) — mask axis closed, box axis confirmed

The exp94/102 localizer bind conflated two constraints: VLM box quality AND the
AMG pool's mask quality *inside* the box. Under GT boxes the pool pick reaches
only m 0.6552 — but that is a statement about POOL masks. SAM's box-prompted
decoder (the Grounded-SAM primitive, unused by the shipped pipeline) is a
different mask source: box in, three tight masks out. Measured on the same
262-row val-matched held-out sample (`exp103_sam_boxp.py`,
`exp104_sam_refine.py`; artifact `exp/out_final/gref_sam_boxp_sample.json`):

**With per-object GT boxes:** SAM oracle over its 3 outputs **0.8845** m
(deployable argmax-score 0.6928) vs the pool pick's 0.6552 — the AMG pool
WAS a mask-quality constraint at high box quality. A shipped construction
with perfect boxes + SAM masks would clear the gIoU requirement (0.6212).

**With the actual LLaVA term boxes:** SAM is *worse* than the pool rule —
oracle 0.4765 vs 0.5335, deployable 0.254-0.457. SAM's output is fully
box-determined, so box imprecision (IoU 0.580, 63.5% hit) propagates with no
score-based recovery, whereas the pool+CLIP picker can recover (the base
score dominates the in-box feature).

**SAM as a refiner of pool picks** (exp104: bbox of the picked mask as the
prompt): does not help — bbox-of-union 0.4593, bbox-of-term-pick 0.4681,
per-term oracle 0.4758; the pool u SAM-largest hybrid gained +0.005..+0.017
over the rule (0.5335 → 0.5384/0.5507), below the 262-row noise SE (~0.022).
Measured, NOT shipped — a val pass for a sub-noise delta would risk the
accepted 40.96/71.11 for no floor movement.

**Conclusion.** The gIoU gap is localizer BOX quality alone: both mask
sources are now measured, the mask side is eliminated as a constraint
(perfect boxes + SAM clear the bar), and the box side has no available
stronger model in this environment (§29/§30). For the pair of floors nothing
changed: cIoU still needs pool-oracle-scale selection at ~96% realization
(§26).

## 32. Prompt-paraphrase box ensembling (model change 14 candidate) — closed by scale

The localizer was only ever queried with one template; paraphrase ensembling is
the standard training-free box-quality lever and the last untested move on the
binding axis (exp102: gIoU needs box quality ~72% of the way from the
incumbent's 0.58 to GT). `exp105_box_ensemble.py` generated boxes for the
sample's terms under two extra paraphrases (cached `nounbox_p2/p3`; the shipped
`nounbox` cache untouched).

**Partial measurement (16 unique terms, 4 images), then stopped:** P1 0.509 vs
P2 0.528 box IoU (+0.019); best-of-2 oracle 0.548 (+0.039); **mean pairwise
agreement 0.838** — the templates largely agree, so there is little to average.
Run cost measured at 35 s/query under full-GPU contention (all four GPUs busy;
projected ~9.5 h for 487 terms x 2 prompts).

**Why stopped — the scale argument, not the sample:** the requirement needs box
IoU ~0.88 (+0.30). Via the measured box->m conversion (~0.29 m per box-IoU
unit, exp102 endpoints), the full oracle gain projects to **+0.004 gIoU against
the 3.23-point gap**, and the observed template agreement (0.838) bounds what
any paraphrase family can add. No prompt-level intervention moves box quality
by +0.30. Artifact: `exp/out_final/gref_box_ensemble_sample.json`.

## 33. GDINO+SAM box/mask sources — SHIPPED (change 13): full val 47.14/73.95

The turn's arc, all measured: exp103/104 closed the mask axis (SAM under GT
boxes 0.885 oracle; under LLaVA boxes 0.477 < pool 0.5335; as a refiner, no
help). exp105 closed prompt paraphrase ensembling by scale (agreement 0.838;
oracle gain +0.039 box IoU -> ~+0.004 gIoU against the 3.23-point gap).
**exp106 found the untested detector family: GroundingDINO-base (on disk,
COCO-trained) gives per-term box IoU 0.7046 / hit 71.6% vs LLaVA's 0.580 /
63.5%, oracle 0.8419, 56.6% of terms with >1 detection.** exp107 built the
deployable rule: pool pick (sxi) in the LLaVA nounbox (GDINO fallback) UNION
SAM's three box-prompt masks from the top-1 GDINO detection -> sample m 0.6518
(knife-edge V5) and 0.6433 for the study winner. An initial study objective
min(cIoU,gIoU) was cIoU-limited everywhere and optimized the wrong axis
(killed at trial 7, restarted with max proj_gIoU).

**exp108 (300 trials, fit/calib by ref): winner = llava-box sxi (alpha 1.2205,
beta 0.6172, top-2 margin 0.729) + SAM all3 (cap 0.25) + GDINO det0 prompts;
calib m 0.6866 vs shipped-on-calib 0.5818 (margin +0.105); proj gIoU 76.49.**
exp108b full-sample verification: winner row-mean m 0.6433, per-rid 0.6340,
delta over the shipped-equivalent +0.0967 [95% CI +0.064,+0.129], P>0 1.000.
exp110 (300 trials) re-studied abstention: converged on the shipped operating
point (no better point). **exp109b full val (one pass): cIoU 47.14, gIoU 73.95**
(was 40.96/71.11; pair 36.75/69.63; ensemble 34.94/69.77). Masses re-measured:
r_delivered 0.5679 (was 0.5030), requirement 0.8305 (was 0.8467) — cIoU gap
narrowed but still ~94% of the pool oracle needed; gIoU 0.39 short of 74.34.

**Post-ship probes (measured, not shipped):** exp111 — the held-out 1-term
subpopulation (n=47) reaches m 0.6410 with the shipped rule; adding the 2nd
GDINO detection's pick + SAM masks gives 0.6734 (+0.032, projected +0.11
gIoU on val, still below the floor). exp112 — SAM(LLaVA box) hurts (-0.035),
SAM(det1) neutral (+0.002), both hurt (-0.017) on the 262 sample.

**Remaining boundary:** cIoU 47.14 vs 68.94 (realization-bound; the new
construction's GT-box oracle is estimated ~0.82 r, still below the 0.8305
requirement); gIoU 73.95 vs 74.34 (0.39; the gap is train->val transfer on
the targeted side — sample m 0.6433 vs val m 0.609 — plus the 1-term
subpopulation, worth +0.11 if shipped).

## 34. The construction oracle, MEASURED on full val — the cIoU foreclosure flips

The corrected cIoU framing carried one extrapolated number: the GT-box SAM
construction oracle (~0.82 r, inferred from exp103's mean-IoU 0.885), on
which the "foreclosed relative to the measured construction" bracket rested.
exp114 measured it directly on all 5,324 val targeted records (per-object GT
boxes -> SAM 3 outputs -> best union over 3^n combos, accumulated through the
exp109b mass identity): **mean targeted IoU 0.8827, cumulative r 0.8818,
cIoU ceiling 71.75 > 68.94**. The extrapolation was too pessimistic by ~0.06 r.

Consequence: cIoU is **realization-bound under both oracles** (pool selection
0.8842; construction 0.8818) — the requirement 0.8305 is ~94% of either, the
rule delivers ~64%. The paper's structural claim ("the construction family
would have to change") was removed; the honest statement is selection
quality against a ~0.88 oracle. Artifact:
`exp/out_final/gref_construction_oracle.json`; macros \wallConstructionOrc /
\wallConstructionCap.

## 35. Selection decomposition — which part of the gap (exp115, held-out)

Reviewer-requested mechanism for "the gap is one of selection quality".
2x2 on the 262-row sample: {delivered boxes, per-object GT boxes} x
{shipped pick, per-term oracle pick over (pool-in-box u SAM)}.

  del_shipped  0.5681      gt_shipped  0.8022
  del_oracle   0.6377      gt_oracle   0.7663

Box axis +0.234 mean targeted IoU; mask-choice axis +0.070. (gt_oracle <
gt_shipped because the oracle definition picks ONE candidate per term while
the shipped pick unions SAM's three outputs — the union's over-coverage
helps; the direction of both axes is what this establishes.) The dominant
training-free loss is WHICH BOX TO ASK, not the mask decision. Artifact:
`exp/out_final/gref_selection_decomposition.json`; paper sentence added.

## 36. Sentence-level GroundingDINO (exp116) — NEGATIVE

With the loss located on the box axis (exp115: +0.234 with GT boxes), the last
untested box query was the full SENTENCE (GDINO is caption-trained; the term
parse might be throwing information away). Measured on the 262-row sample:
top-1 sentence detection + SAM union 0.4529; top-2 0.5469; top-3 0.5443;
oracle over detection subsets 0.5305 — all far below the shipped term-level
rule's 0.6433. The term decomposition onto spatialized nouns is doing real
work; re-querying at the sentence level is closed.

## 37. CHANGE 14 SHIPPED — the gIoU floor is BEATEN (74.73 > 74.34)

The joint study (exp117, 300 trials) re-opened four joints that earlier
studies had fixed one at a time: score source, pick box, SAM source, and the
single-term subpopulation (12.5% of val targeted), on a mixed population at
val prevalence (262 >=2-term rows + 47 held-out 1-term rows). Winner:
**picks inside BOTH the LLaVA nounbox and the GroundingDINO top-1 box**
(sxi bs*(1.811+0.234*bxi), top-2 margin 0.631) **unioned with SAM masks from
the detector's top-1 AND top-2 detections** (cap 0.5). Mixture m 0.6803 vs
shipped-equiv 0.6380 (+0.0424).

One frozen full-val pass (exp118; val-side SAM(det1) precomputed by exp119):
**cIoU 49.59 / gIoU 74.73** — gIoU beats InstAlign's 74.34, the first
published floor on gRefCOCO the training-free class clears (N_acc 86.17 >
trained 79.7, T_acc 87.85, m 0.633 > required 0.6212). cIoU remains:
r_delivered 0.5869 vs requirement 0.8158 (identity closes exactly), needing
92% of the ~0.88 oracle realization. History: 71.11 (termbox) → 73.95 (GDINO
+SAM v13) → 74.73 (v14).

**Operational note (cost 40 min):** every GDINO load hung on a stale lock in
`~/.cache/torch_extensions/py310_cu121/MultiScaleDeformableAttention/lock`
(left by an interrupted nvcc build under system load-40); removing the lock
restored 11.7 s loads. Also: exp118's `--out` default overwrote the change-13
artifact — restored from the wall; the change-14 metrics live in
`gref_joint_full_val.json`.

**Change-14 decomposition (exp120 factorial, after review):** the gain is
ADDITIVE, not "compounding": SAM(det1) source +0.0124 mixture, detector-box
picks +0.0024, interaction -0.0008 ~ 0; the re-tuned sxi coefficients carry
the rest (+0.0284 same-cell). Both populations gain (ge2 +0.0362, 1-term
+0.0850) — the gain is not merely the newly-included subpopulation. Artifact:
`exp/out_final/gref_factorial.json`.

## 38. cIoU-targeted re-study (model change 15 candidate) — r-optimality confirmed, not shipped

The v14 config was chosen on an m (gIoU-side) objective. exp121 ran the full
300-trial study on the cIoU-side quantity instead (mixture cumulative r, with
an m >= 0.6212 safety floor). Result: the winner gains +0.0053 r_mix over the
v14 configuration (0.6384 vs 0.6331) at a cost of -0.0064 m_mix (0.6739 vs
0.6803). The trade would shrink the gIoU-floor margin from 0.39 to ~0.18 for
a cIoU gain that leaves the floor 18+ points away. MEASURED, NOT SHIPPED.

The finding that matters: even when the objective IS the failing metric's
proxy, 300 trials find essentially nothing. The v14 rule is r-optimal in its
space; the cIoU requirement (val r 0.8158 vs delivered 0.5869, +0.23 needed)
is a selection-capability wall, not a tuning failure. This closes the tuning
axis for the cIoU floor the same way exp105 closed paraphrase ensembling and
exp110 closed abstention.

## 39. The second detection as a PICK source (exp122) — closed, exact null

v14 used det0/det1 as SAM sources but only ll/d0 as pick boxes; det1-picks were
the last unexplored cell. Discriminating cells (mixture m / r): ll 0.6788 /
0.6314; d0 0.6760 / 0.6285; d1 0.6619 / 0.6022; ll+d1 0.6798 / 0.6326 — vs
v14's ll+d0 0.6803 / 0.6331. Adding det1 (or det2) as a pick source changes
the mixture EXACTLY zero: the top-2 picks in the extra boxes duplicate the
picks already taken from ll and d0. The single-box cells differ as expected,
so the null is real. This was the last cell of the v14 configuration space;
the space is now exhausted, and the cIoU boundary stands as selection
capability, not tuning (exp121) or configuration (exp122).

## 40. CHANGE 15 SHIPPED — singles-only no-target fallback: cIoU 49.59 → 50.99

The last mechanical component of the cIoU denominator: FP_U (predicted area on
the 1,230 predicted-no-target records) was 10.49M px of pure dilution, and
exp123 measured on held-out no-target rows that 99.8% of the fallback argmaxes
are PAIRS, while singles-only halves the predicted area (8,079 -> 4,007 px).
Targeted records never use the fallback on val (fallback_used == fn), so the
change cannot touch m, N_acc, T_acc or gIoU — confirmed exactly: exp124's
frozen val pass gives cIoU 50.99 (+1.40), gIoU 74.73 identical, identity
closes exactly. FP_U 10.49M -> 6.86M; r_delivered 0.5869 (unchanged — pure
denominator), r_required 0.8158 -> 0.7934. Shipped row:
cIoU 50.99 / gIoU 74.73. The cIoU floor now needs ~90% of the ~0.88 oracle;
the selection-capability boundary is unaffected (the fallback contributes no
intersection either way).

## 41. Union-precision by box clipping (exp125) — NEGATIVE

The cIoU identity shows 41% of predicted mass outside GT (I 65.6M of S_U
111.8M), suggesting a precision lever: clip the union to its own boxes.
Measured on the held-out mixture (base r 0.6331 / m 0.6803):
clip_final (union AND union-of-source-boxes) 0.6303 / 0.6759;
clip_each (each mask clipped to its own prompt/pick box) 0.6288 / 0.6748.
Both HURT: the boxes are looser than the masks they contain (box IoU 0.58-0.70
vs object), so clipping removes more GT than background. The over-coverage is
the deliberate multi-mask union, not box spillover — the r/m tradeoff along
that axis was already explored by exp121's r-objective (r-optimal).
Artifact: `exp/out_final/gref_clip_precision.json`.

## 42. Sentence-count coverage (exp126) — the reviewer's named lever, measured null

The round-H reviewer flagged the count-aware coverage of multi-instance
referents as the one named, mechanism-matched lever (deferred, not denied).
Operationalized with the cheapest count signal — the cached sentence-level
GDINO detections (their count IS the instance count; no VLM counting needed) —
by adding SAM masks from the sentence's top-2 detections to the v14 union.
Held-out mixture: base r 0.6331 / m 0.6803; plus sentence SAM r 0.6348 /
m 0.6806 (+0.0017 r, sub-noise); sentence SAM alone r 0.5200 (the sentence
source is weak, as exp116 found). The detection-count information is already
fully exploited by the v14 det0+det1 coverage. All three of the reviewer's
reopen triggers now carry either a measurement (this one) or an environment
condition (released InstAlign; stronger grounder). Artifact:
`exp/out_final/gref_sentence_coverage.json`.

## 43. Abstention re-study under the final rule (exp127) — third convergence

The abstention operating point had last been studied under the change-13
masks (exp110). Since the v14/v15 rule changed the per-record IoU
distribution (m 0.6094 -> 0.6330) and the gIoU trade depends on it, the
300-trial held-out study was re-run with the FINAL rule's masks: winner =
(k_abs=1, p_abs=0.3404, use_ens=True) — the shipped point, exactly, with the
objective flat (81.71 weighted gIoU) across the top trials. Third independent
convergence (exp73-era, exp110, exp127). The abstention axis is closed under
the shipped rule; `optuna_study_gref16.json` (300 trials) records it.
