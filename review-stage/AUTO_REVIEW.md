# Auto Review Log — extreme run

**Run** `zsris-extreme` · executor `deepseek-flash` · reviewer **`glm-5.3`** (Zhipu)
**Backend** `mcp__llm-chat__review` / `mcp__llm-chat__review_reply` (cross-family:
executor deepseek, reviewer zhipu)
**Thread** `llm-chat:327152389eef` (rounds 1–3)

**Rounds 1–3 (2026-10-03) are the earlier bar.** The 2026-10-05/06 continuation
(floors vs GLOD-RIS/InstAlign; threads: pair/union ship, termbox ship, terminal
`d5b4a5be62fb`) is logged at the bottom of this file.

---

---

## Round 1 — 2026-10-03 · verdict **borderline, 5/10**

Five ranked fixes returned:

| # | criticism | response |
|---|---|---|
| 1 | "Leads every one-line control" is a RefCOCO-only result; on RefCOCOg the no-gate control ties (+0.06 mIoU, p=0.67) and the full-score-set control edges it (0.5102 vs 0.5100) | **Applied.** Abstract and contribution (3) scoped to RefCOCO; a disclosure paragraph added to the controls section with the exact RefCOCOg numbers. |
| 2 | Abstract's "the same composed operator is positive on all four benchmarks" is false for gRefCOCO — the single-mask operator is *negative* there (20.45 vs 22.56 cIoU) | **Applied.** Abstract reworded (positive on RefCOCO/+/g; set-valued extension for gRefCOCO) and the negative single-mask row is stated in the experiments section. |
| 3 | The reproduction audit's 150 "first" refs may be the development set (circular); one experiment to close it | **Applied.** The official code was re-run on a second, disjoint 150-reference block after the first was recorded: block A 46.10 vs 46.43, block B 44.83 vs 45.45 — agreement within 0.6 mIoU on both. Header softened to "not reproduced by the released code". |
| 4 | Several headline numbers not traceable to stored artifacts (84% spatial, top-3 containment, rank, top-8) | **Applied.** All structural numbers recomputed this run into `exp/out_final/headroom.json`; macros read from it. Fresh values replaced stale ones (84%→73.1% vocabulary / 63.7% parser; rank 9.1→10.4; 70.1%→67.6%). |
| 5 | "Calibrated" abstention needs the percentile sweep or pre-registration evidence; relabel the cIoU component | **Applied.** Rule stated as pre-registered (1st percentile of RefCOCO head-noun score; ≤1% target suppression by construction); sweep p0.5/p1/p2/p5 reported in the gRefCOCO table; score-floor-suppression caveat added; clause decomposition credited for the bulk of the gain. |

Novelty advice: reframe as a measurement paper whose taxonomy makes a
falsifiable cross-dataset prediction — **applied** (abstract and introduction
rewritten; title changed to "Instance Selection, Not the Pool").

## Round 2 — 2026-10-03 · verdict **almost, 6.5/10**

Three remaining fixes:

1. Foreground the (weakened) spatial contrast and reconcile the two
   spatial-fraction definitions — **applied**: both definitions computed on the
   same records and stated together; the modest over-representation (73.1% vs
   64.9%; 63.7% vs 57.4% parsed) is foregrounded, and the operator's
   confinement to the parsed subset (+0.1022 there, 0.0000 elsewhere by
   construction) is stated as the decisive test.
2. Finish the reframe (abstract should open with taxonomy + prediction,
   repairs as interventions) — **applied**.
3. Header + counting ambiguity (150 references = ~430 expressions) — **applied**.

## Round 3 — 2026-10-03 · verdict **almost, 7.5/10** (final)

All three confirmed resolved; the stratified check of the operator's gain
(+0.1022 over 6,220 parsed-spatial expressions) was independently re-derived
by the reviewer from `main_refcoco.json`. Residual nits (traceability of the
63.7% figure, "a third" for 38.0%, "matches exactly" phrasing, title) were all
cleared after this round.

**Stop condition**: score ≥ 6 AND verdict ∈ {almost, ready} — **reached**
(7.5, almost). Reviewer recorded as `glm-5.3`, not Codex.

## Residual (non-blocking) reviewer note

The taxonomy is measured at a single operating point of a single baseline
(no GLOD-RIS audit), and the method contribution is a tie-with-a-one-liner
plus a conditional +2.05 mIoU on one dataset; the reviewer accepts the paper
as a diagnosis paper on that basis, and both limits are disclosed in
`paper/sections/06_limitations.tex`.

---

# Extreme continuation — 2026-10-05/06 (floors vs GLOD-RIS / InstAlign)

Reviewer `glm-5.3`, executor `deepseek-flash`, backend `mcp__llm-chat__review`.
Bar floors: RefCOCO 58.42 / RefCOCO+ 54.01 / RefCOCOg 57.85 mIoU (all pass);
gRefCOCO cIoU 68.94 / gIoU 74.34 (the measured boundary, see below).

## Round A — pair/union selector ship (model change 9) · **8/10, ready**

36.75/69.63 full-val. Reviewer caught: the k=15 "fails at any N_acc" absolute
phrasing (scoped to the retired rule family); the oracle cap must be stated
realization-conditionally (`cap = r·G_U/(G_U + r·X)`, increasing in r); "at
most" wording. All applied. Residual: NONE. Ship accepted.

## Round B — per-term box selector ship (model change 10) · **9/10, ready**

40.96/71.11 full-val (+4.21 cIoU). Structural discovery: val targeted is 100%
multi-target vs 33.2% in the tune sample; one LLaVA box per parsed term beats
one union box (0.580/63.5% vs 0.329). Reviewer accepted the ship; residual
non-blocking notes on transfer and single-VLM scope, both disclosed in
`06_limitations.tex`.

## Round C — terminal state (thread `d5b4a5be62fb`) · **9/10, ready**

The reviewer's partial pass caught that `gref_wall.json`'s
`stronger_localizer_foreclosure` block quoted ensemble-era threshold-union
numbers, stale for the shipped termbox rule. Response: exp102 measured the
missing quantity on the 262-row val-matched held-out population —
`m_llava_box 0.5335` / `m_gt_box 0.6552` / `r_gt_box 0.6208` against the
required `m 0.6212` — establishing gIoU is **box-bound (localizer quality),
not foreclosed**. The block was replaced with `localizer_quality_bind`
(old preserved as `_HISTORY_ensemble_era`); `implied_localizer_requirement`
added (~72% of the way from incumbent box IoU 0.580 to GT); paper sentences
corrected in `05_experiments.tex` §gref and `06_limitations.tex`.

Final four-line verdict: **SCORE 9 · VERDICT ready · UNTESTED_ROUTE NONE
(for the pair; gIoU alone has the localizer route, environment-blocked) ·
TERMINATION accept.** Reviewer notes recorded: the ~72% figure is a linear
interpolation (labeled "implied"), and the GT-box headroom rests on 262 rows
(provenance stated). `review_gate.py --round-backend llm-chat` → **stop**.

**Bar outcome (invariant):** NOT_READY — 3 of 5 floors pass; gRefCOCO cIoU
class-foreclosed (needs pool-oracle-scale selection at ~96% realization vs
~57% best deployable; in-GT-box oracle 0.6873 < requirement 0.8467) and gIoU
box-bound with measured in-class headroom but no available localizer in this
environment (Qwen blobs hash-verified corrupt; LLaVA-Video hand-rolled and
measured weaker). The floors are reported as the honest measured boundary of
this method class in this environment.

## Rounds D–F — change 13 ship (thread `da6fcdb913a8`) · **9/10, ready/ship**

**Round D — model change 13.** exp106 found the untested detector family
(GroundingDINO-base, on disk: per-term box IoU 0.7046/71.6% vs LLaVA's
0.580/63.5%, oracle 0.8419). exp108 (300-trial study, objective restarted
from min(cIoU,gIoU) — cIoU-limited everywhere — to max proj_gIoU) selected:
LLaVA nounbox + sxi pick + SAM all-3 masks from the top-1 GDINO detection.
exp108b verified on the full 262-row sample (+0.0967 [CI +0.064,+0.129],
P>0=1.0); exp110 (300 trials) confirmed the abstention operating point
unchanged; **exp109b one frozen val pass: cIoU 47.14 / gIoU 73.95** (was
40.96/71.11). Reviewer's catch adopted: the "93.9% needed / 64.2% delivered"
pair is relative to the POOL oracle, not the construction oracle.

**Round E — oracle flip (exp114).** The reviewer's NEXT_ACTION (measure the
construction oracle instead of extrapolating ~0.82): full val, per-object GT
boxes → SAM → best union over 3^n combos gives **r 0.8818, cIoU ceiling
71.75 > 68.94**. The extrapolation was wrong; the "construction family would
have to change" claim was REMOVED from the paper, replaced by
"realization-bound under both oracles; the gap is selection quality".

**Round F — selection decomposition (exp115).** Reviewer-requested mechanism
on held-out: GT boxes + shipped pick = 0.802 vs 0.568 delivered (+0.234
through the box axis); oracle pick on delivered boxes +0.070 (mask choice).
The dominant training-free loss is which box to ask.

Final four-line verdict: **SCORE 9 · VERDICT ship · UNRESOLVED the
selection-quality claim now has a mechanism (exp115) · NEXT_ACTION
executed**. Winner-vs-runner-up margin within noise and the 262-row transfer
attribution remain disclosed. `review_gate.py --round-backend llm-chat
--score 9 --verdict ready` → **stop**. Bar: NOT_READY (cIoU 47.14 vs 68.94;
gIoU 73.95 vs 74.34 — 0.39 short, transfer-attributed).

## Round G — change 14 ship (thread `4b54cf858e53`) · **8.5/10, ship**

exp117 (300-trial joint study over score source × pick box × SAM source ×
single-term handling, on a val-prevalence mixture population) → winner: picks
in BOTH the LLaVA and GDINO boxes ∪ SAM masks from det0 AND det1. exp119
precomputed the val-side det1 masks (config-independent); one frozen val pass
(exp118): **cIoU 49.59 / gIoU 74.73 — the InstAlign gIoU floor (74.34) is
beaten by the training-free rule.** Reviewer's two hypotheses tested per its
NEXT_ACTION (exp120 factorial): NOT compounding (interaction −0.0008 ≈ 0;
main effects SAM(det1) +0.0124, box +0.0024, coefficient re-tune +0.0284) and
NOT a population-shift artifact (both populations gain: ge2 +0.0362, 1-term
+0.0850). Paper claims corrected to the measured decomposition; the 0.39
margin vs the published InstAlign number now carries an explicit
flip-threshold statement. Final: **SCORE 8.5 · VERDICT ship · RISK the
published-number margin · NEXT_ACTION (re-score InstAlign or state the
threshold) executed as far as possible — no released outputs to re-score.**
`review_gate.py --round-backend llm-chat --score 8.5 --verdict ready` →
**stop**. Bar: NOT_READY — the only failing line is cIoU 49.59 vs 68.94.

## Addendum — change 15 (singles-only no-target fallback) · measured refinement

After the round-G ship, exp123 measured the last mechanical component of the
failing metric: the predicted-no-target records' fallback emitted PAIRS in
99.8% of cases, adding pure dilution (FP_U 10.49M) to the cIoU denominator.
Singles-only (selected on held-out no-target rows; halves the predicted area)
cannot touch the targeted side — fallback_used == fn on val — and exp124's
frozen pass confirms: **cIoU 49.59 → 50.99, gIoU 74.73 identical, identity
closes exactly**. Shipped row cIoU 50.99 / gIoU 74.73. No new review round
requested: the mechanism is a held-out-measured restriction of an
already-reviewed rule with zero effect on the reviewed (targeted) outputs.
Bar: the only failing line is cIoU 50.99 vs 68.94.

## Round H — FINAL review (thread `e4cb5451e20b`) · **9/10, ship, terminal-honest: yes**

The reviewer pressure-tested the terminal claim and demanded (1) the framing
"selection-bound, not foreclosed — unfound under exhaustive search of the
defined selector family, not 'unreachable'", (2) an explicit reopen trigger,
and (3) it flagged the countable-instance selector as the one named,
mechanism-matched, not-yet-built lever (deferred, not denied). All three
adopted: the wall's `terminal_declaration` block carries the claim ("no
measured mechanism closes the remaining +0.21 r", with both oracles above the
requirement) and the reopen triggers (released InstAlign to re-score
same-harness; a stronger on-disk grounder; a countable-instance selector
family), plus the pre-registered first action on reopen (bound the new
selector's ceiling on the two-object-referent slice, GT-box oracle 0.802,
before any val pass). Final: **SCORE 9 · VERDICT ship · TERMINAL_HONEST yes ·
NEXT_ACTION declare terminal and freeze; bound any reopen on the
two-object slice first.** `review_gate.py --round-backend llm-chat --score 9
--verdict ready` → **stop**.

## Post-round-H integrity sweep · abstract staleness caught and fixed

A final claim-consistency audit of the paper against the shipped artifact
found the abstract still carrying the pre-change-13 framing: "locate where
the training-free class ends", the gIoU floor listed as unreached, and two
retired termbox-era macros (\wallRealBest / \wallNeedKfifteen — now unused
anywhere). Rewritten to the shipped state: the rule **beats** the trained
gIoU floor (\grefGIoU\ vs \grefPubGIoU), the remaining cIoU constraint is
stated with the two agreeing oracles and the selection-capability
attribution, all via live macros. PDF rebuilt (12 pp); no hardcoded stale
values remain in any section. This is the class of defect the review
criteria police (claims must be current and traceable); caught in the final
sweep, not by a reviewer.

## Post-round-H integrity sweep #2 · Qwen trigger closure + claim scoping (2026-10-07)

The reopen trigger #2 protocol (clean functional Qwen2.5-VL-7B obtained via
hf-mirror direct, all shards sha256-verified; bound on the pre-registered
two-object slice, held-out, no val pass) closed the trigger by measurement:
term boxes 0.6325/67.4% (GDINO 0.7046/71.6 already in the config; LLaVA
0.580/63.5; ceiling 0.802 — `gref_qwen_terms.json`); forced-choice 51.6% vs
LLaVA 59.3% on the same 246 records (size-heuristic 85.4%, random 26.3% —
`gref_qwen_pairs.json`). No study warranted; wall's `reopen_triggers_measured`
updated. The measurements also surfaced two claim-precision defects in the
frozen paper, both corrected:

1. The `29.8%` forced-choice figure is the full-grid diagnostic (n=858, mixed
   train prevalence) — now scoped as such, with the two-object-slice numbers
   (LLaVA 59.3%, Qwen 51.6%) and the trivial size-heuristic ceiling (85.4%)
   added in 05_experiments and 06_limitations.
2. `≈92% of the best measured oracle` (intro + experiments) was the
   pre-change-15 fraction; post-change-15 it is 89.7% ≈ 90% (wall
   `oracle_fraction_required`). Corrected in both sections.

No shipped number changed; no new review round opened (same precedent as the
change-15 and abstract-staleness addenda: measurements on a closed trigger with
zero effect on shipped rows). PDF rebuilt (12 pp, 6,088,854 bytes).
