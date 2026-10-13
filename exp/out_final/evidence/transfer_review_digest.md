# Review digest — transfer-inversion cycle (2026-10-13)

Machine artifacts: `exp/out_final/transfer_wall.json` (gated),
`exp/out_final/main_grefcoco.json` (shipped row + history + reference row),
`exp/out_final/sota_table.json` (gRefCOCO rows), `exp/out_final/learned_prior.json`
(previous cycle's arms), `exp/out_transfer/{train,val}_parts.npz` (cached part
tables), `exp/out_final/transfer_loo_accounting.json` + `transfer_loo_rare.json`
(review must-fix diagnostics). Paper: `05_experiments.tex` §"The transfer
inversion" (new fourth act), the gres-table caption, `main.tex` abstract.

## What was done (directive 2026-10-13: make the transfer inversion the object)

1. **Characterization** (four falsifiable decompositions, measured on cached part
   tables; train (tune records file) = 10,351 parts / 780 targeted rows,
   val = 89,708 parts / 4,677 rows):
   - population/composition: train targeted rows 68.33% single-target / 30.90%
     two-target vs val 99.53% two-target; mean parts per row 13.27 vs 19.18;
   - prior-quality: train-fitted model calibrated on train (pred 0.427 / actual
     0.412) but over-prediction on val from the second score quintile up
     (pred 0.366 / actual 0.326; top quintile pred 0.57 / actual 0.42; the
     lowest quintile under-predicts, pred 0.17 / actual 0.26);
   - feature drift: term_index median 0 -> 1; inbox_det0 p90 0.41 -> 0.16;
     inbox_llava p90 0.14 -> 0.36;
   - image level: median masks/image 51 vs 47 (rejected as the driver).
2. **Transfer-aware recipe** (one arm, train-side only): stratified-balanced part
   weights (uniform over the per-ref train-side n_targets composition: strata
   1:259, 2:132, 3:4, 4:2, 5:1, 6:1, 11:1 over the 400 targeted tune refs — a
   reference-level distribution, whereas §1's composition is row-level over the
   part table; same tune file, different units — base
   400/7 = 57.14; per-part weights 0.221 / 0.433 / 14.29 / 28.57 / 57.14; realized
   over the 7,443 fit parts the three rare refs with fit parts carry 37.1% of the
   fit mass, the single k=5 ref alone 25.7%) + within-row rank score features +
   acceptance threshold tuned on the held-out 30% carve-out RESTRICTED to its
   two-target stratum. Val (single frozen application): **cIoU 52.7061 / gIoU
   75.0817** (n=14,229) — above the training-free record on both point estimates
   (51.3182 / 74.9098; cIoU-decisive and gIoU parity-level per the cluster
   bootstrap, §7) and far
   above the arms that failed (naive 38.792, rank-only 50.52). Held-out two-target
   cIoU 58.90 (recipe) / 60.05 (+policy).
3. **Probe leads folded in as arms**: the searched combination policy
   (unconditional top-1 SAM part on top of the threshold policy) changes the val
   outcome by **0.00** (the top-1 SAM parts already clear tau=0.35); the
   weak-signal democracy vote, measured on a seeded 800-row val sample with
   2,400 cached MLLM calls, is a **negative**: vote-pick 38.37 vs prior-only
   top part 41.00 on the same rows, paired delta -2.624 +/- 0.818 (SE),
   310/800 picks changed.
4. **Shipped-row change** (change 17): `main_grefcoco.json` shipped = the learned
   recipe (52.7061/75.0817), declared trained, with `best_published` raised to
   the previous record (51.3182/74.9098) per the 2026-10-09 escalation rule; the
   training-free rule is preserved as `prompt_prior_reference` (51.3182/74.9098)
   and as a labelled row in `sota_table.json` ("Ours: termbox union rule
   (training-free reference)"), with the learned row marked as the shipped,
   declared-trained one. Arm selection used the held-out carve-out only; val was
   evaluated once per frozen configuration.
5. **Leave-one-rare-ref-out diagnostics** (2026-10-13 review must-fix; exp156):
   pre-registered as diagnostics only — the shipped row does not change
   regardless of outcomes. Seven runs (control + paired drops): remove each rare
   ref with fit parts (k3 ref-31716, k4 ref-45269, k5 ref-42328, the last being
   the only k5 ref so its removal also changes the base/normalization) plus
   composition-only controls (drop a held-out rare ref from the composition but
   not the fit set: k3 ref-64040, k4 ref-81979) and a base-perturbation-only
   control (drop the k6 ref with no parts). Paired differences give per-ref mass
   effects with the normalization effect separated out. Reading (in
   transfer_loo_rare.json and the paper): record-clearing is robust to any
   single rare ref (51.83–52.86, all above 51.32), but the +0.56 weight
   increment is not independently validated — it concentrates on the single
   k4 ref, whose removal lands the recipe 0.32 BELOW the tau-only arm — and
   the tau-only arm itself clears the record on cIoU only (gIoU 74.77 sits
   at/below 74.91).
6. **Reproducibility note** (found while running 5): re-running the exact shipped
   command must set `OMP_NUM_THREADS=4` (as the shipped runs did). Without it the
   logistic-regression fits drift in the last decimal, two margin-boundary rows
   flip their rank-1 second part, and the train table rebuilds with 10,352 parts
   (val 52.684 / held-out 58.57). With OMP=4 the control reproduces the shipped
   numbers exactly (10,351 parts, 7,443 fit parts, held-out 58.9004, val
   52.7061/75.0817) — verified by an independent recheck run (gref_recheck2.json).
7. **Cluster bootstrap over images** for the shipped-vs-record margin (exp158;
   per-row dumps from both frozen val passes): 10,000 image-cluster resamples.
   cIoU delta +1.38 [+0.60, +2.21] (excludes 0 — the cIoU record claim is
   decisive); gIoU delta +0.17 [-0.03, +0.38] (includes 0 — stated in the paper
   as parity or slightly above, not a gIoU record claim).

## Review questions

1. Claim-vs-artifact consistency: do the digest's numbers match
   transfer_wall.json / main_grefcoco.json / sota_table.json /
   transfer_loo_rare.json exactly?
2. Is the arm-selection discipline sound (weights from train metadata only;
   threshold on the held-out two-target stratum; one val application per frozen
   configuration) and is the trained-declaration complete and unambiguous?
3. Is the verdict ("the transfer inversion is broken by the transfer-aware
   recipe") supported, and does the mechanism attribution (threshold-on-stratum
   dominant; stratified weights a smaller rare-ref-dependent increment) hold
   against the single-correction ablations and the leave-one-out?
4. Must-fix list, verdict, score.
