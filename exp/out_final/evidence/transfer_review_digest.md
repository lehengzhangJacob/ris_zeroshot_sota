# Review digest — transfer-inversion cycle (2026-10-13)

Machine artifacts: `exp/out_final/transfer_wall.json` (gated),
`exp/out_final/main_grefcoco.json` (shipped row + history + reference row),
`exp/out_final/sota_table.json` (gRefCOCO rows), `exp/out_final/learned_prior.json`
(previous cycle's arms), `exp/out_transfer/{train,val}_parts.npz` (cached part
tables). Paper: `05_experiments.tex` §"The transfer inversion" (new fourth act),
the gres-table caption, `main.tex` abstract.

## What was done (directive 2026-10-13: make the transfer inversion the object)

1. **Characterization** (four falsifiable decompositions, measured on cached part
   tables; train = held-out train carve-out 10,351 parts / 780 targeted rows,
   val = 89,708 parts / 4,677 rows):
   - population/composition: train targeted rows 68.33% single-target / 30.90%
     two-target vs val 99.53% two-target; mean parts per row 13.27 vs 19.18;
   - prior-quality: train-fitted model calibrated on train (pred 0.427 / actual
     0.412) but monotone over-prediction on val (pred 0.366 / actual 0.326;
     high-score quintile pred 0.57 / actual 0.42);
   - feature drift: term_index median 0 -> 1; inbox_det0 p90 0.41 -> 0.16;
     inbox_llava p90 0.14 -> 0.36;
   - image level: median masks/image 51 vs 47 (rejected as the driver).
2. **Transfer-aware recipe** (one arm, train-side only): composition-matched part
   weights (uniform over the train-side n_targets composition) + within-row rank
   score features + acceptance threshold tuned on the held-out 30% carve-out
   RESTRICTED to its two-target stratum. Val (single frozen application):
   **cIoU 52.7061 / gIoU 75.0817** (n=14,229) — above the training-free record
   51.3182 / 74.9098 and far above the arms that failed (naive 38.792,
   rank-only 50.52). Held-out two-target cIoU 58.90 (recipe) / 60.05 (+policy).
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

## Review questions

1. Claim-vs-artifact consistency: do the digest's numbers match
   transfer_wall.json / main_grefcoco.json / sota_table.json exactly?
2. Is the arm-selection discipline sound (composition weights from train
   metadata only; threshold on the held-out two-target stratum; one val
   application), and is the trained-declaration complete and unambiguous?
3. Is the verdict ("the transfer inversion is broken by the transfer-aware
   recipe") supported, or does it overreach given the mechanism attribution?
4. Must-fix list, verdict, score.
