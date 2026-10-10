# Paper Improvement Log

**Paper** `paper/main.tex` → `paper/main.pdf` (10 pages)
**Venue** IEEE Transactions (IEEEtran, journal) · **Assurance** draft
**Reviewer** `glm-5.3` via `mcp__llm-chat__review` (executor `deepseek-flash`), thread
`llm-chat:379b639b9599`

---

## Round 0 → Round 1

**Review round 0.** The reviewer received all seven section sources verbatim. Its
response concentrated on numerical consistency between sections and tables, which is
exactly where the defects were.

### Defects raised and disposition

| # | defect | disposition |
|---|---|---|
| 1 | **Under-segmentation number inconsistency.** §I paired "2.4 %" with "17.6 %" as if both were full-split operating points. The 2.4 % is the 716-expression unguided sample; the full-split unguided value is 14.3 % (Table I). The conclusion then compressed this into a "2–18 %" range that also contradicted Table pool (20.5–22.6 % on the selection split). | **Fixed.** §I now reads 14.3 % / 17.6 % at the two full-split RefCOCO operating points and "never exceeds 22.6 % in any configuration we measured". Abstract and conclusion now use 5–23 % (the measured range across datasets, from `cross_dataset.json`). |
| 2 | **"At two operating points" overclaimed for RefCOCOg / RefCOCO$+$.** Table cross is released-point only; the two-operating-point taxonomy exists only for RefCOCO (Table I) and the 716-expression sample. | **Fixed** in §I (contribution bullet), §V-H and §VII: two operating points on RefCOCO, one on RefCOCO$+$/g, two pool densities. |
| 3 | **57.4 % vs 67 % spatial-expression rates appear unreconciled**, a reader sees two different numbers for the same quantity. | **Fixed** in §III-C: the 67 % is the broad lexical probe, the 57.4 % is the operator's narrower relation parser and the coverage figure used in Table cross. The two are now explicitly distinguished at first use. |
| 4 | **Rounding: displayed deltas do not reproduce from displayed mIoU values** (e.g. 0.5029 − 0.4482 = 0.0547, table shows +0.0546). | **Fixed** with an explicit note in the main table caption that deltas are computed from unrounded values. The audit below confirms every delta matches its artifact exactly. |
| 5 | Oracle-hit rate (0.454 → 0.516), stratification deltas, quartile deltas, part-expression counts and the snap-complementarity counts appear in prose but in no table. | **Accepted as-is.** These are supporting figures, the paper already carries five tables, and page budget is at 10. Recorded here rather than acted on. |
| 6 | The reviewer could not check the abstract (it lives in `main.tex`, not a section file). | **Self-audited.** Every abstract number was checked against the artifacts in the audit below. |
| 7 | Claim calibration: does the paper lean on the method result while claiming the diagnosis is the contribution? | **No change needed.** The reviewer's own reading was that the claim/evidence mapping held once defect 1 was fixed; §I, §VI and §VII all lead with the diagnosis and state the operator's boundary in the abstract. |

### Numerical audit after fixes

All 49 headline numbers in the paper were re-extracted from their artifacts
(`final_refcoco.json`, `taxonomy_refcoco_guided.json`, `headroom_refcoco_guided.json`,
`cross_dataset.json`, `review_fixes.json`, `pool_robustness.json`) and compared to the
values printed in the LaTeX: **49 checked, 49 exact, 0 mismatches.**

### Format

Compiled with `pdflatex` + `bibtex` (TeX Live scheme-small, installed locally at
`~/.local/texlive`; IEEEtran.cls supplied alongside the source because the scheme's
`ieeetran` package provides only the BibTeX style).
**0 errors, 0 overfull boxes, 10 pages, 0 undefined references or citations.**

## Remaining known issues (not defects of the manuscript)

- The research reviewer's largest open item — an MLLM chooser baseline over the same
  candidate set — is **not run**, for the reasons stated in §VI, and the paper discloses
  it as the largest open flank with a measured ceiling bound (+0.227 mIoU) in its place.
- Ordinal expressions abstain (4.1 % of RefCOCO val).
- The reproduction of the baseline is 5.09 mIoU below the published base row on RefCOCO
  val, localised to the frozen feature path in §V-J.
