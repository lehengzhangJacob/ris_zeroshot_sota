# Review digest — GLOD-RIS independent reproduction (2026-10-11)

Machine-read artifacts to check this digest against:
`exp/out_final/glodris_repro.json` (complete=true), `paper/tables/repro.tex`,
`paper/tables/sota.tex` (last row), `ENDGAME.md` §"GLOD-RIS independent reproduction".

## What was run (operator instruction 2026-10-10/11)

Full official val splits of the independent GLOD-RIS reimplementation
(`/home/msj_team/Jacob/sys/glod_ris_repro`; author code is unreleased). Guidance
from GLOD-RIS's own MLLM (GLM-4.1V-Thinking-Flash, Zhipu API, our prompt), every
response cached; parsing/transport failures parked and counted, never aborting a
run. Implementation frozen across all three runs (implementation_sha256
71cb44240ba025f6). Per-dataset run_hash differs by design (dataset manifest is in
the spec). Nothing was tuned on val; the reported reproduction is the unflagged
full pipeline.

## Numbers (from exp/out_final/glodris_repro.json)

| dataset | n | mIoU | oIoU | published mIoU/oIoU | delta mIoU | parked (DOD-only) |
|---|---|---|---|---|---|---|
| RefCOCO | 10834 | 32.6781 | 27.9194 | 58.42 / 50.85 | -25.7419 | 15 |
| RefCOCO+ | 10758 | 30.9572 | 26.1309 | 54.01 / 45.96 | -23.0528 | 14 |
| RefCOCOg | 4896 | 37.2403 | 31.9795 | 57.85 / 50.03 | -20.6097 | 0 |

Oracle mIoU (RefCOCO val, best pool mask) = 82.2113.

## Component controls (RefCOCO val, n=10834, this run) — diagnostics only

full pipeline 32.6781 / 27.9194; no GLM guidance at all 35.0892 / 27.1037;
category only 36.1082 / 28.8437; localization only 28.9837 / 24.3796; no
orthogonal decoupling 31.9530 / 27.3351. (Values copied verbatim from
`refcoco_component_controls` in the JSON.)

Interpretation used in the paper: the Gaussian-grid localization term degrades our
guidance integration; category term and orthogonal decoupling each contribute about
+1 mIoU. Feature-path anchor: released HybridGL ancestor scores 44.82 mIoU on the
same split/harness (published 49.48), so replacing the unreleased trained feature
path with frozen off-the-shelf components (SAM ViT-H + CLIP ViT-B/16) costs roughly
ten mIoU (44.82 - 35.09 ~ 9.7).

Provenance: the parked/fallback counts (15/14/0 parked; 92/169/64 all-zero grids;
0/0/0 category-missing) are recomputed by `exp/exp142_glodris_repro.py` from the
per-expression record files at aggregation time — never hand-inserted; re-running
`python3 exp/exp142_glodris_repro.py` reproduces them from the records.

## Paper text added (verbatim, §5 "Reproduction audit")

\paragraph{Independent reproduction of GLOD-RIS.}
GLOD-RIS's code is unreleased, so its rows in Table~\ref{tab:sota} are quoted
across harnesses. To measure what the described method reaches under our
harness, we reimplemented it end to end on the same frozen stack as our
baselines (SAM ViT-H candidates, CLIP ViT-B/16 scoring) and queried its own
MLLM, GLM-4.1V-Thinking-Flash, through the Zhipu API with our own prompt;
every response is cached, and parsing and transport failures are parked and
counted rather than aborting a run (15, 14 and 0 expressions of the three
splits ran the guidance-free path). The reproduction \emph{completes} on all
three official val splits but lands far below the published numbers (-25.7,
-23.1, -20.6 mIoU; Table~\ref{tab:repro}), and two measured facts localize the
gap. (i)~\emph{Candidate generation is not the limit}: the oracle mIoU over
the SAM pool is 82.2, and the largest single component loss is in how the
selector consumes the MLLM guidance -- removing the Gaussian-grid localization
term raises RefCOCO val from 32.68 to 36.11 mIoU, while the category term and
the orthogonal decoupling each contribute about +1 mIoU
(Table~\ref{tab:repro}), i.e., the guidance integration our reimplementation
derives from the paper's equations degrades selection rather than improving
it. (ii)~\emph{Feature path}: the released ancestor scores 44.82 on the same
split in this harness, so roughly ten mIoU points are lost simply when the
unreleased trained feature path is replaced by frozen off-the-shelf
components. We therefore keep the published GLOD-RIS rows as the comparison
and report our method's margin over them as cross-harness; the reproduction
row in Table~\ref{tab:sota} is marked as measured in this run so the shortfall
is visible rather than hidden.

Table float `tab:repro` inputs `paper/tables/repro.tex` (reproduction rows +
controls). Table `tab:sota` gained the row "GLOD-RIS (our independent
reproduction)$^\dagger$" with 32.68/27.92, 30.96/26.13, 37.24/31.98 and the
caption note that it is discussed in the reproduction audit.

## Review questions

1. Do the claims in the paragraph match the artifact exactly (numbers, park
   counts, oracle, deltas, the ~10-mIoU feature-path arithmetic)?
2. Is the cross-harness framing correct and complete — does anything anywhere
   still imply a same-environment comparison against GLOD-RIS?
3. Are the controls correctly labelled as diagnostics (no tuning on val), and is
   the localization-only vs full vs no-guidance reading sound?
4. Must-fix list (if any), then whether this can ship.
