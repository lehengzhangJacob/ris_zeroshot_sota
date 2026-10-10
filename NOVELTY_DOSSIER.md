# Novelty Check Dossier — "Score-gated connected growth" for training-free zero-shot RIS

Date: 2026-10-02. Executor family: DeepSeek (deepseek-flash). Reviewer: Zhipu glm-5.3 (cross-family).

## 1. Proposed method (as stated by the author)

Zero-shot RIS pipeline: frozen SAM emits a candidate mask pool; a frozen CLIP/VLM scores each
candidate against the referring expression; the baseline returns `argmax` over scores.

Claimed empirical observation: the CLIP region-text score has too little dynamic range to pick a
single winner — measured score differences of ~0.01 separate a mask with IoU 0.00 from one with
IoU 0.93.

Proposed refinement ("score-gated connected growth"), training-free, one hyper-parameter eps~0.3:
1. Gate: candidate SET = all masks whose score is within `eps * (max_score - min_score)` of the top
   score (gate is relative to the within-image score RANGE, so no dataset-specific constant).
2. Grow: take CONNECTED GROWTH of the argmax seed — iteratively union in any gated mask that
   overlaps the current growing region; output that grown mask.
No MLLM, no parser, works on top of any per-proposal scorer.

Measured on RefCOCO val (250 refs / 716 held-out expressions): mIoU 0.350 -> 0.412 (+17.6%),
oIoU 0.271 -> 0.310.

## 2. Core claims that carry the delta
C1. Region-text CLIP score is near-flat over the mask pool, so argmax is an unreliable selector.
C2. Fix selection by a score SET gated relative to the within-image score range (eps*(max-min)).
C3. Resolve the tie structurally by connected growth (overlap-constrained iterative union from the
    argmax seed), not by score.
C4. Entirely training-free and scorer-agnostic: no MLLM, no parser, no dataset-specific constant.
C5. Reported +17.6% mIoU / +14.4% oIoU over the argmax baseline on RefCOCO val.

## 3. Prior work found in Phase B (all arXiv IDs verified via arXiv API unless marked)

### Directly on the selection/merge axis
- **MARS: a Multimodal Alignment and Ranking System for Few-Shot Segmentation**, arXiv:2504.07942,
  WACV 2026 (Catalano et al.). VERIFIED FULL TEXT. Abstract: "plug-and-play ranking system that
  leverages multimodal cues to filter and merge mask proposals robustly. Starting from a set of mask
  predictions for a single query image, we score, filter, and merge them to improve results."
  Eq. (20): static gate th_s = 0.55 on the MARS score. Eq. (21): if empty, **dynamic gate relative to
  the per-image max**: keep {m_i : MARS(m_i) >= max_j MARS(m_j) * 0.95}. Eq. (22): final mask =
  **logical OR over all gated proposals** (plain union; NO connectivity/growth constraint, no seed).
  Scores are the average of 4 multimodal scores (Local/Global x Conceptual/Visual; AlphaCLIP +
  DINOv2 + ViP-LLaVA). Task is few-shot segmentation (COCO-20i, Pascal-5i, LVIS-92i, FSS-1000),
  not referring; the score is not a raw CLIP region-text cosine.
- **E-SAM: Training-Free Segment Every Entity Model**, arXiv:2503.12094, ICCV 2025 (Oral). VERIFIED
  ABSTRACT. Freezes SAM; post-processes AMG masks. EMR module builds a superpixel-centroid cosine
  similarity matrix and an adjacent-mask similarity matrix, then **merges two similar adjacent masks
  into one entity-level mask only if a mask in the Mask Gallery encompasses both**; USR adds point
  prompts where IoU with the entity map exceeds rho. Gate is visual/containment, NOT text; task is
  class-agnostic entity segmentation; no seed-and-grow from an argmax.
- **ProposalCLIP: Unsupervised Open-Category Object Proposal Generation via Exploiting CLIP Cues**,
  arXiv:2201.06696, CVPR 2022. Verified ID/title. Graph-based proposal merging: nodes = proposals,
  an edge exists iff IoU > 0.5 AND CLIP visual-feature cosine > 0.9; maximal connected subgraphs are
  merged and re-scored. Gate is proposal-proposal visual similarity + IoU, not text-conditioned;
  task is class-agnostic proposal generation, not referring.
- **"Dual Affinity Graph Construction for Open-Vocabulary Semantic Segmentation"**, Journal of KIISE
  (JOK) 53(7):680-687, 2026, DOI 10.5626/JOK.2026.53.7.680; extended version "Dual Visual-Textual
  Graph Construction for Open-Vocabulary Semantic Segmentation", IEEE Xplore doc 11359636.
  Verified via KCI/JOK/IEEE records (Korean-language venue; small). SAM masks as visual nodes, CLIP
  target-class embeddings as textual nodes; edges combine inter-mask visual affinity and
  visual-textual structural affinity; clusters are meant to represent whole objects rather than
  discriminative parts. Text-conditioned merging of SAM parts, but via graph clustering on affinity
  profiles, not a seed's connected growth and not a score-range gate.
- **MaskClustering: View Consensus based Mask Graph Clustering for Open-Vocabulary 3D Instance
  Segmentation**, arXiv:2401.07745 (2024). VERIFIED ABSTRACT. Training-free; masks are nodes,
  view-consensus rate is the edge weight, iterative clustering merges masks into 3D instances. 3D.
- **"Zero-shot Referring Image Segmentation using Referring Expression Augmentation and Mask
  Aggregation"**, JOK 53(1):15-21, 2026, DOI 10.5626/JOK.2026.53.1.15 (Korean journal; verified via
  KCI/JOK records). LLM-inferred attributes + mask aggregation; reported gains are +0.41 oIoU /
  +0.74 mIoU. Not a growth algorithm.
- **GF-SAM "Bridge the Points: Graph-based Few-shot Segment Anything Semantically"**, arXiv:2410.06964,
  NeurIPS 2024 (Zhang, Gao, Jiao, Liu, Wei). Verified via NeurIPS proceedings page + ar5iv. Builds a
  DIRECTED GRAPH between SAM-generated masks and point prompts from mask coverage, decomposes
  **weakly connected components** into natural clusters of masks/points, then applies **positive
  gating** ("retains masks capturing a greater proportion of potential target areas") and
  overshooting gating to aggregate high-confidence masks. This is connectivity-based grouping of a
  SAM mask pool with gating, but the gate is COVERAGE-based (not a text score), there is no seed
  argmax nor score-margin set, and the task is few-shot semantic segmentation.
- **PinPoint: Training-Free Referring Segmentation via Prompt Disambiguation**, arXiv:2605.26689
  (2026). Training-free VLM+SAM referring pipeline; argues the gap is dominated by prompt
  ambiguity / SAM instability at boundaries; fixes the point-prompt selection, not the mask set.

### Directly on "argmax selection is the bottleneck"
- **Venice-H1: Failure-Aware Query Re-Ranking with Multi-Scale Grid Signatures for Referring Image
  Segmentation**, arXiv:2606.22546 (2026). VERIFIED ABSTRACT. States that RIS systems "generate
  multiple candidate masks per expression but rely on a simple heuristic -- typically the argmax
  detection score -- to select the final output" and that query selection is a failure-case
  bottleneck (3-11% mIoU best-query selection gap; heuristic selection succeeds on only 82-93%).
  Fix is a LEARNED Transformer re-ranker (11.3M params) with a Failure Gate, not structural/training-free.
- **RefChess: Training-Free Contextual Search for Zero-Shot Referring Image Segmentation**, ICML 2026
  poster #64068 (Tong et al.), code github.com/Tongshiyan/RefChess. Verified via the ICML virtual
  page only (ICML 2026, already published as of Oct 2026). Training-free; attacks exactly the claim
  that "existing methods typically score proposals independently and can be distracted by visually
  similar candidates that partially satisfy the expression"; uses Monte-Carlo Tree Search over
  sampled distractor contexts with a stability-aware reward (language decomposition, vision-language
  similarity, object-centric cues, spatial guidance). Different mechanism (search, no growth/union).
- **Prompt Sensitivity in Vision-Language Grounding: How Small Changes in Wording Affect Object
  Detection**, arXiv:2604.17126 (ICCAI 2026). VERIFIED ID/TITLE. DETR proposals + CLIP ViT-B/32
  argmax selection; >55% of images change grounding under near-synonym prompts; text similarity
  explains only 34% of disagreement; identifies the **argmax selection mechanism** as the primary
  source of brittleness because it "amplifies small perturbations in similarity scores into discrete
  changes in spatial output" (non-Lipschitz). No structural fix proposed (recommends reporting variance).

### Adjacent zero-shot RIS baselines / selection-era methods
- TAS (Text Augmented Spatial-aware), Findings of EMNLP 2023; HybridGL, arXiv:2504.00356, CVPR 2025
  (VERIFIED ABSTRACT: training-free SAM proposals + CLIP global-local mask features + spatial
  guidance augmentation, then pick the best mask); LGD, arXiv:2504.14467 (MLLM-generated attribute/
  surrounding descriptions combined into 3 matching scores); BMS = Bidirectional Mask Selection,
  IEEE TCSVT 2024, DOI 10.1109/TCSVT.2024.3460874 (negative masks + training-free mask adaptive
  fusion); IteRPrimE, arXiv:2503.00936, AAAI 2025 (iterative Grad-CAM + post-processing that
  "select[s] a high-quality, contiguous instance mask ... which encapsulates the target object");
  S+S = "Segment, Select, Correct", arXiv:2310.13479, ECCV 2024 Workshops (zero-shot segment+select,
  then TRAIN a corrector -- evidence that selection is the known bottleneck); Tarot-SAM3,
  arXiv:2604.07916 (training-free SAM3 + MLLM; Mask Self-Refining corrects over/under-segmentation
  by re-prompting SAM3 with shifted/added points); GFR-SAM, arXiv:2607.11732 (generate-filter-refine,
  training-free, camouflaged); RefAM (training-free, diffusion attention sinks); GLOD-RIS, IEEE
  Xplore 11711038 (2026; MLLM Gaussian localization + orthogonal decoupling, not training-free-only).
- RAS "Refer to Any Segmentation Mask Group", arXiv:2506.05342, ICCV 2025 — trained mask-centric LMM
  for mask-GROUP prediction (different task; training required).
- LoTeR (CVIU 2026, training-free, localized text-prompt refinement selecting SAM point prompts).

### Seed-then-expand / growth family
- Grounded SAM, arXiv:2401.14159 (2024): text detector boxes -> SAM masks; threshold on box score
  (absolute), no growth. Semantic-SAM, arXiv:2307.04767 (ICCV 2023): multi-granularity, trained.
  RSPrompter, arXiv:2306.16269 (2023): trained SAM prompting for RS. PerSAM, arXiv:2305.03048,
  ICLR 2024: training-free personalization; cascaded post-refinement re-prompts SAM with its own
  mask + a box; PerSAM-F merges SAM's 3 scale masks by LEARNED weighted summation.
- T2S "Text-to-seed generation", arXiv:2608.26624 (2026): training-free OVSS as text-guided seed
  localization (Stable Diffusion attention) + SAM point-prompt expansion. Expansion operator is SAM,
  not a text-score-gated union; no candidate pool gating.
- GeoSelect, arXiv:2607.03869 (2026): training-free referring RS via a typed spatial program over a
  scored candidate set (uses a text-only LLM as a compiler). Selection is program execution.
- Classical precedents (not RIS): seeded region growing (Adams & Bischof 1994), hysteresis/dual-
  threshold connected-component growing (Canny 1986), binary partition tree / region merging.

### Low-dynamic-range / flat-score family (axis 5)
- CorrCLIP, arXiv:2411.10086 (ICCV 2025): "Range Reconstruction" explicitly constrains CLIP's
  patch-patch correlation range; but it is about patch correlation coefficients in dense CLIP
  attention, not the region-text score distribution over a proposal pool.
- MaskCLIP ("Extract Free Dense Labels from CLIP", arXiv:2112.01071), SCLIP (arXiv:2312.01597),
  ProxyCLIP (arXiv:2408.04883), CLIPSeg, CLIP-ES (CVPR 2023), GroupViT (arXiv:2202.11094): modify
  CLIP internals or learn grouping; none treats a proposal pool's score RANGE as the gate.
- On the Brittleness of CLIP Text Encoders, arXiv:2511.04247 (2025): documents CLIP ranking
  instability under perturbation; modality gap literature (Mind the Gap, NeurIPS 2022) explains
  compressed cross-modal cosines. Neither is RIS-proposal-specific.
- HybridGL/TAS motivate the fix by CLIP's "weak fine-grained region-text matching ability and the
  domain gap between masked images and natural images" -- a problem statement, but the fix is
  feature/matching-side, not selection-side normalization.

## 4. Questions for the reviewer
1. Is "score-gated connected growth" novel as a method, given MARS (relative-to-max gate + logical OR
   union of gated proposals, WACV 2026) and E-SAM / ProposalCLIP / Dual-Affinity-Graph mask merging?
2. What is the closest prior work and exactly what is the remaining delta?
3. Is the delta a publishable contribution, or is it "MARS with a connectivity constraint"?
4. Does the claimed empirical observation (flat CLIP region-text scores; ~0.01 differences between
   IoU 0.00 and IoU 0.93 masks) add standalone value if the mechanism is not novel?
5. Which specific paper is the single most dangerous piece of prior art?

=== NOVELTY VERDICT LIMITS (these bound how you judge, never how widely you search) ===
Search exhaustively; judge calibrated. Two failures waste months equally:
passing an idea a published paper already contains, and killing a viable idea
because the territory has neighbors.
1. Proximity is information, not a verdict. Someone working nearby goes in the
   report; it is not by itself a reason to reject.
2. ABANDON has exactly one qualification: a specific published paper already
   contains this result — name that paper. No named paper, no ABANDON.
3. Crowded-but-deltaed is PROCEED: state the delta in one sentence a reviewer
   could verify. Thin or contested delta is PROCEED WITH CAUTION — say what
   would make it carry, not why it should die. CAUTION is not a safe middle:
   if you cannot name the specific thing that makes the delta thin, the
   verdict is PROCEED.
4. Concurrent or competing work is not a veto. That is a race — report it and
   let the user decide whether to run it.
5. A direct attack on a central problem is legitimate novelty when nobody has
   executed it well. "This area is hot" does not mean "this area is taken."
6. This check is an early gate, never the last one — more triage, pilots, or
   external review still stand between any idea and a paper, whatever order
   this run uses. A wrongly passed idea dies cheaply at one of them; a wrongly
   killed idea is never seen again. When torn between two verdicts, choose the
   more permissive one.
Say plainly when an idea clears the check. Do not manufacture overlap.
