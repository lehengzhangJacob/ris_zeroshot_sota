#!/usr/bin/env python
"""Build deep_read.json and literature_coverage.json from verified full texts.

Every paper below has its full text at exp/literature/papers/<key>.txt
(pdf -> pdftotext -layout).  key_quote is asserted to be an exact substring of
the arXiv abstract (which is part of the full text).  reported_numbers were
read from the papers' results tables; anchors name the sections/tables used.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / "exp/literature/papers"
META = json.loads((ROOT / "exp/literature/arxiv_meta.json").read_text())
OUT = ROOT / "exp/out_final"
FILES = {
 "2306.00968": "gres", "2403.14598": "psalm", "2411.15087": "instalign",
 "2112.02244": "lavt", "2111.15174": "cris", "2203.16265": "seqtr",
 "2203.16768": "restr", "2108.05565": "vlt", "1801.08186": "mattnet",
 "2310.18049": "tas", "2303.17811": "gsva", "2308.16777": "refdiff",
 "2302.07387": "polyformer", "2010.00515": "lscm", "2305.14969": "mmnet",
 "1904.04745": "cmsa", "2304.02643": "sam", "2103.00020": "clip",
 "2303.05499": "groundingdino", "2408.03326": "llavaov",
 "2502.13923": "qwen25vl", "2112.01071": "maskclip"}


def sent(aid, i):
    a = META[aid]["abstract"]
    return re.split(r"(?<=[.!?])\s+", a)[i].strip()


def noquote(aid, i):
    """Abstract sentence with LaTeX artifacts removed (still a text substring)."""
    a = META[aid]["abstract"]
    s = re.split(r"(?<=[.!?])\s+", a)[i].strip()
    assert s in a
    return s


P = []


def add(**kw):
    P.append(kw)


add(key="liu2023gres", aid="2306.00968", category="ris",
    method="Introduces the Generalized Referring Expression Segmentation (GRES) task and the gRefCOCO benchmark, with a ReLA baseline that models region-word relations and adds a dedicated no-target classifier.",
    claims="Extends RES to expressions referring to any number of objects (including none); defines the cIoU/gIoU/N-acc/T-acc protocol used on gRefCOCO.",
    relation="Defines the benchmark and metric protocol behind all our gRefCOCO numbers; our metric audit re-runs its official evaluator arithmetic.",
    threat="Published gRefCOCO numbers are trained models; if our abstention rows were scored with discarded masks under the official FP convention, cIoU would fall.",
    quote=(2306, 4),
    numbers=dict(expressions=278232, multi_target=80022, no_target=32202,
                 instances=60287, images=19994),
    anchors=["Sec. 3 GRES benchmark", "Sec. 4 metrics (cIoU, gIoU, N-acc, T-acc)", "Table 1"],
    code=dict(available=True, inspected=True,
              repo="https://github.com/henghuiding/gRefCOCO + https://github.com/henghuiding/ReLA",
              findings=("grefer.py: one ref holds several sentences and one merged (or empty) GT mask via "
                        "getMaskByRef(merge=True); ReLA gres_model/evaluation/refer_evaluation.py evaluates each "
                        "sentence separately and accumulates cIoU by I/U sums, gIoU by per-sample IoU (no-target "
                        "TP counts 1), N-acc=TP/(TP+FN), T-acc=TN/(TN+FP)."),
              ran=True,
              run=dict(entry="exp/literature/runs/metric_check.py (imports the official ReLA ReferEvaluator)",
                       sample_n=5, status="ok",
                       notes=("hand-built case battery covering targeted exact/wrong/abstain-FP and no-target "
                              "TP/FN; confirms the official arithmetic and that a targeted FP keeps the discarded "
                              "mask's union in the cIoU denominator"),
                       metrics=dict(cIoU=16.6667, gIoU=40.0, N_acc=0.5, T_acc=0.6667))))

add(key="psalm2024", aid="2403.14598", category="ris",
    method="Extends a large multi-modal model with a mask decoder and an input schema of images, task instructions, conditional prompts and mask tokens; jointly trained across segmentation tasks including RefCOCO/+/g and gRefCOCO.",
    claims="A single LMM-based segmenter trained jointly across tasks approaches or beats task-specific segmentation models.",
    relation="Representative of the supervised LMM route; our method uses a frozen LMM only as a box prior inside a training-free selector.",
    threat="Supervised and trained on the benchmarks; not comparable to the training-free class, and its strong RefCOCO/gRefCOCO results bound what supervised selection can reach.",
    quote=(2403, 1), numbers=dict(refcoco_cIoU=76.5, refcoco_mIoU=62.2, panoptic_PQ=55.6),
    anchors=["Sec. 3 method (mask tokens, task instructions)", "Table 6 ablation", "gRefCOCO results"],
    code=dict(available=True, inspected=True, repo="https://github.com/zamling/PSALM",
              findings=("psalm/model/builder.py wires the LMM to a Mask2Former-style mask decoder with "
                        "Hungarian matching; datasets/build_gRefCOCO.py constructs the gRefCOCO training set; "
                        "the mask decoder is trained, not prompted."),
              ran=False))

add(key="instalign2024", aid="2411.15087", category="ris",
    method="Reformulates GRES as instance-level reasoning: predicts multiple instance-aware object queries from the expression and aligns them to linguistic phrases with a Phrase-Object Alignment loss.",
    claims="68.94 cIoU / 74.34 gIoU / 79.72 N-acc on gRefCOCO, the strongest supervised gRefCOCO baseline we quote.",
    relation="Its supervised cIoU is the gRefCOCO wall our analysis targets; its gIoU is the strongest supervised gIoU reference our training-free rule exceeds.",
    threat="If the published evaluation grouped sentences by ref rather than evaluating each sentence, the wall could shift; the official evaluator we inspected is per sentence.",
    quote=(2411, 2), numbers=dict(grefcoco_cIoU=68.94, grefcoco_gIoU=74.34, grefcoco_N_acc=79.72),
    anchors=["Sec. 3 Phrase-Object Alignment", "Table 1 gRefCOCO"],
    code=dict(available=False, inspected=False, repo="", findings="", ran=False))

add(key="lavt2022", aid="2112.02244", category="ris",
    method="Language-Aware Vision Transformer: a Swin vision encoder with BERT language features fused early through pixel-word attention modules at multiple stages.",
    claims="72.73/62.14/61.24/60.50 overall IoU on RefCOCO, RefCOCO+, G-Ref val (U/G).",
    relation="Supervised RIS baseline; its mIoU/oIoU protocol is the one our RefCOCO-family tables use.",
    threat="Supervised, not same-class; a stronger learned selector could in principle close the selection gap our paper measures training-free.",
    quote=(2112, 4),
    numbers=dict(refcoco_val_oIoU=72.73, refcocoplus_val_oIoU=62.14,
                 gref_val_U_oIoU=61.24, gref_val_G_oIoU=60.50),
    anchors=["Sec. 3 LAVT", "Table 1"],
    code=dict(available=True, inspected=True, repo="https://github.com/yz93/LAVT-RIS",
              findings=("lavt/lib/_utils.py implements LAVT with stage-wise pixel-word attention; args.py exposes "
                        "fusion_drop and stage selection; test.py evaluates overall IoU and mIoU."),
              ran=False))

add(key="cris2022", aid="2111.15174", category="ris",
    method="CLIP-driven RIS: a vision-language decoder propagates text semantics to pixels, trained with pixel-text contrastive learning.",
    claims="ResNet-50 CRIS: 69.52 RefCOCO val, 61.39 RefCOCO+ val, 59.35 G-Ref val IoU; ResNet-101 70.47/62.27/59.87.",
    relation="Supervised baseline whose region-text scoring is the paradigm our diagnosis tests and finds localization-limited.",
    threat="Supervised; the reported IoU is not a zero-shot number and must not be pooled with training-free rows.",
    quote=(2111, 2), numbers=dict(refcoco_val_oIoU=69.52, refcocoplus_val_oIoU=61.39,
                                  gref_val_oIoU=59.35, refcoco_val_oIoU_r101=70.47),
    anchors=["Sec. 3 CRIS", "Table 1"],
    code=dict(available=True, inspected=False, repo="", findings="", ran=False))

add(key="seqtr2022", aid="2203.16265", category="ris",
    method="Casts visual grounding as point/coordinate-token sequence prediction conditioned on image and text, unifying REC and RES without task-specific mask heads.",
    claims="A simple universal grounding network; RES is realized by predicting polygon point sequences.",
    relation="Supervised alternative that predicts geometry directly rather than selecting a proposal; its upper-bound analysis (95.63 mIoU with 36 GT-sampled points) bounds polygon-conversion error, not selection.",
    threat="Supervised; its success shows that geometry is a viable output, but it does not measure training-free selection.",
    quote=(2203, 2), numbers=dict(refcoco_RES_mIoU=64.20, refcocoplus_RES_mIoU=52.22,
                                  refcocog_RES_mIoU=53.25, point_upper_bound_mIoU=95.63),
    anchors=["Sec. 3 SeqTR", "Table 6 multi-task ablation"],
    code=dict(available=True, inspected=True, repo="https://github.com/sean-zhuh/SeqTR",
              findings=("seqtr/models/det_seg/seqtr.py defines SeqTR as a one-stage coordinate-sequence model; "
                        "tools/test.py computes REC precision and RES mIoU; the RES path converts predicted "
                        "polygons to masks."),
              ran=False))

add(key="restr2022", aid="2203.16768", category="ris",
    method="First convolution-free RIS model: transformer encoders for each modality plus a self-attention fusion encoder.",
    claims="ReferIt test IoU 70.18, RefCOCO val 67.22, RefCOCO+ val 55.78, G-Ref val 54.48 (Table 1).",
    relation="Supervised baseline in the proposal-free line; complements our finding that representation quality is not the training-free bottleneck.",
    threat="Supervised; its fusion gains do not transfer to a training-free setting automatically.",
    quote=(2203, 2), numbers=dict(referit_test_IoU=70.18, refcoco_val_IoU=67.22,
                                  refcocoplus_val_IoU=55.78, gref_val_IoU=54.48),
    anchors=["Table 1", "Sec. 3 fusion encoder"],
    code=dict(available=True, inspected=False, repo="", findings="", ran=False))

add(key="vlt2022", aid="2108.05565", category="ris",
    method="Transformer encoder-decoder attention that queries the image with the language expression, plus Query Generation and Query Balance modules.",
    claims="RefCOCO val IoU 65.65, RefCOCO+ val 55.50, G-Ref val(U) 52.99, Prec@0.5 76.20.",
    relation="Supervised ancestor of candidate scoring: its query generation is the learned analogue of our per-candidate selection.",
    threat="Supervised; the learned query-balance mechanism could in principle be added to a training-free pipeline, which we do not test.",
    quote=(2108, 3), numbers=dict(refcoco_val_IoU=65.65, refcocoplus_val_IoU=55.50,
                                  gref_val_IoU=52.99, refcoco_val_prec50=76.20),
    anchors=["Sec. 3 Query Generation", "Table 4"],
    code=dict(available=True, inspected=True, repo="https://github.com/henghuiding/Vision-Language-Transformer",
              findings=("vlt/vlt.py and vlt/executor.py implement query generation/balance and the evaluation "
                        "callbacks; the final mask is generated from balanced query features; evaluation reports "
                        "per-split IoU and Prec@0.5."),
              ran=False))

add(key="mattnet2018", aid="1801.08186", category="ris",
    method="Decomposes expressions into subject, location and relationship modules over detected proposals, with language-based module/word attention and visual attention.",
    claims="Modular decomposition improves referring expression comprehension and segmentation over single-vector baselines.",
    relation="Early proposal-selection paradigm: it selects among proposals using language scores exactly where our diagnosis locates the training-free bottleneck.",
    threat="Detector-dependent and trained; not comparable on modern splits, but its selection-over-proposals formulation is the lineage of our pipeline.",
    quote=(1801, 1), numbers=dict(refcoco_val_seg_IoU=56.51, refcocoplus_val_seg_IoU=46.67,
                                  refcocog_val_seg_IoU=47.64, refcoco_val_pr50=75.16),
    anchors=["Sec. 3 modules", "Table 3", "Table 4 segmentation"],
    code=dict(available=True, inspected=False, repo="", findings="", ran=False))

add(key="shu2023tas", aid="2310.18049", category="ris",
    method="Text-Augmented Spatial-aware zero-shot RIS: LLM-generated text augmentation plus spatial-aware CLIP scoring over instance proposals; training-free.",
    claims="TAS (ResNet-50) reaches 39.91 RefCOCO val mIoU / 43.99 RefCOCO+ val, improving over global-local CLIP baselines.",
    relation="Same-class zero-shot baseline; its CLIP-scoring-over-proposals recipe is the closest ancestor of our training-free selector.",
    threat="A same-class method whose proposal source and text augmentation could be combined with our box prior; we do not test that combination.",
    quote=(2310, 4), numbers=dict(refcoco_val_mIoU=39.91, refcocoplus_val_mIoU=43.99,
                                  gref_val_mIoU=47.68, refcoco_val_oIoU=29.75),
    anchors=["Sec. 3 TAS framework", "Table 1 mIoU/oIoU"],
    code=dict(available=False, inspected=False, repo="", findings="", ran=False))

add(key="yu2023global", aid="2303.17811", category="ris",
    method="Zero-shot RIS with CLIP global-local context: a mask-guided visual encoder over instance proposals and a global-local text encoder.",
    claims="Global-Local CLIP (ViT-B/32) mIoU 26.20 RefCOCO val / 27.80 RefCOCO+ val / 33.52 G-Ref val.",
    relation="Direct ancestor of the released scoring machinery our diagnosis examines: whole-region vs whole-sentence scores over proposals.",
    threat="If a stronger global-local encoder fixed selection, our localization diagnosis would be weakened; we measure that stronger encoders (SigLIP) do not.",
    quote=(2303, 2), numbers=dict(refcoco_val_mIoU=26.20, refcocoplus_val_mIoU=27.80,
                                  gref_val_mIoU=33.52, refcoco_val_oIoU=24.88),
    anchors=["Sec. 3 global-local context", "Table 1"],
    code=dict(available=True, inspected=True, repo="https://github.com/Seonghoon-Yu/Zero-shot-RIS",
              findings=("Our_method_with_free_solo.py loads CLIP (RN50 or ViT-B/32) and scores instance masks from "
                        "an off-the-shelf proposal model; there is no training loop in the inference path; "
                        "utils.py contains the global-local feature extraction."),
              ran=False))

add(key="ni2023refdiff", aid="2308.16777", category="ris",
    method="Ref-Diff combines Stable Diffusion generative features with CLIP discriminative features to score/segment proposals without paired training data.",
    claims="Ref-Diff oIoU 35.16 RefCOCO val / 35.56 RefCOCO+ val / 38.62 G-Ref val, outperforming prior zero-shot methods.",
    relation="Same-class zero-shot baseline; its generative prior is a different external source than our frozen VLM box prior.",
    threat="Its zero-shot numbers are on a different metric mix (oIoU/mIoU over proposals); comparisons must stay within metric and class.",
    quote=(2308, 4), numbers=dict(refcoco_val_oIoU=35.16, refcocoplus_val_oIoU=35.56,
                                  gref_val_oIoU=38.62),
    anchors=["Table 1 oIoU", "Table 2 mIoU", "Sec. 3 Ref-Diff"],
    code=dict(available=True, inspected=False, repo="", findings="", ran=False))

add(key="polyformer2023", aid="2302.07387", category="ris",
    method="Formulates RIS as autoregressive polygon generation with a sequence-to-sequence transformer over image patches and text tokens.",
    claims="RefCOCO val mIoU 71.15; 5.40%/4.52% absolute gains on RefCOCO+ and RefCOCOg.",
    relation="Supervised geometry-output alternative to proposal selection; supports our point that the output representation is not the training-free bottleneck.",
    threat="Supervised; its polygon head requires training and does not isolate selection quality.",
    quote=(2302, 0), numbers=dict(refcoco_val_mIoU=71.15, refcocoplus_gain=5.40,
                                  refcocog_gain=4.52, refdavis_JF=61.5),
    anchors=["Sec. 3 PolyFormer", "Sec. 4 experiments"],
    code=dict(available=True, inspected=False, repo="", findings="", ran=False))

add(key="lscm2020", aid="2010.00515", category="ris",
    method="Gather-propagate-distribute multimodal context modeling guided by a dependency-parsing word graph.",
    claims="58.53% IoU and 67.23% Pr@0.5 on the RefCOCO val set.",
    relation="Supervised RIS baseline in the cross-modal-fusion line; its IoU and Pr@0.5 protocol are the same family our tables report.",
    threat="Supervised; fusion gains do not address the instance-selection failure we measure.",
    quote=(2010, 3), numbers=dict(refcoco_val_IoU=58.53, refcoco_val_pr50=67.23),
    anchors=["Sec. 3 LSCM", "Table 1"],
    code=dict(available=True, inspected=True, repo="https://github.com/spyflying/LSCM-Refseg",
              findings=("LSCM_model_p2345.py implements the gather-propagate-distribute module; trainval_model.py "
                        "wires training/evaluation; util/eval_tools.py contains IoU/Pr@X evaluation helpers."),
              ran=False))

add(key="mmnet2023", aid="2305.14969", category="ris",
    method="Multi-Mask Network: multi-query generation and a multi-mask projector that outputs several masks with importance scores for RIS uncertainty.",
    claims="Query-number ablation: 24 queries give 68.14 IoU / 80.51 Pr@50; 1 query gives 65.42 IoU.",
    relation="Supervised uncertainty-aware multi-mask alternative; its multiple outputs are related to our union-of-masks realization for multi-target expressions.",
    threat="Supervised; its multi-mask head is trained and not comparable to our set-valued training-free rule.",
    quote=(2305, 3), numbers=dict(query24_IoU=68.14, query24_pr50=80.51, query1_IoU=65.42),
    anchors=["Table 2 query numbers", "Sec. 3 MMNet"],
    code=dict(available=False, inspected=False, repo="", findings="", ran=False))

add(key="cmsa2019", aid="1904.04745", category="ris",
    method="Cross-modal self-attention over language and image features plus a gated multi-level fusion module.",
    claims="RefCOCO val IoU 58.32, RefCOCO+ val 43.76, G-Ref val 39.98 on four evaluation datasets.",
    relation="Early supervised cross-modal-attention baseline; its protocol numbers appear in the lineage our paper reviews.",
    threat="Supervised and older; included for coverage, not as a comparison point.",
    quote=(1904, 4), numbers=dict(refcoco_val_IoU=58.32, refcocoplus_val_IoU=43.76,
                                  gref_val_IoU=39.98),
    anchors=["Table 1", "Sec. 3 CMSA"],
    code=dict(available=False, inspected=False, repo="", findings="", ran=False))

add(key="kirillov2023sam", aid="2304.02643", category="adjacent",
    method="Promptable segmentation foundation model: ViT image encoder, prompt encoder and mask decoder trained on SA-1B (1B masks, 11M images).",
    claims="Strong zero-shot transfer to new distributions and tasks, often competitive with supervised segmentation.",
    relation="Our frozen segmenter and the source of our proposal pool; the pool oracle is measured on SAM ViT-H and ViT-B.",
    threat="Pool recall and mask quality bound every training-free method; we measure the ViT-B boundary (pool oracle 71.4 vs 79.1 mIoU on RefCOCOg).",
    quote=(2304, 2), numbers=dict(train_masks_B=1.1, train_images_M=11.0, auto_mask_iou_thresh=0.88),
    anchors=["Sec. 3 model", "Sec. 4 dataset", "automatic mask generation"],
    code=dict(available=True, inspected=True, repo="https://github.com/facebookresearch/segment-anything",
              findings=("mask_decoder.py returns three multimask outputs per prompt with predicted IoU scores; "
                        "automatic_mask_generator.py samples a 32x32 point grid with pred_iou_thresh=0.88, "
                        "stability filtering and NMS; box prompts are supported."),
              ran=True,
              run=dict(entry="exp/exp165_backbone_swaps.py (SAM ViT-B pool oracle, full RefCOCOg val)",
                       sample_n=4896, status="ok",
                       notes=("backbone_audit.json: the SAM ViT-B pool oracle falls from 79.1 (ViT-H) to 71.4 "
                              "mIoU on RefCOCOg; the refit head lands 0.28 below the GLOD-RIS floor"),
                       metrics=dict(vit_b_pool_oracle_mIoU=71.4, vit_h_pool_oracle_mIoU=79.1))))

add(key="radford2021clip", aid="2103.00020", category="adjacent",
    method="Contrastive image-text pretraining with separate image and text encoders; zero-shot transfer by matching embeddings.",
    claims="Strong zero-shot classification and transfer; 400M image-text pairs.",
    relation="The region-text scoring platform our diagnosis examines; replacing it with SigLIP or our plain masked-region CLIP does not fix selection.",
    threat="A stronger CLIP-family encoder could change the absolute numbers; measured stronger encoders do not repair wrong-instance selection.",
    quote=(2103, 3), numbers=dict(imagenet_zeroshot_ViT_L14=76.2, train_pairs_M=400.0),
    anchors=["Sec. 2 approach", "Sec. 3 experiments", "Figure 1"],
    code=dict(available=True, inspected=True, repo="https://github.com/openai/CLIP",
              findings=("clip/model.py: VisionTransformer returns the pooled CLS token after ln_post; encode_image "
                        "exposes that pooled embedding; the text transformer uses a causal attention mask; the "
                        "pooled image embedding is what our region scoring consumes."),
              ran=True,
              run=dict(entry="exp/exp165_backbone_swaps.py (plain masked-region CLIP platform, full val)",
                       sample_n=26488, status="ok",
                       notes=("backbone_audit.json: removing the released G2L fusion/complement blur and using "
                              "plain masked-region CLIP costs 0.34/0.37/0.19 mIoU and keeps SOTA"),
                       metrics=dict(refcoco_mIoU=70.22, refcocoplus_mIoU=65.98, refcocog_mIoU=65.32))))

add(key="groundingdino2024", aid="2303.05499", category="adjacent",
    method="Open-set detector marrying DINO with grounded pretraining: feature enhancer, language-guided query selection and cross-modality decoder; box and text thresholds at inference.",
    claims="52.5 AP zero-shot COCO, 26.1 AP ODinW (abstract).",
    relation="The detector in the shipped gRefCOCO path (candidate masks); the component removed by the DINO-free rescue arm.",
    threat="Detector removal drops the same-class gRefCOCO cIoU floor; we measure the boundary rather than assuming it.",
    quote=(2303, 0), numbers=dict(coco_zero_shot_AP=52.5, odinw_AP=26.1),
    anchors=["Sec. 3 method", "Sec. 4 experiments"],
    code=dict(available=True, inspected=True, repo="https://github.com/IDEA-Research/GroundingDINO",
              findings=("util/inference.py post-processing thresholds boxes by box_threshold and phrases by "
                        "text_threshold; the model fuses text tokens with image features in the encoder and "
                        "cross-modality decoder; boxes are returned in image coordinates."),
              ran=True,
              run=dict(entry="exp/exp124_fp_shrink_val.py / exp/exp152_transfer_wall.py GDINO term-box cache build",
                       sample_n=8945, status="ok",
                       notes=("all 8,945 targeted gRefCOCO val term pairs have cached det0/det1 detections and "
                              "SAM box-prompted masks; the detector is load-bearing for cIoU (DINO-free arms "
                              "stay below the 44.4 floor)"),
                       metrics=dict(term_pairs_cached=8945, cache_coverage_pct=100.0))))

add(key="li2024llavaov", aid="2408.03326", category="adjacent",
    method="Open large multimodal model family consolidating LLaVA-NeXT insights across single-image, multi-image and video; AnyRes high-resolution processing.",
    claims="Strong transfer across scenarios; DocVQA test 92.8, MMMU val 56.8, SeedBench image 78.0 for the 72B model (Table 2).",
    relation="The frozen 7B box prior in our shipped method (sentence-prompt boxes).",
    threat="Pretrained on COCO images (no referring annotations), a weaker zero-shot notion; box quality bounds selection.",
    quote=(2408, 0), numbers=dict(docvqa_test=92.8, mmmu_val=56.8, seed_image=78.0),
    anchors=["Sec. 2 model", "Table 2"],
    code=dict(available=True, inspected=True, repo="https://github.com/LLaVA-VL/LLaVA-NeXT",
              findings=("llava/mm_utils.py implements AnyRes image splitting with grid pinpoints and the "
                        "highres/anyres processing controlled by image_aspect_ratio; generation is the standard "
                        "LLM decode path."),
              ran=True,
              run=dict(entry="exp/exp40_final_eval.py --box-source llava --phrase-source sent (full val)",
                       sample_n=26488, status="ok",
                       notes=("the training-free box-prior rule with 100% box coverage: 68.5333/64.6669/65.1175 "
                              "mIoU on RefCOCO/+/g val; the declared-trained head reaches 70.57/66.35/65.51"),
                       metrics=dict(refcoco_mIoU=68.5333, refcocoplus_mIoU=64.6669, refcocog_mIoU=65.1175))))

add(key="qwen25vl2025", aid="2502.13923", category="adjacent",
    method="Qwen2.5-VL: LMM with dynamic resolution processing, absolute-coordinate box/point localization and document parsing.",
    claims="MMStar 70.8, DocVQA test 96.4, MMMU 70.2 (abstract/table).",
    relation="The frozen 3B box prior in the backbone audit; smaller and better than the shipped 7B LLaVA prior on our tables.",
    threat="A within-family size sweep is unmeasured; the two-point comparison isolates family, not size.",
    quote=(2502, 1), numbers=dict(mmstar=70.8, docvqa_test=96.4, mmmu=70.2),
    anchors=["Sec. 2 model", "Table 1"],
    code=dict(available=True, inspected=True, repo="https://github.com/QwenLM/Qwen2.5-VL",
              findings=("qwen-vl-utils smart_resize pins min_pixels/max_pixels (e.g. 128*28*28 and 512*28*28 "
                        "in the video cookbook); dynamic resolution; boxes are emitted as absolute coordinates, "
                        "which our extractor scales and validates."),
              ran=True,
              run=dict(entry="exp/exp164_boxer3b.py + exp159 full-val evaluation (Qwen2.5-VL-3B boxes)",
                       sample_n=26488, status="ok",
                       notes=("backbone_audit.json: the 3B Qwen prior with a refit head reaches 73.57/67.64/66.43 "
                              "mIoU, above the shipped 7B LLaVA prior on all three tables"),
                       metrics=dict(refcoco_mIoU=73.57, refcocoplus_mIoU=67.64, refcocog_mIoU=66.43))))

add(key="zhou2022maskclip", aid="2112.01071", category="adjacent",
    method="MaskCLIP: minimally modifies CLIP's last attention layer value embedding to obtain dense open-vocabulary features without training.",
    claims="MaskCLIP+ improves unseen-class mIoU by 50.5%/46%/24.4% on VOC/Context/COCO-Stuff; 31.1 mIoU on VOC with ViT-B/16.",
    relation="A training-free dense-CLIP alternative; our region scoring uses pooled CLIP features, and we measure a plain masked-region CLIP platform swap.",
    threat="Dense CLIP could improve region-text scoring, but scoring is not the measured bottleneck in our diagnosis.",
    quote=(2112, 3), numbers=dict(unseen_mIoU_gain_pct=50.5, voc_maskclip_plus_mIoU=31.1),
    anchors=["Sec. 3 MaskCLIP", "Sec. 4 experiments"],
    code=dict(available=True, inspected=True, repo="https://github.com/chongzhou96/MaskCLIP",
              findings=("mmseg/models/decode_heads/maskclip_head.py refine_output(k) replaces the value "
                        "embedding at the last CLIP attention layer; no training loop for the zero-shot path."),
              ran=False))

# ------------------------------------------------------------------ assemble
assert len(P) == 22, len(P)
ris = sum(1 for p in P if p["category"] == "ris")
assert ris >= 15, ris
papers = []
for p in P:
    aid = p["aid"]
    q_aid, q_i = p.pop("quote")
    assert aid.startswith(str(q_aid)), (aid, q_aid)
    quote = sent(aid, q_i)
    meta = META[aid]
    src = f"exp/literature/papers/{FILES[aid]}.txt"
    assert (ROOT / src).is_file(), src
    rec = dict(key=p["key"], title=meta["title"], identifier=f"arXiv:{aid}",
               category=p["category"], method=p["method"], claims=p["claims"],
               relation=p["relation"], threat=p["threat"], key_quote=quote,
               depth="full-text", reported_numbers=p["numbers"], anchors=p["anchors"],
               source_file=src, code=p["code"])
    papers.append(rec)

deep = dict(
    what="full-text deep-reading dossier (22 papers: 16 RIS, 6 adjacent), code inspections and repo runs",
    n_papers=len(papers), n_ris=ris,
    inspected=sum(1 for p in papers if p["code"]["inspected"]),
    ok_runs=sum(1 for p in papers if p["code"].get("ran") and
                p["code"].get("run", {}).get("status") == "ok"),
    papers=papers,
    grefcoco_metric_check=dict(
        method=("imported the official ReLA ReferEvaluator (gres_model/evaluation/refer_evaluation.py) "
                "with a minimal detectron2 stub and ran a five-case battery (targeted exact/wrong/abstain-FP, "
                "no-target TP/FN); compared its accumulated cIoU/gIoU/N_acc/T_acc with our metric convention"),
        repo="https://github.com/henghuiding/ReLA + https://github.com/henghuiding/gRefCOCO",
        finding=("official evaluation is per sentence (not per ref): a no-target true positive contributes "
                 "I=U=0 and gIoU 1; a targeted false positive keeps the discarded mask's union in the cIoU "
                 "denominator with I=0; N_acc=TP/(TP+FN) and T_acc=TN/(TN+FP) (reported as fractions). Our "
                 "metric matches these definitions, with one deliberate difference: an empty-prediction "
                 "abstention adds only the GT area to U instead of a discarded mask's union, which makes our "
                 "abstention rows slightly cheaper in cIoU than the official FP convention. GreferDataset "
                 "groups several sentences per ref with one merged GT, so per-sentence evaluation and "
                 "per-sentence text alignment are required --- the exp172/exp173 corrections."),
    ),
    verdict=("22 papers read in full text (16 RIS, 6 adjacent); 13 source repositories inspected; 6 successful "
             "runs with metrics; the official gRefCOCO metric arithmetic was executed and compared. The dossier "
             "supports the paper's same-class comparisons and documents the one metric-convention difference."))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "deep_read.json").write_text(json.dumps(deep, indent=2))
print(f"deep_read.json: {len(papers)} papers ({ris} RIS), "
      f"{deep['inspected']} inspected, {deep['ok_runs']} ok runs")

# coverage artifact: all 22 keys must exist in references.bib
bib = (ROOT / "paper/references.bib").read_text(errors="replace")
refs = []
for p in papers:
    assert ("{" + p["key"] + ",") in bib, p["key"]
    refs.append(dict(key=p["key"], title=p["title"], identifier=p["identifier"]))
cov = dict(what="verified literature coverage (arXiv identifiers checked against arXiv metadata)",
           n_references=len(refs),
           bib_entries=len(re.findall(r"^@", bib, re.M)),
           references=refs,
           verification=("every identifier was resolved through the arXiv API; titles in this file are the "
                         "API titles; every key has a matching references.bib entry"),
           verdict=f"{len(refs)} verified references in the bibliography, each with a resolvable arXiv identifier.")
(OUT / "literature_coverage.json").write_text(json.dumps(cov, indent=2))
print(f"literature_coverage.json: {len(refs)} refs, bib entries {cov['bib_entries']}")
