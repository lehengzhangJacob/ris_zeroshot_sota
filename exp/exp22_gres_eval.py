#!/usr/bin/env python
"""EXP-22 — gRefCOCO (GRES) evaluation of the released pipeline and the
set-valued extension.

Protocol follows the official ReLA evaluator (gres_model/evaluation/
refer_evaluation.py) sample for sample:

  * sample = (reference, sentence);
  * no-target samples: an empty prediction is a true positive (gIoU += 1,
    cIoU += 0/0 -> excluded from the cIoU sums), a non-empty prediction is a
    false negative (gIoU += 0, cIoU union += |pred|);
  * targeted samples: IoU(pred, union-of-GT) contributes to both gIoU (mean
    over all samples) and cIoU (total I / total U); an empty prediction is a
    false positive (I = 0);
  * N-acc = TP/(TP+FN) over no-target samples, T-acc = TN/(TN+FP) over
    targeted samples.

Prediction rows are pure functions of the stored records plus the cached
proposal masks (`cache_file`), so everything here is a re-analysis of one
scoring pass. Clause head phrases are scored with the frozen text encoder
against the cached proposal features.

Abstention threshold: calibrated on the *RefCOCO* records of the same run
(1st percentile of the top-1 head-noun score, where every expression has a
target) and applied unchanged to gRefCOCO; the sweep is reported as a
secondary analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, boot  # noqa: E402
from method import Selector, parse_relation  # noqa: E402
from method_gres import gres_select, clause_heads  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
OUT = Path("exp/out_final")


class MaskCache:
    def __init__(self):
        self.by_file = {}
        self.meta = {}

    def masks(self, rec):
        cf = rec["cache_file"]
        if cf not in self.by_file:
            with np.load(cf, allow_pickle=False) as d:
                shape = tuple(int(x) for x in d["mask_shape"])
                self.by_file[cf] = np.unpackbits(d["mask_bits"], count=int(np.prod(shape))
                                                 ).reshape(shape).astype(bool)
                self.meta[cf] = d["foreground"].astype(np.float32)
        return self.by_file[cf]

    def feats(self, rec):
        self.masks(rec)
        return self.meta[rec["cache_file"]]


class PhraseScorer:
    """Frozen-text-encoder score of every proposal in a record's pool."""

    def __init__(self, device="cuda:0"):
        import importlib.util
        for rel in ("third_party/modified_CLIP", "third_party/segment-anything"):
            p = str(REPRO / "third_party/HybridGL" / rel)
            if p not in sys.path:
                sys.path.insert(0, p)
        import clip
        import torch
        self.torch = torch
        self.clip = clip
        self.device = device
        self.model, _ = clip.load(str(REPRO / "checkpoints/ViT-B-16.pt"), device=device, jit=False)
        self.model.eval()
        self.cache = {}
        self.feats_norm = {}

    def _text(self, phrase):
        if phrase not in self.cache:
            with self.torch.inference_mode():
                tok = self.clip.tokenize(phrase, truncate=True).to(self.device)
                vec = self.model.encode_text(tok).float()
                vec = vec / vec.norm(dim=1, keepdim=True)
            self.cache[phrase] = vec
        return self.cache[phrase]

    def __call__(self, rec, phrase, masks_cache):
        key = rec["cache_file"]
        if key not in self.feats_norm:
            f = self.torch.from_numpy(masks_cache.feats(rec)).to(self.device)
            self.feats_norm[key] = f / f.norm(dim=1, keepdim=True)
        with self.torch.inference_mode():
            logit = self.model.logit_scale.exp()
            s = (logit * self.feats_norm[key] @ self._text(phrase).t()).reshape(-1)
        return s.float().cpu().numpy()


def union_iou(masks, idxs, gt):
    if not idxs:
        pred = np.zeros(gt.shape, bool)
    else:
        pred = masks[idxs].any(0)
    I = int(np.logical_and(pred, gt).sum())
    U = int(np.logical_or(pred, gt).sum())
    return I, U


def gres_metrics(samples):
    """samples: list of dicts with iou (targeted) / nt flags. Official formulas."""
    total = len(samples)
    accum_iou = 0.0
    accum_I = accum_U = 0.0
    tp = fn = tn = fp = 0
    for s in samples:
        if s["gt_nt"]:
            if s["pred_nt"]:
                tp += 1
                accum_iou += 1.0
            else:
                fn += 1
                accum_U += s["U"]
        else:
            if s["pred_nt"]:
                fp += 1
            else:
                tn += 1
                accum_iou += s["iou"]
                accum_I += s["I"]
                accum_U += s["U"]
    gIoU = 100.0 * accum_iou / total if total else 0.0
    cIoU = 100.0 * accum_I / accum_U if accum_U else 0.0
    N_acc = tp / (tp + fn) if (tp + fn) else 0.0
    T_acc = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(n=total, cIoU=cIoU, gIoU=gIoU, N_acc=100 * N_acc, T_acc=100 * T_acc,
                TP=tp, FN=fn, TN=tn, FP=fp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--refcoco-records", type=Path,
                    default=Path("exp/out_extreme/records_refcoco.jsonl"))
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.records.open()]
    if args.limit:
        recs = recs[:args.limit]
    print(f"{len(recs)} gRefCOCO samples", flush=True)

    gd = GreferDataset(REPRO / "data/refer", "val")
    ref_by_id = {r["ref_id"]: r for r in gd.refs}
    gts = {}
    for r in recs:
        rid = r["ref_id"]
        if rid not in gts:
            gts[rid] = gd.ground_truth(ref_by_id[rid])

    mc = MaskCache()
    scorer = PhraseScorer(device=args.device)
    own = Selector()

    # ---- abstention calibration on RefCOCO val (targets always exist there) --
    thr = None
    pct_name = "p1"
    if args.refcoco_records.exists():
        head_scores = []
        for line in args.refcoco_records.open():
            rr = json.loads(line)
            head_scores.append(float(np.max(rr["sc_noun"])))
        thr = float(np.percentile(head_scores, 1.0))
        thr_05 = float(np.percentile(head_scores, 0.5))
        thr_2 = float(np.percentile(head_scores, 2.0))
        thr_5 = float(np.percentile(head_scores, 5.0))
        print(f"abstain thr percentiles on RefCOCO top sc_noun (n={len(head_scores)}): "
              f"p0.5={thr_05:.3f} p1={thr:.3f} p2={thr_2:.3f} p5={thr_5:.3f}", flush=True)

    # ---- rows ---------------------------------------------------------------
    def predict_released(r):
        return [r["final"]]

    def predict_argmax(r):
        return [r["pure"]]

    def predict_snap(r):
        return [int(r["ps_idx"][r["final"]])]

    def predict_ours(r):
        return [int(own(Rec(r)))]

    def predict_ours_snap(r):
        return [int(r["ps_idx"][int(own(Rec(r)))])]

    _phrase_cache = {}

    def clause_phrases(r):
        key = r["sent"]
        if key not in _phrase_cache:
            _phrase_cache[key] = clause_heads(r["sent"], _nlp)
        return _phrase_cache[key]

    _score_cache = {}

    def phrase_score(r, phrase):
        key = (r["cache_file"], phrase)
        if key not in _score_cache:
            _score_cache[key] = scorer(r, phrase, mc)
        return _score_cache[key]

    def predict_clause(r, margin=0.0, rel=True, snap=True, abstain=None):
        rec2 = dict(r)
        rec2["pc"] = _pc(r)
        rec2["sc_content"] = np.asarray(r["sc_content"])
        idxs = gres_select(rec2, lambda p: phrase_score(r, p), _nlp,
                           clause_phrases=clause_phrases(r), margin=margin,
                           abstain_thr=abstain, snap=snap, use_relations=rel)
        return idxs

    import spacy
    _nlp = spacy.load("en_core_web_lg")
    _pc_cache = {}

    def _pc(r):
        cf = r["cache_file"]
        if cf not in _pc_cache:
            masks = mc.masks(r)
            n = len(masks)
            flat = masks.reshape(n, -1).astype(np.float32)
            inter = flat @ flat.T
            ar = flat.sum(1)
            union = ar[:, None] + ar[None, :] - inter
            _pc_cache[cf] = (inter / np.maximum(union, 1)).astype(np.float32)
            del flat, inter, union
        return _pc_cache[cf]

    ROWS = {
        "released": lambda r: predict_released(r),
        "argmax_g2l": lambda r: predict_argmax(r),
        "snap": lambda r: predict_snap(r),
        "ours_single": lambda r: predict_ours(r),
        "ours_single+snap": lambda r: predict_ours_snap(r),
        "clause_only": lambda r: predict_clause(r, margin=0.0),
        "clause+plural m0.3": lambda r: predict_clause(r, margin=0.3),
        "clause+plural m0.6": lambda r: predict_clause(r, margin=0.6),
        "clause+plural m1.0": lambda r: predict_clause(r, margin=1.0),
        "clause+plural m1.5": lambda r: predict_clause(r, margin=1.5),
        "full m1.0 + abstain": lambda r: predict_clause(r, margin=1.0, abstain=thr),
        "clause+abstain": lambda r: predict_clause(r, margin=0.0, abstain=thr),
        "clause+abstain p0.5": lambda r: predict_clause(r, margin=0.0, abstain=thr_05),
        "clause+abstain p2": lambda r: predict_clause(r, margin=0.0, abstain=thr_2),
        "clause+abstain p5": lambda r: predict_clause(r, margin=0.0, abstain=thr_5),
        "clause+plural m0.3 + abstain": lambda r: predict_clause(r, margin=0.3, abstain=thr),
        "ctrl: no relation": lambda r: predict_clause(r, margin=1.0, rel=False),
        "ctrl: no snap": lambda r: predict_clause(r, margin=1.0, snap=False),
    }

    results = {name: [] for name in ROWS}
    for i, r in enumerate(recs):
        rid = r["ref_id"]
        gt = gts[rid]
        masks = mc.masks(r)
        for name, fn in ROWS.items():
            idxs = fn(r)
            I, U = union_iou(masks, idxs, gt)
            gt_nt = bool(r["no_target"])
            results[name].append(dict(gt_nt=gt_nt, pred_nt=(len(idxs) == 0), I=I, U=U,
                                      iou=(I / U if U else 0.0)))
        if i and i % 1000 == 0:
            print(f"  [{i}]", flush=True)

    ref_ids = sorted({r["ref_id"] for r in recs})
    ref_nt = {r["ref_id"]: bool(r["no_target"]) for r in recs}
    out = {"n": len(recs), "n_refs": len(ref_ids),
           "n_refs_no_target": int(sum(ref_nt.values())),
           "n_refs_multi": int(len(ref_ids) - sum(ref_nt.values())),
           "n_no_target": int(sum(r["no_target"] for r in recs)),
           "source_refs": 4237,
           "abstain_threshold": thr, "calibration": pct_name, "rows": {}}
    print(f"\n=== gRefCOCO val (n={len(recs)}) ===")
    print(f"{'row':<28}{'cIoU':>8}{'gIoU':>8}{'N-acc':>8}{'T-acc':>8}")
    for name in ROWS:
        m = gres_metrics(results[name])
        out["rows"][name] = m
        print(f"{name:<28}{m['cIoU']:>8.2f}{m['gIoU']:>8.2f}{m['N_acc']:>8.1f}{m['T_acc']:>8.1f}")

    # targeted-only IoU (the RefCOCO-family analogue) and oracle
    tgt = [i for i, r in enumerate(recs) if not r["no_target"]]
    out["targeted_only"] = {}
    for name in ROWS:
        vals = [results[name][i]["iou"] for i in tgt]
        out["targeted_only"][name] = dict(n=len(vals), mean_iou=float(np.mean(vals)))
    print("\n=== targeted-only (multi-target) mean IoU ===")
    for name in ROWS:
        print(f"  {name:<28}{out['targeted_only'][name]['mean_iou']:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "main_grefcoco.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.output / 'main_grefcoco.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
