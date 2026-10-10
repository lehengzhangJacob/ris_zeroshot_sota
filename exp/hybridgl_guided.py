"""The released HybridGL scorer, driven from the cached pool.

Reproduces `Hybridgl_main.py` (commit f7eb19bb) score for score, but reads the
SAM ViT-H pool and the G2L features from `glod.vision`'s cache instead of
regenerating them, and keeps every per-proposal quantity instead of only the
final argmax. The released constants are used verbatim: r = 0.5 text ensemble,
k1 = 3 semantic candidates, k2 = 6 negative-noun candidates, alpha = 0.6,
black in {1.5, 1.8, 1.95}.

Two additions that the released code does not log but downstream analysis
needs: the noun-phrase-only score (a category probe with no spatial content)
and a spatial-stripped text score.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
HGL = REPRO / "third_party/HybridGL"
if str(HGL) not in sys.path:
    sys.path.insert(0, str(HGL))

from utils import (extract_dir_phrase, extract_noun_phrase, extract_nouns,  # noqa: E402
                   extract_rela_word, gen_dir_mask, relation_boxes)

R = 0.5
K1, K2 = 3, 6
ALPHA = 0.6

SPATIAL_WORDS = {"left", "right", "leftmost", "rightmost", "top", "bottom", "upper", "lower",
                 "above", "below", "middle", "center", "centre", "mid", "central",
                 "front", "back", "behind", "next", "near", "beside", "between", "under",
                 "over", "side", "corner", "topmost", "l"}
STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "with", "and", "that", "is", "are",
        "this", "these", "those", "there", "it", "its", "his", "her", "their", "by", "for",
        "from", "as", "into", "onto", "up", "down", "out", "off"}


def stripped(sent: str) -> str:
    w = [x for x in re.findall(r"[A-Za-z0-9]+", sent) if x.lower() not in SPATIAL_WORDS]
    return " ".join(w) if w else sent


def content_only(sent: str) -> str:
    w = [x for x in re.findall(r"[A-Za-z0-9]+", sent)
         if x.lower() not in SPATIAL_WORDS and x.lower() not in STOP]
    return " ".join(w) if w else sent


class GuidedScorer:
    """Holds GEM + spacy + the encoders; scores one (image, expression) at a time."""

    def __init__(self, enc, device="cuda:0"):
        import gem
        import spacy
        self.enc = enc
        self.device = torch.device(device)
        self.gem = gem.create_gem_model(model_name="ViT-B/16", pretrained="openai", device=device)
        self.gem_pre = gem.get_gem_img_transform()
        self.nlp = spacy.load("en_core_web_lg")
        self.softmax0 = torch.nn.Softmax(0)

    # ---- text side -------------------------------------------------------
    def _text_feats(self, sent):
        s = sent.lower()
        toks = [t.text for t in self.nlp(s) if t.text != " "]
        s2 = " ".join(toks)
        noun_phrase, _, _ = extract_noun_phrase(s2, self.nlp, need_index=True)
        other_nps, nouns = extract_nouns(s2, self.nlp)
        dirflag = extract_dir_phrase(s2, self.nlp, False)
        relaflag = extract_rela_word(s2, self.nlp)
        return dict(sentence=s2, noun_phrase=noun_phrase, other_nouns=other_nps,
                    nouns=nouns, dirflag=dirflag, relaflag=relaflag)

    def score(self, rgb, masks, boxes, feat, sent):
        """Return a dict of per-proposal score vectors and the released selection.

        `feat` is the HybridGL G2L output (the masked-region feature, i.e. the
        `foreground` array). GLOD-RIS's DOD fusion is a *different* scorer and
        must not be passed here.
        """
        dev = self.device
        parsed = self._text_feats(sent)
        np_text = parsed["noun_phrase"]

        clip = self.enc.clip
        model = self.enc.encoder.model

        with torch.inference_mode():
            def txt(t):
                return model.encode_text(clip.tokenize(t, truncate=True).to(dev)).float()

            sent_f = txt(parsed["sentence"])
            noun_f = txt(np_text)
            ensemble = R * sent_f + (1 - R) * noun_f
            other = torch.zeros(1, sent_f.shape[1], device=dev)
            cnt = 0
            other_vecs = {}
            for o in parsed["other_nouns"]:
                vec = txt("a photo of " + o)
                other_vecs[o] = vec
                other = other + vec
                cnt += 1
            if cnt:
                other = other / cnt

            feat = torch.from_numpy(feat).float().to(dev)
            feat = feat / feat.norm(dim=1, keepdim=True)
            logit = model.logit_scale.exp()

            def calc(t):
                t = t / t.norm(dim=1, keepdim=True)
                return (logit * feat @ t.t()).reshape(-1)

            sc = calc(ensemble)
            sc_neg = calc(other) if cnt else torch.full_like(sc, -1.0)
            sc_noun = calc(noun_f)
            noun_scores = {np_text: sc_noun.cpu().numpy()}
            for o, vec in other_vecs.items():
                noun_scores[o] = calc(vec).cpu().numpy()
            sc_strip = calc(txt(stripped(parsed["sentence"])))
            sc_content = calc(txt(content_only(parsed["sentence"])))

            s_sc = self.softmax0(sc)
            s_neg = self.softmax0(sc_neg)
            n = len(masks)
            k1 = min(K1, n)
            k2 = min(K2, n)
            top_idx = torch.topk(s_sc.reshape(-1), k=k1).indices
            neg_idx = torch.topk(s_neg.reshape(-1), k=k2).indices

            # ---- spatial relationship guidance (released) ----
            dots = np.zeros(k1)
            rf = parsed["relaflag"]
            if len(parsed["nouns"]) == 0:
                for i in range(k1):
                    for j in top_idx:
                        dots[i] += relation_boxes(boxes[int(top_idx[i])], boxes[int(j)],
                                                  float(s_sc[int(top_idx[i])]), float(s_sc[int(j)]), rf)
            else:
                for i in range(k1):
                    for j in neg_idx:
                        dots[i] += relation_boxes(boxes[int(top_idx[i])], boxes[int(j)],
                                                  float(s_sc[int(top_idx[i])]), float(s_neg[int(j)]), rf)
            tops = self.softmax0(torch.tensor(dots, device=dev, dtype=torch.float32))

            # ---- spatial coherence guidance (GEM + position) ----
            h, w = masks.shape[1], masks.shape[2]
            # the released code relies on the DataLoader (batch_size=1) for the
            # leading dimension; GEM's forward requires [1, 3, H, W]
            tensor_img = self.gem_pre(rgb).unsqueeze(0).to(dev)
            imgattn = self.gem(tensor_img, [np_text])[0]
            imgattn = T.Resize((h, w), antialias=True)(imgattn)[0]
            imgattn = (imgattn - imgattn.min()) / (imgattn.max() - imgattn.min())
            pmask = gen_dir_mask(parsed["dirflag"], imgattn.shape[0], imgattn.shape[1], dev)
            imgattn = imgattn * pmask
            imgattn = imgattn / imgattn.mean()
            black = 1.95 if rf == "big" else (1.5 if rf == "small" else 1.8)

            mt = torch.from_numpy(masks).to(dev, torch.float32)
            a = mt.sum((1, 2))
            pos = (imgattn[None] * mt).sum((1, 2)) / a.clamp(min=1)
            b = ((1 - mt) * imgattn[None]).sum((1, 2)) / (mt.shape[1] * mt.shape[2] - a).clamp(min=1)
            score_gem = (2 - black) * pos - black * b

            fused_tops = tops.clone()
            fused_tops = fused_tops * (1 - ALPHA) + ALPHA * score_gem[top_idx]
            final = int(top_idx[int(torch.argmax(fused_tops))])
            pure = int(top_idx[0])

        return dict(
            parsed=parsed,
            sc=sc.cpu().numpy(), sc_neg=sc_neg.cpu().numpy(),
            sc_noun=sc_noun.cpu().numpy(), sc_strip=sc_strip.cpu().numpy(),
            sc_content=sc_content.cpu().numpy(),
            score_gem=score_gem.cpu().numpy(),
            noun_scores=noun_scores,
            top_idx=top_idx.cpu().numpy(), fused_tops=fused_tops.cpu().numpy(),
            pure=pure, final=final,
        )
