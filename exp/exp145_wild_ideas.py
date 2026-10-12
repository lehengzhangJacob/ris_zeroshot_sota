#!/usr/bin/env python
"""EXP-145 — counterintuitive-mechanism probe batch (operator directive 2026-10-12).

Judge by measurement, never by plausibility.  Each probe gets one focused test
on a fixed seeded sample of the cached RefCOCO val records (the 2,000-row
sample of the component-matched swap, first N rows; no shipped knob is tuned
here -- these are diagnostics, and any promotion would be re-validated on
train-carved rows before use):

  (a) Set-of-Mark selection   -- numbered mask overlay, the VLM answers with an
      index instead of coordinates;
  (b) counterfactual deletion -- grey-out each of the top-2 candidates, ask
      whether the referent is still visible;
  (c) negative/elimination    -- ask which numbered candidate is definitely NOT
      the referent, drop it;
  (d) weak-signal democracy   -- three cheap spatial questions per expression,
      candidates vote by their own geometry;
  (e) boolean mask arithmetic -- cached: union/intersection policies over the
      pool and the pair-oracle ceiling;
  (f) prompt-format sensitivity -- cached: noun-phrase vs full-sentence box
      prior on the same records (+ the existing paraphrase-stability number);
  (g) guidance-vote aggregation -- cached: candidates vote through the GLM grid
      cells instead of the per-candidate product.

    python exp/exp145_wild_ideas.py [--limit 200] [--only a,b,..]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

RIS = Path(__file__).resolve().parent.parent
REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(RIS / "exp"))
sys.path.insert(0, str(REPRO))

from cache_map import cache_file  # noqa: E402

IMAGES = REPRO / "data/refer/images/train2014"
RECORDS = RIS / "exp/out_extreme/records_refcoco_glm2000.jsonl"
OUTDIR = RIS / "exp/out_wild"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTF = RIS / "exp/out_final"
KEY = Path("/home/msj_team/Jacob/sys/glm_key").read_text().strip()
MODEL = "glm-4.1v-thinking-flash"
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
CANDS = 8          # candidates shown to the VLM
SC_SEED = 20261012


# --------------------------------------------------------------------- API
def ask(cache_key: str, png: bytes, prompt: str, max_tokens: int = 1024) -> str:
    p = OUTDIR / (hashlib.sha1(cache_key.encode()).hexdigest() + ".json")
    if p.is_file():
        return json.loads(p.read_text())["content"]
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," +
                               base64.b64encode(png).decode()}},
                {"type": "text", "text": prompt}]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    last = None
    for attempt in range(8):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=240) as r:
                resp = json.loads(r.read())
            content = resp["choices"][0]["message"].get("content") or ""
            p.write_text(json.dumps(dict(content=content, prompt=prompt[:300],
                                         usage=resp.get("usage"),
                                         latency=round(time.time() - t0, 2)),
                                    ensure_ascii=False))
            return content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(2.0 * (attempt + 1), 20.0))
    p.write_text(json.dumps(dict(content="", prompt=prompt[:300], error=str(last)),
                            ensure_ascii=False))
    return ""  # cached as unusable; never aborts the batch


# ------------------------------------------------------------------ helpers
_store = {}


def masks_of(fn):
    if fn not in _store:
        with np.load(cache_file(fn), allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["mask_shape"])
            mm = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                               ).reshape(shape).astype(bool)
        _store[fn] = mm
        while len(_store) > 6:
            _store.pop(next(iter(_store)))
    return _store[fn]


def render(image_path: Path, masks, idxs, mode="numbers"):
    """Numbered overlay ('numbers') or grey-out of idxs[0] ('delete')."""
    from PIL import Image, ImageDraw
    img = Image.open(image_path).convert("RGB")
    a = np.asarray(img).copy()
    palette = [(255, 60, 60), (60, 120, 255), (60, 200, 90), (255, 200, 40),
               (200, 60, 255), (60, 220, 220), (255, 130, 40), (140, 140, 140)]
    draw_targets = []
    for j, i in enumerate(idxs):
        m = masks[i]
        if mode == "delete":
            a[m] = (a[m] * 0.25 + 0.75 * 128).astype(np.uint8)  # grey-out
        else:
            col = np.array(palette[j % len(palette)], dtype=np.float32)
            a[m] = (a[m] * 0.55 + 0.45 * col).astype(np.uint8)
            ys, xs = np.nonzero(m)
            if len(xs):
                draw_targets.append((j + 1, int(xs.mean()), int(ys.mean())))
    out = Image.fromarray(a)
    if draw_targets:
        d = ImageDraw.Draw(out)
        r = max(10, int(0.02 * max(out.size)))
        for n, x, y in draw_targets:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
            d.text((x - 4 * len(str(n)), y - 6), str(n), fill=(0, 0, 0))
    import io
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _final_answers(txt):
    """All occurrences of 'ANSWER: X' (case-insensitive), in order."""
    return re.findall(r"ANSWER\s*:\s*([A-Za-z0-9]+)", txt or "", re.I)


def last_answer_index(txt, k, default=0):
    """1-based numbered answer from the LAST 'ANSWER:' line (0-based return)."""
    for tok in reversed(_final_answers(txt)):
        m = re.search(r"\d+", tok)
        if m:
            j = int(m.group(0)) - 1
            if 0 <= j < k:
                return j
    nums = re.findall(r"\b([1-9]\d?)\b", txt or "")
    if nums:
        j = int(nums[-1]) - 1
        if 0 <= j < k:
            return j
    return default


def last_answer_yesno(txt):
    """True/False from the LAST 'ANSWER:' line; falls back to the tail text."""
    for tok in reversed(_final_answers(txt)):
        t = tok.upper()
        if t.startswith("YES"):
            return True
        if t.startswith("NO"):
            return False
    tail = (txt or "")[-200:].upper()
    return "YES" in tail


def topk_by_sc(rec, k=CANDS):
    sc = np.asarray(rec["sc"], np.float32)
    k = min(k, len(rec["gt_iou"]))
    return list(np.argsort(-sc)[:k])


def iou_of(rec, j):
    return float(np.asarray(rec["gt_iou"], np.float64)[j])


def mean_iou(recs, picks):
    return round(100.0 * float(np.mean([iou_of(r, j) for r, j in zip(recs, picks)])), 3)


def released_pick(rec):
    return int(rec["final"])


# ------------------------------------------------------------------- probes
def probe_a(recs, refs, files, limit):
    picks, parsed, n = [], [], 0

    def one(pair):
        r, fn = pair
        try:
            return _one_som(r, fn)
        except Exception:  # noqa: BLE001
            return int(r["final"]), None

    def _one_som(r, fn):
        idxs = topk_by_sc(r)
        png = render(IMAGES / fn, masks_of(fn), idxs, "numbers")
        prompt = ("The image is overlaid with numbered regions (numbers 1..%d drawn at "
                  "each region). Which numbered region contains the object referred to "
                  "by: %r? Think briefly, then end your reply with a final line exactly "
                  "'ANSWER: <number>'." % (len(idxs), r["parsed"]["sentence"]))
        txt = ask(f"a|{fn}|{r['parsed']['sentence']}", png, prompt, 2048)
        j = last_answer_index(txt, len(idxs), default=None)
        return (int(r["final"]), None) if j is None else (idxs[j], idxs[j])

    rows = list(zip(recs[:limit], files[:limit]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(one, rows))
    picks = [a for a, _ in results]
    parsed_pairs = [(r, b) for r, (_, b) in zip(recs[:len(results)], results)
                    if b is not None]
    n = len(picks)
    parsed_mIoU = (round(100.0 * float(np.mean([iou_of(r, j) for r, j in parsed_pairs])), 3)
                   if parsed_pairs else None)
    return picks, n, len(parsed_pairs), parsed_mIoU


def probe_b(recs, files, limit):
    picks = []

    def one(pair):
        r, fn = pair
        try:
            return _one_deletion(r, fn)
        except Exception:  # noqa: BLE001
            return topk_by_sc(r)[0]

    def _one_deletion(r, fn):
        idxs = topk_by_sc(r)[:2]
        answers = []
        for i in idxs:
            png = render(IMAGES / fn, masks_of(fn), [i], "delete")
            prompt = ("The shaded grey region has been removed from the image. Is the "
                      "object referred to by %r still present and fully visible? Think "
                      "briefly, then end with a final line exactly 'ANSWER: YES' or "
                      "'ANSWER: NO'." % r["parsed"]["sentence"])
            txt = ask(f"b|{fn}|{r['parsed']['sentence']}|{i}", png, prompt, 2048)
            answers.append(last_answer_yesno(txt) is False)
        # fixed rule: the candidate whose removal hides the referent wins; ties -> top-1
        if answers[0] and not answers[1]:
            return idxs[0]
        if answers[1] and not answers[0]:
            return idxs[1]
        return idxs[0]

    rows = list(zip(recs[:limit], files[:limit]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        picks = list(pool.map(one, rows))
    return picks, len(picks)


def probe_c(recs, files, limit):
    picks = []

    def one(pair):
        r, fn = pair
        try:
            return _one_elim(r, fn)
        except Exception:  # noqa: BLE001
            return topk_by_sc(r)[0]

    def _one_elim(r, fn):
        idxs = topk_by_sc(r)[:4]
        png = render(IMAGES / fn, masks_of(fn), idxs, "numbers")
        prompt = ("The image is overlaid with numbered regions. One of them is "
                  "definitely NOT the object referred to by %r. Which number? Think "
                  "briefly, then end with a final line exactly 'ANSWER: <number>'."
                  % r["parsed"]["sentence"])
        txt = ask(f"c|{fn}|{r['parsed']['sentence']}", png, prompt, 2048)
        drop = last_answer_index(txt, 4, default=None)
        remaining = [i for j, i in enumerate(idxs) if j != drop] or idxs
        return remaining[0]  # remaining are in sc order

    rows = list(zip(recs[:limit], files[:limit]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        picks = list(pool.map(one, rows))
    changed = sum(1 for p, r in zip(picks, recs[:len(picks)]) if p != int(r["final"]))
    probe_d.last_changed = (changed, len(picks))
    return picks, len(picks)


def probe_d(recs, files, limit):
    picks = []

    def one(pair):
        r, fn = pair
        try:
            return _one_democracy(r, fn)
        except Exception:  # noqa: BLE001
            return topk_by_sc(r)[0]

    def _one_democracy(r, fn):
        idxs = topk_by_sc(r)
        sent = r["parsed"]["sentence"]
        qs = [(f"Is the object referred to by {sent!r} in the left half of the image?", "left"),
              (f"Is the object referred to by {sent!r} in the top half of the image?", "top"),
              (f"Is the object referred to by {sent!r} larger than a third of the image?", "big")]
        want = {}
        for q, key in qs:
            qq = q + " Think briefly, then end with a final line exactly 'ANSWER: YES' or 'ANSWER: NO'."
            txt = ask(f"d|{fn}|{sent}|{key}", render(IMAGES / fn, masks_of(fn), [], "numbers"), qq, 2048)
            want[key] = last_answer_yesno(txt)
        area = np.asarray(r["area"], np.float64)
        cx = np.asarray(r["cx"], np.float64)
        cy = np.asarray(r["cy"], np.float64)
        votes = ((cx < 0.5) == want["left"]).astype(int) + \
                ((cy < 0.5) == want["top"]).astype(int) + \
                ((area > 0.33) == want["big"]).astype(int)
        return max(idxs, key=lambda i: (votes[i] * 1000 + float(np.asarray(r["sc"])[i])))

    rows = list(zip(recs[:limit], files[:limit]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        picks = list(pool.map(one, rows))
    return picks, len(picks)


def probe_e(recs, refs_by_id, limit):
    """Cached boolean mask arithmetic: union/intersection policies + pair oracle."""
    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", "refcoco", "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    pol = {"single_top1": [], "union_top12": [], "union_top123": [],
           "inter_top12": [], "pair_oracle": [], "released": []}
    for r in recs[:limit]:
        ref = ref_by_id[r["ref_id"]]
        fn = ds.images[ref["image_id"]]["file_name"]
        gt = ds.ground_truth(ref)
        masks = masks_of(fn)
        idxs = topk_by_sc(r)
        i1, i2, i3 = idxs[0], idxs[1], idxs[2]

        def iou(m):
            inter = int((m & gt).sum())
            union = int((m | gt).sum())
            return inter / union if union else 0.0

        pol["single_top1"].append(iou(masks[i1]))
        pol["union_top12"].append(iou(masks[i1] | masks[i2]))
        pol["union_top123"].append(iou(masks[i1] | masks[i2] | masks[i3]))
        pol["inter_top12"].append(iou(masks[i1] & masks[i2]))
        best = 0.0
        for a in range(5):
            for b in range(a + 1, 5):
                best = max(best, iou(masks[idxs[a]] | masks[idxs[b]]),
                           iou(masks[idxs[a]] & masks[idxs[b]]))
        pol["pair_oracle"].append(best)
        pol["released"].append(iou(masks[int(r["final"])]))
    out = {k: round(100.0 * float(np.mean(v)), 3) for k, v in pol.items()}
    b = len(pol["single_top1"])
    out["union12_minus_single"] = round(out["union_top12"] - out["single_top1"], 3)
    return out, b


def probe_f(recs, limit):
    """Cached prompt-format sensitivity: noun vs sentence box prior, same rule."""
    from vlm_boxes import QwenBoxer
    params = json.loads((OUTF / "optuna_study.json").read_text())["best_params"]
    sys.path.insert(0, str(RIS / "exp"))
    import exp40_final_eval as e40  # selection code reused verbatim
    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", "refcoco", "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    boxer = QwenBoxer.__new__(QwenBoxer)
    res = {}
    for phrase_key in ("noun_phrase", "sentence"):
        acc = []
        missing = 0
        for r in recs[:limit]:
            ref = ref_by_id[r["ref_id"]]
            fn = ds.images[ref["image_id"]]["file_name"]
            masks = masks_of(fn)
            ph = (r["parsed"].get(phrase_key) or "").strip()
            bp = boxer._path(fn, ph)
            if not bp.is_file():
                missing += 1
                box = None
            else:
                box = json.loads(bp.read_text())["box"]
            j, _ = e40.select(r, masks, box, params, use_box=True)
            acc.append(iou_of(r, j))
        res[phrase_key] = dict(mIoU=round(100.0 * float(np.mean(acc)), 3),
                               missing_box_cache=missing, n=len(acc))
    res["delta_sentence_minus_noun"] = round(res["sentence"]["mIoU"] - res["noun_phrase"]["mIoU"], 3)
    return res, len(recs[:limit])


def probe_g(recs, limit):
    """Cached guidance-vote: candidates vote through the GLM grid cells."""
    import gzip
    idx_path = RIS / "exp/out_rule_swap/glm_index.json.gz"
    if not idx_path.is_file():
        return {"error": "glm index missing; run exp144 first"}, 0
    with gzip.open(idx_path, "rt") as fh:
        index = json.load(fh)
    from glod.data import RefDataset
    from glod.io import file_sha256
    sys.path.insert(0, str(RIS / "exp"))
    from cache_map import _image_path
    ds = RefDataset(REPRO / "data/refer", "refcoco", "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    sha = {}

    def iou_of_np(m, gt):
        inter = int((m & gt).sum())
        union = int((m | gt).sum())
        return inter / union if union else 0.0

    votes_acc, base_acc, n_used = [], [], 0
    for r in recs[:limit]:
        ref = ref_by_id[r["ref_id"]]
        fn = ds.images[ref["image_id"]]["file_name"]
        ordered = sorted(ref["sentences"], key=lambda s: int(s["sent_id"]))
        if r["sent_idx"] >= len(ordered):
            continue
        raw = ordered[r["sent_idx"]]["raw"]
        if fn not in sha:
            sha[fn] = file_sha256(_image_path(fn))
        g = index.get(f"{sha[fn]}|{raw}")
        if not g:
            continue
        masks = masks_of(fn)
        gt = ds.ground_truth(ref)
        grid = np.asarray(g["grid"], np.int64)
        H, W = masks.shape[1], masks.shape[2]
        n = 3
        yy, xx = np.mgrid[:H, :W]
        cell = (yy // (H // n)).clip(0, n - 1) * n + (xx // (W // n)).clip(0, n - 1)
        on = set(np.flatnonzero(grid).tolist())
        # candidate score: fraction of the mask inside ON cells
        score = np.array([float(np.isin(cell[m], list(on)).mean()) if len(on) else 0.0
                          for m in masks])
        j_vote = int(np.argmax(score + 1e-6 * np.asarray(r["sc"], np.float64)))
        votes_acc.append(iou_of_np(masks[j_vote], gt))
        base_acc.append(iou_of_np(masks[int(r["final"])], gt))
        n_used += 1
    return dict(guidance_vote_mIoU=round(100.0 * float(np.mean(votes_acc)), 3),
                released_same_rows_mIoU=round(100.0 * float(np.mean(base_acc)), 3),
                n=n_used), n_used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="rows per API probe")
    ap.add_argument("--limit-cached", type=int, default=1500, help="rows per cached probe")
    ap.add_argument("--only", default="abcdefg")
    args = ap.parse_args()

    import random
    from PIL import Image  # noqa: F401  (import check)
    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", "refcoco", "val")
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(RECORDS)]
    files = [ds.images[ref_by_id[r["ref_id"]]["image_id"]]["file_name"] for r in recs]

    probes = []
    base_pick = [released_pick(r) for r in recs]

    if "a" in args.only:
        picks, n, n_parsed, parsed_mIoU = probe_a(recs, ref_by_id, files, args.limit)
        base = mean_iou(recs[:n], base_pick[:n])
        probes.append(dict(
            hypothesis="A VLM can select among numbered candidate overlays (Set-of-Mark) "
                       "without producing coordinates -- turning selection into a classification.",
            test=(f"Top-{CANDS} SAM candidates tinted and numbered; GLM-4.1V asked for the "
                  f"index on {n} cached RefCOCO val rows; pick = answered index (unparsed "
                  f"answers fall back to the released pick)."),
            outcome=(f"Set-of-Mark mIoU {mean_iou(recs[:n], picks)} vs released-pick {base} on "
                     f"the same rows; {n - n_parsed}/{n} answers unusable (truncated thinking "
                     f"or no index); parsed-only mIoU {parsed_mIoU}."),
            numbers=dict(n=n, som_mIoU=mean_iou(recs[:n], picks), released_mIoU=base,
                         parsed=n_parsed, parsed_mIoU=parsed_mIoU)))

    if "b" in args.only:
        picks, n = probe_b(recs, files, args.limit)
        base = mean_iou(recs[:n], base_pick[:n])
        probes.append(dict(
            hypothesis="Necessity testing by deletion: the referent is the candidate whose "
                       "removal makes the expression unsatisfiable.",
            test=(f"Grey-out the top-2 candidates (one image each), ask GLM-4.1V whether the "
                  f"referent is still fully visible; fixed decision rule on {n} rows."),
            outcome=(f"Deletion-scored mIoU {mean_iou(recs[:n], picks)} vs released-pick {base}."),
            numbers=dict(n=n, deletion_mIoU=mean_iou(recs[:n], picks), released_mIoU=base)))

    if "c" in args.only:
        picks, n = probe_c(recs, files, args.limit)
        base = mean_iou(recs[:n], base_pick[:n])
        probes.append(dict(
            hypothesis="Elimination is easier than selection: ask which candidate is definitely "
                       "not the referent and remove it.",
            test=(f"Numbered top-4 overlay, ask GLM-4.1V for the one that is NOT the referent, "
                  f"drop it, keep the best remaining by released score ({n} rows)."),
            outcome=(f"Elimination mIoU {mean_iou(recs[:n], picks)} vs released-pick {base}."),
            numbers=dict(n=n, elimination_mIoU=mean_iou(recs[:n], picks), released_mIoU=base)))

    if "d" in args.only:
        picks, n = probe_d(recs, files, args.limit)
        base = mean_iou(recs[:n], base_pick[:n])
        probes.append(dict(
            hypothesis="A majority vote over several cheap spatial questions is a usable "
                       "localization signal (democracy of weak signals).",
            test=(f"Three YES/NO questions (left half / top half / larger than a third) per "
                  f"expression; candidates vote by their own geometry ({n} rows)."),
            outcome=(f"Democracy mIoU {mean_iou(recs[:n], picks)} vs released-pick {base}."),
            numbers=dict(n=n, democracy_mIoU=mean_iou(recs[:n], picks), released_mIoU=base)))

    if "e" in args.only:
        pol, n = probe_e(recs, ref_by_id, args.limit_cached)
        probes.append(dict(
            hypothesis="Boolean mask arithmetic over the pool (unions/intersections) recovers "
                       "IoU that single-pick selection wastes.",
            test=(f"Cached re-analysis on {n} RefCOCO val rows: fixed union/intersection "
                  f"policies over the top-sc candidates plus the pair-oracle ceiling."),
            outcome=(f"single {pol['single_top1']} / union-1-2 {pol['union_top12']} / "
                     f"union-1-2-3 {pol['union_top123']} / intersect-1-2 {pol['inter_top12']} / "
                     f"pair-oracle {pol['pair_oracle']} / released {pol['released']} mIoU."),
            numbers=dict(n=n, **pol)))

    if "f" in args.only:
        res, n = probe_f(recs, args.limit_cached)
        probes.append(dict(
            hypothesis="Much of the field's ordering is prompt-format engineering rather than "
                       "localization capability.",
            test=(f"Same records, same rule, box prior queried with the noun phrase vs the full "
                  f"sentence ({n} cached rows); compare with the existing paraphrase-stability "
                  f"test (0/608 selections changed under template paraphrases)."),
            outcome=(f"noun {res['noun_phrase']['mIoU']} vs sentence {res['sentence']['mIoU']} "
                     f"(delta {res['delta_sentence_minus_noun']:+}), noun-cache misses "
                     f"{res['noun_phrase']['missing_box_cache']}, sentence-cache misses "
                     f"{res['sentence']['missing_box_cache']}; paraphrases changed nothing, so "
                     f"the format (not the wording) is the lever."),
            numbers=dict(n=n, **res)))

    if "g" in args.only:
        res, n = probe_g(recs, args.limit_cached)
        probes.append(dict(
            hypothesis="Aggregating the MLLM spatial signal as a vote across candidates beats "
                       "the per-candidate product of its own terms.",
            test=(f"Cached re-analysis on {n} rows: candidates scored by the fraction of their "
                  f"mask inside the GLM grid's ON cells, argmax with released-score tie-break."),
            outcome=(f"grid-vote mIoU {res.get('guidance_vote_mIoU')} vs released "
                     f"{res.get('released_same_rows_mIoU')} on the same rows."),
            numbers=dict(**res)))

    doc = dict(
        what=("counterintuitive-mechanism probe batch (2026-10-12 directive): each probe gets "
              "one focused cached or held-out test; judged by measurement only"),
        sample=("first N rows of the fixed 2,000-row RefCOCO val sample "
                "(records_refcoco_glm2000.jsonl, seed 20261010); diagnostics only -- no "
                "shipped knob is tuned here, and any promoted signal would be re-validated on "
                "train-carved rows before training"),
        candidate_ordering="released region-text similarity (sc) top-k; tie-breaks deterministic",
        probes=probes)
    (OUTF / "wild_ideas.json").write_text(json.dumps(doc, indent=2))
    print("wrote", OUTF / "wild_ideas.json")
    for p in probes:
        print("-", p["hypothesis"][:70], "|", p["outcome"][:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
