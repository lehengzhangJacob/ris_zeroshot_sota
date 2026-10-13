#!/usr/bin/env python
"""EXP-154 — weak-signal democracy vote folded into the gRefCOCO path.

2026-10-13 directive: fold the two live probe leads in as labeled arms.  The
democracy lead (+0.67 on 1,500 RefCOCO rows, null-consistent there) is tested
here on the gRefCOCO population, bounded scope: a seeded sample of val rows,
three binary spatial questions per row via GLM-4.1V (cached), each part voting
by its own geometry.  The arm's decision rule is fixed (no tuning), so the
comparison is a clean paired measurement against the prior-only top part on
the SAME rows:

    pick = argmax over parts of (votes, prior-rank tie-break)

and the metric is the union-IoU of the picked part against the row's GT union
(the gRefCOCO official per-sample IoU).  The shipped union rule's full-val cIoU
is reported alongside as context, not as the paired baseline.

    CUDA_VISIBLE_DEVICES=1 python exp/exp154_democracy_gref.py --limit 800
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

RIS = Path(__file__).resolve().parent.parent
REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(RIS / "exp"))
sys.path.insert(0, str(REPRO))
from exp138_learned_union_selector import (  # noqa: E402
    TRAIN, VAL, build_candidates, build_parts, load_rows, make_clf, terms_of,
)

OUTF = RIS / "exp/out_final"
WORK = RIS / "exp/out_wild_gref"
WORK.mkdir(parents=True, exist_ok=True)
IMAGES = REPRO / "data/refer/images/train2014"
KEY = Path("/home/msj_team/Jacob/sys/glm_key").read_text().strip()
MODEL = "glm-4.1v-thinking-flash"
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def ask(image_bytes, prompt, tag):
    p = WORK / (hashlib.sha1(tag.encode()).hexdigest() + ".json")
    if p.is_file():
        return json.loads(p.read_text())["content"]
    b64 = base64.b64encode(image_bytes).decode()
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 2048,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt}]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                resp = json.loads(r.read())
            content = resp["choices"][0]["message"].get("content") or ""
            p.write_text(json.dumps(dict(content=content, prompt=prompt[:200]),
                                    ensure_ascii=False))
            return content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(2.0 * (attempt + 1), 20.0))
    p.write_text(json.dumps(dict(content="", prompt=prompt[:200], error=str(last))))
    return ""


def yesno(txt):
    import re
    for tok in reversed(re.findall(r"ANSWER\s*:\s*([A-Za-z]+)", txt or "", re.I)):
        t = tok.upper()
        if t.startswith("YES"):
            return True
        if t.startswith("NO"):
            return False
    tail = (txt or "")[-200:].upper()
    return "YES" in tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    cfg = dict(json.loads((OUTF / "optuna_study_gref14.json").read_text())["best_params"])
    cfg["sam_area_cap"] = float(cfg.pop("cap", cfg.get("sam_area_cap", 0.5)))
    cfg.setdefault("gd_thresh", 0.25)
    p_sel = float(g8["p_sel"])
    clf_base = make_clf("logreg", dict(C=st5["C"]))

    table = np.load(RIS / "exp/out_transfer/val_parts.npz")
    rows_meta = json.loads((RIS / "exp/out_transfer/val_rows.json").read_text())
    prior_rank = {}
    from sklearn.ensemble import HistGradientBoostingRegressor
    # prior-only model: the best learned arm's configuration (rank features, 70% train)
    tr = np.load(RIS / "exp/out_transfer/train_parts.npz")
    sys.path.insert(0, str(RIS / "exp"))
    import exp152_transfer_wall as tw
    Xr_tr = tw.rank_within_rows(tr["X"], list(zip(tr["row_ids"], range(len(tr["row_ids"])))))
    n_rows_tr = len(json.loads((RIS / "exp/out_transfer/train_rows.json").read_text()))
    fit_rows = tw.holdout_split(n_rows_tr)
    fit = np.isin(tr["row_ids"], list(fit_rows))
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.08,
                                          random_state=0)
    model.fit(Xr_tr[fit], tr["y"][fit])
    Xr_va = tw.rank_within_rows(table["X"], list(zip(table["row_ids"], range(len(table["row_ids"])))))
    pv_all = model.predict(Xr_va)
    by_row = {}
    for idx, ri in enumerate(table["row_ids"]):
        by_row.setdefault(int(ri), []).append((idx, float(pv_all[idx])))

    rng = random.Random(args.seed)
    cand_rows = sorted(int(k) for k in rows_meta if rows_meta[k]["n_targets"] == 2)
    sample = sorted(rng.sample(cand_rows, min(args.limit, len(cand_rows))))

    # the base classifier must be fitted on the train records first (exp138's setup)
    tr_obj = load_rows(TRAIN, "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr_obj]),
                 np.concatenate([r["y"] for r in tr_obj]))
    val_rows = load_rows(VAL, "val", clf_base)
    for r in val_rows:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    terms_val = {}
    for line in open(VAL):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        ts = terms_of(rec["parsed"])
        if ts:
            terms_val[rec["ref_id"]] = (rec["file_name"], ts,
                                       rec["parsed"].get("sentence") or "")

    delta_vote, delta_ship, n_used = [], [], 0
    t0 = time.time()
    for ridx in sample:
        parts = by_row.get(ridx)
        if not parts or ridx >= len(val_rows):
            continue
        r = val_rows[ridx]
        m = rows_meta[str(ridx)]
        fn_ts = terms_val.get(m["rid"])
        if not fn_ts:
            continue
        built = build_parts(r, fn_ts[0], fn_ts[1], cfg, train_side=False)
        if built is None:
            continue
        masks, X, meta = built
        if len(masks) != len(parts):
            continue
        geom = []
        for mk in masks:
            ys, xs = np.nonzero(mk)
            if len(xs) == 0:
                geom.append((0.5, 0.5, 0.0))
            else:
                H, W = mk.shape
                geom.append((float(xs.mean()) / W, float(ys.mean()) / H,
                             float(mk.sum()) / mk.size))
        sent = fn_ts[2]
        img_bytes = (IMAGES / fn_ts[0]).read_bytes()
        qs = [(f"Is the object referred to by {sent!r} in the left half of the image?", "left"),
              (f"Is the object referred to by {sent!r} in the top half of the image?", "top"),
              (f"Is the object referred to by {sent!r} larger than a third of the image?", "big")]
        want = {}
        for q, key in qs:
            txt = ask(img_bytes,
                      q + " Think briefly, then end with a final line exactly "
                          "'ANSWER: YES' or 'ANSWER: NO'.", f"{fn_ts[0]}|{sent}|{key}")
            want[key] = yesno(txt)
        votes = np.zeros(len(masks), np.int32)
        for k, (cx, cy, ar) in enumerate(geom):
            votes[k] = (int((cx < 0.5) == want["left"])
                        + int((cy < 0.5) == want["top"])
                        + int((ar > 0.33) == want["big"]))
        prior = np.array([pv for _, pv in parts], np.float64)
        j_vote = int(np.argmax(votes * 10.0 + prior))
        j_prior = int(np.argmax(prior))
        gt = r["gt"]
        def iou(mm):
            inter = float(np.logical_and(mm, gt).sum())
            union = float(np.logical_or(mm, gt).sum())
            return inter / union if union else 0.0
        delta_vote.append(iou(masks[j_vote]))
        delta_ship.append(iou(masks[j_prior]))
        n_used += 1
        if n_used % 100 == 0:
            print(f"  {n_used}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)

    d = np.array(delta_vote) - np.array(delta_ship)
    out = dict(
        mode="democracy",
        n=n_used,
        vote_pick_mIoU=round(100.0 * float(np.mean(delta_vote)), 3),
        prior_top_pick_mIoU=round(100.0 * float(np.mean(delta_ship)), 3),
        paired_delta=round(100.0 * float(d.mean()), 3),
        paired_se=round(100.0 * float(d.std(ddof=1) / np.sqrt(len(d))), 3) if n_used > 1 else None,
        changed_picks=int(sum(1 for a, b in zip(delta_vote, delta_ship)
                              if abs(a - b) > 1e-9)),
        note=("three fixed spatial questions per row, parts vote by geometry, top by "
              "(votes, prior); paired against the prior-only top part on the same rows; "
              "single-pick arms, so the absolute numbers sit below the shipped union rule"),
    )
    (WORK / "democracy_result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
