#!/usr/bin/env python
"""EXP-173 — detector-free gRefCOCO search and arms with per-record terms.

Corrects the term-alignment bug of exp169 (terms were looked up by ref_id;
84.7% of targeted rows used another sentence's terms).  Modes:

  cache   precompute the held-out tune-split structures (packed masks, per-term
          box features, pair-candidate fallback) with each row's OWN sentence
  search  320-trial Optuna search over the DINO-free configuration space on the
          held-out tune split (official cIoU objective; no val contact)
  val     one pass over a val shard evaluating the fixed arms plus the
          searched configuration, with per-record alignment

    python exp/exp173_gref_search_aligned.py --mode cache
    python exp/exp173_gref_search_aligned.py --mode search --trials 320
    python exp/exp173_gref_search_aligned.py --mode val --shard 0 --nshards 4
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import probe_no  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
from exp169_gref_rescue import base_box_path  # noqa: E402
from method_gres import clauses  # noqa: E402

OUTF = Path("exp/out_final")
TUNE = Path("exp/out_extreme/records_grefcoco_train_tune.jsonl")
VAL = Path("exp/out_extreme/records_grefcoco_14229.jsonl")
CACHE = Path("exp/out_extreme/gref_search_tune_cache.pkl")
STRIDE = 2


# ---------------------------------------------------------------- structures
def pack2d(a):
    a = np.ascontiguousarray(a, dtype=bool)
    return np.packbits(a.reshape(-1)), a.shape


def unpack2d(p, shape):
    n = shape[0] * shape[1]
    return np.unpackbits(p, count=n).reshape(shape).astype(bool)


def bxi_box(masks, bs, box, halved=False):
    """Box--mask IoU column exactly as box_feats F[:,1] (stride-2 frame)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if not halved:
        x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
        x1, y1 = min(masks.shape[2], -(-x1 // STRIDE)), min(masks.shape[1], -(-y1 // STRIDE))
    else:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(masks.shape[2], x1), min(masks.shape[1], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    areas = masks.sum((1, 2)).astype(np.float64)
    area_box = float((x1 - x0) * (y1 - y0))
    inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
    return (inb / (areas + area_box - inb + 1e-9)).astype(np.float32)


def fit_base(tune=TUNE):
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(tune, "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    return clf_base, tr


def fit_pair(tr, clf_base):
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    return pair_clf, g8


def term_arrays(r, fn, terms, halved=False):
    out = []
    for t in terms:
        box = None
        if halved:
            box = r["_box"]
        else:
            p = lpath(fn, t)
            if p.is_file():
                box = json.loads(p.read_text()).get("box")
        if box is None:
            continue
        b = bxi_box(r["masks"], r["_bs"], box, halved=halved)
        if b is not None:
            out.append((t, b))
    return out


# ------------------------------------------------------------------- cache
def build_cache():
    t0 = time.time()
    clf_base, tr = fit_base()
    pair_clf, g8 = fit_pair(tr, clf_base)
    print(f"tune rows {len(tr)}, clfs fitted ({time.time()-t0:.0f}s)", flush=True)
    cache = []
    for i, r in enumerate(tr):
        rec = r["rec"]
        fn = rec["file_name"]
        sent = (rec["parsed"].get("sentence") or "").strip()
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
        masks = r["masks"]
        H, W = masks.shape[1], masks.shape[2]
        packed = np.stack([pack2d(m)[0] for m in masks]) if len(masks) else \
            np.zeros((0, (H * W + 7) // 8), np.uint8)
        gt_p, gt_shape = pack2d(r["gt"])
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        if r["no_target"]:
            single = [kk for kk, c in enumerate(ids) if c[1] < 0]
            fb = max(single, key=lambda kk: pp[kk]) if single else int(np.argmax(pp))
        else:
            fb = int(np.argmax(pp))
        fb_mask = candidate_mask(masks, ids[fb])
        entry = dict(
            n=len(masks), shape=(H, W), masks=packed, gt=gt_p, gt_shape=gt_shape,
            gt_area=float(r["gt_area"]), no_target=bool(r["no_target"]),
            ens=bool(any(r["ens"])), pp_max=float(pp.max()),
            fallback=pack2d(fb_mask)[0], bs=r["_bs"][:len(masks)].astype(np.float32),
            noun=term_arrays(r, fn, terms_of(rec["parsed"])),
            sentence=term_arrays(r, fn, [sent], halved=True) if sent else [],
        )
        cls = []
        for c in clauses(sent):
            p = base_box_path(fn, c)
            if not p.is_file():
                continue
            box = json.loads(p.read_text()).get("box")
            if box is None:
                continue
            b = bxi_box(masks, r["_bs"], box)
            if b is not None:
                cls.append((c, b, probe_no("exists", fn, c) is True))
        entry["clause"] = cls
        cache.append(entry)
        r["masks"] = None
        if (i + 1) % 200 == 0:
            print(f"  cache {i+1}/{len(tr)} ({time.time()-t0:.0f}s)", flush=True)
    CACHE.write_bytes(pickle.dumps(cache, protocol=4))
    print(f"cache written: {len(cache)} rows, {CACHE.stat().st_size/1e9:.2f} GB "
          f"({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ metrics
def row_metrics(rows):
    I = sum(x["I"] for x in rows)
    U = sum(x["U"] for x in rows)
    iou = sum(x["iou"] for x in rows)
    n_nt = sum(1 for x in rows if x["no_target"])
    n_nt_ab = sum(1 for x in rows if x["no_target"] and x["abstain"])
    return dict(cIoU=100.0 * I / U if U else 0.0, gIoU=100.0 * iou / len(rows),
                N_acc=100.0 * n_nt_ab / n_nt if n_nt else 0.0, n=len(rows),
                _raw=dict(I=I, U=U, iou=iou, n=len(rows), n_nt=n_nt,
                          n_nt_ab=n_nt_ab))


def eval_cache(cache, cfg):
    """Objective on the held-out tune rows (official cIoU, mean gIoU, N-acc)."""
    src = cfg["box_source"]
    alpha, beta = cfg["alpha"], cfg["beta"]
    topk, margin = cfg["topk"], cfg["margin"]
    p_sel = cfg["p_sel"]
    presence = cfg.get("presence", False)
    rows = []
    for e in cache:
        abstain = (e["pp_max"] < p_sel) or e["ens"]
        if abstain:
            rows.append(dict(no_target=e["no_target"], abstain=True, I=0.0,
                             U=0.0 if e["no_target"] else e["gt_area"],
                             iou=1.0 if e["no_target"] else 0.0))
            continue
        if src == "noun":
            terms = [(t, b) for t, b in e["noun"]]
        elif src == "sentence":
            terms = e["sentence"]
        else:
            terms = [(t, b) for t, b, absent in e["clause"]
                     if not (presence and absent)]
        parts = []
        for _, b in terms:
            score = e["bs"] * (alpha + beta * b)
            order = np.argsort(score)[::-1]
            js = [int(order[0])]
            if topk == 2 and len(order) > 1 and score[order[1]] >= margin * score[order[0]]:
                js.append(int(order[1]))
            parts.extend(js)
        if parts:
            pred = None
            for j in parts:
                m = unpack2d(e["masks"][j], e["shape"])
                pred = m if pred is None else (pred | m)
        else:
            pred = unpack2d(e["fallback"], e["shape"])
        gt = unpack2d(e["gt"], e["gt_shape"])
        I = float(np.logical_and(pred, gt).sum())
        U = float(np.logical_or(pred, gt).sum())
        rows.append(dict(no_target=e["no_target"], abstain=False, I=I, U=U,
                         iou=0.0 if e["no_target"] else (I / U if U else 0.0)))
    return row_metrics(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cache", "search", "val"])
    ap.add_argument("--trials", type=int, default=320)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.mode == "cache":
        build_cache()
        return 0

    if args.mode == "search":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cache = pickle.loads(CACHE.read_bytes())
        cfg14 = dict(json.loads((OUTF / "optuna_study_gref14.json").read_text())["best_params"])
        g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]

        def objective(trial):
            cfg = dict(
                box_source=trial.suggest_categorical(
                    "box_source", ["noun", "clause", "clause_presence", "sentence"]),
                alpha=trial.suggest_float("alpha", 0.0, 3.0),
                beta=trial.suggest_float("beta", 0.0, 0.5),
                topk=trial.suggest_categorical("topk", [1, 2]),
                margin=trial.suggest_float("margin", 0.5, 1.0),
                p_sel=trial.suggest_float("p_sel", 0.2, 0.7),
            )
            cfg["presence"] = cfg["box_source"] == "clause_presence"
            cfg["box_source"] = "clause" if cfg["box_source"] == "clause_presence" \
                else cfg["box_source"]
            m = eval_cache(cache, cfg)
            trial.set_user_attr("gIoU", m["gIoU"])
            trial.set_user_attr("N_acc", m["N_acc"])
            return m["cIoU"]

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=20261019))
        t0 = time.time()
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
        bp = dict(study.best_params)
        bp["presence"] = bp["box_source"] == "clause_presence"
        bp["box_source"] = "clause" if bp["box_source"] == "clause_presence" \
            else bp["box_source"]
        out = dict(
            n_trials=len(study.trials), best_params=bp,
            best_value=float(study.best_value),
            best_attrs={k: float(v) for k, v in study.best_trial.user_attrs.items()},
            split="held-out gRefCOCO train-tune rows (1,784 records, per-record terms)",
            objective="official cIoU on tune rows (abstention included)",
            sampler="TPESampler(seed=20261019)",
            trials=[dict(number=t.number, value=t.value, params=t.params,
                         state=str(t.state)) for t in study.trials],
            seconds=round(time.time() - t0, 1),
            generated="exp173_gref_search_aligned.py --mode search")
        outpath = args.out or (OUTF / "gref_search_aligned.json")
        outpath.write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("n_trials", "best_params",
                                             "best_value", "best_attrs", "seconds")},
                         indent=2), flush=True)
        return 0

    # ---- val pass ---------------------------------------------------------
    cfg14 = dict(json.loads((OUTF / "optuna_study_gref14.json").read_text())["best_params"])
    if "cap" in cfg14 and "sam_area_cap" not in cfg14:
        cfg14["sam_area_cap"] = cfg14.pop("cap")
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    search = json.loads((OUTF / "gref_search_aligned.json").read_text())
    searched = dict(search["best_params"])
    p_fixed = float(g8["p_sel"])
    arms = {
        "base_noun": dict(box_source="noun", alpha=cfg14["alpha"], beta=cfg14["beta"],
                          topk=cfg14["topk"], margin=cfg14["margin"], p_sel=p_fixed,
                          presence=False),
        "full_clause": dict(box_source="clause", alpha=cfg14["alpha"], beta=cfg14["beta"],
                            topk=cfg14["topk"], margin=cfg14["margin"], p_sel=p_fixed,
                            presence=False),
        "full_clause_presence": dict(box_source="clause", alpha=cfg14["alpha"],
                                     beta=cfg14["beta"], topk=cfg14["topk"],
                                     margin=cfg14["margin"], p_sel=p_fixed,
                                     presence=True),
        "sentence_box": dict(box_source="sentence", alpha=cfg14["alpha"],
                             beta=cfg14["beta"], topk=cfg14["topk"],
                             margin=cfg14["margin"], p_sel=p_fixed, presence=False),
        "pair_union": dict(box_source="pair", alpha=0.0, beta=0.0, topk=1,
                           margin=1.0, p_sel=p_fixed, presence=False),
        "searched": searched,
    }

    clf_base = make_clf("logreg", dict(C=json.loads(
        (OUTF / "optuna_study_gref5.json").read_text())["best_params"]["C"]))
    tr = load_rows(TUNE, "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]), np.concatenate([r["y"] for r in tr]))
    pair_clf, _ = fit_pair(tr, clf_base)
    del tr
    print(f"[val] clfs fitted; loading rows (shard {args.shard}/{args.nshards})", flush=True)
    val = load_rows(VAL, "val", clf_base)
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    val = val[args.shard::args.nshards]
    if args.limit:
        val = val[: args.limit]
    print(f"[val] {len(val)} rows", flush=True)

    per = {a: [] for a in arms}
    t0 = time.time()
    for k, r in enumerate(val):
        rec = r["rec"]
        fn = rec["file_name"]
        sent = (rec["parsed"].get("sentence") or "").strip()
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        no_target = r["no_target"]
        gt_area = r["gt_area"]
        gt = r["gt"]
        # shared per-row structures (computed once, used by all arms)
        struct = dict(bs=r["_bs"], masks=r["masks"])
        struct["noun"] = term_arrays(r, fn, terms_of(rec["parsed"]))
        struct["sentence"] = term_arrays(r, fn, [sent], halved=True) if sent else []
        cls = []
        for c in clauses(sent):
            p = base_box_path(fn, c)
            if not p.is_file():
                continue
            box = json.loads(p.read_text()).get("box")
            if box is None:
                continue
            b = bxi_box(r["masks"], r["_bs"], box)
            if b is not None:
                cls.append((c, b, probe_no("exists", fn, c) is True))
        struct["clause"] = cls
        plain_pair = candidate_mask(r["masks"], ids[int(np.argmax(pp))])
        if no_target:
            single = [kk for kk, c in enumerate(ids) if c[1] < 0]
            fbi = max(single, key=lambda kk: pp[kk]) if single else int(np.argmax(pp))
        else:
            fbi = int(np.argmax(pp))
        fb_mask = candidate_mask(r["masks"], ids[fbi])
        for name, cfg in arms.items():
            abstain = (pp.max() < cfg["p_sel"]) or any(r["ens"])
            if abstain:
                per[name].append(dict(no_target=no_target, abstain=True, I=0.0,
                                      U=0.0 if no_target else gt_area,
                                      iou=1.0 if no_target else 0.0))
                continue
            if cfg["box_source"] == "pair":
                pred = plain_pair
            else:
                src = cfg["box_source"]
                if src == "noun":
                    terms = struct["noun"]
                elif src == "sentence":
                    terms = struct["sentence"]
                else:
                    terms = [(t, b) for t, b, absent in struct["clause"]
                             if not (cfg.get("presence") and absent)]
                parts = []
                alpha, beta = cfg["alpha"], cfg["beta"]
                topk, margin = cfg["topk"], cfg["margin"]
                for _, b in terms:
                    score = struct["bs"] * (alpha + beta * b)
                    order = np.argsort(score)[::-1]
                    js = [int(order[0])]
                    if topk == 2 and len(order) > 1 and \
                            score[order[1]] >= margin * score[order[0]]:
                        js.append(int(order[1]))
                    parts.extend(js)
                if parts:
                    pred = None
                    for j in parts:
                        m = struct["masks"][j]
                        pred = m if pred is None else (pred | m)
                else:
                    pred = fb_mask
            I = float(np.logical_and(pred, gt).sum())
            U = float(np.logical_or(pred, gt).sum())
            per[name].append(dict(no_target=no_target, abstain=False, I=I, U=U,
                                  iou=0.0 if no_target else (I / U if U else 0.0)))
        if (k + 1) % 2000 == 0:
            print(f"  val {k+1}/{len(val)} ({time.time()-t0:.0f}s)", flush=True)

    res = {a: row_metrics(rows) for a, rows in per.items()}
    out = dict(shard=args.shard, nshards=args.nshards, arms=res,
               searched_params=searched, fixed_params=dict(
                   alpha=cfg14["alpha"], beta=cfg14["beta"], topk=cfg14["topk"],
                   margin=cfg14["margin"], p_sel=p_fixed),
               alignment="per-record parsed sentence and file_name",
               generated="exp173_gref_search_aligned.py --mode val")
    outpath = args.out or (OUTF / f"gref_search_val_part{args.shard}.json")
    outpath.write_text(json.dumps(out, indent=2))
    for a in arms:
        print(f"  {a:22s} cIoU {res[a]['cIoU']:7.3f} gIoU {res[a]['gIoU']:7.3f} "
              f"N {res[a]['N_acc']:6.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
