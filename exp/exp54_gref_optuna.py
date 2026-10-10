#!/usr/bin/env python
"""EXP-54 — 300-trial Optuna study for the gRefCOCO set+abstain model.

Held-out split only: gRefCOCO TRAIN refs (exp/tuning/grefcoco_refs.json),
never val.  The model under test is the changed one — set-valued selection
driven by the VLM box, with the VLM `none` answer (plus a gain-margin second
chance) as the abstention branch.  Objective: maximise
min(cIoU/68.94, gIoU/74.34) on the held-out records, i.e. how close the tuned
model gets to the published floors.

    python exp/exp54_gref_optuna.py --trials 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gref_setselect import rank01  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
STRIDE = 2


def box_path(fn, ph):
    key = hashlib.sha1(f"{MODEL_TAG}|{fn}|{ph.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def box_for(fn, ph):
    p = box_path(fn, ph)
    return (json.loads(p.read_text())["box"] if p.is_file() else None)


class Study:
    """Preloaded records: stride-2 masks, scores, boxes, GTs."""

    def __init__(self, records: Path, limit: int, seed: int, split: str = "train"):
        recs = [json.loads(l) for l in open(records)]
        recs = [r for r in recs if r.get("parsed")]
        # only records whose box answer is already cached.  A cached `none` is a
        # decided answer; only a MISSING cache entry means the VLM was never
        # asked, which would masquerade as an abstention and corrupt the study.
        n0 = len(recs)
        recs = [r for r in recs
                if box_path(r["file_name"],
                            (r["parsed"].get("sentence") or "").strip()).is_file()]
        print(f"box-cached records: {len(recs)} of {n0}", flush=True)
        rng = random.Random(seed)
        if limit and limit < len(recs):
            by_nt = {True: [r for r in recs if r["no_target"]],
                     False: [r for r in recs if not r["no_target"]]}
            n_nt = min(len(by_nt[True]), limit // 2)
            n_t = min(len(by_nt[False]), limit - n_nt)
            recs = rng.sample(by_nt[True], n_nt) + rng.sample(by_nt[False], n_t)
        gd = GreferDataset(REPRO / "data/refer", split)
        by_id = {r["ref_id"]: r for r in gd.refs}
        self.gt_cache, self.mask_cache, self.box_cache = {}, {}, {}
        self.rows = []
        from cache_map import cache_file
        for r in recs:
            rid = r["ref_id"]
            ref = by_id.get(rid)
            if ref is None:
                continue
            cf = r.get("cache_file") or str(cache_file(r["file_name"]))
            if cf not in self.mask_cache:
                with np.load(cf, allow_pickle=False) as z:
                    shape = tuple(int(x) for x in z["mask_shape"])
                    mm = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                                       ).reshape(shape).astype(bool)
                self.mask_cache[cf] = mm[:, ::STRIDE, ::STRIDE]
            if rid not in self.gt_cache:
                gt = gd.ground_truth(ref)
                self.gt_cache[rid] = gt[::STRIDE, ::STRIDE]
            ph = (r["parsed"].get("sentence") or "").strip()
            bk = (r["file_name"], ph.lower())
            if bk not in self.box_cache:
                self.box_cache[bk] = box_for(r["file_name"], ph)
            box = self.box_cache[bk]
            n = len(r["sc"])
            masks = self.mask_cache[cf]
            H, W = masks.shape[1], masks.shape[2]   # stride-2 dims
            if box is not None:
                # the box is in FULL-resolution pixels; convert to stride space
                # before clamping, or boxes past the half-width get truncated
                x0, y0, x1, y1 = [int(round(v)) for v in box]
                x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
                x1, y1 = min(W, -(-x1 // STRIDE)), min(H, -(-y1 // STRIDE))
                box = [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None
            sub = area_in = area_out = None
            if box is not None:
                x0, y0, x1, y1 = box
                sub = masks[:, y0:y1, x0:x1]
                area_in = sub.sum((1, 2)).astype(np.float64)
                area_out = masks.sum((1, 2)).astype(np.float64) - area_in
            self.rows.append(dict(
                cf=cf, rid=rid, no_target=bool(r["no_target"]),
                sc=np.asarray(r["sc"][:n], np.float32),
                sc_noun=np.asarray(r["sc_noun"][:n], np.float32),
                sc_strip=np.asarray(r["sc_strip"][:n], np.float32),
                gem=np.asarray(r["score_gem"][:n], np.float32),
                box=box, sub=sub, area_in=area_in, area_out=area_out,
                gt_area=float(r["gt_area"]) * (self.gt_cache[rid].size)))
        print(f"study rows: {len(self.rows)} "
              f"({sum(r['no_target'] for r in self.rows)} no-target)", flush=True)

    def _base(self, rec, src, gem_mix):
        if src == "sc_x_gem":
            return rank01(rec["sc"]) + gem_mix * rank01(rec["gem"])
        return {"sc": rec["sc"], "sc_noun": rec["sc_noun"],
                "sc_strip": rec["sc_strip"], "gem": rec["gem"]}[src]

    def evaluate(self, params):
        rows = []
        for rec in self.rows:
            masks = self.mask_cache[rec["cf"]]
            gt = self.gt_cache[rec["rid"]]
            base = self._base(rec, params["dec_source"], params["gem_mix"])
            k = min(int(params["k_rank"]), len(base))
            cand = np.argsort(base)[::-1][:k]
            box = rec["box"]
            sel = []
            if box is not None:
                x0, y0, x1, y1 = box
                if params["rule"] == "top2_disjoint":
                    sel = [int(cand[0])]
                    m1 = masks[sel[0]]
                    for j in cand[1:]:
                        m = masks[j]
                        uni = np.logical_or(m, m1).sum()
                        if uni and np.logical_and(m, m1).sum() / uni < params["max_iou"]:
                            sel.append(int(j))
                            break
                else:  # greedy box-cover with gain margin
                    sub = rec["sub"][cand]
                    area_in = rec["area_in"][cand]
                    area_out = rec["area_out"][cand]
                    box_area = float((y1 - y0) * (x1 - x0))
                    cover = np.zeros((y1 - y0, x1 - x0), bool)
                    cov_out = []          # outside-box masks already chosen
                    chosen_pos = []       # positions within `cand`
                    gains = []
                    for _ in range(int(params["max_k"])):
                        g = np.full(len(cand), -np.inf)
                        for j in range(len(cand)):
                            if j in chosen_pos or area_in[j] + area_out[j] <= 0:
                                continue
                            newly_in = np.logical_and(sub[j], ~cover).sum()
                            newly_out = area_out[j]
                            for mo in cov_out:
                                newly_out -= np.logical_and(
                                    masks[cand[j]], mo).sum()
                            g[j] = (newly_in - params["lam"] * newly_out) / box_area
                        j = int(np.argmax(g))
                        if not np.isfinite(g[j]) or g[j] <= 0:
                            break
                        gains.append(float(g[j]))
                        chosen_pos.append(j)
                        cover |= sub[j]
                        mo = masks[cand[j]].copy()
                        mo[y0:y1, x0:x1] = False
                        cov_out.append(mo)
                    sel = [int(cand[j]) for j in chosen_pos]
                    if params["rule"] == "boxcover_margin":
                        if not gains or max(gains) < params["gain_min"]:
                            sel = []
            if sel:
                pred = masks[sel].any(0)
            else:
                pred = np.zeros_like(gt)
            I = int(np.logical_and(pred, gt).sum())
            U = int(np.logical_or(pred, gt).sum())
            rows.append(dict(gt_nt=rec["no_target"], abstained=(not sel),
                             I=I, U=U, iou=(I / U if U else 0.0),
                             gt_area=rec["gt_area"]))
        # official metrics
        total = len(rows)
        acc_iou = acc_I = acc_U = 0.0
        tp = fn = tn = fp = 0
        for s in rows:
            if s["gt_nt"]:
                if s["abstained"]:
                    tp += 1
                    acc_iou += 1.0
                else:
                    fn += 1
                    acc_U += s["U"]
            else:
                if s["abstained"]:
                    fp += 1
                    acc_U += s["gt_area"]
                else:
                    tn += 1
                    acc_iou += s["iou"]
                    acc_I += s["I"]
                    acc_U += s["U"]
        cIoU = 100.0 * acc_I / acc_U if acc_U else 0.0
        gIoU = 100.0 * acc_iou / total if total else 0.0
        N_acc = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
        T_acc = 100.0 * tn / (tn + fp) if (tn + fp) else 0.0
        return dict(cIoU=cIoU, gIoU=gIoU, N_acc=N_acc, T_acc=T_acc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/optuna_study_gref.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--eval-only", action="store_true",
                    help="apply --study's best params to --records and report metrics")
    ap.add_argument("--study", type=Path,
                    default=Path("exp/out_final/optuna_study_gref.json"))
    args = ap.parse_args()

    study = Study(args.records, args.limit, args.seed, split=args.split)

    if args.eval_only:
        params = json.loads(args.study.read_text())["best_params"]
        m = study.evaluate(params)
        print(f"params: {params}")
        print(f"n={len(study.rows)}  cIoU={m['cIoU']:.2f}  gIoU={m['gIoU']:.2f}  "
              f"N_acc={m['N_acc']:.1f}  T_acc={m['T_acc']:.1f}")
        out = dict(records=str(args.records), split=args.split, n=len(study.rows),
                   params=params, metrics=m,
                   source_study=str(args.study))
        dst = Path(args.out)
        dst.write_text(json.dumps(out, indent=2))
        print(f"wrote {dst}")
        return 0

    def objective(trial):
        params = dict(
            rule=trial.suggest_categorical(
                "rule", ["boxcover", "boxcover_margin", "top2_disjoint"]),
            dec_source=trial.suggest_categorical(
                "dec_source", ["sc", "sc_noun", "sc_strip", "sc_x_gem"]),
            gem_mix=trial.suggest_float("gem_mix", 0.0, 2.0),
            k_rank=trial.suggest_categorical("k_rank", [10, 15, 25, 50]),
            lam=trial.suggest_float("lam", 0.1, 1.0),
            max_k=trial.suggest_categorical("max_k", [1, 2, 3, 4, 6]),
            max_iou=trial.suggest_float("max_iou", 0.2, 0.8),
            gain_min=trial.suggest_float("gain_min", 0.0, 0.15),
        )
        m = study.evaluate(params)
        v = min(m["cIoU"] / 68.94, m["gIoU"] / 74.34)
        trial.set_user_attr("metrics", m)
        return v

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    st = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    st.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    el = time.time() - t0
    print(f"{args.trials} trials in {el:.0f}s ({el/args.trials:.1f}s/trial)", flush=True)
    best = st.best_trial
    print(f"best value {best.value:.4f}")
    print(f"best params {best.params}")
    print(f"best metrics {best.user_attrs['metrics']}")
    top = sorted(st.trials, key=lambda t: -t.value)[:5]
    out = dict(n_trials=args.trials, best_value=best.value,
               best_params=best.params, best_metrics=best.user_attrs["metrics"],
               split="gRefCOCO train (held out; val never touched)",
               n_study_records=len(study.rows), stride=STRIDE,
               top5=[dict(value=t.value, params=t.params, metrics=t.user_attrs["metrics"])
                     for t in top],
               generated="exp54_gref_optuna.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
