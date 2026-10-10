#!/usr/bin/env python
"""EXP-66 — selection + abstention study on top of VLM region verification.

The verify cache (exp65) gives a yes/no for each of a record's top-4
classifier proposals.  Per record we precompute the official-metric
contributions of every candidate SUBSET (2^4), so the study searches

    selection : which subset rule  (verified-only, verified|fallback,
                verified UNION current-p_inc set, current-p_inc set)
    abstention: box-none | exists-no | all-candidates-verified-no,
                with the current empty-selection rule as an option

over 300 Optuna trials on the held-out fit/calib split (scored at val
prevalence), then freezes and takes one full-val pass.

    python exp66_verify_select.py --trials 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import load_record, make_clf, score_rows  # noqa: E402
from exp65_verify_extract import vpath, TOPK  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
VAL_NT_FRAC = 8896 / 14220


def build_rows(records: Path, split: str, clf, params):
    gd = GreferDataset(REPRO / "data/refer", split)
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    rows = []
    for line in open(records):
        rec = json.loads(line)
        row = load_record(rec, gd, by_id, mc, gc)
        if row is None:
            continue
        row["file_name"] = rec["file_name"]
        row["phrase"] = (rec["parsed"].get("sentence") or "").strip()
        rows.append(row)
    for r in rows:
        p = clf.predict_proba(r["X"])[:, 1]
        r["scores"] = p
        r["max_p"] = float(p.max())
        order = np.argsort(p)[::-1][:TOPK]
        r["cand"] = [int(j) for j in order]
        r["verify"] = []
        for j in r["cand"]:
            vp = vpath(r["file_name"], j, r["phrase"])
            no = None
            if vp.is_file():
                raw = (json.loads(vp.read_text()).get("raw") or "").strip().lower()
                if raw[:1] in ("y", "n"):
                    no = raw.startswith("n")
            r["verify"].append(no)
        # subset contributions: bitmask over candidates
        subs = {}
        for m in range(1, 1 << len(r["cand"])):
            idxs = [r["cand"][i] for i in range(len(r["cand"])) if m >> i & 1]
            pred = r["masks"][idxs].any(0)
            I = float(np.logical_and(pred, r["gt"]).sum())
            U = float(np.logical_or(pred, r["gt"]).sum())
            subs[m] = (I, U, I / U if U else 0.0)
        r["subs"] = subs
        # current p_inc selection
        for p_inc in (0.65, 0.75, 0.776, 0.85):
            sel = np.flatnonzero(r["scores"] >= p_inc)
            if sel.size:
                pred = r["masks"][sel].any(0)
                I = float(np.logical_and(pred, r["gt"]).sum())
                U = float(np.logical_or(pred, r["gt"]).sum())
                r.setdefault("pinc", {})[p_inc] = (True, I, U, I / U if U else 0.0)
            else:
                r.setdefault("pinc", {})[p_inc] = (False, 0.0, 0.0, 0.0)
    return rows


def pick(rule, r, p_inc, cand_idx=TOPK):
    """Return (has_pred, I, U, iou_reg, kind) under a selection rule."""
    v = r["verify"][:cand_idx]
    mask = 0
    for i, no in enumerate(v):
        if no is False:      # verified YES
            mask |= 1 << i
    if rule == "verified":
        if mask == 0:
            return (False, 0.0, 0.0, 0.0, "ver")
        return (True,) + r["subs"][mask] + ("ver",)
    if rule == "ver_fb":     # verified, else the top-1 candidate
        m = mask if mask else 1
        return (True,) + r["subs"][m] + ("ver",)
    if rule == "union":      # verified UNION current p_inc mask
        has, I2, U2, iou2 = r["pinc"][p_inc]
        if mask == 0 and not has:
            return (False, 0.0, 0.0, 0.0, "ver")
        if mask == 0:
            return (True, I2, U2, iou2, "ver")
        if not has:
            return (True,) + r["subs"][mask] + ("ver",)
        idxs = [r["cand"][i] for i in range(len(v)) if mask >> i & 1]
        sel = np.flatnonzero(r["scores"] >= p_inc)
        pred = r["masks"][idxs + list(sel)].any(0)
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        return (True, I, U, I / U if U else 0.0, "ver")
    # "base": ignore verification, current p_inc rule
    has, I, U, iou = r["pinc"][p_inc]
    return (has, I, U, iou, "base")


def evaluate(rows, rule, p_inc, abs_mode, allno_abstain, w_nt, w_t, weighted=True):
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0.0
    wsum = 0.0
    for r in rows:
        w = (w_nt if r["no_target"] else w_t) if weighted else 1.0
        wsum += w
        has, I, U, iou, _ = pick(rule, r, p_inc)
        abstain = r["vlm_none"] or (r.get("exists_no") is True)
        if not abstain and abs_mode == "empty" and not has:
            abstain = True
        if not abstain and allno_abstain:
            v = r["verify"][:TOPK]
            if v and all(x is True for x in v):
                abstain = True
        if abstain:
            if r["no_target"]:
                tp += w
                acc_iou += w
            else:
                fp += w
                acc_U += w * r["gt_area"]
            continue
        if not has:               # no mask available -> empty prediction
            if r["no_target"]:
                tp += w
                acc_iou += w
            else:
                fp += w
                acc_U += w * r["gt_area"]
            continue
        if r["no_target"]:
            fn += w
            acc_U += w * U
        else:
            tn += w
            acc_iou += w * iou
            acc_I += w * I
            acc_U += w * U
    return dict(cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                gIoU=100.0 * acc_iou / wsum if wsum else 0.0,
                N_acc=100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
                T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--val-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--classifier", type=Path,
                    default=Path("exp/out_final/optuna_study_gref5.json"))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/optuna_study_gref6.json"))
    args = ap.parse_args()

    st = json.loads(args.classifier.read_text())["best_params"]
    params = dict(C=st["C"]) if st["clf"] == "logreg" else dict(
        n_estimators=st["n_estimators"], max_depth=st["max_depth"],
        learning_rate=st["learning_rate"])
    clf = make_clf(st["clf"], params)
    t0 = time.time()
    tr_raw = build_rows(args.train_records, "train", clf, params) if False else None
    # classifier must be fit before scoring: build train features first, fit, then score
    gd_tr = GreferDataset(REPRO / "data/refer", "train")
    by_tr = {r["ref_id"]: r for r in gd_tr.refs}
    mc, gc = {}, {}
    feats = []
    for line in open(args.train_records):
        r = load_record(json.loads(line), gd_tr, by_tr, mc, gc)
        if r is not None:
            feats.append(r)
    clf.fit(np.concatenate([r["X"] for r in feats]),
            np.concatenate([r["y"] for r in feats]))
    print(f"classifier fit ({time.time()-t0:.0f}s)", flush=True)
    train_rows = build_rows(args.train_records, "train", clf, params)
    print(f"train rows {len(train_rows)} ({time.time()-t0:.0f}s)", flush=True)
    val_rows = build_rows(args.val_records, "val", clf, params)
    print(f"val rows {len(val_rows)} ({time.time()-t0:.0f}s)", flush=True)

    rng = random.Random(args.seed)
    refs = sorted({r["rid"] for r in train_rows})
    rng.shuffle(refs)
    n_fit = int((1 - args.calib_frac) * len(refs))
    fit_refs = set(refs[:n_fit])
    calib = [r for r in train_rows if r["rid"] not in fit_refs]
    n_nt_c = sum(r["no_target"] for r in calib)
    n_t_c = len(calib) - n_nt_c
    w_nt = (VAL_NT_FRAC / (1 - VAL_NT_FRAC)) * (n_t_c / max(1, n_nt_c))
    print(f"calib {len(calib)} rows ({n_nt_c} nt); w_nt={w_nt:.2f}", flush=True)

    def objective(trial):
        rule = trial.suggest_categorical("rule", ["verified", "ver_fb", "union", "base"])
        p_inc = trial.suggest_categorical("p_inc", [0.65, 0.75, 0.776, 0.85])
        abs_mode = trial.suggest_categorical("abs_mode", ["empty", "none"])
        allno = trial.suggest_categorical("allno", [True, False])
        m = evaluate(calib, rule, p_inc, abs_mode, allno, w_nt, 1.0)
        trial.set_user_attr("metrics", m)
        return min(m["cIoU"] / 68.94, m["gIoU"] / 74.34)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    best = study.best_trial
    print(f"{args.trials} trials in {time.time()-t0:.0f}s")
    print(f"best calib (val prevalence): {best.params} -> {best.user_attrs['metrics']}")
    bp = best.params
    vm = evaluate(val_rows, bp["rule"], bp["p_inc"], bp["abs_mode"], bp["allno"],
                  1.0, 1.0, weighted=False)
    print(f"FULL VAL: rule={bp['rule']} p_inc={bp['p_inc']} abs={bp['abs_mode']} "
          f"allno={bp['allno']} -> cIoU={vm['cIoU']:.2f} gIoU={vm['gIoU']:.2f} "
          f"N_acc={vm['N_acc']:.1f} T_acc={vm['T_acc']:.1f} (n={len(val_rows)})")
    diag = {}
    for rule in ("verified", "ver_fb", "union", "base"):
        for p in (0.75, 0.776):
            diag[f"{rule}|{p}"] = evaluate(val_rows, rule, p, "empty", False, 1.0, 1.0,
                                           weighted=False)
    out = dict(n_trials=args.trials, best_params=bp, calib_metrics=best.user_attrs["metrics"],
               val_transfer=vm, n_val=len(val_rows), val_diagnostic=diag,
               generated="exp66_verify_select.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
