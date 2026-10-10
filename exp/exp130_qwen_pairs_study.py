#!/usr/bin/env python
"""EXP-130 — forced-choice discrimination on the two-object slice: LLaVA vs Qwen.

Re-analysis of the exp72 (LLaVA, namespace llava-onevision-qwen2-7b-ov|pair|)
and exp129 (Qwen, namespace qwen2.5vl|pair|) caches over the SAME jobs: the
291-row two-object sample, top-4 proposals by the shipped study-5 classifier,
6 unordered pairs x 2 orders. For each model: per-candidate mean preference
(both orders averaged, in [0,1]), winner = argmax; agreement = winner equals
the GT-best proposal in the top-4 (exp73's diagnostic; LLaVA's full-grid
reference: 29.8%, chance 25%). Parsed with a tolerant regex on both sides so
the head-to-head is fair; the strict startswith count is also reported for
LLaVA (exp73 comparability).

CPU only. Reads caches; writes exp/out_final/gref_qwen_pairs.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp72_pairwise_extract import TOPK  # noqa: E402
from vlm_boxes import CACHE_DIR  # noqa: E402

OUTF = Path("exp/out_final")


def lpath(file_name, i, j, phrase):
    key = hashlib.sha1(
        f"llava-onevision-qwen2-7b-ov|pair|{file_name}|{i}|{j}|{phrase.lower()}"
        .encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def qpath(file_name, i, j, phrase):
    key = hashlib.sha1(
        f"qwen2.5vl|pair|{file_name}|{i}|{j}|{phrase.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def answer(p, tolerant=True):
    """Return 1.0 if the FIRST image won, 0.0 if the SECOND, None unknown."""
    if not p.is_file():
        return None
    raw = (json.loads(p.read_text()).get("raw") or "").strip().lower()
    if not raw:
        return None
    if tolerant:
        m = re.search(r"\b(first|second)\b", raw)
        if not m:
            return None
        return 1.0 if m.group(1) == "first" else 0.0
    if raw.startswith("first"):
        return 1.0
    if raw.startswith("second"):
        return 0.0
    return None


def main() -> int:
    from exp62_clause_selector import load_record, make_clf
    from grefer_data import GreferDataset

    st = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    tune_by_rid = {}
    for l in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        tr = json.loads(l)
        tune_by_rid[tr["ref_id"]] = tr
    train_rows = [r for r in (load_record(tr, gd, by_id, dict(mc), {})
                              for tr in tune_by_rid.values()) if r is not None]
    clf = make_clf("logreg", dict(C=st["C"]))
    clf.fit(np.concatenate([r["X"] for r in train_rows]),
            np.concatenate([r["y"] for r in train_rows]))

    recs = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    models = {"llava": (lpath, True), "qwen": (qpath, True)}
    stats = {m: dict(n_records=0, agree=0, n_answered=0, order_answers=0)
             for m in models}
    strict_llava = dict(agree=0, n_records=0)
    size_heur = dict(agree=0, n_records=0)
    rand_base, rand_records = 0.0, 0
    for rec in recs:
        tr = tune_by_rid.get(rec["ref_id"])
        if tr is None:
            continue
        row = load_record(tr, gd, by_id, mc, gc)
        if row is None:
            continue
        ph = (tr["parsed"].get("sentence") or "").strip()
        if not ph:
            continue
        p = clf.predict_proba(row["X"])[:, 1]
        top = [int(j) for j in np.argsort(p)[::-1][:TOPK]]
        ious = {c: float(np.logical_and(row["masks"][c], row["gt"]).sum())
                / max(1.0, float(np.logical_or(row["masks"][c], row["gt"]).sum()))
                for c in top}
        arg_gt = max(top, key=lambda c: ious[c])
        for mname, (pathf, tol) in models.items():
            pref = {c: [] for c in top}
            for a, b in combinations(top, 2):
                for i2, j2 in ((a, b), (b, a)):
                    v = answer(pathf(rec["file_name"], i2, j2, ph), tolerant=tol)
                    if v is None:
                        continue
                    stats[mname]["order_answers"] += 1
                    pref[i2].append(v)          # i2 == the first image
            have = [c for c in top if pref[c]]
            if not have:
                continue
            winner = max(have, key=lambda c: float(np.mean(pref[c])))
            stats[mname]["n_records"] += 1
            stats[mname]["agree"] += int(winner == arg_gt)
            stats[mname]["n_answered"] += sum(len(v) for v in pref.values())
        # size-heuristic baseline: winner = largest mask area among candidates
        # with LLaVA answers (a trivial bias the VLM preference could mimic)
        have_l = [c for c in top if any(
            answer(lpath(rec["file_name"], i2, j2, ph), tolerant=True) is not None
            for i2, j2 in [(c, d) for d in top if d != c] + [(d, c) for d in top if d != c])]
        if have_l:
            areas = {c: float(row["masks"][c].sum()) for c in have_l}
            size_heur["n_records"] += 1
            size_heur["agree"] += int(max(have_l, key=lambda c: areas[c]) == arg_gt)
            rand_base += 1.0 / len(have_l)
            rand_records += 1
        # strict LLaVA count (exp73 comparability)
        pref = {c: [] for c in top}
        for a, b in combinations(top, 2):
            for i2, j2 in ((a, b), (b, a)):
                v = answer(lpath(rec["file_name"], i2, j2, ph), tolerant=False)
                if v is not None:
                    pref[i2].append(v)
        have = [c for c in top if pref[c]]
        if have:
            winner = max(have, key=lambda c: float(np.mean(pref[c])))
            strict_llava["n_records"] += 1
            strict_llava["agree"] += int(winner == arg_gt)

    out = dict(
        n_sample_records=len(recs),
        chance_pct=25.0,
        llava_full_grid_reference_pct=29.8,
        models={m: dict(
            records_with_answers=stats[m]["n_records"],
            order_answers=stats[m]["order_answers"],
            gt_best_agreement_pct=(100.0 * stats[m]["agree"]
                                   / stats[m]["n_records"]
                                   if stats[m]["n_records"] else None))
            for m in models},
        llava_strict_parse=dict(
            records=strict_llava["n_records"],
            agreement_pct=(100.0 * strict_llava["agree"] / strict_llava["n_records"]
                           if strict_llava["n_records"] else None)),
        baselines=dict(
            size_heuristic_agreement_pct=(100.0 * size_heur["agree"]
                                          / size_heur["n_records"]
                                          if size_heur["n_records"] else None),
            size_heuristic_records=size_heur["n_records"],
            random_over_answered_candidates_pct=(100.0 * rand_base / rand_records
                                                 if rand_records else None)),
        note=("tolerant parse = first regex hit of 'first'|'second' on both sides; "
              "Qwen extraction (exp129) may still be running - re-run for the final number"),
        generated="exp130_qwen_pairs_study.py")
    (OUTF / "gref_qwen_pairs.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
