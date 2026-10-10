#!/usr/bin/env python
"""EXP-6 — Robustness ablations for the confusion-set selector.

Every ablation re-analyses the stored records; no model is re-run. Reports the
delta over argmax, how often the selector lands on the oracle proposal, and the
per-relation-type breakdown, so a reviewer can see which part of the vocabulary
carries the gain and what the operator is actually doing when it fires.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_evaluate import Pool, boot, confusion, resolve, parent_strict  # noqa: E402

# Vocabulary tiers, smallest first.
VOCAB = {
    "lr": ({"left": -1, "right": 1}, {}, set()),
    "lrmid": ({"left": -1, "leftmost": -1, "right": 1, "rightmost": 1},
              {}, {"middle", "center", "centre", "mid", "central"}),
    "lrmid_ord": ({"left": -1, "leftmost": -1, "right": 1, "rightmost": 1},
                  {}, {"middle", "center", "centre", "mid", "central"},
                  {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                   "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}),
    "full": ({"left": -1, "leftmost": -1, "l": -1, "right": 1, "rightmost": 1},
             {"top": -1, "upper": -1, "above": -1, "topmost": -1,
              "bottom": 1, "lower": 1, "below": 1},
             {"middle", "center", "centre", "mid", "central"},
             {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
              "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}),
}


def parse_vocab(sent, spec):
    import re
    H = spec[0]
    V = spec[1] if len(spec) > 1 else {}
    M = spec[2] if len(spec) > 2 else set()
    O = spec[3] if len(spec) > 3 else {}
    w = re.findall(r"[a-z0-9]+", sent.lower())
    rel = dict(h=0, v=0, mid=False, ordinal=None, axis=None, direction=None)
    for i, x in enumerate(w):
        if x in H:
            rel["h"] = H[x]
        if x in V:
            rel["v"] = V[x]
        if x in M:
            rel["mid"] = True
        if x in O:
            rel["ordinal"] = O[x]
            for y in w[i + 1:i + 4]:
                if y in H:
                    rel["axis"], rel["direction"] = "h", H[y]
                if y in V:
                    rel["axis"], rel["direction"] = "v", V[y]
    return rel if (rel["h"] or rel["v"] or rel["mid"] or rel["ordinal"]) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp1d/records_collect_s101.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp6"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.35)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pools = [Pool(json.loads(l)) for l in args.records.open()]
    n = len(pools)
    base = np.array([p.gt[p.base] for p in pools])
    oracle = np.array([p.gt[p.orc] for p in pools])

    def run(k=args.k, delta=args.delta, split_iou=0.30, key="content", vocab="full", swap=False):
        ious = np.zeros(n)
        is_orc = np.zeros(n, bool)
        fired = np.zeros(n, bool)
        for t, p in enumerate(pools):
            rel = parse_vocab(p.r["sent"], VOCAB[vocab]) if vocab in VOCAB else p.rel
            if def_rel := (rel is None):
                ious[t] = p.gt[p.base]
                continue
            if swap:
                rel = dict(rel, h=-rel["h"], v=-rel["v"])
            cand, _ = confusion(p, key, k, delta, split_iou)
            if len(cand) < 2:
                ious[t] = p.gt[p.base]
                continue
            c = resolve(p, cand, rel)
            idx = int(c) if c is not None else p.base
            ious[t] = p.gt[idx]
            is_orc[t] = idx == p.orc
            fired[t] = idx != p.base
        return ious, is_orc, fired

    out = {}

    def rec(name, **kw):
        ious, is_orc, fired = run(**kw)
        d, lo, hi, pv = boot(base, ious)
        out[name] = dict(kw=kw, mIoU=float(ious.mean()), delta=d, ci=[lo, hi], p=pv,
                         fired=int(fired.sum()),
                         precision_on_oracle=float(is_orc[fired].mean()) if fired.sum() else None,
                         n_oracle_in_set=int(is_orc.sum()))
        return out[name]

    rec("main")
    for v in VOCAB:
        rec(f"vocab_{v}", vocab=v)
    for k in (2, 3, 4, 5, 6, 8, 12):
        rec(f"k{k}", k=k)
    for d in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.01):
        rec(f"d{d}", delta=d)
    for s in (0.10, 0.20, 0.30, 0.50, 0.70):
        rec(f"iouthr{s}", split_iou=s)
    for key in ("content", "nospace", "full"):
        rec(f"keyscore_{key}", key=key)
    rec("swap", swap=True)

    # per-relation-type breakdown of the main rule
    ious, is_orc, fired = run()
    reltype = np.array(["none" if parse_vocab(p.r["sent"], VOCAB["full"]) is None else
                        ("ordinal" if parse_vocab(p.r["sent"], VOCAB["full"])["ordinal"] else
                         "h" if parse_vocab(p.r["sent"], VOCAB["full"])["h"] else
                         "v" if parse_vocab(p.r["sent"], VOCAB["full"])["v"] else "mid")
                        for p in pools])
    strat = {}
    for rt in ("none", "h", "v", "mid", "ordinal"):
        m = reltype == rt
        if not m.sum():
            continue
        strat[rt] = dict(n=int(m.sum()), delta=float((ious - base)[m].mean()),
                         fired=int(fired[m].sum()),
                         precision_on_oracle=float(is_orc[m & fired].mean()) if (m & fired).sum() else None,
                         base_mIoU=float(base[m].mean()), new_mIoU=float(ious[m].mean()))
    out["_strata_by_relation"] = strat
    out["_oracle_in_confusion_set"] = float(is_orc.mean())
    out["_base_mIoU"] = float(base.mean())
    out["_oracle_mIoU"] = float(oracle.mean())

    with (args.output / "ablations_s101.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"n={n} base={base.mean():.4f} oracle={oracle.mean():.4f}")
    print(f"{'variant':<20}{'mIoU':>8}{'delta':>9}{'CI_lo':>9}{'fired':>7}{'P(oracle|fire)':>16}")
    for name, v in out.items():
        if name.startswith("_"):
            continue
        po = f"{v['precision_on_oracle']:.3f}" if v["precision_on_oracle"] is not None else "-"
        print(f"{name:<20}{v['mIoU']:>8.4f}{v['delta']:>+9.4f}{v['ci'][0]:>+9.4f}{v['fired']:>7d}{po:>16}")
    print("\nby relation type:")
    for rt, s in strat.items():
        po = f"{s['precision_on_oracle']:.3f}" if s["precision_on_oracle"] is not None else "-"
        print(f"  {rt:<9} n={s['n']:4d} base={s['base_mIoU']:.3f} new={s['new_mIoU']:.3f} "
              f"delta={s['delta']:+.4f} fired={s['fired']:4d} P(oracle|fire)={po}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
