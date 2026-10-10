#!/usr/bin/env python
"""EXP-23 — Final RefCOCO-family tables from this run's records.

Reads `exp/out_extreme/records_<tag>.jsonl` (one scoring pass, every per
proposal quantity stored) and writes

  exp/out_final/main_<tag>.json          the paper's main + sensitivity + control rows
  exp/out_final/evidence/<tag>_<row>.json   one evidence file per row

Rows are pure re-analyses of the stored pass; the selector is `method.py`
(category-score candidate set -> relation gate incl. two-axis -> released
coherence ranking) composed with the containment snap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, boot, PART_WORDS, reltype, taxonomy  # noqa: E402
from method import Selector, select  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "exp/out_final")
    ap.add_argument("--select-refs", type=Path, default=ROOT / "exp/select_refs_s101.json")
    args = ap.parse_args()

    recs = [Rec(json.loads(l)) for l in args.records.open()]
    pairs = np.load(args.pairs)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    n = len(recs)
    print(f"{args.tag}: {n} records", flush=True)

    sel = Selector()
    i_of = {id(r): i for i, r in enumerate(recs)}
    K8 = [np.argsort(-r.S["sc_content"], kind="stable")[:8] for r in recs]
    K8full = [np.argsort(-r.S["sc"], kind="stable")[:8] for r in recs]

    def ours(i):
        return int(sel(recs[i]))

    def no_gate(i):
        r = recs[i]
        if r.rel is None:
            return r.final
        c = K8[i]
        return int(c[int(np.argmax(r.S_gem[c]))])

    def no_gate_full(i):
        r = recs[i]
        if r.rel is None:
            return r.final
        c = K8full[i]
        return int(c[int(np.argmax(r.S_gem[c]))])

    def geom_extreme(i):
        r = recs[i]
        rel = r.rel
        if rel is None or rel["ordinal"]:
            return r.final
        c = K8[i]
        if rel["h"]:
            p = int(np.argmin(r.cx[c])) if rel["h"] < 0 else int(np.argmax(r.cx[c]))
        elif rel["v"]:
            p = int(np.argmin(r.cy[c])) if rel["v"] < 0 else int(np.argmax(r.cy[c]))
        elif rel["mid"]:
            p = int(np.argmin(np.abs(r.cx[c] - 0.5)))
        else:
            return r.final
        return int(c[p])

    def swap(i):
        r = recs[i]
        if r.rel is None:
            return r.final
        rel2 = dict(r.rel, h=-r.rel["h"], v=-r.rel["v"])
        c = K8[i]
        idx = select(c, rel2, r.cx, r.cy, r.S_gem, 0.5)
        return int(idx) if idx is not None else r.final

    def allpool(i):
        r = recs[i]
        if r.rel is None or r.rel["ordinal"]:
            return r.final
        idx = select(list(range(len(r.S_gem))), r.rel, r.cx, r.cy, r.S_gem, 0.5)
        return int(idx) if idx is not None else r.final

    rows = {
        "argmax_g2l": lambda i: recs[i].pure,
        "hybridgl_released": lambda i: recs[i].final,
        "snap_parent_strict": lambda i: int(recs[i].ps[recs[i].final]),
        "ours": ours,
        "ours_snap": lambda i: int(recs[i].ps[ours(i)]),
        "ctrl_no_gate": no_gate,
        "ctrl_no_gate_snap": lambda i: int(recs[i].ps[no_gate(i)]),
        "ctrl_fullset": no_gate_full,
        "ctrl_fullset_snap": lambda i: int(recs[i].ps[no_gate_full(i)]),
        "ctrl_geom_extreme": geom_extreme,
        "ctrl_swap": swap,
        "ctrl_allpool": allpool,
        "ours_k5": Selector(k=5),
        "ours_k12": Selector(k=12),
        "ours_k16": Selector(k=16),
        "ours_k24": Selector(k=24),
        "ours_p34": Selector(pct=0.34),
        "ours_p66": Selector(pct=0.66),
        "ours_rank_sem": Selector(score="sc_content"),
        "ours_rank_sc": Selector(score="sc"),
        "ours_set_full": Selector(key="sc"),
        "ours_set_strip": Selector(key="sc_strip"),
        "ours_set_noun": Selector(key="sc_noun"),
        "ours_ordfire": Selector(ordinals=True),
    }
    # callable rows take a record, not an index
    callable_rows = {k for k, v in rows.items() if isinstance(v, Selector)}

    idx = {}
    for name, fn in rows.items():
        if name in callable_rows:
            idx[name] = np.array([int(rows[name](r)) for r in recs])
        else:
            idx[name] = np.array([fn(i) for i in range(n)])
    gt = [r.gt for r in recs]
    ys = {name: np.array([gt[i][idx[name][i]] for i in range(n)]) for name in rows}
    yref = ys["hybridgl_released"]

    def oiou(idxv):
        inter = np.array([recs[i].inter[idxv[i]] for i in range(n)]).sum()
        union = np.array([recs[i].union[idxv[i]] for i in range(n)]).sum()
        return float(inter / union)

    part = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & PART_WORDS)
                     for r in recs])
    rty = np.array([reltype(r.rel) for r in recs])
    gt_area = np.array([r.gt_area for r in recs])
    qi = np.digitize(gt_area, np.quantile(gt_area, [0.25, 0.5, 0.75]), right=True)

    out_rows = {}
    for name in rows:
        d, lo, hi, p = boot(yref, ys[name])
        da, alo, ahi, ap = boot(ys["argmax_g2l"], ys[name])
        entry = dict(
            mIoU=float(ys[name].mean()), oIoU=oiou(idx[name]),
            delta_vs_released=d, ci_vs_released=[lo, hi], p_vs_released=p,
            delta_vs_argmax=da, ci_vs_argmax=[alo, ahi], p_vs_argmax=ap,
            fired=int((idx[name] != idx["hybridgl_released"]).sum()),
            win=int((ys[name] > yref).sum()), loss=int((ys[name] < yref).sum()),
            oracle_hit=float(np.mean([ys[name][i] == gt[i][recs[i].orc] for i in range(n)])),
        )
        if name in ("ours", "ours_snap", "snap_parent_strict", "hybridgl_released"):
            entry.update(
                part_delta=float((ys[name] - yref)[part].mean()), part_n=int(part.sum()),
                strata={k: float((ys[name] - yref)[rty == k].mean())
                        for k in ("none", "horizontal", "vertical", "middle", "ordinal")
                        if (rty == k).sum()},
                strata_n={k: int((rty == k).sum())
                          for k in ("none", "horizontal", "vertical", "middle", "ordinal")
                          if (rty == k).sum()},
                quartile_delta=[float((ys[name] - yref)[qi == q].mean()) for q in range(4)])
        out_rows[name] = entry
        print(f"  {name:<22}{entry['mIoU']:.4f} {entry['oIoU']:.4f}  {d:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}] p={p:.4f}", flush=True)

    tax = taxonomy(recs, lambda r: r.final)
    ours_vs_snap = boot(ys["snap_parent_strict"], ys["ours"])
    ours_vs_snap_snap = boot(ys["snap_parent_strict"], ys["ours_snap"])
    cond = boot(ys["ctrl_no_gate_snap"], ys["ours_snap"])

    out = {
        "tag": args.tag, "n": n,
        "records": str(args.records), "pairs": str(args.pairs),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scorer": "released HybridGL guidance reimplementation (exp/hybridgl_guided.py)",
        "selector": "exp/method.py Selector() (k=8, category set, relation gate incl. two-axis, "
                    "released coherence ranking) + containment snap",
        "oracle_mIoU": float(np.mean([r.gt[r.orc] for r in recs])),
        "oracle_ge50_frac": float(np.mean([r.gt[r.orc] >= 0.5 for r in recs])),
        "mean_pool_size": float(np.mean([len(r.gt) for r in recs])),
        "spatial_expr_frac": float(np.mean([r.rel is not None for r in recs])),
        "taxonomy": tax,
        "rows": out_rows,
        "comparisons": {
            "ours_minus_snap": dict(delta=ours_vs_snap[0], ci=[ours_vs_snap[1], ours_vs_snap[2]],
                                    p=ours_vs_snap[3]),
            "ours_snap_minus_snap": dict(delta=ours_vs_snap_snap[0],
                                         ci=[ours_vs_snap_snap[1], ours_vs_snap_snap[2]],
                                         p=ours_vs_snap_snap[3]),
            "ours_snap_minus_ctrl_no_gate_snap": dict(delta=cond[0], ci=[cond[1], cond[2]], p=cond[3]),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"main_{args.tag}.json").write_text(json.dumps(out, indent=2))

    ev = args.output / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    for name in rows:
        entry = dict(out_rows[name])
        entry.update(tag=args.tag, row=name, n=n, records=str(args.records),
                     script="exp/exp23_final.py",
                     records_mtime=time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime(args.records.stat().st_mtime)),
                     generated_at=out["generated_at"])
        (ev / f"{args.tag}_{name}.json").write_text(json.dumps(entry, indent=2))
    print(f"wrote {args.output / f'main_{args.tag}.json'} and {len(rows)} evidence files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
