#!/usr/bin/env python
"""EXP-10 — Rank the snap-free selector family only.

The composition rows in EXP-9 are dominated by the containment snap, which is a
known one-line granularity fix and is reported only as a baseline here. This
script ranks the selector *without* any snapping, selects on the selection
split, and reports the winner on the full split and the held-out complement.
The snap is added afterwards purely as a labelled ablation row.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, boot, confusion, PART_WORDS  # noqa: E402
from exp9_guided_v3 import pick, cand_set  # noqa: E402


def make_rule(mode, k, delta, pct, by, snap):
    def f(r):
        base = int(r.ps[r.final]) if snap else r.final
        if r.rel is None:
            return base
        cand = cand_set(r, mode, k, delta, True)
        if len(cand) < 2:
            return base
        idx = pick(r, cand, r.rel, pct, by)
        if idx is None:
            return base
        return int(r.ps[idx]) if snap else int(idx)
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp4/records_guided_val.jsonl"))
    ap.add_argument("--pairs", type=Path, default=Path("exp/out_exp4/pairs_guided_val.npz"))
    ap.add_argument("--select-refs", type=Path, default=Path("exp/select_refs_s101.json"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp10"))
    ap.add_argument("--tag", default="refcoco_guided")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    recs = [Rec(json.loads(l)) for l in args.records.open()]
    pairs = np.load(args.pairs)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    sel_ids = set(json.loads(args.select_refs.read_text()))
    sel = [r for r in recs if r.r["ref_id"] in sel_ids]
    held = [r for r in recs if r.r["ref_id"] not in sel_ids]
    print(f"{len(recs)} records; select={len(sel)} heldout={len(held)}", flush=True)

    BASE = {"argmax_g2l": lambda r: r.pure,
            "parent_strict": lambda r: int(r.ps[r.pure]),
            "hybridgl_guided": lambda r: r.final,
            "parent_strict_guided": lambda r: int(r.ps[r.final])}

    rules = {}
    for mode, k, delta, pct, by in itertools.product(
            ("conf", "topk", "unionTopk"), (5, 8, 12, 16, 24, 32), (0.20, 0.35, 0.50, 1.01),
            (0.5, 0.34), ("gem", "gem_sem", "sem")):
        rules[f"{mode}{k}_d{delta}_p{int(pct*100)}_{by}"] = make_rule(mode, k, delta, pct, by, False)
    print(f"{len(rules)} snap-free rules", flush=True)

    ysel = {nm: np.array([sel[t].gt[rules[nm](sel[t])] for t in range(len(sel))]) for nm in rules}
    r0 = ysel["hybridgl_guided"] if "hybridgl_guided" in ysel else np.array(
        [sel[t].gt[sel[t].final] for t in range(len(sel))])
    ranked = sorted(rules, key=lambda k: -ysel[k].mean())
    print(f"\nSELECTION SPLIT n={len(sel)} — top 20 snap-free")
    print(f"{'rule':<30}{'mIoU':>8}{'d_guided':>10}")
    for nm in ranked[:20]:
        print(f"{nm:<30}{ysel[nm].mean():>8.4f}{(ysel[nm]-r0).mean():>+10.4f}")

    chosen = ranked[0]
    print(f"\nCHOSEN: {chosen}", flush=True)

    keep = list(BASE) + [chosen] + [nm for nm in ranked[1:4]]
    # the composition row (selector + known granularity snap), clearly labelled
    cs = chosen
    comp_name = chosen + "+snap"
    keep.append(comp_name)

    summary = {}
    for sname, rs in (("all", recs), ("select", sel), ("heldout", held)):
        y, inter, un = {}, {}, {}
        for nm in keep:
            fn = (make_rule(*parse_name(cs), snap=True) if nm == comp_name else
                  (rules[nm] if nm in rules else BASE[nm]))
            y[nm] = np.array([rs[t].gt[fn(rs[t])] for t in range(len(rs))])
            inter[nm] = np.array([rs[t].inter[fn(rs[t])] for t in range(len(rs))])
            un[nm] = np.array([rs[t].union[fn(rs[t])] for t in range(len(rs))])
        part = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & PART_WORDS) for r in rs])
        nor = np.array([r.rel is None for r in rs])
        ref = y["hybridgl_guided"]
        arg = y["argmax_g2l"]
        rows = {}
        for nm in keep:
            d, lo, hi, p = boot(ref, y[nm])
            da, alo, ahi, ap = boot(arg, y[nm])
            rows[nm] = dict(mIoU=float(y[nm].mean()), oIoU=float(inter[nm].sum() / un[nm].sum()),
                            delta=d, ci=[lo, hi], p=p, delta_vs_argmax=da,
                            ci_vs_argmax=[alo, ahi], p_vs_argmax=ap,
                            part_delta=float((y[nm] - ref)[part].mean()) if part.sum() else None,
                            no_rel_delta=float((y[nm] - ref)[nor].mean()) if nor.sum() else None,
                            fired=int((y[nm] != ref).sum()),
                            win=int((y[nm] > ref).sum()), loss=int((y[nm] < ref).sum()))
        summary[sname] = dict(n=len(rs), rows=rows)
        with (args.output / f"rows_{args.tag}_{sname}.json").open("w") as fh:
            json.dump(rows, fh, indent=2)

    for sname in ("all", "heldout"):
        print(f"\n=== {sname.upper()} n={summary[sname]['n']} ===")
        s = summary[sname]["rows"]
        print(f"{'rule':<32}{'mIoU':>8}{'oIoU':>8}{'d_guided':>10}{'CI':>20}{'p':>8}{'fired':>7}{'part':>8}")
        for nm in sorted(s, key=lambda k: -s[k]["mIoU"]):
            v = s[nm]
            pd = v["part_delta"]
            print(f"{nm:<32}{v['mIoU']:>8.4f}{v['oIoU']:>8.4f}{v['delta']:>+10.4f}"
                  f"{('['+format(v['ci'][0],'+.4f')+','+format(v['ci'][1],'+.4f')+']'):>20}"
                  f"{v['p']:>8.4f}{v['fired']:>7d}{(pd if pd is not None else float('nan')):>+8.4f}")
    a = summary["all"]["rows"]
    o = a[chosen]
    gate = dict(delta_ge_2=bool(o["delta"] >= 0.02), ci_excludes_0=bool(o["ci"][0] > 0),
                oIoU_not_degraded=bool(o["oIoU"] >= a["hybridgl_guided"]["oIoU"] - 1e-9),
                part_ok=bool(o["part_delta"] is None or o["part_delta"] >= -0.01),
                parent_strict_row_present=True)
    gate["PASS"] = all(gate.values())
    print(f"\nchosen={chosen}\ngate={json.dumps(gate)}")
    with (args.output / f"gate_{args.tag}.json").open("w") as fh:
        json.dump(dict(chosen=chosen, gate=gate,
                       ranking=[(n, float(ysel[n].mean())) for n in ranked[:30]]), fh, indent=2)
    return 0


def parse_name(name):
    """'conf8_d0.35_p50_gem' -> ('conf', 8, 0.35, 0.5, 'gem')"""
    parts = name.split("_")
    mode = "".join(c for c in parts[0] if not c.isdigit())
    k = int("".join(c for c in parts[0] if c.isdigit()))
    delta = float(parts[1][1:])
    pct = int(parts[2][1:]) / 100.0
    by = "_".join(parts[3:])
    return mode, k, delta, pct, by


if __name__ == "__main__":
    raise SystemExit(main())
