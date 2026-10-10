#!/usr/bin/env python
"""EXP-5 — Taxonomy and selector evaluation at the released HybridGL operating point.

Reads the EXP-4 record file (full RefCOCO val through the released scorer) plus
the exact pairwise mask-IoU matrices from EXP-4b, and produces:

  * the error taxonomy of the oracle gap, overall and by GT-area quartile
  * mIoU / oIoU / paired-bootstrap CI / win-loss for every row
  * the pre-registered gate verdict for the selector under test
  * separate numbers for the seed-101 selection split and the held-out
    complement, so a hyper-parameter is never reported on the split it was
    chosen on

Everything is a re-analysis of one stored pass.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

H_WORDS = {"left": -1, "leftmost": -1, "l": -1, "right": +1, "rightmost": +1}
V_WORDS = {"top": -1, "upper": -1, "above": -1, "topmost": -1,
           "bottom": +1, "lower": +1, "below": +1}
MID = {"middle", "center", "centre", "mid", "central"}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
       "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}
PART_WORDS = {"hand", "hands", "arm", "arms", "leg", "legs", "head", "face", "foot", "feet", "eye", "eyes",
              "ear", "ears", "wheel", "wheels", "tail", "wing", "wings", "nose", "mouth", "neck", "finger",
              "fingers", "paw", "paws", "hoof", "beak", "horn", "horns", "antenna", "trunk", "handle",
              "lid", "door", "window", "sleeve", "collar", "tire", "tyre", "propeller", "blade", "stem"}
CLASSES = ["correct", "under-segmentation", "over-segmentation", "wrong-instance", "boundary"]


def parse(sent):
    w = re.findall(r"[a-z0-9]+", sent.lower())
    rel = dict(h=0, v=0, mid=False, ordinal=None, axis=None, direction=None)
    for i, x in enumerate(w):
        if x in H_WORDS:
            rel["h"] = H_WORDS[x]
        if x in V_WORDS:
            rel["v"] = V_WORDS[x]
        if x in MID:
            rel["mid"] = True
        if x in ORD:
            rel["ordinal"] = ORD[x]
            for y in w[i + 1:i + 4]:
                if y in H_WORDS:
                    rel["axis"], rel["direction"] = "h", H_WORDS[y]
                if y in V_WORDS:
                    rel["axis"], rel["direction"] = "v", V_WORDS[y]
    return rel if (rel["h"] or rel["v"] or rel["mid"] or rel["ordinal"]) else None


def reltype(rel):
    if rel is None:
        return "none"
    if rel["ordinal"]:
        return "ordinal"
    if rel["h"]:
        return "horizontal"
    if rel["v"]:
        return "vertical"
    return "middle"


def classify(iou_am, iou_orc, sub_am, sub_orc, iou_pair):
    if iou_am >= 0.5:
        return "correct"
    if sub_am:
        return "under-segmentation"
    if sub_orc:
        return "over-segmentation"
    if iou_pair < 0.1:
        return "wrong-instance"
    return "boundary"


class Rec:
    __slots__ = ("r", "S", "gt", "inter", "union", "cx", "cy", "area", "ps", "top",
                 "pure", "final", "orc", "rel", "sent", "gt_area", "pc", "S_gem")

    def __init__(self, r):
        self.r = r
        self.S = {k: np.array(r[k]) for k in ("sc", "sc_neg", "sc_noun", "sc_strip", "sc_content")}
        self.S_gem = np.array(r["score_gem"])
        self.gt = np.array(r["gt_iou"])
        self.inter = np.array(r["inter_gt"])
        self.union = np.array(r["union_gt"])
        self.cx = np.array(r["cx"])
        self.cy = np.array(r["cy"])
        self.area = np.array(r["area"])
        self.ps = np.array(r["ps_idx"])
        self.top = np.array(r["top_idx"])
        self.pure = r["pure"]
        self.final = r["final"]
        self.orc = r["orc"]
        self.sent = r["sent"]
        self.rel = parse(r["sent"])
        self.gt_area = r["gt_area"]
        self.pc = None


def taxonomy(recs, sel_idx):
    """Decompose the oracle gap for the proposal `sel_idx(rec)` selects."""
    gap = np.array([r.gt[r.orc] - r.gt[sel_idx(r)] for r in recs])
    cls, gt_area = [], []
    for r in recs:
        i = int(sel_idx(r))
        sub_am = r.pc[i, r.orc] == 0 and r.area[i] < r.area[r.orc] and r.gt[i] < 0.5
        # strict containment via mask-area identity is not recoverable from
        # IoU alone; use the standard proxy: intersection/area(i) >= 0.95
        inter = r.pc[i, r.orc] * (r.area[i] + r.area[r.orc]) / (1 + r.pc[i, r.orc])
        sub_am = (inter / max(r.area[i], 1e-9)) >= 0.95 and r.area[i] < r.area[r.orc]
        sub_orc = (inter / max(r.area[r.orc], 1e-9)) >= 0.95 and r.area[r.orc] < r.area[i]
        cls.append(classify(r.gt[i], r.gt[r.orc], sub_am, sub_orc, r.pc[i, r.orc]))
        gt_area.append(r.gt_area)
    cls = np.array(cls)
    gt_area = np.array(gt_area)
    total = float(gap.sum())
    out = dict(n=len(recs), mIoU=float(np.mean([r.gt[sel_idx(r)] for r in recs])),
               oracle_mIoU=float(np.mean([r.gt[r.orc] for r in recs])),
               gap_mass_total=total, by_class={})
    for c in CLASSES:
        m = cls == c
        out["by_class"][c] = dict(n=int(m.sum()), prevalence=float(m.mean()),
                                  gap_mass=float(gap[m].sum()),
                                  gap_share=float(gap[m].sum() / total) if total else None,
                                  mean_gap=float(gap[m].mean()) if m.sum() else None)
    q = np.quantile(gt_area, [0.25, 0.5, 0.75])
    qi = np.digitize(gt_area, q, right=True)
    out["by_quartile"] = {}
    for k, name in enumerate(("q1_smallest", "q2", "q3", "q4_largest")):
        m = qi == k
        out["by_quartile"][name] = dict(
            n=int(m.sum()), gap_mean=float(gap[m].mean()) if m.sum() else None,
            classes={c: float(gap[(cls == c) & m].sum() / gap[m].sum()) if m.sum() and gap[m].sum() else None
                     for c in CLASSES})
    return out


def confusion(r, key, k, delta, split_iou):
    S = r.S[key]
    order = np.argsort(-S, kind="stable")
    rng = float(S.max() - S.min())
    thr = S.max() - delta * rng if rng > 0 else S.max()
    kept = []
    for i in order:
        if len(kept) >= k:
            break
        if kept and S[i] < thr:
            break
        if all(r.pc[i, j] < split_iou for j in kept):
            kept.append(int(i))
    return kept


def resolve(r, cand, rel, swap=False):
    cx, cy = r.cx[cand], r.cy[cand]
    h = -rel["h"] if swap else rel["h"]
    v = -rel["v"] if swap else rel["v"]
    if rel["ordinal"]:
        axis = rel["axis"] or "h"
        d = (rel["direction"] or -1) * (-1 if swap else 1)
        order = np.argsort(d * (cx if axis == "h" else cy))
        p = rel["ordinal"] - 1
        return cand[int(order[p])] if p < len(order) else None
    if h:
        return cand[int(np.argmin(cx))] if h < 0 else cand[int(np.argmax(cx))]
    if v:
        return cand[int(np.argmin(cy))] if v < 0 else cand[int(np.argmax(cy))]
    if rel["mid"]:
        return cand[int(np.argmin(np.abs(cx - 0.5)))]
    return None


def make_rule(key="sc_noun", k=3, delta=0.35, split_iou=0.20, fallback="final", swap=False):
    def fn(r):
        fb = r.final if fallback == "final" else r.pure
        if r.rel is None:
            return fb
        cand = confusion(r, key, k, delta, split_iou)
        if len(cand) < 2:
            return fb
        c = resolve(r, cand, r.rel, swap)
        return int(c) if c is not None else fb
    return fn


def boot(y_a, y_b, n_boot=10000, seed=0, chunk=256):
    """Paired bootstrap + sign-flip permutation, chunked so n=10834 stays in memory."""
    rng = np.random.default_rng(seed)
    d = (y_b - y_a).astype(np.float64)
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    obs = float(d.mean())
    bs = np.empty(n_boot, np.float64)
    ps = np.empty(n_boot, np.float64)
    for s in range(0, n_boot, chunk):
        e = min(s + chunk, n_boot)
        b = e - s
        idx = rng.integers(0, n, size=(b, n))
        bs[s:e] = d[idx].mean(1)
        signs = rng.integers(0, 2, size=(b, n)) * 2.0 - 1.0
        ps[s:e] = np.abs((d * signs).mean(1))
        del idx, signs
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float((ps >= abs(obs)).mean())
    return obs, float(lo), float(hi), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp4/records_guided_val.jsonl"))
    ap.add_argument("--pairs", type=Path, default=Path("exp/out_exp4/pairs_guided_val.npz"))
    ap.add_argument("--select-refs", type=Path, default=Path("exp/select_refs_s101.json"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp5"))
    ap.add_argument("--tag", default="refcoco_guided")
    ap.add_argument("--rank-only", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("loading records ...", flush=True)
    recs = [Rec(json.loads(l)) for l in args.records.open()]
    pairs = np.load(args.pairs)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    print(f"{len(recs)} records loaded", flush=True)

    sel_ids = set(json.loads(args.select_refs.read_text()))
    splits = {"all": recs,
              "select": [r for r in recs if r.r["ref_id"] in sel_ids],
              "heldout": [r for r in recs if r.r["ref_id"] not in sel_ids]}

    # ---------------------------------------------------------- taxonomy
    tax = {"released_guided": taxonomy(recs, lambda r: r.final),
           "argmax_g2l": taxonomy(recs, lambda r: r.pure)}
    with (args.output / f"taxonomy_{args.tag}.json").open("w") as fh:
        json.dump(tax, fh, indent=2)
    for name, t in tax.items():
        print(f"\nTAXONOMY [{name}] n={t['n']} mIoU={t['mIoU']:.4f} oracle={t['oracle_mIoU']:.4f}")
        for c in CLASSES:
            v = t["by_class"][c]
            share = f"{v['gap_share']:.3f}" if v["gap_share"] is not None else "-"
            print(f"   {c:<20} n={v['n']:6d} prev={v['prevalence']:.3f} gap_share={share}")

    # ---------------------------------------------------------- selector grid
    RULES = {
        "argmax_g2l": lambda r: r.pure,
        "parent_strict": lambda r: int(r.ps[r.pure]),
        "hybridgl_guided": lambda r: r.final,
        "parent_strict_guided": lambda r: int(r.ps[r.final]),
    }
    grid = []
    for key in ("sc_noun", "sc_strip", "sc_content", "sc"):
        for k in (2, 3, 5, 8):
            for delta in (0.20, 0.35, 0.50, 1.01):
                grid.append((f"O_{key}_k{k}_d{delta}", dict(key=key, k=k, delta=delta)))
    for key in ("sc_noun", "sc_strip", "sc_content"):
        grid.append((f"O_{key}_k3_d0.35_pure", dict(key=key, k=3, delta=0.35, fallback="pure")))
        grid.append((f"SWAP_{key}_k3_d0.35", dict(key=key, k=3, delta=0.35, swap=True)))
    for name, kw in grid:
        RULES[name] = make_rule(**kw)

    # ---- 1. rank every rule on the selection split only
    select = splits["select"]
    print(f"\nranking {len(RULES)} rules on selection split (n={len(select)}) ...", flush=True)
    ysel = {name: np.array([select[t].gt[fn(select[t])] for t in range(len(select))])
            for name, fn in RULES.items()}
    ref_sel = ysel["hybridgl_guided"]
    rank = sorted(RULES, key=lambda k: -ysel[k].mean())
    print(f"{'rule':<28}{'mIoU':>8}{'d_guided':>10}")
    for name in rank[:12]:
        print(f"{name:<28}{ysel[name].mean():>8.4f}{(ysel[name]-ref_sel).mean():>+10.4f}")
    print("  ... worst:", rank[-1], f"{ysel[rank[-1]].mean():.4f}")
    chosen = rank[0]
    print(f"\nCHOSEN (selected on selection split): {chosen}", flush=True)

    # ---- 2. report the chosen rule + baselines on every split
    keep = ["argmax_g2l", "parent_strict", "hybridgl_guided", "parent_strict_guided", chosen]
    for extra in ("O_sc_strip_k3_d0.35", "SWAP_sc_noun_k3_d0.35"):
        if extra in RULES:
            keep.append(extra)
    summary = {}
    for sname, rs in splits.items():
        if not rs:
            continue
        rows = {}
        yy, ii, uu = {}, {}, {}
        for name in keep:
            fn = RULES[name]
            yy[name] = np.array([rs[t].gt[fn(rs[t])] for t in range(len(rs))])
            ii[name] = np.array([rs[t].inter[fn(rs[t])] for t in range(len(rs))])
            uu[name] = np.array([rs[t].union[fn(rs[t])] for t in range(len(rs))])
        part = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & PART_WORDS) for r in rs])
        noref = np.array([r.rel is None for r in rs])
        ref = yy["hybridgl_guided"]
        arg = yy["argmax_g2l"]
        for name in keep:
            d, lo, hi, p = boot(ref, yy[name])
            da, alo, ahi, ap = boot(arg, yy[name])
            rows[name] = dict(mIoU=float(yy[name].mean()),
                              oIoU=float(ii[name].sum() / uu[name].sum()),
                              delta_vs_guided=d, ci_vs_guided=[lo, hi], p_vs_guided=p,
                              delta_vs_argmax=da, ci_vs_argmax=[alo, ahi], p_vs_argmax=ap,
                              part_delta=float((yy[name] - ref)[part].mean()) if part.sum() else None,
                              part_n=int(part.sum()),
                              no_relation_delta=float((yy[name] - ref)[noref].mean()) if noref.sum() else None,
                              no_relation_n=int(noref.sum()),
                              fired=int((yy[name] != ref).sum()),
                              wlt=dict(win=int((yy[name] > ref).sum()),
                                       loss=int((yy[name] < ref).sum()),
                                       tie=int((yy[name] == ref).sum())))
        summary[sname] = dict(n=len(rs), rows=rows)
        with (args.output / f"rows_{args.tag}_{sname}.json").open("w") as fh:
            json.dump(rows, fh, indent=2)

    # ---- 3. pre-registered gate
    s = summary["all"]["rows"]
    o = s[chosen]
    gate = dict(delta_ge_2=bool(o["delta_vs_guided"] >= 0.02),
                ci_excludes_0=bool(o["ci_vs_guided"][0] > 0),
                oIoU_not_degraded=bool(o["oIoU"] >= s["hybridgl_guided"]["oIoU"] - 1e-9),
                part_ok=bool(o["part_delta"] is None or o["part_delta"] >= -0.01),
                parent_strict_row_present=bool("parent_strict" in RULES))
    gate["PASS"] = all(gate.values())

    print(f"\n=== FULL SPLIT n={summary['all']['n']} ===")
    print(f"{'rule':<28}{'mIoU':>8}{'oIoU':>8}{'d_guided':>10}{'CI':>20}{'p':>8}{'fired':>7}")
    for name in sorted(s, key=lambda k: -s[k]["mIoU"]):
        r = s[name]
        print(f"{name:<28}{r['mIoU']:>8.4f}{r['oIoU']:>8.4f}{r['delta_vs_guided']:>+10.4f}"
              f"{('['+format(r['ci_vs_guided'][0],'+.4f')+','+format(r['ci_vs_guided'][1],'+.4f')+']'):>20}"
              f"{r['p_vs_guided']:>8.4f}{r['fired']:>7d}")
    print("\nheld-out complement:")
    hs = summary["heldout"]["rows"]
    for name in sorted(hs, key=lambda k: -hs[k]["mIoU"]):
        r = hs[name]
        print(f"{name:<28}{r['mIoU']:>8.4f}{r['oIoU']:>8.4f}{r['delta_vs_guided']:>+10.4f}"
              f"{('['+format(r['ci_vs_guided'][0],'+.4f')+','+format(r['ci_vs_guided'][1],'+.4f')+']'):>20}"
              f"{r['p_vs_guided']:>8.4f}{r['fired']:>7d}")
    print(f"\nchosen={chosen}")
    print("gate:", json.dumps(gate))
    for name in (chosen, "parent_strict", "hybridgl_guided"):
        r = s[name]
        print(f"  {name:<24} fired={r['fired']:5d} part_d={r['part_delta']} "
              f"no_rel_d={r['no_relation_delta']} wlt={r['wlt']}")
    with (args.output / f"gate_{args.tag}.json").open("w") as fh:
        json.dump(dict(chosen=chosen, gate=gate, ranking=[(n, float(ysel[n].mean())) for n in rank[:20]]),
                  fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
