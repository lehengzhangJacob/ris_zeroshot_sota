#!/usr/bin/env python
"""EXP-18 — Decision probe (CPU re-analysis of stored records).

P1  Kill-controls: does the relation gate survive *conditional on the snap*?
P2  Strata: which spatial sub-patterns the parser acts on, which are left out.
P3  Loss diagnostics + precision levers for the selector.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, boot  # noqa: E402
from method import Selector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "exp/out_exp4/records_guided_val.jsonl"
PAIRS = ROOT / "exp/out_exp4/pairs_guided_val.npz"

ANCHOR_PAT = re.compile(
    r"\b(next\s+to|near|beside|between|behind|in\s+front\s+of|on\s+top\s+of|"
    r"left\s+of|right\s+of|above|below|under|top\s+of)\b")
SUP_W = {"leftmost": ("h", -1), "rightmost": ("h", +1),
         "topmost": ("v", -1), "bottommost": ("v", +1)}


def relstratum(r):
    if r.rel is None:
        return "none"
    if r.rel["ordinal"]:
        return "ordinal"
    w = set(re.findall(r"[a-z]+", r.sent.lower()))
    if w & set(SUP_W):
        return "superlative"
    if r.rel["h"] and r.rel["v"]:
        return "two_axis"
    if ANCHOR_PAT.search(r.sent.lower()):
        return "anchored"
    if r.rel["h"]:
        return "horizontal"
    if r.rel["v"]:
        return "vertical"
    return "middle"


def main() -> int:
    recs = [Rec(json.loads(l)) for l in REC.open()]
    pairs = np.load(PAIRS)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    n = len(recs)
    print(f"{n} records, {sum(r.rel is not None for r in recs)} spatial", flush=True)

    gt = [r.gt for r in recs]
    ps = [r.ps for r in recs]
    K8 = [np.argsort(-r.S["sc_content"], kind="stable")[:8] for r in recs]
    K8full = [np.argsort(-r.S["sc"], kind="stable")[:8] for r in recs]

    def side_mask(r, cand, rel, pct=0.5):
        cx, cy = r.cx[cand], r.cy[cand]
        if rel["h"]:
            v, sign = cx, rel["h"]
        elif rel["v"]:
            v, sign = cy, rel["v"]
        elif rel["mid"]:
            v, sign = -np.abs(cx - 0.5), 1
        else:
            return np.zeros(len(cand), bool)
        thr = np.quantile(v, pct) if sign < 0 else np.quantile(v, 1 - pct)
        return v <= thr if sign < 0 else v >= thr

    # ---------------- P1 -----------------------------------------------------
    selr = Selector()

    def ours(i):
        return int(selr(recs[i]))

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

    rows = {
        "hybridgl_guided": lambda i: recs[i].final,
        "argmax_g2l": lambda i: recs[i].pure,
        "parent_strict (snap)": lambda i: int(ps[i][recs[i].final]),
        "ours": ours,
        "ours + snap": lambda i: int(ps[i][ours(i)]),
        "no_gate (cat-top8+coh)": no_gate,
        "no_gate + snap": lambda i: int(ps[i][no_gate(i)]),
        "fullset (full-top8+coh)": no_gate_full,
        "fullset + snap": lambda i: int(ps[i][no_gate_full(i)]),
    }
    idx = {name: np.array([fn(i) for i in range(n)]) for name, fn in rows.items()}
    ys = {name: np.array([gt[i][idx[name][i]] for i in range(n)]) for name in rows}
    yref = ys["hybridgl_guided"]

    def oiou(idxv):
        inter = np.array([recs[i].inter[idxv[i]] for i in range(n)]).sum()
        union = np.array([recs[i].union[idxv[i]] for i in range(n)]).sum()
        return float(inter / union)

    print("\n=== P1 kill-controls (full RefCOCO val) ===", flush=True)
    print(f"{'row':<26}{'mIoU':>8}{'oIoU':>8}{'d_mIoU':>9}{'CI':>20}{'win':>6}{'loss':>6}")
    for name in rows:
        d, lo, hi, p = boot(yref, ys[name])
        print(f"{name:<26}{ys[name].mean():>8.4f}{oiou(idx[name]):>8.4f}{d:>+9.4f}"
              f"{'[%+.4f,%+.4f]' % (lo, hi):>20}"
              f"{int((ys[name] > yref).sum()):>6}{int((ys[name] < yref).sum()):>6}", flush=True)

    for a, b in (("no_gate + snap", "ours + snap"),
                 ("no_gate (cat-top8+coh)", "ours"),
                 ("fullset + snap", "ours + snap"),
                 ("parent_strict (snap)", "ours")):
        d, lo, hi, p = boot(ys[a], ys[b])
        print(f"({b}) - ({a}): {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}", flush=True)

    # ---------------- P2 -----------------------------------------------------
    strat = np.array([relstratum(r) for r in recs])
    y_ours = ys["ours"]

    def spec_super(i):
        r = recs[i]
        w = set(re.findall(r"[a-z]+", r.sent.lower()))
        got = [(ax, sg) for word, (ax, sg) in SUP_W.items() if word in w]
        if not got:
            return r.final
        ax, sg = got[0]
        c = K8[i]
        v = r.cx[c] if ax == "h" else r.cy[c]
        return int(c[int(np.argmin(sg * v))])

    def spec_two(i):
        r = recs[i]
        c = K8[i]
        rel = r.rel
        if rel is None:
            return r.final
        keep = side_mask(r, c, dict(rel, mid=False, ordinal=None))
        keep2 = side_mask(r, c, dict(rel, h=0, mid=False, ordinal=None))
        sub = c[keep & keep2]
        if not len(sub):
            sub = c[keep] if keep.any() else c
        return int(sub[int(np.argmax(r.S_gem[sub]))])

    def spec_anchorless(i):
        r = recs[i]
        c = K8[i]
        return int(c[int(np.argmax(r.S_gem[c]))])

    specs = {"superlative": spec_super, "two_axis": spec_two, "anchored": spec_anchorless}
    print("\n=== P2 strata ===", flush=True)
    print(f"{'stratum':<14}{'n':>7}{'fired':>7}{'ours_d':>9}{'spec_d':>9}")
    for s in ("horizontal", "vertical", "middle", "superlative", "two_axis",
              "anchored", "ordinal", "none"):
        m = strat == s
        if not m.sum():
            continue
        fired = int((idx["ours"][m] != idx["hybridgl_guided"][m]).sum())
        od = float((y_ours - yref)[m].mean())
        sd = "-"
        if s in specs:
            sv = np.array([gt[i][specs[s](i)] for i in range(n)])
            sd = f"{float((sv - yref)[m].mean()):+.4f}"
        print(f"{s:<14}{int(m.sum()):>7}{fired:>7}{od:>+9.4f}{sd:>9}", flush=True)

    # ---------------- P3 -----------------------------------------------------
    diff = y_ours - yref
    hurt, helped = diff < -1e-9, diff > 1e-9
    released_in_side = np.zeros(n, bool)
    gem_margin = np.zeros(n)
    iou_cr = np.zeros(n)
    for i, r in enumerate(recs):
        sub = K8[i][side_mask(r, K8[i], r.rel)] if r.rel is not None else K8[i]
        released_in_side[i] = r.final in set(sub.tolist())
        if len(sub) > 1:
            g = np.sort(r.S_gem[sub])[::-1]
            gem_margin[i] = float(g[0] - g[1])
        iou_cr[i] = float(r.pc[idx["ours"][i], r.final])
    print(f"\n=== P3 diagnostics: helps {int(helped.sum())}, hurts {int(hurt.sum())} ===")
    for label, m in (("helped", helped), ("hurt", hurt)):
        print(f"  {label:<7} n={int(m.sum()):>5} released_in_side={released_in_side[m].mean():.3f}"
              f" gem_margin_med={np.median(gem_margin[m]):.3f}"
              f" IoU(chosen,released)_med={np.median(iou_cr[m]):.3f}", flush=True)
    print(f"  hurts with released NOT in side set: {int((hurt & ~released_in_side).sum())}/{int(hurt.sum())}",
          flush=True)

    # ---------------- P3b precision levers -----------------------------------
    print("\n=== P3b precision levers ===", flush=True)

    def variant(mode):
        out = np.empty(n, int)
        for i, r in enumerate(recs):
            if r.rel is None or r.rel["ordinal"]:
                out[i] = r.final
                continue
            c = K8[i]
            sub = c[side_mask(r, c, r.rel)]
            if not len(sub):
                out[i] = r.final
                continue
            g = r.S_gem[sub]
            winner = int(sub[int(np.argmax(g))])
            rel_in = released_in_side[i]
            if mode == "gem":
                out[i] = winner
            elif mode == "forced_only":
                out[i] = winner if not rel_in else r.final
            elif mode == "margin":
                ok = True
                if len(sub) > 1:
                    gg = np.sort(g)[::-1]
                    ok = (gg[0] - gg[1]) >= 0.05
                out[i] = winner if (ok or not rel_in) else r.final
            elif mode == "cat_not_worse":
                out[i] = (r.final if (rel_in and r.S["sc_content"][winner] < r.S["sc_content"][r.final])
                          else winner)
            elif mode == "gem_plus_cat":
                zc = (r.S["sc_content"][sub] - r.S["sc_content"][sub].mean()) / (r.S["sc_content"][sub].std() + 1e-9)
                zg = (g - g.mean()) / (g.std() + 1e-9)
                out[i] = int(sub[int(np.argmax(zg + 0.5 * zc))])
            else:
                raise ValueError(mode)
        return out

    for mode in ("gem", "forced_only", "margin", "cat_not_worse", "gem_plus_cat"):
        idxv = variant(mode)
        yr = np.array([gt[i][idxv[i]] for i in range(n)])
        ysnap = np.array([gt[i][ps[i][idxv[i]]] for i in range(n)])
        d, lo, hi, p = boot(yref, yr)
        ds, los, his, psx = boot(yref, ysnap)
        print(f"  {mode:<14} ours {yr.mean():.4f} ({d:+.4f} [{lo:+.4f},{hi:+.4f}])"
              f"  +snap {ysnap.mean():.4f} ({ds:+.4f} [{los:+.4f},{his:+.4f}])", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
