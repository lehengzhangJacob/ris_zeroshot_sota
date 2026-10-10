#!/usr/bin/env python
"""EXP-15 — Analyses requested in review round 1.

(a) Localise the reproduction gap. The published HybridGL ablation (Table 5) gives
    G2L alone = 31.71 oIoU / 40.27 mIoU and G2L + Rel + Gco + Gpos = 41.81 / 49.48
    on RefCOCO val, so the *guidance increment* is +10.10 oIoU / +9.21 mIoU. If our
    reproduction matches that increment but starts from a lower base, the shortfall
    is in the feature/score path, not in the guidance we re-implemented.

(b) Test whether the selector differs from the containment snap formally, rather
    than by comparing two point estimates: paired bootstrap on the per-expression
    difference of the two deltas.

(c) On RefCOCO+ (no location words by construction), check the residual spatial
    stratum — a neutral overall result is only informative if the operator still
    acts on the expressions that do carry a spatial word.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, boot, PART_WORDS, reltype  # noqa: E402
from method import Selector  # noqa: E402

PUBLISHED = dict(refcoco=dict(g2l=dict(oIoU=0.3171, mIoU=0.4027), full=dict(oIoU=0.4181, mIoU=0.4948)),
                 refcocog=dict(g2l=dict(oIoU=0.3912, mIoU=0.4864), full=dict(oIoU=0.4247, mIoU=0.5125)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("exp/out_final"))
    args = ap.parse_args()
    own = Selector()
    out = {}

    # ---------------- (a) reproduction gap localisation
    for tag, ds, pub in (("refcoco", "guided_val", "refcoco"), ("refcocog", "guided_refcocog", "refcocog")):
        s = json.loads((Path("exp/out_exp4") / f"summary_{ds}.json").read_text())
        p_g2l, p_full = PUBLISHED[pub]["g2l"], PUBLISHED[pub]["full"]
        out[tag] = dict(
            published=dict(g2l=p_g2l, full=p_full,
                           guidance_increment_oIoU=p_full["oIoU"] - p_g2l["oIoU"],
                           guidance_increment_mIoU=p_full["mIoU"] - p_g2l["mIoU"]),
            reproduced=dict(g2l=s["pure"], full=s["guided"],
                            guidance_increment_oIoU=s["guided"]["oIoU"] - s["pure"]["oIoU"],
                            guidance_increment_mIoU=s["guided"]["mIoU"] - s["pure"]["mIoU"]),
            base_gap_oIoU=s["pure"]["oIoU"] - p_g2l["oIoU"],
            base_gap_mIoU=s["pure"]["mIoU"] - p_g2l["mIoU"],
            increment_gap_oIoU=(s["guided"]["oIoU"] - s["pure"]["oIoU"]) - (p_full["oIoU"] - p_g2l["oIoU"]),
            increment_gap_mIoU=(s["guided"]["mIoU"] - s["pure"]["mIoU"]) - (p_full["mIoU"] - p_g2l["mIoU"]),
        )
    print("=== (a) reproduction-gap localisation ===")
    for tag, v in out.items():
        print(f"  [{tag}] published G2L {v['published']['g2l']['oIoU']*100:.2f}/"
              f"{v['published']['g2l']['mIoU']*100:.2f} -> full {v['published']['full']['oIoU']*100:.2f}/"
              f"{v['published']['full']['mIoU']*100:.2f}  (increment +{v['published']['guidance_increment_oIoU']*100:.2f}"
              f"/+{v['published']['guidance_increment_mIoU']*100:.2f})")
        print(f"  [{tag}] reproduced G2L {v['reproduced']['g2l']['oIoU']*100:.2f}/"
              f"{v['reproduced']['g2l']['mIoU']*100:.2f} -> full {v['reproduced']['full']['oIoU']*100:.2f}/"
              f"{v['reproduced']['full']['mIoU']*100:.2f}  (increment +{v['reproduced']['guidance_increment_oIoU']*100:.2f}"
              f"/+{v['reproduced']['guidance_increment_mIoU']*100:.2f})")
        print(f"  [{tag}] base gap {v['base_gap_oIoU']*100:+.2f}/{v['base_gap_mIoU']*100:+.2f}   "
              f"guidance-increment gap {v['increment_gap_oIoU']*100:+.2f}/{v['increment_gap_mIoU']*100:+.2f}")

    # ---------------- (b) does the selector differ from the snap?
    recs = [Rec(json.loads(l)) for l in (Path("exp/out_exp4") / "records_guided_val.jsonl").open()]
    pairs = np.load(Path("exp/out_exp4") / "pairs_guided_val.npz")
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    ref = np.array([r.gt[r.final] for r in recs])
    ours = np.array([r.gt[own(r)] for r in recs])
    snap = np.array([r.gt[int(r.ps[r.final])] for r in recs])
    d, lo, hi, p = boot(snap, ours)
    # where each operator wins
    o_win = (ours > snap) & (ours > ref)
    s_win = (snap > ours) & (snap > ref)
    o_win_snap_loss = (ours > ref) & (snap < ref)
    s_win_ours_loss = (snap > ref) & (ours < ref)
    out["selector_vs_snap"] = dict(
        delta=d, ci=[lo, hi], p=p,
        ours_better=int(o_win.sum()), snap_better=int(s_win.sum()),
        ours_gains_where_snap_hurts=int(o_win_snap_loss.sum()),
        snap_gains_where_ours_hurts=int(s_win_ours_loss.sum()),
        tie=int((ours == snap).sum()), n=len(recs))
    print("\n=== (b) selector vs containment snap (paired) ===")
    print(f"  ours - snap = {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
    print(f"  ours strictly better on {o_win.sum()}, snap strictly better on {s_win.sum()}, ties {int((ours==snap).sum())}")
    print(f"  ours gains where snap hurts: {o_win_snap_loss.sum()}; snap gains where ours hurts: {s_win_ours_loss.sum()}")
    po = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & PART_WORDS) for r in recs])
    print(f"  part expressions: ours Δ {(ours-ref)[po].mean():+.4f} vs snap Δ {(snap-ref)[po].mean():+.4f}")

    # ---------------- (c) RefCOCO+ residual spatial stratum
    p_path = Path("exp/out_exp4/records_guided_refcocoplus.jsonl")
    if p_path.exists():
        import subprocess
        recs2 = [Rec(json.loads(l)) for l in p_path.open()]
        pairs2 = np.load(Path("exp/out_exp4") / "pairs_guided_refcocoplus.npz")
        for r in recs2:
            r.pc = pairs2[str(r.r["ref_id"])].astype(np.float32)
        ref2 = np.array([r.gt[r.final] for r in recs2])
        ours2 = np.array([r.gt[own(r)] for r in recs2])
        snap2 = np.array([r.gt[int(r.ps[r.final])] for r in recs2])
        rt = np.array([reltype(r.rel) for r in recs2])
        d2, lo2, hi2, p2 = boot(ref2, ours2)
        sp = rt != "none"
        out["refcocoplus"] = dict(
            n=len(recs2), spatial_frac=float(sp.mean()),
            released_mIoU=float(ref2.mean()), ours_mIoU=float(ours2.mean()),
            ours_delta=d2, ci=[lo2, hi2], p=p2,
            snap_delta=float(snap2.mean() - ref2.mean()),
            ours_delta_spatial=float((ours2 - ref2)[sp].mean()) if sp.sum() else None,
            n_spatial=int(sp.sum()),
            ours_delta_spatial_h=float((ours2 - ref2)[rt == "horizontal"].mean())
            if (rt == "horizontal").sum() else None,
            n_horizontal=int((rt == "horizontal").sum()))
        print("\n=== (c) RefCOCO+ ===")
        print(json.dumps(out["refcocoplus"], indent=2))
    else:
        print("\n(c) RefCOCO+ records not present yet")

    # ---------------- (d) candidate-set ceilings
    # Bounds what ANY chooser -- including an MLLM over the same set -- could
    # achieve, so the "why not an MLLM" question is bounded rather than open.
    ceil = {}
    for nm, fn in (
        ("released_top3 (what the guidance may choose from)",
         lambda r: [int(i) for i in r.top]),
        ("method candidate set (top-8 category score)",
         lambda r: np.argsort(-r.S["sc_content"], kind="stable")[:8].tolist()),
        ("method set, relation-consistent half",
         lambda r: np.argsort(-r.S["sc_content"], kind="stable")[:8].tolist()),
        ("whole pool (algorithmic ceiling)", lambda r: list(range(len(r.gt)))),
    ):
        vals = []
        for r in recs:
            cand = fn(r)
            vals.append(max(r.gt[c] for c in cand))
        ceil[nm] = float(np.mean(vals))
    ceil["released_achieved"] = float(ref.mean())
    ceil["ours_achieved"] = float(ours.mean())
    ceil["assumption"] = ("relation-consistent-half ceiling uses the same set as "
                          "the candidate set; the side filter is only meaningful per expression")
    out["ceilings"] = ceil
    print("\n=== (d) candidate-set ceilings (max achievable mIoU with a perfect chooser) ===")
    for k, v in ceil.items():
        print(f"  {k:<56}{v:+.4f}" if isinstance(v, float) and k != "assumption" else f"  {k}: {v}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "review_fixes.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
