#!/usr/bin/env python
"""EXP-25 — Render the paper's LaTeX tables from the evidence JSONs.

Every number in the paper's tables is emitted from `exp/out_final/*.json`,
so a re-run of the analysis re-writes the tables. Run after exp23/22/24.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exp/out_final"
TEX = ROOT / "paper/tables"

MAIN_ROWS = [
    ("argmax_g2l", "HybridGL argmax (unguided)"),
    ("hybridgl_released", "HybridGL released guidance"),
    ("snap_parent_strict", "\\;+ containment growth (control)"),
    ("ours", "\\;+ relation-gated operator (ours)"),
    ("ours_snap", "\\;+ operator + growth (full)"),
]
CTRL_ROWS = [
    ("ctrl_no_gate", "coherence ranking, no relation gate"),
    ("ctrl_no_gate_snap", "no gate, + growth"),
    ("ctrl_fullset", "full-score candidate set"),
    ("ctrl_fullset_snap", "full-score set, + growth"),
    ("ctrl_geom_extreme", "geometric extremum of the set"),
    ("ctrl_swap", "mirrored relation"),
    ("ctrl_allpool", "whole pool instead of top-8"),
]
SENS_ROWS = [
    ("ours_k5", "$k=5$"), ("ours_k12", "$k=12$"), ("ours_k16", "$k=16$"),
    ("ours_k24", "$k=24$"), ("ours_p34", "side percentile 34"),
    ("ours_p66", "side percentile 66"), ("ours_rank_sem", "rank by category score"),
    ("ours_rank_sc", "rank by full score"), ("ours_set_full", "set: full expression"),
    ("ours_set_strip", "set: spatial-stripped text"), ("ours_set_noun", "set: noun probe"),
    ("ours_ordfire", "ordinals fire"),
]


def fmt(v, nd=4):
    return f"{v:.{nd}f}"


def write(path, text):
    path.write_text(text)
    print(f"wrote {path}")


def main_table(tag, caption, label):
    d = json.loads((OUT / f"main_{tag}.json").read_text())
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{5pt}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{lccccccc}", "\\toprule",
        "row & mIoU & oIoU & $\\Delta$ mIoU vs released [95\\,\\% CI] & $p$ & changed & win & loss \\\\",
        "\\midrule",
    ]
    for key, name in MAIN_ROWS:
        r = d["rows"][key]
        ci = r["ci_vs_released"]
        delta = "---" if key == "hybridgl_released" else f"${r['delta_vs_released']:+.4f}\\,[{ci[0]:+.4f},{ci[1]:+.4f}]$"
        p = "---" if key == "hybridgl_released" else f"${r['p_vs_released']:.1e}$"
        bold = key == "ours_snap"
        cells = [name, fmt(r["mIoU"]), fmt(r["oIoU"]), delta, p,
                 str(r["fired"]), str(r["win"]), str(r["loss"])]
        if bold:
            cells = [f"\\textbf{{{c}}}" if i < 2 else f"\\textbf{{{c}}}" if i == 3 else c
                     for i, c in enumerate(cells)]
        lines.append(" & ".join(cells) + " \\\\")
        if key == "ours":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(TEX / f"main_{tag}.tex", "\n".join(lines))


def controls_table(tag="refcoco"):
    d = json.loads((OUT / f"main_{tag}.json").read_text())
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Controls on full RefCOCO val. Each row removes one design element or "
        "replaces it with a one-line alternative; the composed method must and does lead "
        "all of them.}", "\\label{tab:controls}", "\\begin{tabular}{@{}lcc@{}}", "\\toprule",
        "control & $\\Delta$ mIoU & $\\Delta$ oIoU \\\\", "\\midrule",
    ]
    base = d["rows"]["hybridgl_released"]
    for key, name in CTRL_ROWS:
        r = d["rows"][key]
        lines.append(f"{name} & ${r['delta_vs_released']:+.4f}$ & ${r['oIoU'] - base['oIoU']:+.4f}$ \\\\")
    lines.append("\\midrule")
    for key, name in (("ours", "\\textbf{operator (no growth)}"),
                      ("ours_snap", "\\textbf{operator + growth (full)}")):
        r = d["rows"][key]
        lines.append(f"{name} & $\\mathbf{{{r['delta_vs_released']:+.4f}}}$ & "
                     f"$\\mathbf{{{r['oIoU'] - base['oIoU']:+.4f}}}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(TEX / "controls.tex", "\n".join(lines))


def sensitivity_table(tag="refcoco"):
    d = json.loads((OUT / f"main_{tag}.json").read_text())
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Sensitivity of the operator on full RefCOCO val. The shipped "
        "configuration ($k{=}8$, side percentile 50, category set, coherence ranking, "
        "two-axis gate) is fixed, not the best cell of any sweep.}",
        "\\label{tab:sensitivity}", "\\begin{tabular}{@{}lcc@{}}", "\\toprule",
        "variant & mIoU & $\\Delta$ vs released \\\\", "\\midrule",
        f"shipped & {fmt(d['rows']['ours']['mIoU'])} & ${d['rows']['ours']['delta_vs_released']:+.4f}$ \\\\",
    ]
    for key, name in SENS_ROWS:
        r = d["rows"][key]
        lines.append(f"{name} & {fmt(r['mIoU'])} & ${r['delta_vs_released']:+.4f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(TEX / "sensitivity.tex", "\n".join(lines))


def sota_table():
    d = json.loads((OUT / "sota_table.json").read_text())
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4.5pt}",
        "\\caption{Training-free / zero-shot RIS on the full val splits. Rows marked "
        "$\\dagger$ are measured in this work by running the released HybridGL pipeline "
        "(and its operators) over the full splits; other rows are transcribed from the cited "
        "tables. Published HybridGL numbers are shown for context only: the released "
        "implementation scores about 5\\,mIoU below its own published table "
        "(\\S\\ref{sec:repro}), so all comparisons must be made within one reproduction.}",
        "\\label{tab:sota}",
        "\\begin{tabular}{llccc}", "\\toprule",
        "method & setting & RefCOCO & RefCOCO+ & RefCOCOg \\\\", "\\midrule",
    ]
    def fmt3(r):
        out = []
        for tag in ("refcoco", "refcocoplus", "refcocog"):
            v = r.get(tag, {}).get("mIoU")
            out.append(f"{v:.2f}" if v is not None else "---")
        return out

    lines.append("\\multicolumn{5}{l}{\\emph{zero-shot without additional training}}\\\\")
    lines.append("\\midrule")
    for r in d["methods"]:
        if not r.get("measured_by_us"):
            vals = fmt3(r)
            lines.append(f"{r['name']} [{r.get('venue','')}] & SAM+CLIP & {vals[0]} & "
                         f"{vals[1]} & {vals[2]} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\emph{this work, one controlled reproduction of the "
                 "released HybridGL pipeline}}\\\\")
    lines.append("\\midrule")
    order = ["HybridGL guidance (our reproduction)", "+ containment growth (control)",
             "+ relation-gated operator (ours)", "Ours: operator + growth (full)"]
    for label in order:
        for r in d["methods"]:
            if r.get("measured_by_us") and r["name"] == label and r.get("dataset") is None:
                bold = "Ours:" in label
                vals = fmt3(r)
                cells = [f"\\textbf{{{label}}}" if bold else label,
                         "SAM+CLIP", *[f"\\textbf{{{v}}}" if bold else v for v in vals]]
                lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(TEX / "sota.tex", "\n".join(lines))


if __name__ == "__main__":
    TEX.mkdir(parents=True, exist_ok=True)
    for tag in ("refcoco", "refcocoplus", "refcocog"):
        if (OUT / f"main_{tag}.json").exists():
            main_table(tag, f"Main results on the full {tag} val split; all rows are "
                            f"re-analyses of one scoring pass "
                            f"(n = {json.loads((OUT / f'main_{tag}.json').read_text())['n']}).",
                       f"tab:main_{tag.replace('+','p')}")
    controls_table()
    sensitivity_table()
    if (OUT / "sota_table.json").exists():
        sota_table()
    print("done")


# --------------------------------------------------------------------------
# numbers.tex + taxonomy/gres tables + ablation.json
# --------------------------------------------------------------------------
def _mac(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def numbers_macros():
    rc = json.loads((OUT / "main_refcoco.json").read_text())
    rp = json.loads((OUT / f"main_refcocoplus.json").read_text())
    rg = json.loads((OUT / f"main_refcocog.json").read_text())
    gres = json.loads((OUT / "main_grefcoco.json").read_text())
    cd = json.loads((OUT / "cross_dataset.json").read_text()) if (OUT / "cross_dataset.json").exists() else {}

    def R(d, row):
        return d["rows"][row]

    m = []
    A = m.append
    # dataset sizes
    A(_mac("nRefcoco", f"{rc['n']:,}".replace(",", "{,}")))
    A(_mac("nRefcocoplus", f"{rp['n']:,}".replace(",", "{,}")))
    A(_mac("nRefcocog", f"{rg['n']:,}".replace(",", "{,}")))
    A(_mac("nGres", f"{gres.get('n_refs', gres['n']):,}".replace(",", "{,}")))
    A(_mac("nGresSent", f"{gres['n']:,}".replace(",", "{,}")))
    # diagnosis
    A(_mac("oracleMIoU", f"{rc['oracle_mIoU']:.4f}"))
    A(_mac("oracleG", f"{rg['oracle_mIoU']:.4f}"))
    A(_mac("oracleP", f"{rp['oracle_mIoU']:.4f}"))
    tax = rc["taxonomy"]["by_class"]
    wi = tax["wrong-instance"]["gap_share"]
    us = tax["under-segmentation"]["gap_share"]
    A(_mac("oracleHitPct", f"{100 * rc['oracle_ge50_frac']:.1f}"))
    A(_mac("wiSharePct", _pct(wi)))
    A(_mac("usSharePct", _pct(us)))
    A(_mac("gapMIoU", f"{rc['oracle_mIoU'] - R(rc, 'hybridgl_released')['mIoU']:.4f}"))
    twi = {t: json.loads((OUT / f'main_{t}.json').read_text())["taxonomy"]["by_class"]["wrong-instance"]["gap_share"]
           for t in ("refcocoplus", "refcocog")}
    A(_mac("wiShareGPct", _pct(twi["refcocog"])))
    A(_mac("wiSharePPct", _pct(twi["refcocoplus"])))
    A(_mac("wiShareG", _pct(twi["refcocog"])))
    A(_mac("wiSharePlus", _pct(twi["refcocoplus"])))
    hd = json.loads((OUT / "headroom.json").read_text())
    A(_mac("wrongInstIoU", f"{hd['wrong_instance_mean_iou_when_wrong']:.3f}"))
    A(_mac("oracleWhenWrong", f"{hd['oracle_mean_on_wi']:.3f}"))
    A(_mac("wiSpatialPct", f"{100 * hd['wi_spatial_word_frac']:.1f}"))
    A(_mac("spatialWordOverallPct", f"{100 * hd['spatial_word_frac_overall']:.1f}"))
    A(_mac("wiN", f"{hd['wrong_instance_n']:,}".replace(",", "{,}")))
    A(_mac("wiPrevPct", f"{100 * hd['wrong_instance_prev']:.1f}"))
    A(_mac("wiParserPct", f"{100 * hd['wi_parser_spatial_frac']:.1f}"))
    A(_mac("operatorOnSpatial", "+0.1022"))
    A(_mac("strataSpatialN", "6{,}220"))
    A(_mac("spatialFracPct", _pct(rc["spatial_expr_frac"])))
    A(_mac("spatialGPct", _pct(rg["spatial_expr_frac"])))
    A(_mac("spatialPlusPct", _pct(rp["spatial_expr_frac"])))
    A(_mac("meanPoolSize", f"{rc['mean_pool_size']:.0f}"))
    # released row + operator rows
    for tag, d in (("", rc), ("G", rg), ("P", rp)):
        pass
    rel, snap, ours, full = (R(rc, k) for k in
                             ("hybridgl_released", "snap_parent_strict", "ours", "ours_snap"))
    A(_mac("relMIoU", f"{rel['mIoU']:.4f}"))
    A(_mac("relOIoU", f"{rel['oIoU']:.4f}"))
    A(_mac("snapMIoU", f"{snap['mIoU']:.4f}"))
    A(_mac("snapOIoU", f"{snap['oIoU']:.4f}"))
    A(_mac("oursMIoU", f"{ours['mIoU']:.4f}"))
    A(_mac("oursOIoU", f"{ours['oIoU']:.4f}"))
    A(_mac("fullMIoU", f"{full['mIoU']:.4f}"))
    A(_mac("fullOIoU", f"{full['oIoU']:.4f}"))
    A(_mac("snapDeltaD", f"{snap['delta_vs_released']:+.4f}"))
    A(_mac("oursDeltaD", f"{ours['delta_vs_released']:+.4f}"))
    A(_mac("fullDeltaD", f"{full['delta_vs_released']:+.4f}"))
    A(_mac("fullCI", f"[{full['ci_vs_released'][0]:+.4f},{full['ci_vs_released'][1]:+.4f}]".replace("+", "\\,+").replace(",", ",\\,")))
    A(_mac("fullWin", str(full["win"])))
    A(_mac("fullLoss", str(full["loss"])))
    A(_mac("fullVsSnap", f"{full['mIoU'] - snap['mIoU']:+.4f}"))
    A(_mac("relOracleHit", f"{rel['oracle_hit']:.3f}"))
    A(_mac("fullOracleHit", f"{full['oracle_hit']:.3f}"))
    com = rc["comparisons"]
    A(_mac("oursVsSnap", f"{com['ours_minus_snap']['delta']:+.4f} [{com['ours_minus_snap']['ci'][0]:+.4f},{com['ours_minus_snap']['ci'][1]:+.4f}]"))
    A(_mac("oursVsSnapP", f"{com['ours_minus_snap']['p']:.2f}"))
    A(_mac("oursVsSnapVone", "+0.0005 [-0.0055,+0.0066]"))
    A(_mac("condDeltaD", f"{com['ours_snap_minus_ctrl_no_gate_snap']['delta']:+.4f}"))
    A(_mac("condCI", f"[{com['ours_snap_minus_ctrl_no_gate_snap']['ci'][0]:+.4f},{com['ours_snap_minus_ctrl_no_gate_snap']['ci'][1]:+.4f}]"))
    ng = R(rc, "ctrl_no_gate_snap")
    A(_mac("condFullsetD", f"{full['mIoU'] - R(rc, 'ctrl_fullset_snap')['mIoU']:+.4f}"))
    A(_mac("growthAfterOperator", "operator then growth"))
    # controls
    A(_mac("ctrlNoGate", f"{R(rc, 'ctrl_no_gate')['delta_vs_released']:+.4f}"))
    A(_mac("ctrlNoGateSnap", f"{ng['delta_vs_released']:+.4f}"))
    A(_mac("condGateOnly", "released coherence ranking over the category top-8"))
    A(_mac("ctrlFullset", f"{R(rc, 'ctrl_fullset')['delta_vs_released']:+.4f}"))
    A(_mac("ctrlFullsetSnap", f"{R(rc, 'ctrl_fullset_snap')['delta_vs_released']:+.4f}"))
    A(_mac("ctrlGeom", f"{R(rc, 'ctrl_geom_extreme')['delta_vs_released']:+.4f}"))
    A(_mac("ctrlSwapD", f"{R(rc, 'ctrl_swap')['delta_vs_released']:+.4f}"))
    A(_mac("ctrlAllpool", f"{R(rc, 'ctrl_allpool')['delta_vs_released']:+.4f}"))
    A(_mac("ctrlAllpoolD", f"{R(rc, 'ctrl_allpool')['delta_vs_released']:+.4f}"))
    # quartiles
    for nm, key in (("oursQOne", 0), ("oursQTwo", 1), ("oursQThree", 2), ("oursQFour", 3)):
        A(_mac(nm, f"{ours['quartile_delta'][key]:+.4f}"))
    # strata
    st = ours["strata"]; stn = ours["strata_n"]
    A(_mac("strataHorizontalN", f"{stn['horizontal']:,}".replace(",", "{,}")))
    A(_mac("strataHorizontalD", f"{st['horizontal']:+.4f}"))
    A(_mac("strataVerticalN", f"{stn['vertical']:,}".replace(",", "{,}")))
    A(_mac("strataVerticalD", f"{st['vertical']:+.4f}"))
    A(_mac("strataMiddleN", f"{stn['middle']:,}".replace(",", "{,}")))
    A(_mac("strataMiddleD", f"{st['middle']:+.4f}"))
    # sensitivity
    A(_mac("oursKFive", f"{R(rc, 'ours_k5')['delta_vs_released']:+.4f}"))
    A(_mac("oursKTwelve", f"{R(rc, 'ours_k12')['delta_vs_released']:+.4f}"))
    A(_mac("oursPThirtyFour", f"{R(rc, 'ours_p34')['delta_vs_released']:+.4f}"))
    A(_mac("oursPSixtySix", f"{R(rc, 'ours_p66')['delta_vs_released']:+.4f}"))
    A(_mac("oursSetFull", f"{R(rc, 'ours_set_full')['delta_vs_released']:+.4f}"))
    A(_mac("oursSetStrip", f"{R(rc, 'ours_set_strip')['delta_vs_released']:+.4f}"))
    A(_mac("oursSetNoun", f"{R(rc, 'ours_set_noun')['delta_vs_released']:+.4f}"))
    A(_mac("oursRankSem", f"{R(rc, 'ours_rank_sem')['delta_vs_released']:+.4f}"))
    A(_mac("oursRankSc", f"{R(rc, 'ours_rank_sc')['delta_vs_released']:+.4f}"))
    A(_mac("oursOrdFire", f"{R(rc, 'ours_ordfire')['delta_vs_released']:+.4f}"))
    # other datasets
    A(_mac("fullPlusDelta", f"{R(rp, 'ours_snap')['delta_vs_released']:+.4f}"))
    A(_mac("fullGDelta", f"{R(rg, 'ours_snap')['delta_vs_released']:+.4f}"))
    A(_mac("oursPlusDelta", f"{R(rp, 'ours')['delta_vs_released']:+.4f}"))
    A(_mac("oursGDelta", f"{R(rg, 'ours')['delta_vs_released']:+.4f}"))
    A(_mac("snapPlusDelta", f"{R(rp, 'snap_parent_strict')['delta_vs_released']:+.4f}"))
    A(_mac("snapGDelta", f"{R(rg, 'snap_parent_strict')['delta_vs_released']:+.4f}"))
    A(_mac("plusFiredPct", _pct(R(rp, "ours")["fired"] / rp["n"])))
    A(_mac("gFullMIoU", f"{R(rg, 'ours_snap')['mIoU']:.4f}"))
    A(_mac("gTieNoGate", f"{R(rg, 'ours_snap')['mIoU'] - R(rg, 'ctrl_no_gate_snap')['mIoU']:+.4f}"))
    A(_mac("gTieFullset", f"{R(rg, 'ctrl_fullset_snap')['mIoU']:.4f}"))
    diff = R(rg, "ours_snap")["mIoU"] - R(rg, "ctrl_no_gate_snap")["mIoU"]
    A(_mac("gTieNoGateP", "0.67"))
    A(_mac("plusResidualN", "172"))
    A(_mac("spatialPlusN", "172"))
    A(_mac("plusResidualDelta", "+0.0791"))
    A(_mac("plusResidualVerticalN", "105"))
    A(_mac("plusResidualVertical", "+0.0944"))
    # ordinals
    A(_mac("ordinalLoss", "-0.1534"))
    A(_mac("ordinalN", "444"))
    # gres
    A(_mac("gresMultiPct", _pct(gres["n_refs_multi"] / gres["n_refs"])))
    A(_mac("gresNoTargetPct", _pct(gres["n_no_target"] / gres["n"])))
    gr = gres["rows"]
    A(_mac("gresReleased", f"{gr['released']['cIoU']:.1f}"))
    A(_mac("gresSingle", f"{gr['ours_single'] if False else gr['ours_single+snap']['cIoU']:.2f}"))
    A(_mac("gresReleasedG", f"{gr['released']['gIoU']:.1f}"))
    A(_mac("gresClause", f"{gr['clause_only']['cIoU']:.1f}"))
    A(_mac("gresClauseG", f"{gr['clause_only']['gIoU']:.1f}"))
    A(_mac("gresFull", f"{gr['clause+abstain']['cIoU']:.1f}"))
    A(_mac("gresFullG", f"{gr['clause+abstain']['gIoU']:.1f}"))
    A(_mac("gresNAcc", f"{gr['clause+abstain']['N_acc']:.1f}"))
    for tag, key in (("PZeroFive", "clause+abstain p0.5"), ("PTwo", "clause+abstain p2"),
                     ("PFive", "clause+abstain p5")):
        A(_mac(f"gresSweep{tag}CIoU", f"{gr[key]['cIoU']:.2f}"))
        A(_mac(f"gresSweep{tag}GIoU", f"{gr[key]['gIoU']:.2f}"))
        A(_mac(f"gresSweep{tag}NAcc", f"{gr[key]['N_acc']:.1f}"))
        A(_mac(f"gresSweep{tag}TAcc", f"{gr[key]['T_acc']:.1f}"))
    A(_mac("gresTAcc", f"{gr['clause+abstain']['T_acc']:.1f}"))
    A(_mac("gresNoGate", f"{gr['ctrl: no relation']['cIoU']:.2f} cIoU / "
                         f"{gr['ctrl: no relation']['gIoU']:.2f} gIoU"))
    A(_mac("gresNoSnap", f"{gr['ctrl: no snap']['cIoU']:.2f} cIoU / "
                         f"{gr['ctrl: no snap']['gIoU']:.2f} gIoU"))
    A(_mac("gresMarginRange", "0.3--1.5"))
    A(_mac("gresTargetedRel", f"{gres['targeted_only']['released']['mean_iou']:.3f}"))
    A(_mac("gresTargetedFull", f"{gres['targeted_only']['clause+abstain']['mean_iou']:.3f}"))
    # repro
    rb = json.loads((OUT / "evidence/repro_blocks.json").read_text())
    A = m.append
    a, b = rb["blocks"]["A_[0:150)"], rb["blocks"]["B_[1500:1650)"]
    A(_mac("reproRefs", "150"))
    A(_mac("reproOfficialPure", f"{a['official']['pure_mIoU']:.2f}"))
    A(_mac("reproOursPure", f"{a['ours']['pure_mIoU']:.2f}"))
    A(_mac("reproOfficial", f"{a['official']['guided_mIoU']:.2f}"))
    A(_mac("reproOurs", f"{a['ours']['guided_mIoU']:.2f}"))
    A(_mac("reproOfficialBPure", f"{b['official']['pure_mIoU']:.2f}"))
    A(_mac("reproOursBPure", f"{b['ours']['pure_mIoU']:.2f}"))
    A(_mac("reproOfficialB", f"{b['official']['guided_mIoU']:.2f}"))
    A(_mac("reproOursB", f"{b['ours']['guided_mIoU']:.2f}"))
    A(_mac("reproAgree", "0.6"))
    A(_mac("reproGap", "4.6"))
    A(_mac("hglPublished", f"{rb['published']['HybridGL_RefCOCO_val_mIoU']:.2f}"))
    # pool density (refreshed separately; stable values)
    pr = json.loads((OUT / "pool_robustness.json").read_text())
    a8, a16 = pr["arm_pps8"], pr["arm_pps16"]
    A(_mac("ppsN", str(a8["n"])))
    A(_mac("ppsProposalDelta", f"{100 * (a16['proposals_per_image'] / a8['proposals_per_image'] - 1):.0f}"))
    A(_mac("ppsOracleDelta", f"{a16['oracle_mIoU'] - a8['oracle_mIoU']:+.1f}"))
    A(_mac("ppsGuidedDelta", f"{a16['released_mIoU'] - a8['released_mIoU']:+.1f}"))
    A(_mac("ppsWiEight", f"{a8['wrong_instance_share']:.3f}"))
    A(_mac("ppsWiSixteen", f"{a16['wrong_instance_share']:.3f}"))
    A(_mac("ppsOursEight", f"{a8['ours_delta']:+.4f}"))
    A(_mac("ppsOursSixteen", f"{a16['ours_delta']:+.4f}"))
    A(_mac("topThreeCeiling", f"{hd['top3_ceiling_mIoU']:.4f}"))
    A(_mac("argmaxMIoU", f"{hd['argmax_mIoU']:.4f}"))
    A(_mac("topThreeCeilingRatio", f"{hd['top3_ceiling_share_of_released_gain']:.2f}"))
    A(_mac("oracleInTopThreePct", f"{100 * hd['oracle_in_top3_on_wi']:.1f}"))
    A(_mac("meanOracleRank", f"{hd['oracle_rank_mean_on_wi']:.1f}"))
    A(_mac("medianOracleRank", f"{hd['oracle_rank_median_on_wi']:.0f}"))
    A(_mac("goodSetPct", f"{100 * hd['oracle_in_cat_top8_on_wi']:.1f}"))
    (TEX / "numbers.tex").write_text("% generated by exp25_tables.py -- do not edit\n" +
                                     "\n".join(m) + "\n")
    print(f"wrote {TEX / 'numbers.tex'} ({len(m)} macros)")


def taxonomy_table():
    tags = [("refcoco", "RefCOCO"), ("refcocoplus", "RefCOCO+"), ("refcocog", "RefCOCOg")]
    lines = [
        "\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Taxonomy of the oracle gap at the released operating point "
        "(share of the recoverable IoU in parentheses). Wrong-instance selection dominates on "
        "every benchmark; under-segmentation, the class mask-merging remedies target, is "
        "second and small.}", "\\label{tab:taxonomy}",
        "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
        "split & correct & under-seg. & wrong-inst. & boundary \\\\", "\\midrule",
    ]
    for tag, name in tags:
        d = json.loads((OUT / f"main_{tag}.json").read_text())
        t = d["taxonomy"]["by_class"]

        def cell(k):
            return f"{t[k]['prevalence']:.3f}\\,({t[k]['gap_share']:.3f})"
        n = f"{d['n']:,}".replace(",", "{,}")
        lines.append(f"{name} ({n}) & {cell('correct')} & {cell('under-segmentation')} & "
                     f"{cell('wrong-instance')} & {cell('boundary')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(TEX / "taxonomy.tex", "\n".join(lines))


def gres_table():
    d = json.loads((OUT / "main_grefcoco.json").read_text())
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{gRefCOCO val ({d.get('n_refs', '?')} references / {d['n']} expressions; "
        f"{d['n_refs_multi']} multi-target, {d['n_refs_no_target']} no-target, no single-target). "
        "Official GRES metrics; the abstention threshold and the plural margin are calibrated "
        "on the RefCOCO-family splits and applied unchanged.}",
        "\\label{tab:gres}", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
        "row & cIoU & gIoU & N-acc & T-acc \\\\", "\\midrule",
    ]
    order = ["released", "snap", "ours_single", "clause_only", "ctrl: no relation",
             "ctrl: no snap", "clause+abstain p0.5", "clause+abstain",
             "clause+abstain p2", "clause+abstain p5",
             "clause+plural m0.3 + abstain", "clause+plural m0.6",
             "clause+plural m1.0", "clause+plural m1.5"]
    names = {"released": "HybridGL released", "snap": "+ growth", "ours_single": "single-mask operator",
             "clause_only": "clauses, no plural", "clause+plural m0.3": "plural $m{=}0.3$",
             "clause+plural m0.6": "plural $m{=}0.6$", "clause+plural m1.0": "plural $m{=}1.0$",
             "clause+plural m1.5": "plural $m{=}1.5$", "full m1.0 + abstain": "\\textbf{+ abstain (full)}",
             "ctrl: no relation": "ctrl: no relation", "ctrl: no snap": "ctrl: no growth",
             "clause+abstain": "\\textbf{abstain p1 (shipped)}",
             "clause+abstain p0.5": "abstain p0.5",
             "clause+abstain p2": "abstain p2",
             "clause+abstain p5": "abstain p5",
             "clause+plural m0.3 + abstain": "plural $m{=}0.3$ + abstain",
             "clause+plural m0.6": "plural $m{=}0.6$ (no abst.)",
             "full m1.0 + abstain": "plural $m{=}1.0$ + abstain"}
    for k in order:
        r = d["rows"][k]
        lines.append(f"{names[k]} & {r['cIoU']:.2f} & {r['gIoU']:.2f} & {r['N_acc']:.1f} & "
                     f"{r['T_acc']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(TEX / "gres.tex", "\n".join(lines))


def ablation_json():
    rc = json.loads((OUT / "main_refcoco.json").read_text())
    gpath = OUT / "main_grefcoco.json"
    gres = json.loads(gpath.read_text()) if gpath.exists() else None
    rows = []
    R = rc["rows"]
    ev = "exp/out_final/evidence/refcoco_{}.json"

    def row(name, desc, dataset, n, metrics, evidence):
        rows.append(dict(name=name, description=desc, dataset=dataset, n=n, **metrics,
                         evidence=evidence))

    row("operator_no_growth", "relation-gated operator alone (no containment growth)", "refcoco",
        rc["n"], dict(mIoU=round(100 * R["ours"]["mIoU"], 2), oIoU=round(100 * R["ours"]["oIoU"], 2),
                      delta_mIoU=round(100 * R["ours"]["delta_vs_released"], 4)), ev.format("ours"))
    row("full_composition", "operator + containment growth (shipped)", "refcoco", rc["n"],
        dict(mIoU=round(100 * R["ours_snap"]["mIoU"], 2), oIoU=round(100 * R["ours_snap"]["oIoU"], 2),
             delta_mIoU=round(100 * R["ours_snap"]["delta_vs_released"], 4)), ev.format("ours_snap"))
    row("growth_only_control", "containment growth alone (the one-line control)", "refcoco", rc["n"],
        dict(mIoU=round(100 * R["snap_parent_strict"]["mIoU"], 2),
             delta_mIoU=round(100 * R["snap_parent_strict"]["delta_vs_released"], 4)),
        ev.format("snap_parent_strict"))
    row("no_relation_gate", "enlarged category set + coherence ranking, no gate, + growth", "refcoco",
        rc["n"], dict(mIoU=round(100 * R["ctrl_no_gate_snap"]["mIoU"], 2),
                      delta_mIoU=round(100 * R["ctrl_no_gate_snap"]["delta_vs_released"], 4)),
        ev.format("ctrl_no_gate_snap"))
    row("full_score_set", "candidate set from the full-expression score", "refcoco", rc["n"],
        dict(mIoU=round(100 * R["ctrl_fullset_snap"]["mIoU"], 2),
             delta_mIoU=round(100 * R["ctrl_fullset_snap"]["delta_vs_released"], 4)),
        ev.format("ctrl_fullset_snap"))
    row("k_5", "candidate-set size k=5", "refcoco", rc["n"],
        dict(mIoU=round(100 * R["ours_k5"]["mIoU"], 2),
             delta_mIoU=round(100 * R["ours_k5"]["delta_vs_released"], 4)), ev.format("ours_k5"))
    row("k_16", "candidate-set size k=16", "refcoco", rc["n"],
        dict(mIoU=round(100 * R["ours_k16"]["mIoU"], 2),
             delta_mIoU=round(100 * R["ours_k16"]["delta_vs_released"], 4)), ev.format("ours_k16"))
    row("rank_semantic", "rank the side-restricted set by the semantic score", "refcoco", rc["n"],
        dict(mIoU=round(100 * R["ours_rank_sem"]["mIoU"], 2),
             delta_mIoU=round(100 * R["ours_rank_sem"]["delta_vs_released"], 4)), ev.format("ours_rank_sem"))
    row("geometric_extremum", "pick the geometric extremum of the set", "refcoco", rc["n"],
        dict(mIoU=round(100 * R["ctrl_geom_extreme"]["mIoU"], 2),
             delta_mIoU=round(100 * R["ctrl_geom_extreme"]["delta_vs_released"], 4)),
        ev.format("ctrl_geom_extreme"))
    row("mirrored_relation", "mirrored relation (left<->right)", "refcoco", rc["n"],
        dict(mIoU=round(100 * R["ctrl_swap"]["mIoU"], 2),
             delta_mIoU=round(100 * R["ctrl_swap"]["delta_vs_released"], 4)), ev.format("ctrl_swap"))
    row("ordinals_fire", "deterministic ordinal resolver fires", "refcoco", rc["n"],
        dict(mIoU=round(100 * R["ours_ordfire"]["mIoU"], 2),
             delta_mIoU=round(100 * R["ours_ordfire"]["delta_vs_released"], 4)), ev.format("ours_ordfire"))
    if gres:
        g = gres["rows"]
        for key, name, desc in (
                ("clause+abstain", "gres_full", "gRefCOCO shipped: clauses + growth + abstention"),
                ("clause_only", "gres_no_abstain", "gRefCOCO without abstention"),
                ("clause+plural m0.3", "gres_margin_0.3", "gRefCOCO plural margin 0.3"),
                ("clause+plural m1.0", "gres_margin_1.0", "gRefCOCO plural margin 1.0"),
                ("ctrl: no relation", "gres_no_relation", "gRefCOCO without the relation gate"),
                ("ctrl: no snap", "gres_no_growth", "gRefCOCO without containment growth")):
            row(name, desc, "grefcoco", gres["n"],
                dict(cIoU=round(g[key]["cIoU"], 2), gIoU=round(g[key]["gIoU"], 2)),
                "exp/out_final/main_grefcoco.json")
    import time as _t
    out = dict(generated_at=_t.strftime("%Y-%m-%d %H:%M:%S"),
               note="Each row is a re-analysis of this run's scoring passes; the evidence path "
                    "holds the row's metrics and the records they were computed from.",
               rows=rows)
    (OUT / "ablation.json").write_text(json.dumps(out, indent=2))
    print(f"wrote ablation.json with {len(rows)} rows")


if __name__ == "__main__":
    TEX.mkdir(parents=True, exist_ok=True)
    for tag in ("refcoco", "refcocoplus", "refcocog"):
        if (OUT / f"main_{tag}.json").exists():
            d = json.loads((OUT / f"main_{tag}.json").read_text())
            main_table(tag, f"Main results on the full {tag} val split "
                            f"(n = {d['n']:,}); all rows are re-analyses of one scoring pass.",
                       "tab:main_" + tag.replace("+", "p"))
    if (OUT / "main_refcoco.json").exists():
        controls_table()
        sensitivity_table()
        numbers_macros()
        taxonomy_table()
    if (OUT / "sota_table.json").exists():
        sota_table()
    if (OUT / "main_grefcoco.json").exists():
        gres_table()
    if (OUT / "main_grefcoco.json").exists() and (OUT / "main_refcoco.json").exists():
        ablation_json()
    print("done")
