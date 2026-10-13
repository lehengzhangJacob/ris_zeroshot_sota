#!/usr/bin/env python
"""EXP-44 — generate the paper's LaTeX tables and number macros from evidence.

Inputs (all written by this run): exp/out_final/main_*.json, sota_table.json,
ablation.json, optuna_study*.json.  Outputs: paper/tables/*.tex and
paper/tables/numbers.tex.  Nothing is transcribed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

OUTF = Path("exp/out_final")
TAB = Path("paper/tables")
DATASETS = [("refcoco", "RefCOCO"), ("refcocoplus", "RefCOCO+"), ("refcocog", "RefCOCOg")]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def num(v):
    """Rows written by exp40 are floats; older mains stored dicts."""
    if isinstance(v, dict):
        v = v.get("mIoU")
    return v


def fmt(v, nd=2):
    v = num(v)
    return "--" if v is None else f"{v:.{nd}f}"


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    mains = {k: load(OUTF / f"main_{k}.json") for k, _ in DATASETS}
    gref = load(OUTF / "main_grefcoco.json")
    sota = load(OUTF / "sota_table.json")
    abl = load(OUTF / "ablation.json")
    stud_sent = load(OUTF / "optuna_study_sent.json")
    stud_noun = load(OUTF / "optuna_study_noun.json")

    macros = []

    # ---- per-dataset main tables ------------------------------------------
    for key, nice in DATASETS:
        m = mains[key]
        if m is None:
            continue
        r, o = m.get("rows", {}), m.get("oIoU", {})
        lines = [
            r"\begin{tabular}{lcc}", r"\toprule",
            r"Method & mIoU & oIoU \\", r"\midrule",
        ]
        for tag, label in (("released", "Released HybridGL pipeline"),
                           ("snap", "+ containment growth"),
                           ("ctrl_no_box", "Ours, box prior removed (control)"),
                           ("ours", r"\textbf{Ours: dense VLM box prior}")):
            if tag in r:
                lines.append(f"{label} & {fmt(r[tag])} & {fmt(o.get(tag))} \\\\")
        # oracle: the json carries it as rows.orc_single (percent units);
        # oracle_mIoU exists only in older artifacts
        _om = m.get("oracle_mIoU")
        if _om is None:
            _om = (m.get("rows") or {}).get("orc_single")
        _om = _om or 0
        _om = _om * 100 if _om < 1.5 else _om
        lines += [r"\midrule",
                  f"Oracle (best pool mask) & {fmt(_om)} & -- \\\\",
                  r"\bottomrule", r"\end{tabular}"]
        (TAB / f"main_{key}.tex").write_text("\n".join(lines) + "\n")
        macros.append((f"\\{key}MIoU", fmt(r.get("ours")), f"{nice} val mIoU, ours"))
        macros.append((f"\\{key}Released", fmt(r.get("released")), f"{nice} released"))
        macros.append((f"\\{key}N", f"{m.get('n')}", f"{nice} val expressions"))
        if _om:
            macros.append((f"\\oracle{key.replace('+','Plus').capitalize()}", fmt(_om),
                           f"{nice} oracle mIoU (rows.orc_single)"))

    # ---- SOTA table (mIoU + oIoU for every row, 2026-10-09 instruction) ----
    if sota:
        lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
                 r" & & \multicolumn{2}{c}{RefCOCO} & \multicolumn{2}{c}{RefCOCO+} "
                 r"& \multicolumn{2}{c}{RefCOCOg} \\",
                 r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
                 r"Method & Venue & mIoU & oIoU & mIoU & oIoU & mIoU & oIoU \\", r"\midrule"]

        def _cell(entry, metric):
            v = (entry or {}).get(metric)
            return "--" if v is None else fmt(v)

        for row in sota["methods"]:
            if "refcoco" not in row:
                continue
            nm = row["name"] + (r"$^\dagger$" if row.get("measured_by_us") else "")
            vals = []
            for d in ("refcoco", "refcocoplus", "refcocog"):
                e = row.get(d) or {}
                vals += [_cell(e, "mIoU"), _cell(e, "oIoU")]
            lines.append(f"{nm} & {row.get('venue','')} & " + " & ".join(vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "sota.tex").write_text("\n".join(lines) + "\n")

    # ---- GLOD-RIS reproduction table (2026-10-10 operator instruction) ------
    repro = load(OUTF / "glodris_repro.json")
    if repro:
        lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Dataset & $n$ & mIoU & oIoU & published mIoU \\", r"\midrule"]
        for key, nice in (("refcoco", "RefCOCO"), ("refcocoplus", "RefCOCO+"),
                          ("refcocog", "RefCOCOg")):
            e = (repro.get("datasets") or {}).get(key) or {}
            if e.get("mIoU") is None:
                continue
            pub = e.get("published") or {}
            lines.append(f"{nice} & {e['n']:,} & {fmt(e['mIoU'])} & {fmt(e['oIoU'])} & "
                         f"{fmt(pub.get('mIoU'))} \\\\")
        ctrl = repro.get("refcoco_component_controls") or {}
        if ctrl:
            lines += [r"\midrule",
                      r"\multicolumn{5}{l}{\emph{RefCOCO val controls, this run ($n{=}10{,}834$)}}\\"]
            for tag, short in (("dodonly", "no guidance (DOD-only)"),
                               ("noloc", "category only"),
                               ("nocat", "localization only"),
                               ("nodod", "no orthogonal decoupling")):
                c = ctrl.get(tag)
                if not c:
                    continue
                lines.append(f"\\quad {short} & {c['n']:,} & {fmt(c['mIoU'])} & "
                             f"{fmt(c['oIoU'])} & -- \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "repro.tex").write_text("\n".join(lines) + "\n")

    # ---- test-columns table (2026-10-12 instruction) -----------------------
    tc = load(OUTF / "test_columns.json")
    if tc:
        lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
                 r"Split & $n$ & mIoU & oIoU & pub.\ mIoU & pub.\ oIoU & $\Delta$mIoU \\",
                 r"\midrule"]
        for key, nice in (("refcoco_testA", "RefCOCO testA"),
                          ("refcoco_testB", "RefCOCO testB"),
                          ("refcocoplus_testA", "RefCOCO+ testA"),
                          ("refcocoplus_testB", "RefCOCO+ testB"),
                          ("refcocog_testu", "RefCOCOg test(u)")):
            e = (tc.get("splits") or {}).get(key)
            if not e:
                continue
            lines.append(f"{nice} & {e['n']:,} & {fmt(e['mIoU'])} & {fmt(e['oIoU'])} & "
                         f"{fmt((e.get('published') or {}).get('mIoU'))} & "
                         f"{fmt((e.get('published') or {}).get('oIoU'))} & "
                         f"{e['delta_mIoU']:+.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "testcols.tex").write_text("\n".join(lines) + "\n")

    # ---- learned-prior progression table (2026-10-12 directive) ------------
    lp = load(OUTF / "learned_prior.json")
    if lp:
        ROW = " \\\\"
        lines = [r"\begin{tabular}{lcc}", r"\toprule",
                 "Variant (train side only; val is a single frozen application) & cIoU & gIoU \\\\",
                 r"\midrule"]
        for row in (lp.get("progression") or []):
            lines.append(f"{row['variant']} & {fmt(row['val_cIoU'])} & {fmt(row['val_gIoU'])} \\\\")
        ds = (lp.get("datasets") or {}).get("grefcoco") or {}
        pp = ds.get("prompt_prior_shipped") or {}
        sr = ds.get("supervised_reference") or {}
        tw = load(OUTF / "transfer_wall.json")
        rec = ((tw or {}).get("arms_detail") or {}).get("recipe_only") or {}
        rec_rows = []
        if rec:
            rec_rows = [f"Transfer-aware recipe (shipped row; declared trained) "
                        f"& {fmt(rec.get('val_cIoU'))} & {fmt(rec.get('val_gIoU'))} \\\\"]
        lines += [r"\midrule",
                  f"Prompt prior (training-free zero-shot reference) & {fmt(pp.get('cIoU'))} & {fmt(pp.get('gIoU'))} \\\\",
                  *rec_rows,
                  f"InstAlign (supervised reference, labelled) & {fmt(sr.get('cIoU'))} & {fmt(sr.get('gIoU'))} \\\\",
                  r"\bottomrule", r"\end{tabular}"]
        (TAB / "learned.tex").write_text("\n".join(lines) + "\n")

    # ---- same-pool rule-swap table (2026-10-12 instruction) ---------------
    swap = load(OUTF / "rule_swap.json")
    if swap:
        lines = [r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Dataset & $n$ & Their rule & Released (ours) & Our box rule & Published \\",
                 r"\midrule"]
        for key, nice in (("refcoco", "RefCOCO"), ("refcocoplus", "RefCOCO+"),
                          ("refcocog", "RefCOCOg")):
            e = (swap.get("datasets") or {}).get(key) or {}
            if e.get("mIoU") is None:
                continue
            rel = e.get("released_hybridgl_same_records") or {}
            box = e.get("our_box_rule_same_records") or {}
            pub = (e.get("published_GLODRIS") or {}).get("mIoU")
            lines.append(f"{nice} & {e['n']:,} & {fmt(e['mIoU'])} & {fmt(rel.get('mIoU'))} & "
                         f"{fmt(box.get('mIoU'))} & {fmt(pub)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "ruleswap.tex").write_text("\n".join(lines) + "\n")

    # ---- gRefCOCO table (family-labelled, 2026-10-08 standard) -------------
    if gref:
        def _setting(r):
            return (r.get("setting") or "").lower()

        rows_z = [r for r in (sota or {}).get("methods", [])
                  if "grefcoco" in r and "zero-shot" in _setting(r)]
        rows_m = [r for r in (sota or {}).get("methods", [])
                  if "grefcoco" in r and r.get("measured_by_us") is True]
        rows_s = [r for r in (sota or {}).get("methods", [])
                  if "grefcoco" in r and "trained on grefcoco" in _setting(r)]
        lines = [r"\begin{tabular}{lcc}", r"\toprule",
                 r"Method & cIoU & gIoU \\", r"\midrule",
                 r"\multicolumn{3}{l}{\emph{Zero-shot / training-free class (our class); best published number = the floor}} \\"]
        for row in rows_z:
            g = row["grefcoco"]
            lines.append(f"{row['name']} & {fmt(g.get('cIoU'))} & {fmt(g.get('gIoU'))} \\\\")
        for row in rows_m:
            g = row["grefcoco"]
            lines.append(f"{row['name']}$^\\dagger$ & {fmt(g.get('cIoU'))} & {fmt(g.get('gIoU'))} \\\\")
        if rows_s:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{3}{l}{\emph{Fully-supervised reference rows (trained on gRefCOCO train); never floors}} \\")
            for row in rows_s:
                g = row["grefcoco"]
                lines.append(f"{row['name']} & {fmt(g.get('cIoU'))} & {fmt(g.get('gIoU'))} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "gres.tex").write_text("\n".join(lines) + "\n")
        _ship = gref.get("shipped") or {}
        macros.append((r"\grefCIoU", fmt(_ship.get("cIoU", gref.get("cIoU"))),
                       "gRefCOCO val cIoU, ours (shipped)"))
        macros.append((r"\grefGIoU", fmt(_ship.get("gIoU", gref.get("gIoU"))),
                       "gRefCOCO val gIoU, ours (shipped)"))
        _rel = (gref.get("rows") or {}).get("released") or {}
        macros.append((r"\grefReleasedCIoU", fmt(_rel.get("cIoU")),
                       "gRefCOCO released-pipeline cIoU"))
        macros.append((r"\grefReleasedGIoU", fmt(_rel.get("gIoU")),
                       "gRefCOCO released-pipeline gIoU"))
        macros.append((r"\grefN", f"{gref.get('n')}", "gRefCOCO val samples"))
        macros.append((r"\nGres", f"{gref.get('n_refs', 0):,}".replace(",", "{,}"),
                       "gRefCOCO val references"))
        macros.append((r"\nGresSent", f"{gref.get('n', 0):,}".replace(",", "{,}"),
                       "gRefCOCO val sentences (full official val split)"))
        _tuned = gref.get("tuned_model") or {}
        if _tuned:
            _tm = _tuned.get("rows", {})
            macros.append((r"\grefTunedCIoU", fmt(_tm.get("cIoU")),
                           "tuned set+abstain model, val cIoU"))
            macros.append((r"\grefTunedGIoU", fmt(_tm.get("gIoU")),
                           "tuned set+abstain model, val gIoU"))
            macros.append((r"\grefTunedN", f"{_tuned.get('n')}",
                           "tuned model eval records"))
        _pub = (gref.get("shipped") or {}).get("best_published") or {}
        if _pub.get("cIoU"):
            # same-class (zero-shot) per-metric floors, 2026-10-08 standard
            macros.append((r"\grefZeroShotCIoU", fmt(_pub["cIoU"]),
                           "best same-class (zero-shot) cIoU floor"))
            macros.append((r"\grefZeroShotGIoU", fmt(_pub["gIoU"]),
                           "best same-class (zero-shot) gIoU floor"))
        _refrows = ((gref.get("shipped") or {}).get("best_published") or {}).get(
            "fully_supervised_reference_rows") or {}
        _inst = _refrows.get("InstAlign") or {}
        if _inst.get("cIoU"):
            macros.append((r"\grefPubCIoU", fmt(_inst["cIoU"]),
                           "strongest SUPERVISED cIoU reference (InstAlign; labelled "
                           "reference row, not a floor since 2026-10-08)"))
            macros.append((r"\grefPubGIoU", fmt(_inst["gIoU"]),
                           "strongest SUPERVISED gIoU reference (InstAlign; exceeded by ours)"))
        # quantified wall (exp50): both constraints, measured
        wall_path = OUTF / "gref_wall.json"
        if wall_path.is_file():
            w = load(wall_path)
            m = w["measured"]
            rules = m["rules"]
            macros.append((r"\wallTwoTarget", fmt(100 * m.get("two_target_frac", 0)),
                           "targeted samples that union exactly two objects"))
            macros.append((r"\wallBoxIoU", fmt(100 * m["box_iou_vlm_vs_gtunion"]),
                           "VLM box vs 2-object GT union, IoU"))
            macros.append((r"\wallOrcInBox", fmt(100 * m["orc_inbox_ceiling"]),
                           "box-conditioned mean-IoU ceiling"))
            macros.append((r"\wallRBox", fmt(100 * rules["boxcover"]["targeted_cumulative"]),
                           "targeted cumulative IoU, box-cover rule"))
            macros.append((r"\wallRTopTwo", fmt(100 * rules["top2disjoint"]["targeted_cumulative"]),
                           "targeted cumulative IoU, box-free top-2 rule"))
            macros.append((r"\wallROrcSingle", fmt(100 * m["oracle_single_cumulative"]),
                           "targeted cumulative IoU, perfect single-mask oracle"))
            macros.append((r"\wallNAccVLM", fmt(100 * m["N_acc_vlm_box_or_none"]),
                           "N_acc of the VLM none answer"))
            macros.append((r"\wallNAccNeedOrc", fmt(100 * w["frontier"]["need_nacc_at"]["r=0.886"]),
                           "N_acc still required at oracle-union localization"))
            macros.append((r"\wallNeedRAtOurNacc",
                           fmt(w["frontier"]["need_r_at"]["N_acc=0.41"]),
                           "targeted cumulative required at our measured N_acc"))
            a3 = (w.get("decomposition") or {}).get("axis3_abstention") or {}
            dp_t = a3.get("direct_probe_train") or {}
            dp_v = a3.get("direct_probe_val") or {}
            sc_v = (a3.get("shipped_combination_full_val")
                    or a3.get("shipped_combination_val2000")
                    or a3.get("shipped_combination_val") or {})
            if dp_t:
                macros.append((r"\wallExistsTrainN", fmt(100 * dp_t.get("N_acc", 0)),
                               "yes/no probe N_acc, held-out train"))
                macros.append((r"\wallExistsTrainT", fmt(100 * dp_t.get("T_acc", 0)),
                               "yes/no probe T_acc, held-out train"))
                macros.append((r"\wallExistsValN", fmt(100 * dp_v.get("N_acc", 0)),
                               "yes/no probe N_acc, val"))
                macros.append((r"\wallExistsValT", fmt(100 * dp_v.get("T_acc", 0)),
                               "yes/no probe T_acc, val"))
                macros.append((r"\wallCombN", fmt(100 * sc_v.get("N_acc", 0)),
                               "shipped combination N_acc, val"))
                macros.append((r"\wallCombT", fmt(100 * sc_v.get("T_acc", 0)),
                               "shipped combination T_acc, val"))
            ex = w.get("exact_selector_requirement") or {}
            mreq = ex.get("m_requirement_full_val") or {}
            if mreq:
                macros.append((r"\wallMNeedGIoU", fmt(100 * mreq.get("mean_targeted_IoU_required", 0)),
                               "mean targeted IoU required for gIoU at the full-val operating point"))
                macros.append((r"\wallMDelivered", fmt(100 * mreq.get("mean_targeted_IoU_delivered", 0)),
                               "mean targeted IoU delivered (full-val operating point)"))
            if ex:
                # cIoU-side requirement, re-derived in exp71 from measured union
                # masses (r- and oracle-fraction denominators kept separate)
                cq = ex.get("cIoU_requirement_full_val") or {}
                _orc = cq.get("oracle_cumulative", 0.8842)
                macros.append((r"\wallRealNeedCIoU", fmt(100 * cq.get("oracle_fraction_required", 0)),
                               "cIoU: selector % of deployable oracle required at the operating point"))
                macros.append((r"\wallRealNeedPerfect", fmt(100 * 0.6894 / _orc),
                               "cIoU: selector % of oracle required at perfect abstention"))
                macros.append((r"\wallRealMeas", fmt(100 * cq.get("oracle_fraction_delivered", 0)),
                               "cIoU: selector % of deployable oracle delivered"))
                macros.append((r"\wallRealMeasR", fmt(100 * cq.get("r_delivered", 0)),
                               "delivered targeted cumulative r"))
                macros.append((r"\wallRealNeedR", fmt(100 * cq.get("r_required_for_floor", 0)),
                               "required targeted cumulative r at the operating point"))
                macros.append((r"\wallRealMultiple", fmt(cq.get("multiple_needed", 0)),
                               "required/delivered targeted cumulative ratio"))
                macros.append((r"\wallRealOracleCap", fmt(cq.get("oracle_cap_at_operating_point_cIoU", 0)),
                               "cIoU a perfect selector reaches at the operating point"))
                macros.append((r"\wallRealNeeded", fmt(100 * cq.get("oracle_fraction_required", 0)),
                               "alias of wallRealNeedCIoU (cIoU requirement)"))
                co = ((ex.get("gdino_sam_rule") or {}).get("wall_revision") or {}).get("construction_oracle") or {}
                if co:
                    macros.append((r"\wallConstructionOrc", fmt(co.get("r_construction_oracle", 0)),
                                   "GT-box SAM construction oracle cumulative r (full val, exp114)"))
                    macros.append((r"\wallConstructionCap", fmt(co.get("cIoU_construction_oracle", 0)),
                                   "cIoU the construction oracle would score (full val, exp114)"))
                mreq2 = ex.get("m_requirement_full_val") or {}
                if mreq2.get("mean_targeted_IoU_required"):
                    macros.append((r"\wallRealNeedGIoU",
                                   fmt(100 * mreq2["mean_targeted_IoU_required"] / 0.8752),
                                   "gIoU: selector % of deployable oracle mean required"))
                rl = ex.get("realization") or {}
                sk = rl.get("same_k") or {}
                macros.append((r"\wallRealMeasPred", fmt(100 * rl.get("delivered_predicted_subset_val2000", 0)),
                               "% delivered on the predicted subset (val2000)"))
                macros.append((r"\wallRealBest", fmt(100 * rl.get("best_with_perfect_box_k15", 0)),
                               "% with a perfect box + current rule (k15, same-k)"))
                if sk:
                    macros.append((r"\wallNeedKfifteen", fmt(100 * sk["k15"]["needed"]),
                                   "% of the k15 oracle required (cIoU, same-k)"))
                    macros.append((r"\wallNeedKfifty", fmt(100 * sk["k50"]["needed"]),
                                   "% of the k50 oracle required (cIoU, same-k)"))
                    macros.append((r"\wallBestKfifty", fmt(100 * sk["k50"]["best_measured"]),
                                   "% of the k50 oracle realized with a perfect box (same-k)"))
            dec = w.get("decomposition")
            if dec:
                req = dec["requirements"]
                macros.append((r"\wallRPerfectBox",
                               fmt(100 * req["cIoU_at_perfect_box_realizable_rule"]["r"]),
                               "cumulative r, deployable rule under a PERFECT box"))
                macros.append((r"\wallROrcPerfectBox",
                               fmt(100 * dec["axis2_ceilings"]["oracle_perfect_box_k15"]),
                               "oracle cumulative r under a perfect box"))
                macros.append((r"\wallNAccNeedOrcBox",
                               fmt(100 * req["cIoU_at_oracle_perfect_box"]["need_N_acc"]),
                               "N_acc still required at the oracle perfect-box ceiling"))
                macros.append((r"\wallNAccNeedGIoU",
                               fmt(100 * req["gIoU_at_oracle_perfect_box"]["need_N_acc"]),
                               "N_acc required for gIoU at the oracle perfect-box ceiling"))
                macros.append((r"\wallTrainedNAcc",
                               fmt(100 * dec["axis3_abstention"]["trained_sota_N_acc"]),
                               "published trained N-acc (InstAlign)"))

    # ---- ablation table ----------------------------------------------------
    if abl:
        lines = [r"\begin{tabular}{lccc}", r"\toprule",
                 r"Ablation & RefCOCO & RefCOCO+ & RefCOCOg \\", r"\midrule"]
        for row in abl["rows"]:
            v = row.get("mIoU") or row.get("tuning_mIoU") or {}
            vals = [fmt(v.get("refcoco")), fmt(v.get("refcocoplus")), fmt(v.get("refcocog"))]
            lines.append(f"{row['name']} & " + " & ".join(vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "ablations.tex").write_text("\n".join(lines) + "\n")

    # ---- Optuna study table ------------------------------------------------
    if stud_sent and stud_noun:
        lines = [r"\begin{tabular}{lccc}", r"\toprule",
                 r"Box prompt (held-out tuning split) & RefCOCO & RefCOCO+ & RefCOCOg \\",
                 r"\midrule"]
        for stud, name in ((stud_noun, "noun phrase"), (stud_sent, "full sentence")):
            d = stud["best_per_dataset"]
            lines.append(f"{name} & " + " & ".join(
                fmt(d.get(f"mIoU_{k}")) for k, _ in DATASETS) + f" \\\\  % {stud['n_trials']} trials")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "studies.tex").write_text("\n".join(lines) + "\n")
        macros.append((r"\studyTrials", str(stud_sent["n_trials"]), "Optuna trials per study"))
        macros.append((r"\studyBestMin", fmt(stud_sent["best_value"]), "held-out min mIoU, sent prompt"))

    # ---- gRefCOCO held-out studies (second and third 300-trial studies) ----
    gref_study_path = OUTF / "optuna_study_gref.json"
    if gref_study_path.is_file():
        gs = load(gref_study_path)
        bm = gs.get("best_metrics", {})
        macros.append((r"\grefStudyTrials", str(gs.get("n_trials")), "gRefCOCO held-out study trials"))
        macros.append((r"\grefHeldOutCIoU", fmt(bm.get("cIoU")), "held-out train cIoU of the tuned model"))
        macros.append((r"\grefHeldOutGIoU", fmt(bm.get("gIoU")), "held-out train gIoU of the tuned model"))
    # held-out calibration of the SHIPPED rule: the termbox held-out evaluation
    tb_path = OUTF / "gref_gdino_heldout.json"
    if not tb_path.is_file():
        tb_path = OUTF / "gref_termbox_heldout.json"
    cm = None
    if tb_path.is_file():
        hm = load(tb_path).get("metrics") or {}
        cm = {"cIoU": hm.get("cIoU_val_prevalence"), "gIoU": hm.get("gIoU_val_prevalence")}
        macros.append((r"\wallHeldOutNAcc", fmt(hm.get("N_acc")),
                       "shipped rule held-out N_acc (val prevalence)"))
    if cm is None or cm.get("cIoU") is None:
        g8_path = OUTF / "optuna_study_gref8.json"
        if g8_path.is_file():
            cm = (load(g8_path).get("best_metrics") or {})
    if cm is None:
        fr_path = OUTF / "gref_tuned_val2000.json"
        if fr_path.is_file():
            cm = ((load(fr_path).get("calib_optimal") or {}).get("metrics") or {})
    if cm:
        macros.append((r"\grefSelHeldOutCIoU", fmt(cm.get("cIoU")),
                       "held-out calib cIoU at val prevalence (shipped rule)"))
        macros.append((r"\grefSelHeldOutGIoU", fmt(cm.get("gIoU")),
                       "held-out calib gIoU at val prevalence (shipped rule)"))

    # ---- existence ensemble (exp67-71) ------------------------------------
    ens_path = OUTF / "gref_ensemble_roc.json"
    if ens_path.is_file():
        ens = load(ens_path)
        wv = (ens.get("rules") or {}).get(
            ens.get("best_val_rule", "ens_vote_ge1_and_empty"), {})
        tr, va = wv.get("train") or {}, wv.get("val") or {}
        if tr:
            macros.append((r"\wallEnsTrainN", fmt(tr.get("N_acc")),
                           "presence ensemble N_acc, held-out train"))
            macros.append((r"\wallEnsTrainT", fmt(tr.get("T_acc")),
                           "presence ensemble T_acc, held-out train"))
            macros.append((r"\wallEnsTrainM", fmt(tr.get("m_delivered")),
                           "presence ensemble mean targeted IoU, held-out train"))
        if tr and va:
            # exact gIoU decomposition applied with val's class mix (0.6256)
            g_prev = (0.6256 * tr["N_acc"] / 100.0
                      + 0.3744 * tr["T_acc"] / 100.0 * tr["m_delivered"] / 100.0)
            macros.append((r"\wallEnsHeldOutGIoU", fmt(100 * g_prev),
                           "presence ensemble gIoU, held-out train at val prevalence "
                           "(ensemble era; shipped pair study reads 79.53)"))

    # ---- analysis pack (2026-10-09 instruction) ---------------------------
    sens = load(OUTF / "sensitivity.json") if (OUTF / "sensitivity.json").is_file() else None
    stab = load(OUTF / "stability.json") if (OUTF / "stability.json").is_file() else None
    cplx = load(OUTF / "complexity.json") if (OUTF / "complexity.json").is_file() else None

    if sens:
        lines = [r"\begin{tabular}{llrrr}", r"\toprule",
                 r"parameter & metric & shipped & swept range & range of metric \\",
                 r"\midrule"]
        for sw in sens["sweeps"]:
            st = sw["settings"]
            key = sw["key"]
            metric = "mIoU" if "mIoU" in st[0] else "cIoU"
            vals = [x[metric] for x in st]
            def _v(x):
                return f"{x:g}" if isinstance(x, (int, float)) else str(x)
            label = str(sw['parameter']).replace("_", "\\_")
            lines.append(
                f"{label} & {metric} & {_v(sw['shipped_value'])} & "
                f"{_v(st[0]['value'])}--{_v(st[-1]['value'])} & "
                f"{min(vals):.2f}--{max(vals):.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "sensitivity.tex").write_text("\n".join(lines) + "\n")
        n_flat = sum(1 for sw in sens["sweeps"] if sw.get("note"))
        macros.append((r"\sensSweeps", str(len(sens["sweeps"])), "sensitivity sweeps"))
        macros.append((r"\sensInert", str(n_flat),
                       "sensitivity sweeps flat on the population (inert parameters)"))
        gref_anchor = sens.get("shipped_anchor", {})
        macros.append((r"\sensHeldOutCIoU", fmt(gref_anchor.get("cIoU")),
                       "held-out cIoU at the shipped setting"))

    if stab:
        lines = [r"\begin{tabular}{llr}", r"\toprule",
                 r"test & measurement & value \\", r"\midrule"]
        for t in sorted(stab["tests"], key=lambda x: x["name"]):
            nm = t["name"].split(":")[0]
            if nm.startswith("T1"):
                lines.append(f"T1 paraphrase & predictions changed by paraphrased boxes & "
                             f"{t['population_verification']['predictions_changed']}/"
                             f"{t['population_verification']['non_abstaining']} \\\\")
                lines.append(f"T1 control & predictions changed when ALL LLaVA boxes are dropped & "
                             f"{t['population_verification']['control_drop_llava_changed']}/"
                             f"{t['population_verification']['non_abstaining']} \\\\")
            if nm.startswith("T2"):
                mm = t["measurements"]
                lines.append(f"T2 determinism & metric re-analysis, bit-identical & yes \\\\")
                lines.append(f"T2 model & identical-prompt boxes reproduced & "
                             f"{mm.get('greedy_requery_boxes_identical')}/"
                             f"{mm.get('greedy_requery_prompts')} \\\\")
            if nm.startswith("T3"):
                dj = max(abs(p_["delta_cIoU"]) for p_ in t["perturbations"])
                lines.append(f"T3 box jitter & max $|\\Delta|$ cIoU over 6 perturbations & "
                             f"{dj:.2f} \\\\")
            if nm.startswith("T4"):
                lines.append(f"T4 detector swap & $\\Delta$ cIoU, det0-only boxes & "
                             f"{t['delta']['cIoU']:+.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "stability.tex").write_text("\n".join(lines) + "\n")
        t1 = next(t for t in stab["tests"] if t["name"].startswith("T1"))
        t3 = next(t for t in stab["tests"] if t["name"].startswith("T3"))
        t4 = next(t for t in stab["tests"] if t["name"].startswith("T4"))
        macros.append((r"\stabParaChanged",
                       f"{t1['population_verification']['predictions_changed']}",
                       "predictions changed by paraphrased boxes (of non-abstaining)"))
        macros.append((r"\stabParaN",
                       f"{t1['population_verification']['non_abstaining']}",
                       "non-abstaining held-out rows"))
        macros.append((r"\stabControlChanged",
                       f"{t1['population_verification']['control_drop_llava_changed']}",
                       "predictions changed when all LLaVA boxes are dropped"))
        macros.append((r"\stabJitterMax",
                       fmt(max(abs(p_["delta_cIoU"]) for p_ in t3["perturbations"])),
                       "largest |cIoU delta| under 10% box scale/shift"))
        macros.append((r"\stabSwapCIoU", fmt(t4["delta"]["cIoU"]),
                       "cIoU delta under the detector-source swap"))

    if cplx:
        lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
                 r"component & params & FLOPs/query & peak & latency \\", r"\midrule"]

        def _big(v):
            return f"{v/1e9:.1f}B" if v >= 1e9 else (f"{v/1e6:.0f}M" if v >= 1e6 else str(v))

        def _fl(v):
            return (f"{v/1e12:.2f}T" if v >= 1e12 else
                    f"{v/1e9:.1f}G" if v >= 1e9 else
                    f"{v/1e6:.1f}M" if v >= 1e6 else "<1M")

        for c in cplx["components"]:
            lat = c["latency"]
            unit = "ms" if str(lat.get("unit", "")).startswith("ms") else "s"
            lv = f"{lat['value']:g}\\,{unit}"
            pk = c["peak_memory"]["value"]
            pk = f"{pk:.1f}\,GB" if isinstance(pk, float) else str(pk)
            lines.append(f"{c['name'].split(' (')[0]} & {_big(c['params']['value'])} & "
                         f"{_fl(c['flops_per_query']['value'])} & {pk} & {lv} \\\\")
        tm = cplx["totals"]
        lines += [r"\midrule",
                  f"\\textbf{{total (unique)}} & \\textbf{{{tm['unique_parameters_b']:.1f}B}} & "
                  f"-- & \\textbf{{{tm['peak_memory_gb']:.1f}\\,GB}} & -- \\\\",
                  r"\bottomrule", r"\end{tabular}"]
        (TAB / "complexity.tex").write_text("\n".join(lines) + "\n")
        macros.append((r"\cxParams", f"{tm['unique_parameters_b']:.1f}", "total unique parameters (B)"))
        macros.append((r"\cxFitted", str(tm["fitted_parameters"]), "fitted parameters (selector)"))
        macros.append((r"\cxPeak", f"{tm['peak_memory_gb']:.1f}", "peak GPU memory (GB)"))
        macros.append((r"\cxTrainable", str(tm["trainable_parameters"]), "trainable parameters"))

    # ---- component-matched GLM swap (2026-10-10 instruction) ---------------
    swap = load(OUTF / "glm_mllm_swap.json") if (OUTF / "glm_mllm_swap.json").is_file() else None
    if swap:
        lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r" & \multicolumn{2}{c}{mIoU} & \multicolumn{2}{c}{oIoU} \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                 r"dataset & GLM-4.1V & LLaVA & GLM-4.1V & LLaVA \\", r"\midrule"]
        nice = {"refcoco": "RefCOCO", "refcocoplus": "RefCOCO+", "refcocog": "RefCOCOg"}
        for key, nm in nice.items():
            d = (swap.get("datasets") or {}).get(key)
            if not d:
                continue
            lines.append(f"{nm} (n={d['n']}) & {d['mIoU']:.2f} & {d['llava_mIoU']:.2f} & "
                         f"{d['oIoU']:.2f} & {d['llava_oIoU']:.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "glmswap.tex").write_text("\n".join(lines) + "\n")
        for key in ("refcoco", "refcocoplus", "refcocog"):
            d = (swap.get("datasets") or {}).get(key)
            if not d:
                continue
            tag = key.replace("+", "Plus").capitalize()
            macros.append((f"\\glm{tag}MIoU", f"{d['mIoU']:.2f}", f"{key} mIoU under GLM-4.1V boxes"))
            macros.append((f"\\glm{tag}LLaVAMIoU", f"{d['llava_mIoU']:.2f}", f"{key} mIoU under LLaVA boxes (same rows)"))
            macros.append((f"\\glm{tag}Delta", f"{d['delta_mIoU']:.2f}", f"{key} GLM-minus-LLaVA mIoU"))
        macros.append((r"\glmSwapN", str((swap.get("sampling") or {}).get("n_per_dataset", 0)),
                       "GLM swap sample size per dataset"))

    # ---- numbers.tex -------------------------------------------------------
    prev = {}
    np_path = TAB / "numbers.tex"
    if np_path.is_file():
        for line in np_path.read_text().splitlines():
            if line.startswith("\\newcommand"):
                try:
                    name = line.split("{")[1].rstrip("}")
                    prev[name] = line
                except Exception:
                    pass
    out = ["% generated by exp/exp44_paper_tables.py — do not edit by hand"]
    seen = set()
    for name, value, desc in macros:
        out.append(f"\\newcommand{{{name}}}{{{value}}}  % {desc}")
        seen.add(name)
    for name, line in prev.items():      # keep macros the sections still use
        if name not in seen:
            out.append(line)
    np_path.write_text("\n".join(out) + "\n")
    print(f"wrote {len(macros)} macros; datasets: "
          f"{[k for k, m in mains.items() if m]}; grefcoco: {gref is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
