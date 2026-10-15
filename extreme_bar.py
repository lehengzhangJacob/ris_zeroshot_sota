#!/usr/bin/env python3
"""Reject a finished run until a CVPR-style package exists.

Evidence files must be newer than extreme_started.txt, so yesterday's draft
does not count.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAMP = ROOT / "extreme_started.txt"


def fail(msg):
    print("FAIL:", msg)
    return False


def newer(path):
    path = ROOT / path
    if not path.is_file():
        return fail(f"missing {path.relative_to(ROOT)}")
    if not STAMP.is_file() or path.stat().st_mtime < STAMP.stat().st_mtime:
        return fail(f"stale {path.relative_to(ROOT)}")
    return True


def load_json(rel):
    path = ROOT / rel
    if not newer(rel):
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        fail(f"bad json {rel}: {exc}")
        return None


def tex_has_appendix():
    hits = []
    for path in (ROOT / "paper").rglob("*.tex"):
        if path.stat().st_mtime < STAMP.stat().st_mtime:
            continue
        text = path.read_text(errors="replace")
        if r"\appendix" in text or r"\appendices" in text:
            hits.append(path)
    if not hits:
        return fail("no new tex file contains \\appendix")
    return True


def architecture():
    candidates = [
        ROOT / "paper" / "figures" / "architecture.pdf",
        ROOT / "paper" / "figures" / "architecture.svg",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size >= 20000 and path.stat().st_mtime >= STAMP.stat().st_mtime:
            return True
    return fail("paper/figures/architecture.pdf or .svg missing, small, or stale")


def sota():
    data = load_json("exp/out_final/sota_table.json")
    if data is None:
        return False
    rows = data.get("methods") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) < 6:
        return fail("sota_table.json needs methods[] of length >= 6")
    measured = 0
    for row in rows:
        if not isinstance(row, dict):
            return fail("sota row is not an object")
        if row.get("measured_by_us") is True:
            evidence = row.get("evidence")
            if not isinstance(evidence, str) or not newer(evidence):
                return fail(f"measured SOTA row has no fresh evidence: {row.get('name')}")
            measured += 1
    if measured < 3:
        return fail(f"only {measured} SOTA rows were measured by us; need >= 3")
    return True


def ablation():
    data = load_json("exp/out_final/ablation.json")
    if data is None:
        return False
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) < 5:
        return fail("ablation.json needs rows[] of length >= 5")
    for row in rows:
        evidence = row.get("evidence") if isinstance(row, dict) else None
        if not isinstance(evidence, str) or not newer(evidence):
            return fail("ablation row missing fresh evidence")
    return True


def mains():
    need = {
        "exp/out_final/main_refcoco.json": 10834,
        "exp/out_final/main_refcocoplus.json": 10758,
        "exp/out_final/main_refcocog.json": 4896,
        "exp/out_final/main_grefcoco.json": 2000,
    }
    ok = True
    for rel, n_min in need.items():
        data = load_json(rel)
        if data is None:
            ok = False
            continue
        n = data.get("n")
        if not isinstance(n, int) or n < n_min:
            ok = fail(f"{rel} n={n}, need >= {n_min}")
    return ok


def graveyard():
    rel = "refine-logs/METHOD_GRAVEYARD.md"
    if not newer(rel):
        return False
    text = (ROOT / rel).read_text(errors="replace")
    if text.lower().count("killed") < 2:
        return fail("METHOD_GRAVEYARD.md needs at least two killed methods")
    return True


def protocol_audit():
    """Cross-paper metric-protocol consistency gate (operator instruction 2026-10-08).

    Every shipped number and cross-paper comparison must use the same metric
    protocol as the published numbers compared against. The run records the
    per-dataset protocol source, the metric definition, and the aligned
    re-evaluation in exp/out_final/protocol_audit.json; the paper must be
    rebuilt after that audit (PDF newer than the audit file).
    """
    rel = "exp/out_final/protocol_audit.json"
    if not newer(rel):
        return False
    try:
        data = json.loads((ROOT / rel).read_text())
    except Exception as exc:
        return fail(f"bad json {rel}: {exc}")
    datasets = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(datasets, dict):
        return fail("protocol_audit.json needs a datasets{} object")
    for key in ("refcoco", "refcocoplus", "refcocog", "grefcoco"):
        entry = datasets.get(key)
        if not isinstance(entry, dict):
            return fail(f"protocol_audit.json datasets.{key} missing")
        for field in ("protocol_source", "definition"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                return fail(f"protocol_audit.json {key}.{field} must be a non-empty string")
        if entry.get("aligned") is not True:
            return fail(f"protocol_audit.json {key}.aligned must be true")
    pdf_path = ROOT / "paper" / "main.pdf"
    audit_path = ROOT / rel
    if not pdf_path.is_file() or pdf_path.stat().st_mtime < audit_path.stat().st_mtime:
        return fail("paper/main.pdf must be rebuilt after exp/out_final/protocol_audit.json")
    return True


def metrics_sota():
    """Every compared metric must appear for every row (user instruction 2026-10-09).

    The comparison table must mirror the compared papers' own tables: every
    RefCOCO-family row carries BOTH mIoU and oIoU (published rows take the
    compared papers' numbers, our rows take the aligned re-evaluations).
    """
    data = load_json("exp/out_final/sota_table.json")
    if data is None:
        return False
    rows = data.get("methods") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return fail("sota_table.json needs methods[]")
    for row in rows:
        if not isinstance(row, dict):
            return fail("sota row is not an object")
        for key in ("refcoco", "refcocoplus", "refcocog"):
            entry = row.get(key)
            if entry is None:
                continue
            if not isinstance(entry, dict):
                return fail(f"sota row {row.get('name')} {key} must be an object")
            for metric in ("mIoU", "oIoU"):
                if metric not in entry:
                    return fail(
                        f"sota row {row.get('name')} {key} missing {metric}; "
                        "every RefCOCO-family row needs both mIoU and oIoU"
                    )
    return True


def analysis_pack():
    """Fair-comparison analysis pack (user instruction 2026-10-09).

    sensitivity.json (Parameter Sensitivity Analysis), stability.json (Testing
    Stability), complexity.json (Model Complexity: parameters, FLOPs, memory,
    latency, training time). The paper must be rebuilt after these files.
    """
    specs = [
        ("exp/out_final/sensitivity.json", "sweeps", 3),
        ("exp/out_final/stability.json", "tests", 3),
        ("exp/out_final/complexity.json", "components", 4),
    ]
    newest = 0.0
    for rel, key, n_min in specs:
        if not newer(rel):
            return False
        try:
            data = json.loads((ROOT / rel).read_text())
        except Exception as exc:
            return fail(f"bad json {rel}: {exc}")
        lst = data.get(key) if isinstance(data, dict) else None
        if not isinstance(lst, list) or len(lst) < n_min:
            return fail(f"{rel} needs a {key}[] list of length >= {n_min}")
        if rel.endswith("complexity.json"):
            totals = data.get("totals")
            if not isinstance(totals, dict) or not totals:
                return fail("complexity.json needs a non-empty totals{} object")
            training = data.get("training_time")
            if not isinstance(training, str) or not training.strip():
                return fail("complexity.json needs a non-empty training_time string (state explicitly if none)")
        newest = max(newest, (ROOT / rel).stat().st_mtime)
    pdf_path = ROOT / "paper" / "main.pdf"
    if not pdf_path.is_file() or pdf_path.stat().st_mtime < newest:
        return fail("paper/main.pdf must be rebuilt after the analysis-pack files")
    return True


def glm_swap():
    """Component-matched GLM-4.1V swap (user instruction 2026-10-10).

    GLOD-RIS's exact MLLM (GLM-4.1V-Thinking-Flash, via Zhipu API) must be
    evaluated in the same box-prior path on the three RefCOCO-family official
    val splits; the paper must be rebuilt after the results file.
    """
    rel = "exp/out_final/glm_mllm_swap.json"
    if not newer(rel):
        return False
    try:
        data = json.loads((ROOT / rel).read_text())
    except Exception as exc:
        return fail(f"bad json {rel}: {exc}")
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, str) or not model.strip():
        return fail("glm_mllm_swap.json needs a non-empty model string")
    datasets = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(datasets, dict):
        return fail("glm_mllm_swap.json needs a datasets{} object")
    for key in ("refcoco", "refcocoplus", "refcocog"):
        entry = datasets.get(key)
        if not isinstance(entry, dict):
            return fail(f"glm_mllm_swap.json datasets.{key} missing")
        n = entry.get("n")
        if not isinstance(n, int) or n < 1000:
            return fail(f"glm_mllm_swap.json datasets.{key} n must be an int >= 1000")
        if not isinstance(entry.get("mIoU"), (int, float)):
            return fail(f"glm_mllm_swap.json datasets.{key} needs a numeric mIoU")
    pdf_path = ROOT / "paper" / "main.pdf"
    rel_path = ROOT / rel
    if not pdf_path.is_file() or pdf_path.stat().st_mtime < rel_path.stat().st_mtime:
        return fail("paper/main.pdf must be rebuilt after exp/out_final/glm_mllm_swap.json")
    return True


def glodris_repro():
    """Full-val GLOD-RIS reproduction, second attempt (user instruction 2026-10-10)."""
    rel = "exp/out_final/glodris_repro.json"
    if not newer(rel):
        return False
    try:
        data = json.loads((ROOT / rel).read_text())
    except Exception as exc:
        return fail(f"bad json {rel}: {exc}")
    if data.get("complete") is not True:
        return fail("glodris_repro.json complete must be true (full official val splits)")
    datasets = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(datasets, dict):
        return fail("glodris_repro.json needs a datasets{} object")
    for key, n_min in (("refcoco", 10834), ("refcocoplus", 10758), ("refcocog", 4896)):
        entry = datasets.get(key)
        if not isinstance(entry, dict):
            return fail(f"glodris_repro.json datasets.{key} missing")
        n = entry.get("n")
        if not isinstance(n, int) or n < n_min:
            return fail(f"glodris_repro.json datasets.{key} n={n}, need >= {n_min}")
        for metric in ("mIoU", "oIoU"):
            if not isinstance(entry.get(metric), (int, float)):
                return fail(f"glodris_repro.json datasets.{key} needs numeric {metric}")
        pub = entry.get("published")
        if not isinstance(pub, dict) or not isinstance(pub.get("mIoU"), (int, float)):
            return fail(f"glodris_repro.json datasets.{key} needs published{{mIoU}}")
    pdf_path = ROOT / "paper" / "main.pdf"
    rel_path = ROOT / rel
    if not pdf_path.is_file() or pdf_path.stat().st_mtime < rel_path.stat().st_mtime:
        return fail("paper/main.pdf must be rebuilt after exp/out_final/glodris_repro.json")
    return True


def rule_swap():
    """Same-pool rule-swap control (user instruction 2026-10-12)."""
    rel = "exp/out_final/rule_swap.json"
    if not newer(rel):
        return False
    try:
        data = json.loads((ROOT / rel).read_text())
    except Exception as exc:
        return fail(f"bad json {rel}: {exc}")
    if not isinstance(data, dict):
        return fail("rule_swap.json must be an object")
    for field in ("rule", "guidance_source", "pool"):
        v = data.get(field)
        if not isinstance(v, str) or not v.strip():
            return fail(f"rule_swap.json needs a non-empty {field} string")
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        return fail("rule_swap.json needs a datasets{} object")
    for key, n_min in (("refcoco", 10834), ("refcocoplus", 10758), ("refcocog", 4896)):
        entry = datasets.get(key)
        if not isinstance(entry, dict):
            return fail(f"rule_swap.json datasets.{key} missing")
        n = entry.get("n")
        if not isinstance(n, int) or n < n_min:
            return fail(f"rule_swap.json datasets.{key} n={n}, need >= {n_min}")
        for metric in ("mIoU", "oIoU"):
            if not isinstance(entry.get(metric), (int, float)):
                return fail(f"rule_swap.json datasets.{key} needs numeric {metric}")
    pdf_path = ROOT / "paper" / "main.pdf"
    rel_path = ROOT / rel
    if not pdf_path.is_file() or pdf_path.stat().st_mtime < rel_path.stat().st_mtime:
        return fail("paper/main.pdf must be rebuilt after exp/out_final/rule_swap.json")
    return True


def learned_prior():
    """Learned localization prior (user directive 2026-10-12)."""
    rel = "exp/out_final/learned_prior.json"
    if not newer(rel):
        return False
    try:
        data = json.loads((ROOT / rel).read_text())
    except Exception as exc:
        return fail(f"bad json {rel}: {exc}")
    comp = data.get("trained_component") if isinstance(data, dict) else None
    if not isinstance(comp, dict):
        return fail("learned_prior.json needs trained_component{}")
    p = comp.get("params")
    if not isinstance(p, (int, float)) or p <= 0:
        return fail("learned_prior.json trained_component.params must be a positive number")
    for field in ("training_data", "supervision"):
        v = comp.get(field)
        if not isinstance(v, str) or not v.strip():
            return fail(f"learned_prior.json trained_component.{field} must be a non-empty string")
    datasets = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(datasets, dict):
        return fail("learned_prior.json needs a datasets{} object")
    entry = datasets.get("grefcoco")
    if not isinstance(entry, dict):
        return fail("learned_prior.json datasets.grefcoco missing")
    n = entry.get("n")
    if not isinstance(n, int) or n < 14229:
        return fail(f"learned_prior.json datasets.grefcoco n={n}, need >= 14229 (full val)")
    for metric in ("cIoU", "gIoU"):
        if not isinstance(entry.get(metric), (int, float)):
            return fail(f"learned_prior.json datasets.grefcoco needs numeric {metric}")
    pdf_path = ROOT / "paper" / "main.pdf"
    rel_path = ROOT / rel
    if not pdf_path.is_file() or pdf_path.stat().st_mtime < rel_path.stat().st_mtime:
        return fail("paper/main.pdf must be rebuilt after exp/out_final/learned_prior.json")
    return True


def wild_ideas():
    """Counterintuitive-mechanism probe batch (user directive 2026-10-12)."""
    rel = "exp/out_final/wild_ideas.json"
    if not newer(rel):
        return False
    try:
        data = json.loads((ROOT / rel).read_text())
    except Exception as exc:
        return fail(f"bad json {rel}: {exc}")
    probes = data.get("probes") if isinstance(data, dict) else None
    if not isinstance(probes, list) or len(probes) < 6:
        return fail("wild_ideas.json needs probes[] of length >= 6")
    for i, p in enumerate(probes):
        if not isinstance(p, dict):
            return fail(f"wild_ideas probe {i} is not an object")
        for field in ("hypothesis", "test", "outcome"):
            v = p.get(field)
            if not isinstance(v, str) or not v.strip():
                return fail(f"wild_ideas probe {i} needs a non-empty {field}")
    return True


def test_columns():
    """Full test-column coverage vs the compared table (user question 2026-10-12)."""
    rel = "exp/out_final/test_columns.json"
    if not newer(rel):
        return False
    try:
        data = json.loads((ROOT / rel).read_text())
    except Exception as exc:
        return fail(f"bad json {rel}: {exc}")
    splits = data.get("splits") if isinstance(data, dict) else None
    if not isinstance(splits, dict):
        return fail("test_columns.json needs a splits{} object")
    for key in ("refcoco_testA", "refcoco_testB", "refcocoplus_testA", "refcocoplus_testB", "refcocog_testu"):
        entry = splits.get(key)
        if not isinstance(entry, dict):
            return fail(f"test_columns.json splits.{key} missing")
        n = entry.get("n")
        if not isinstance(n, int) or n < 1000:
            return fail(f"test_columns.json splits.{key} n must be an int >= 1000")
        for metric in ("mIoU", "oIoU"):
            if not isinstance(entry.get(metric), (int, float)):
                return fail(f"test_columns.json splits.{key} needs numeric {metric}")
        pub = entry.get("published")
        if not isinstance(pub, dict) or not isinstance(pub.get("mIoU"), (int, float)):
            return fail(f"test_columns.json splits.{key} needs published{{mIoU}}")
    pdf_path = ROOT / "paper" / "main.pdf"
    rel_path = ROOT / rel
    if not pdf_path.is_file() or pdf_path.stat().st_mtime < rel_path.stat().st_mtime:
        return fail("paper/main.pdf must be rebuilt after exp/out_final/test_columns.json")
    return True


def transfer_wall():
    """Transfer-inversion investigation (user directive 2026-10-13)."""
    rel = "exp/out_final/transfer_wall.json"
    if not newer(rel):
        return False
    try:
        data = json.loads((ROOT / rel).read_text())
    except Exception as exc:
        return fail(f"bad json {rel}: {exc}")
    if not isinstance(data, dict):
        return fail("transfer_wall.json must be an object")
    mech = data.get("mechanism")
    if not isinstance(mech, list) or len(mech) < 3:
        return fail("transfer_wall.json needs mechanism[] of length >= 3")
    for i, m in enumerate(mech):
        if not isinstance(m, dict):
            return fail(f"transfer_wall mechanism {i} is not an object")
        for field in ("hypothesis", "measurement", "verdict"):
            v = m.get(field)
            if not isinstance(v, str) or not v.strip():
                return fail(f"transfer_wall mechanism {i} needs a non-empty {field}")
    arms = data.get("arms")
    if not isinstance(arms, list) or len(arms) < 2:
        return fail("transfer_wall.json needs arms[] of length >= 2")
    for i, a in enumerate(arms):
        if not isinstance(a, dict):
            return fail(f"transfer_wall arm {i} is not an object")
        if not isinstance(a.get("name"), str) or not a.get("name").strip():
            return fail(f"transfer_wall arm {i} needs a non-empty name")
        if not isinstance(a.get("val_cIoU"), (int, float)):
            return fail(f"transfer_wall arm {i} needs numeric val_cIoU")
    verdict = data.get("verdict")
    if not isinstance(verdict, str) or not verdict.strip():
        return fail("transfer_wall.json needs a non-empty verdict string")
    pdf_path = ROOT / "paper" / "main.pdf"
    rel_path = ROOT / rel
    if not pdf_path.is_file() or pdf_path.stat().st_mtime < rel_path.stat().st_mtime:
        return fail("paper/main.pdf must be rebuilt after exp/out_final/transfer_wall.json")
    return True


def component_provenance():
    """Contribution provenance audit (operator directive 2026-10-15)."""
    rel = "exp/out_final/component_provenance.json"
    if not newer(rel):
        return False
    try:
        data = json.loads((ROOT / rel).read_text())
    except Exception as exc:
        return fail(f"bad json {rel}: {exc}")
    if not isinstance(data, dict):
        return fail("component_provenance.json must be an object")
    comps = data.get("components")
    if not isinstance(comps, list) or len(comps) < 8:
        return fail("component_provenance.json needs components[] of length >= 8")
    allowed = {"ours", "released_hybridgl", "generic_frozen"}
    for i, c in enumerate(comps):
        if not isinstance(c, dict):
            return fail(f"component_provenance component {i} is not an object")
        for field in ("name", "role", "evidence"):
            v = c.get(field)
            if not isinstance(v, str) or not v.strip():
                return fail(f"component_provenance component {i} needs a non-empty {field}")
        if c.get("provenance") not in allowed:
            return fail(f"component_provenance component {i} provenance must be one of {sorted(allowed)}")
    hb = data.get("head_rebuild")
    arms = hb.get("arms") if isinstance(hb, dict) else None
    if not isinstance(arms, list) or len(arms) < 2:
        return fail("component_provenance.json needs head_rebuild.arms[] of length >= 2")
    ours_only = 0
    for i, a in enumerate(arms):
        if not isinstance(a, dict):
            return fail(f"head_rebuild arm {i} is not an object")
        if not isinstance(a.get("name"), str) or not a["name"].strip():
            return fail(f"head_rebuild arm {i} needs a non-empty name")
        if not isinstance(a.get("uses_released_scores"), bool):
            return fail(f"head_rebuild arm {i} needs uses_released_scores bool")
        ev = a.get("evidence")
        if not isinstance(ev, str) or not newer(ev):
            return fail(f"head_rebuild arm {i} needs fresh evidence")
        val = a.get("val")
        if not isinstance(val, dict):
            return fail(f"head_rebuild arm {i} needs val{{}}")
        for key in ("refcoco", "refcocoplus", "refcocog"):
            if not isinstance(val.get(key), (int, float)):
                return fail(f"head_rebuild arm {i} val.{key} must be numeric")
        if a["uses_released_scores"] is False:
            ours_only += 1
    if ours_only < 1:
        return fail("head_rebuild needs at least one arm with uses_released_scores=false")
    verdict = data.get("verdict")
    if not isinstance(verdict, str) or not verdict.strip():
        return fail("component_provenance.json needs a non-empty verdict string")
    pdf_path = ROOT / "paper" / "main.pdf"
    if not pdf_path.is_file() or pdf_path.stat().st_mtime < (ROOT / rel).stat().st_mtime:
        return fail("paper/main.pdf must be rebuilt after exp/out_final/component_provenance.json")
    return True


def as_percent(value):
    value = float(value)
    if value <= 1.5:
        value *= 100.0
    return value


def sota_line():
    """Every dataset must beat the best published number, not our own baseline."""
    stamp = ROOT / "sota_bar_started.txt"
    if not stamp.is_file():
        return fail("sota_bar_started.txt missing")
    floors = [
        ("exp/out_final/main_refcoco.json", "mIoU", 58.42, "GLOD-RIS"),
        ("exp/out_final/main_refcocoplus.json", "mIoU", 54.01, "GLOD-RIS"),
        ("exp/out_final/main_refcocog.json", "mIoU", 57.85, "GLOD-RIS"),
        # 2026-10-08 operator change (user instruction): gRefCOCO floor moved from
        # fully-supervised InstAlign (68.94/74.34) to same-methodology zero-shot
        # PSALM (ECCV 2024, Table 12: val cIoU 42.0 / gIoU 43.3). Supervised rows
        # stay as labelled reference rows in the paper, never as floors.
        # Escalation rule ("if a newer same-class paper is higher, raise
        # best_published and beat that"): SELF1E-8B (CVPR 2026, arXiv 2603.19026)
        # zero-shot val cIoU 44.4 > PSALM 42.0, so the cIoU floor is compiled in at
        # 44.4; best_published.cIoU must be >= this and shipped.cIoU must beat it.
        # 2026-10-09 user instruction: cIoU must exceed the run's own record -- the
        # floor is the previous shipped cIoU, so best_published must be raised to it
        # and shipped must strictly beat it (push toward the supervised class via
        # the declared light-supervision allowance).
        ("exp/out_final/main_grefcoco.json", "cIoU", 50.99, "previous shipped record (this run)"),
        ("exp/out_final/main_grefcoco.json", "gIoU", 43.3, "PSALM (zero-shot)"),
    ]
    ok = True
    seen = set()
    for rel, metric, floor, rival in floors:
        if rel not in seen:
            if not newer_than(rel, stamp):
                ok = False
                continue
            seen.add(rel)
        data = json.loads((ROOT / rel).read_text())
        shipped = data.get("shipped") if isinstance(data, dict) else None
        if not isinstance(shipped, dict) or metric not in shipped:
            ok = fail(f"{rel} shipped.{metric} missing")
            continue
        ours = as_percent(shipped[metric])
        published = shipped.get("best_published")
        if not isinstance(published, dict) or metric not in published:
            ok = fail(f"{rel} shipped.best_published.{metric} missing")
            continue
        best = as_percent(published[metric])
        if best + 1e-6 < floor:
            ok = fail(f"{rel} best_published {metric}={best:.2f} is below {rival} {floor}")
            continue
        if ours <= best:
            ok = fail(f"{rel} {metric} {ours:.2f} does not beat published {best:.2f}")
    return ok


def newer_than(rel, stamp):
    path = ROOT / rel
    if not path.is_file():
        return fail(f"missing {rel}")
    if path.stat().st_mtime < stamp.stat().st_mtime:
        return fail(f"stale {rel}")
    return True


def pdf():
    path = ROOT / "paper" / "main.pdf"
    if not path.is_file() or path.stat().st_mtime < STAMP.stat().st_mtime:
        return fail("paper/main.pdf missing or older than this run")
    if path.stat().st_size < 400000:
        return fail("paper/main.pdf is too small to contain the required figures")
    return True


def main():
    if not STAMP.is_file():
        print("FAIL: extreme_started.txt missing")
        return 1
    checks = [tex_has_appendix, architecture, sota, ablation, mains, graveyard, protocol_audit, metrics_sota, analysis_pack, glm_swap, glodris_repro, rule_swap, wild_ideas, learned_prior, test_columns, transfer_wall, component_provenance, sota_line, pdf]
    passed = all(check() for check in checks)
    print("PASS" if passed else "NOT_READY")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
