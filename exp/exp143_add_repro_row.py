#!/usr/bin/env python
"""EXP-143 — add the GLOD-RIS independent-reproduction row to sota_table.json.

The reproduction (2026-10-10/11 operator instruction) is a measured row over
the three official val splits: our reimplementation's numbers, with the
published numbers carried per dataset for the audit.  Appends in place and is
idempotent (drops a previous row with the same name first); exp24/exp43 are
not re-run because their measured-row schema predates the mIoU+oIoU columns
and the current evidence files.
"""
from __future__ import annotations

import json
from pathlib import Path

OUTF = Path("exp/out_final")
NAME = "GLOD-RIS (our independent reproduction)"


def main() -> int:
    repro = json.loads((OUTF / "glodris_repro.json").read_text())
    table = json.loads((OUTF / "sota_table.json").read_text())
    row = dict(name=NAME, venue="this run", measured_by_us=True,
               evidence="exp/out_final/glodris_repro.json",
               setting="zero-shot, SAM ViT-H + CLIP ViT-B/16, GLM-4.1V-Thinking-Flash guidance")
    for key in ("refcoco", "refcocoplus", "refcocog"):
        e = (repro.get("datasets") or {}).get(key) or {}
        if e.get("mIoU") is None:
            continue
        row[key] = dict(mIoU=e["mIoU"], oIoU=e["oIoU"], n=e["n"],
                        published=e.get("published"),
                        source="independent reproduction, full official val split")
    if not any(k in row for k in ("refcoco", "refcocoplus", "refcocog")):
        print("glodris_repro.json has no dataset rows; refusing to add")
        return 1
    table["methods"] = [m for m in table["methods"] if m.get("name") != NAME] + [row]
    cap = table.get("caption", "").rstrip()
    note = ("The GLOD-RIS reproduction row is this run's independent reimplementation "
            "(their code is unreleased); it is reported below the published row and "
            "discussed in the reproduction audit.")
    if "independent reimplementation" not in cap:
        table["caption"] = (cap + (" " if cap else "") + note)
    (OUTF / "sota_table.json").write_text(json.dumps(table, indent=2))
    print(f"added row: {NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
