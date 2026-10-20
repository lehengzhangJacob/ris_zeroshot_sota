#!/usr/bin/env python
"""Build per-sentence PSALM referring-evaluation JSONs from the official RefCOCO
splits (one entry per (ref, sentence), matching our unified protocol)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/msj_team/Jacob/sys/glod_ris_repro")
from glod.data import RefDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/out_final/psalm_eval"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"refcoco": "refcoco", "refcocoplus": "refcoco+", "refcocog": "refcocog"}
for key, name in NAMES.items():
    ds = RefDataset(Path("/home/msj_team/Jacob/sys/glod_ris_repro/data/refer"), name, "val")
    rows = []
    for ref in ds.refs:
        img = ds.images[ref["image_id"]]
        ann = ds.annotations[ref["ann_id"]]
        seg = ann["segmentation"]
        a = dict(id=int(ann["id"]), image_id=int(ref["image_id"]),
                 category_id=int(ann["category_id"]), iscrowd=int(ann.get("iscrowd", 0)),
                 area=float(ann.get("area", 0.0)), bbox=list(ann["bbox"]),
                 segmentation=seg)
        for s in ref["sentences"]:
            rows.append(dict(
                image_info=dict(file_name=img["file_name"], height=int(img["height"]),
                                width=int(img["width"])),
                anns=[a], new_img_id=int(ref["image_id"]),
                instruction=[dict(sent=s["sent"])],
                ref_id=int(ref["ref_id"]), sent_id=int(s["sent_id"])))
    p = OUT / f"{key}_val_persentence.json"
    p.write_text(json.dumps(rows))
    print(key, len(rows), p)
