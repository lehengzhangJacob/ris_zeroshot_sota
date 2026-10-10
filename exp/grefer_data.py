"""gRefCOCO (GRES) loader.

Unlike RefCOCO-family references, a gRefCOCO reference may point to *several*
annotations (multi-target) or to none at all (`no_target`, ann_id == [-1]).
`ground_truth` therefore returns the union mask, or an all-False mask for a
no-target reference; `n_targets` says which case it is.

Format: grefs(unc).json + instances.json, both from the official release
(huggingface.co/datasets/FudanCVL/gRefCOCO). Images are COCO train2014, shared
with the RefCOCO-family splits already on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GreferDataset:
    def __init__(self, root, split="val"):
        self.root = Path(root)
        self.split = split
        self.instances = json.loads((self.root / "grefcoco/instances.json").read_text())
        self.images = {x["id"]: x for x in self.instances["images"]}
        self.annotations = {x["id"]: x for x in self.instances["annotations"]}
        refs = json.loads((self.root / "grefcoco/grefs(unc).json").read_text())
        self.refs = sorted([r for r in refs if r["split"] == split],
                           key=lambda r: (r["ref_id"], r["image_id"]))
        self.total_refs = len(self.refs)

    def image_path(self, ref):
        return self.root / "images/train2014" / ref["file_name"]

    def target_ann_ids(self, ref):
        return [a for a in ref["ann_id"] if a >= 0 and a in self.annotations]

    def ground_truth(self, ref):
        """Union mask over the reference's annotations; all-False when none."""
        ann_ids = self.target_ann_ids(ref)
        if not ann_ids:
            image = self.images[ref["image_id"]]
            return np.zeros((image["height"], image["width"]), dtype=bool)
        from pycocotools import mask as mask_utils
        image = self.images[ref["image_id"]]
        h, w = image["height"], image["width"]
        union = np.zeros((h, w), dtype=bool)
        for ann_id in ann_ids:
            segmentation = self.annotations[ann_id]["segmentation"]
            if isinstance(segmentation, list):
                rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, h, w))
            elif isinstance(segmentation["counts"], list):
                rle = mask_utils.frPyObjects(segmentation, h, w)
            else:
                rle = segmentation
            union |= mask_utils.decode(rle).astype(bool)
        return union

    def n_targets(self, ref):
        return len(self.target_ann_ids(ref))
