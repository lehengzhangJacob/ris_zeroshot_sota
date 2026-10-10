"""GroundingDINO wrapper: per-(image, phrase) box detection, cached on disk.

Replaces GLOD-RIS's external-MLLM 3x3 grid prior with a dense on-board
open-vocabulary detector prior.  Detections are cached per
(file_name, phrase) so repeated experiments are free.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# NOTE: default huggingface.co works through the local proxy; hf-mirror.com does
# not serve these files reliably from this machine. Do not set HF_ENDPOINT here.

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
IMAGES = REPRO / "data/refer/images/train2014"
CACHE_DIR = Path("/home/msj_team/Jacob/research/ris/exp/out_gdino")
MODEL_ID = "IDEA-Research/grounding-dino-base"


class Gdino:
    def __init__(self, device="cuda:0", box_threshold=0.25, text_threshold=0.2):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
        self.model.eval()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, file_name, phrase):
        import hashlib
        key = hashlib.sha1(f"{MODEL_ID}|{self.box_threshold}|{file_name}|{phrase}".encode()).hexdigest()
        return CACHE_DIR / (key + ".json")

    def detect(self, file_name, phrase):
        """Return (boxes_xyxy, scores) for `phrase` in the image (cached)."""
        cache = self._cache_path(file_name, phrase)
        if cache.is_file():
            d = json.loads(cache.read_text())
            return d["boxes"], d["scores"]
        from PIL import Image
        image = Image.open(IMAGES / file_name).convert("RGB")
        text = phrase.strip().lower()
        if not text.endswith("."):
            text = text + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]])[0]
        boxes = [[float(v) for v in b] for b in result["boxes"].tolist()]
        scores = [float(s) for s in result["scores"].tolist()]
        cache.write_text(json.dumps(dict(boxes=boxes, scores=scores)))
        return boxes, scores


if __name__ == "__main__":
    import sys
    import time
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"
    g = Gdino(device=dev)
    print("loaded", MODEL_ID, "on", dev, flush=True)
    tests = [
        ("COCO_train2014_000000000716.jpg", "mouse"),
        ("COCO_train2014_000000000716.jpg", "laptop"),
        ("COCO_train2014_000000000716.jpg", "boy"),
    ]
    for fn, phr in tests:
        t0 = time.time()
        boxes, scores = g.detect(fn, phr)
        print(f"{fn} '{phr}': {len(boxes)} boxes in {time.time()-t0:.2f}s", flush=True)
        for b, s in list(zip(boxes, scores))[:4]:
            print(f"    [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {s:.3f}")
