"""A local VLM as the localization prior (replaces GLOD-RIS's external MLLM).

GLOD-RIS queries GLM-4.1V-Flash through an API for a 3x3 grid + a category
word.  We ask an on-board VLM (LLaVA-OneVision-7B, already cached locally) for
a *dense* box instead — strictly finer than the 3x3 grid, no network
dependency.  Boxes are cached per (file_name, phrase), so all downstream
selection experiments are free re-analyses.

The vision tower pass is cached per image and reused across that image's
expressions (a COCO image carries ~7 expressions), which is where most of the
compute would otherwise go.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
IMAGES = REPRO / "data/refer/images/train2014"
CACHE_DIR = Path("/home/msj_team/Jacob/research/ris/exp/out_vlm_boxes")
MODEL_TAG = "llava-onevision-qwen2-7b-ov"

PROMPT = ("Locate the {phrase} in this image. Reply with ONLY the bounding box as "
          "[x0,y0,x1,y1] in pixel coordinates measured from the top-left corner, "
          "or the single word none if it does not appear in the image.")

# existence probes (tagged cache namespaces; the box cache keys must stay
# byte-identical to the released runs, so the tag is only appended when set)
PROMPT_EXISTS = ("Is the {phrase} present in this image? "
                 "Answer with only the word yes or the word no.")
PROMPT_EXISTS2 = ("Does the image contain the {phrase}? "
                  "Answer with only the word yes or the word no.")
PROMPT_EXISTS3 = ("Look carefully at the image. Is the {phrase} visible? "
                  "Answer with only the word yes or the word no.")
PROMPT_CONF = ("How likely is the {phrase} to be present in this image? "
               "Answer with only a number from 0 (definitely absent) to 100 "
               "(definitely present).")


def parse_box(text, width, height):
    """Accept [x0,y0,x1,y1], (x0,y0),(x1,y1), or the word none.

    Values <= 1.5 are treated as normalised coordinates.
    """
    t = text.strip()
    if re.search(r"\bnone\b", t, re.I) and not re.search(r"\d", t):
        return None
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]
    if len(nums) < 4:
        return None
    x0, y0, x1, y1 = nums[:4]
    if max(abs(v) for v in (x0, y0, x1, y1)) <= 1.5:
        x0, y0, x1, y1 = x0 * width, y0 * height, x1 * width, y1 * height
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(width), x1), min(float(height), y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [x0, y0, x1, y1]


class LlavaBoxer:
    TAG = None       # cache namespace; None keeps the released key format
    PROMPT_TEXT = None  # None -> PROMPT

    def __init__(self, device="cuda:0", max_new_tokens=32):
        import torch
        from PIL import Image  # noqa: F401
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from llavaov import build, processor
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = build(device=device, verbose=False)
        self.proc = processor()
        if getattr(self.proc, "num_image_tokens", None) is None:
            self.proc.num_image_tokens = 729
        self._feat_cache = {}
        self._install_feature_cache()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- image-feature cache: the vision pass runs once per image ----------
    def _install_feature_cache(self, max_images=256):
        # The vision output per image is ~6 MB; unbounded caching OOM'd a
        # 48 GB card on the 2,500-image RefCOCOg test(u) split (2026-10-12).
        # Bounded FIFO: pairs sharing a recent image still hit the cache.
        model = self.model
        orig = model.get_image_features

        def cached(*args, **kwargs):
            key = self._cur_key
            if key is not None and key in self._feat_cache:
                return self._feat_cache[key]
            out = orig(*args, **kwargs)
            if key is not None:
                self._feat_cache[key] = out
                while len(self._feat_cache) > max_images:
                    self._feat_cache.pop(next(iter(self._feat_cache)))
            return out

        model.get_image_features = cached
        self._cur_key = None

    def _path(self, file_name, phrase):
        name = MODEL_TAG if self.TAG is None else f"{MODEL_TAG}|{self.TAG}"
        key = hashlib.sha1(f"{name}|{file_name}|{phrase.lower()}".encode()).hexdigest()
        return CACHE_DIR / (key + ".json")

    def box(self, file_name, phrase):
        p = self._path(file_name, phrase)
        if p.is_file():
            d = json.loads(p.read_text())
            return d["box"], d["raw"]
        from PIL import Image
        image = Image.open(IMAGES / file_name).convert("RGB")
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (self.PROMPT_TEXT or PROMPT).format(phrase=phrase)}]}]
        prompt = self.proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = self.proc(images=image, text=prompt, return_tensors="pt").to(self.device)
        inp = {k: (v.to(self.torch.bfloat16) if v.dtype.is_floating_point else v)
               for k, v in inp.items()}
        # keyed by IMAGE only: the vision features do not depend on the phrase,
        # and sentence prompts are unique per record (no phrase-level reuse)
        self._cur_key = file_name
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
        self._cur_key = None
        raw = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]
        box = parse_box(raw, image.width, image.height)
        p.write_text(json.dumps(dict(box=box, raw=raw[:200], w=image.width, h=image.height)))
        return box, raw

    def warm(self, file_name, phrases):
        """Run all phrases for one image with a warm feature cache."""
        outs = []
        for ph in phrases:
            outs.append(self.box(file_name, ph))
        return outs

    def box_many(self, pairs, batch_size=6, log_every=50):
        """Batched box(): pairs is a list of (file_name, phrase).

        Greedy decode is memory-bandwidth bound, so a batch of B amortises the
        weight read across B queries — roughly a B-fold speedup until the KV
        cache dominates.  Cached pairs are returned straight from disk; new
        answers are written per pair, so an interrupted run resumes for free.
        Returns a list aligned with `pairs`.
        """
        from PIL import Image
        out = [None] * len(pairs)
        todo = []
        for i, (fn, ph) in enumerate(pairs):
            p = self._path(fn, ph)
            if p.is_file():
                d = json.loads(p.read_text())
                out[i] = (d["box"], d["raw"])
            else:
                todo.append((i, fn, ph))
        # group by image: the same image is queried ~9x in gRefCOCO val
        todo.sort(key=lambda t: (t[1], t[2]))
        n_new = 0
        for s in range(0, len(todo), batch_size):
            chunk = todo[s:s + batch_size]
            imgs, texts = [], []
            for (i, fn, ph) in chunk:
                imgs.append(Image.open(IMAGES / fn).convert("RGB"))
                conv = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": (self.PROMPT_TEXT or PROMPT).format(phrase=ph)}]}]
                texts.append(self.proc.apply_chat_template(conv, add_generation_prompt=True))
            inp = self.proc(images=imgs, text=texts, return_tensors="pt",
                            padding=True).to(self.device)
            inp = {k: (v.to(self.torch.bfloat16) if v.dtype.is_floating_point else v)
                   for k, v in inp.items()}
            self._cur_key = None  # batched: the per-image feature cache does not apply
            with self.torch.inference_mode():
                gen = self.model.generate(**inp, max_new_tokens=self.max_new_tokens,
                                          do_sample=False)
            raws = self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for (i, fn, ph), raw, im in zip(chunk, raws, imgs):
                box = parse_box(raw, im.width, im.height)
                self._path(fn, ph).write_text(json.dumps(
                    dict(box=box, raw=raw[:200], w=im.width, h=im.height)))
                out[i] = (box, raw)
                n_new += 1
            if log_every and n_new % log_every < batch_size:
                print(f"  box_many: {n_new}/{len(todo)} new "
                      f"({len(pairs) - len(todo)} cached)", flush=True)
        return out

    def free_image(self, file_name):
        for k in [k for k in self._feat_cache if k.startswith(file_name + "|")]:
            self._feat_cache.pop(k, None)


class LlavaBoxer8bit(LlavaBoxer):
    """8-bit backend (~8 GB) for GPUs where the bf16 model does not fit.

    Uses the locally converted HF-format checkpoint so the standard
    bitsandbytes path applies.  Same cache namespace as LlavaBoxer: boxes are
    identical whichever backend produced them.
    """

    def __init__(self, device="cuda:0", max_new_tokens=32):
        import torch
        from transformers import (LlavaOnevisionForConditionalGeneration,
                                  LlavaOnevisionProcessor, BitsAndBytesConfig)
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        hf_dir = "/home/msj_team/Jacob/models/llava-onevision-qwen2-7b-ov-hf"
        qc = BitsAndBytesConfig(load_in_8bit=True)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            hf_dir, quantization_config=qc, device_map={"": 0}).eval()
        self.proc = LlavaOnevisionProcessor.from_pretrained(hf_dir)
        if getattr(self.proc, "num_image_tokens", None) is None:
            self.proc.num_image_tokens = 729
        self._feat_cache = {}
        self._install_feature_cache()  # a COCO image carries ~9 expressions
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


class Qwen2Boxer(LlavaBoxer):
    """Same interface, Qwen2.5-VL-7B backend (different cache namespace)."""

    def __init__(self, device="cuda:0", max_new_tokens=48):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_id = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.proc = AutoProcessor.from_pretrained(self.model_id, max_pixels=1280 * 28 * 28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16).to(device).eval()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, file_name, phrase):
        key = hashlib.sha1(f"qwen2.5vl|{file_name}|{phrase.lower()}".encode()).hexdigest()
        return CACHE_DIR / (key + ".json")

    def box(self, file_name, phrase):
        p = self._path(file_name, phrase)
        if p.is_file():
            d = json.loads(p.read_text())
            return d["box"], d["raw"]
        from PIL import Image
        image = Image.open(IMAGES / file_name).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (self.PROMPT_TEXT or PROMPT).format(phrase=phrase)}]}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[prompt], images=[image], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
        raw = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]
        box = parse_box(raw, image.width, image.height)
        p.write_text(json.dumps(dict(box=box, raw=raw[:200], w=image.width, h=image.height)))
        return box, raw


class ExistsBoxer(LlavaBoxer):
    """Binary presence probe: `yes` / `no`.  Own cache namespace."""
    TAG = "exists"
    PROMPT_TEXT = PROMPT_EXISTS


class ConfBoxer(LlavaBoxer):
    """Graded presence probe: 0-100.  Own cache namespace."""
    TAG = "conf"
    PROMPT_TEXT = PROMPT_CONF


class ExistsBoxer8bit(LlavaBoxer8bit):
    TAG = "exists"
    PROMPT_TEXT = PROMPT_EXISTS


class ConfBoxer8bit(LlavaBoxer8bit):
    TAG = "conf"
    PROMPT_TEXT = PROMPT_CONF


class Exists2Boxer8bit(LlavaBoxer8bit):
    TAG = "exists2"
    PROMPT_TEXT = PROMPT_EXISTS2


class Exists3Boxer8bit(LlavaBoxer8bit):
    TAG = "exists3"
    PROMPT_TEXT = PROMPT_EXISTS3


QwenBoxer = LlavaBoxer  # backwards-compatible alias for the earlier scripts


if __name__ == "__main__":
    import sys
    import time
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    b = LlavaBoxer(device=dev)
    print("loaded on", dev, flush=True)
    t0 = time.time()
    for fn, phr in [("COCO_train2014_000000000716.jpg", "the laptop"),
                    ("COCO_train2014_000000000716.jpg", "the little boy"),
                    ("COCO_train2014_000000000716.jpg", "the elephant")]:
        box, raw = b.box(fn, phr)
        print(f"'{phr}' -> {box}  raw={raw[:60]!r}", flush=True)
    print(f"3 calls in {time.time()-t0:.1f}s")
