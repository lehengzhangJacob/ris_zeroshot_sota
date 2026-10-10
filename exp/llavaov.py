"""Load the cached LLaVA-OneVision-7B (LLaVA repo format) into HF classes.

The local checkpoint (lmms-lab/llava-onevision-qwen2-7b-ov) uses the LLaVA
team's flat key names; transformers 4.49's LlavaOnevision class expects the
converted layout.  The mapping here is mechanical.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO_DIR = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/*"))[0]
HF_CFG_DIR = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots/*"))[0]
MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


def map_key(k: str) -> str:
    if k.startswith("model.vision_tower.vision_tower."):
        return "vision_tower." + k[len("model.vision_tower.vision_tower."):]
    if k.startswith("model.mm_projector.0."):
        return "multi_modal_projector.linear_1." + k[len("model.mm_projector.0."):]
    if k.startswith("model.mm_projector.2."):
        return "multi_modal_projector.linear_2." + k[len("model.mm_projector.2."):]
    if k.startswith("model.layers.") or k.startswith("model.embed_tokens.") or k.startswith("model.norm."):
        return "language_model.model." + k[len("model."):]
    if k.startswith("lm_head."):
        return "language_model." + k
    if k == "model.image_newline":
        return "image_newline"
    return k


def build(device="cuda:0", dtype=None, verbose=True):
    import torch
    from safetensors import safe_open
    from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration
    dtype = dtype or torch.bfloat16
    cfg = LlavaOnevisionConfig.from_pretrained(HF_CFG_DIR)
    sd = {}
    for shard in sorted(glob.glob(REPO_DIR + "/model-*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                sd[map_key(k)] = f.get_tensor(k)
    # the LLaVA repo checkpoint keeps Qwen2.5's original 152064 vocab; the
    # converted config declares the padded 152128.  Match the checkpoint.
    cfg.text_config.vocab_size = int(sd["language_model.model.embed_tokens.weight"].shape[0])
    model = LlavaOnevisionForConditionalGeneration(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"loaded {len(sd)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
        for m in list(missing)[:8]:
            print("  MISSING", m)
        for u in list(unexpected)[:8]:
            print("  UNEXPECTED", u)
    model = model.to(device, dtype).eval()
    return model


def processor():
    from transformers import LlavaOnevisionProcessor
    return LlavaOnevisionProcessor.from_pretrained(HF_CFG_DIR)


if __name__ == "__main__":
    import sys
    m = build(device=sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    print("params:", sum(p.numel() for p in m.parameters()) / 1e9, "B")
    p = processor()
    print("processor ok:", type(p).__name__)
