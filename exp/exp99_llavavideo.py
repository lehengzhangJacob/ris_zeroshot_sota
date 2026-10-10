#!/usr/bin/env python
"""EXP-99 — hand-roll LLaVA-Video-7B-Qwen2 from its checkpoint (no lmms-lab fork).

The one remaining lever (exp94: box-quality headroom +0.10 r) needs a second
localizer.  Qwen2.5-VL's cached shards are hash-verified corrupt; LLaVA-Video's
weights are complete and self-contained: the checkpoint holds the SigLIP-so400m
vision tower (model.vision_tower.vision_tower.*, 421 tensors), the mlp2x_gelu
projector (model.mm_projector.{0,2}), Qwen2-7B (model.layers.*), model.norm,
lm_head, model.image_newline.  Every component maps onto a native transformers
class, so the architecture can be assembled directly.

Verified the same way the corrupt assembly was caught: a coherent,
image-specific description on known images (the corrupt model hallucinated
different content per prompt).  Then a box prompt.

    CUDA_VISIBLE_DEVICES=3 python exp/exp99_llavavideo.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open
from transformers import (Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig,
                          SiglipVisionModel, AutoTokenizer)

SNAP = glob.glob("/home/msj_team/.cache/huggingface/hub/"
                 "models--lmms-lab--LLaVA-Video-7B-Qwen2/snapshots/*")[0]
IMG = "/home/msj_team/Jacob/sys/glod_ris_repro/data/refer/images/train2014"


def load_state():
    sd = {}
    for f in sorted(glob.glob(SNAP + "/model-*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                sd[k] = h.get_tensor(k)
    return sd


def build(device="cuda:0"):
    cfg = json.loads(Path(SNAP, "config.json").read_text())
    sd = load_state()

    # ---- vision tower (SigLIP so400m patch14-384) ----
    vcfg = SiglipVisionConfig(hidden_size=1152, intermediate_size=4304,
                              num_hidden_layers=27, num_attention_heads=16,
                              image_size=384, patch_size=14)
    vision = SiglipVisionModel(vcfg)
    vmap = {k.replace("model.vision_tower.vision_tower.", ""): v
            for k, v in sd.items() if k.startswith("model.vision_tower.")}
    missing = vision.load_state_dict(vmap, strict=False)
    print("vision load: missing", len(missing.missing_keys),
          "unexpected", len(missing.unexpected_keys), flush=True)

    # ---- projector (mlp2x_gelu) ----
    proj = nn.Sequential(
        nn.Linear(1152, 3584),
        nn.GELU(),
        nn.Linear(3584, 3584),
    )
    proj[0].weight.data.copy_(sd["model.mm_projector.0.weight"])
    proj[0].bias.data.copy_(sd["model.mm_projector.0.bias"])
    proj[2].weight.data.copy_(sd["model.mm_projector.2.weight"])
    proj[2].bias.data.copy_(sd["model.mm_projector.2.bias"])

    # ---- LLM (Qwen2-7B) ----
    qcfg = Qwen2Config(hidden_size=cfg["hidden_size"],
                       intermediate_size=cfg["intermediate_size"],
                       num_hidden_layers=cfg["num_hidden_layers"],
                       num_attention_heads=cfg["num_attention_heads"],
                       num_key_value_heads=cfg["num_key_value_heads"],
                       vocab_size=cfg["vocab_size"],
                       max_position_embeddings=cfg["max_position_embeddings"],
                       rms_norm_eps=cfg["rms_norm_eps"],
                       rope_theta=cfg["rope_theta"],
                       tie_word_embeddings=False)
    llm = Qwen2ForCausalLM(qcfg)
    lmap = {k: v for k, v in sd.items()
            if k.startswith("model.") and not k.startswith("model.vision_tower")
            and not k.startswith("model.mm_projector")}
    lmap["lm_head.weight"] = sd["lm_head.weight"]
    missing = llm.load_state_dict(lmap, strict=False)
    print("llm load: missing", len(missing.missing_keys),
          "unexpected", len(missing.unexpected_keys), flush=True)

    image_newline = sd["model.image_newline"]
    tok = AutoTokenizer.from_pretrained(SNAP)
    dtype = torch.bfloat16
    return (vision.to(device, dtype).eval(), proj.to(device, dtype).eval(),
            llm.to(device, dtype).eval(), image_newline.to(device, dtype), tok,
            cfg)


def encode_image(vision, proj, image_newline, image, device):
    """384 single-view: patch tokens -> projector -> + newline row."""
    from transformers import SiglipImageProcessor
    proc = SiglipImageProcessor(size={"height": 384, "width": 384})
    px = proc(images=image, return_tensors="pt")["pixel_values"].to(device,
                                                                    torch.bfloat16)
    with torch.inference_mode():
        h = vision(pixel_values=px).last_hidden_state[0]      # (729, 1152)
        emb = proj(h)                                          # (729, 3584)
    emb = torch.cat([emb, image_newline.unsqueeze(0)], 0)      # 730 tokens
    return emb


def generate(vision, proj, llm, image_newline, tok, cfg, image, question,
             device="cuda:0", max_new_tokens=40):
    emb_img = encode_image(vision, proj, image_newline, image, device)
    n_img = emb_img.shape[0]
    prompt = (f"<|im_start|>user\n<image>\n{question}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    img_tok = cfg["image_token_index"]
    # build inputs_embeds with the image tokens replaced
    emb_txt = llm.get_input_embeddings()(ids)[0]
    pos = (ids[0] == img_tok).nonzero().flatten()
    assert len(pos) == 1, f"expected 1 image token, got {len(pos)}"
    p = int(pos[0])
    full = torch.cat([emb_txt[:p], emb_img, emb_txt[p + 1:]], 0).unsqueeze(0)
    with torch.inference_mode():
        out = llm.generate(inputs_embeds=full, max_new_tokens=max_new_tokens,
                           do_sample=False)
    return tok.batch_decode(out[:, full.shape[1] - full.shape[1]:] if False
                            else out, skip_special_tokens=True)[0]


def main() -> int:
    dev = "cuda:0"
    vision, proj, llm, image_newline, tok, cfg = build(dev)
    print("assembled OK; image_token_index", cfg["image_token_index"], flush=True)
    for fn, q in [
        ("COCO_train2014_000000000716.jpg", "Describe this image in one sentence."),
        ("COCO_train2014_000000000716.jpg", "What is in the image? List the main objects."),
    ]:
        img = Image.open(f"{IMG}/{fn}").convert("RGB")
        print(f"[{q[:40]}] ->", repr(generate(vision, proj, llm, image_newline,
                                              tok, cfg, img, q, dev)[:180]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
