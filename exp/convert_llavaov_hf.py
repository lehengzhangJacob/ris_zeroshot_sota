#!/usr/bin/env python
"""Convert the cached LLaVA-OneVision-7B (LLaVA repo format) into a local
HF-format directory, so standard transformers loading paths — including
bitsandbytes 8-bit quantization and device_map offload — work on it.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llavaov import HF_CFG_DIR, build  # noqa: E402

OUT = Path("/home/msj_team/Jacob/models/llava-onevision-qwen2-7b-ov-hf")


def main() -> int:
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = build(device="cpu", dtype=torch.bfloat16, verbose=True)
    model.save_pretrained(OUT, safe_serialization=True)
    print(f"saved weights in {time.time()-t0:.0f}s", flush=True)
    for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                 "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
                 "chat_template.json", "generation_config.json", "config.json"):
        src = Path(HF_CFG_DIR) / name
        if src.is_file():
            shutil.copyfile(src, OUT / name)
    # keep the checkpoint's own vocab size in the saved config
    cfg = json.loads((OUT / "config.json").read_text())
    cfg["text_config"]["vocab_size"] = 152064
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
