#!/bin/bash
source /home/anaconda3/etc/profile.d/conda.sh
conda activate glod-ris
unset HF_ENDPOINT
for i in $(seq 1 400); do
  if huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct > /tmp/qwen_dl_$i.log 2>&1; then
    echo "COMPLETE after $i attempts"; exit 0
  fi
  sleep 3
done
echo "GAVE UP"; exit 1
