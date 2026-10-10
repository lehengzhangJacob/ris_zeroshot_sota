#!/bin/bash
# 1) wait for the RefCOCOg extraction, 2) resume RefCOCO+ (bf16, 8-bit fallback),
# 3) run the final evaluation chain.  Every step is a deterministic re-analysis.
set -u
source /home/anaconda3/etc/profile.d/conda.sh
conda activate glod-ris
cd /home/msj_team/Jacob/research/ris
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/msj_team/Jacob/sys/glod_ris_repro:${PYTHONPATH:-}"

echo "[queue] waiting for RefCOCOg at $(date)"
while pgrep -f "records_refcocog" > /dev/null; do sleep 60; done

echo "[queue] RefCOCO+ resume (bf16) at $(date)"
python exp/boxes_for_records.py --records exp/out_extreme/records_refcocoplus.jsonl \
  --dataset refcoco+ --split val --phrase-source sent --device cuda:0 \
  || {
    echo "[queue] bf16 failed, retrying 8-bit at $(date)"
    python exp/boxes_for_records.py --records exp/out_extreme/records_refcocoplus.jsonl \
      --dataset refcoco+ --split val --phrase-source sent --device cuda:0 --load-8bit
  }

echo "[queue] all extraction done at $(date)"
exec /home/msj_team/Jacob/research/ris/exp/finalize.sh
