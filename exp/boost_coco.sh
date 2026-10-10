#!/bin/bash
# When the RefCOCO+ resume finishes, GPU1 is free.  A single bf16 job there is
# ~3x the throughput of the two 8-bit shards combined (1.2 s vs 3.7 s per
# pair), so add it: it re-walks the RefCOCO records, skips the cached pairs
# instantly, and finishes the remainder.  No jobs are killed, so the finalize
# chain's "wait for box processes" condition is never falsely satisfied.
set -u
source /home/anaconda3/etc/profile.d/conda.sh
conda activate glod-ris
cd /home/msj_team/Jacob/research/ris
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[boost] waiting for the RefCOCO+ resume at $(date)"
while pgrep -f "records_refcocoplus" > /dev/null; do sleep 60; done
echo "[boost] launching bf16 RefCOCO booster at $(date)"
python exp/boxes_for_records.py --records exp/out_extreme/records_refcoco.jsonl \
  --dataset refcoco --split val --phrase-source sent --device cuda:0
echo "[boost] booster finished at $(date)"
