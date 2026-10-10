#!/bin/bash
# Wait for the running RefCOCO+ box job, then extract RefCOCOg, then RefCOCO.
source /home/anaconda3/etc/profile.d/conda.sh
conda activate glod-ris
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "boxes_for_records.py --records exp/out_extreme/records_refcocoplus.jsonl" > /dev/null; do
  sleep 60
done
cd /home/msj_team/Jacob/research/ris
python exp/boxes_for_records.py --records exp/out_extreme/records_refcocog.jsonl --dataset refcocog --split val --phrase-source sent --device cuda:0
python exp/boxes_for_records.py --records exp/out_extreme/records_refcoco.jsonl --dataset refcoco --split val --phrase-source sent --device cuda:0
echo "QUEUE COMPLETE"
