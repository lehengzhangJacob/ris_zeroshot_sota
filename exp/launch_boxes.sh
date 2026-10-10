#!/bin/bash
# RefCOCO+ (bf16) on GPU1, RefCOCO sharded 8-bit on GPU0/GPU2.
source /home/anaconda3/etc/profile.d/conda.sh
conda activate glod-ris
cd /home/msj_team/Jacob/research/ris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 nohup python exp/boxes_for_records.py --records exp/out_extreme/records_refcocoplus.jsonl --dataset refcoco+ --split val --phrase-source sent --device cuda:0 > exp/out_val_boxes_plus.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=0 nohup python exp/boxes_for_records.py --records exp/out_extreme/records_refcoco.jsonl --dataset refcoco --split val --phrase-source sent --shard 0 --nshards 2 --device cuda:0 --load-8bit > exp/out_val_coco_s0.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=2 nohup python exp/boxes_for_records.py --records exp/out_extreme/records_refcoco.jsonl --dataset refcoco --split val --phrase-source sent --shard 1 --nshards 2 --device cuda:0 --load-8bit > exp/out_val_coco_s1.log 2>&1 &
P3=$!
echo "launched $P1 $P2 $P3"
wait $P1
CUDA_VISIBLE_DEVICES=1 python exp/boxes_for_records.py --records exp/out_extreme/records_refcocog.jsonl --dataset refcocog --split val --phrase-source sent --device cuda:0
echo "ALL DONE"
