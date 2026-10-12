#!/bin/bash
# test-column driver, split-list driven:  bash run_test_columns2.sh <vis> "ds:split:tag" ...
cd /home/msj_team/Jacob/research/ris
PY=/home/msj_team/.conda/envs/glod-ris/bin/python
VIS=$1; shift
mkdir -p exp/out_test_columns
D=exp/out_test_columns/driver_${VIS}.log
echo "=== [$(date)] driver vis=$VIS start" > $D
for spec in "$@"; do
  ds=${spec%%:*}; rest=${spec#*:}; sp=${rest%%:*}; tag=${rest#*:}
  echo "=== [$(date)] records $ds/$sp" >> $D
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$VIS $PY exp/exp4_guided_full.py \
      --dataset "$ds" --split "$sp" --tag "tc_${tag}" --output exp/out_test_columns \
      >> exp/out_test_columns/records_${tag}.log 2>&1
  echo "records rc=$? :: $(grep -o '\"n\": [0-9]*' exp/out_test_columns/records_${tag}.log | tail -1)" >> $D
  echo "=== [$(date)] boxes $ds/$sp" >> $D
  CUDA_VISIBLE_DEVICES=$VIS $PY exp/boxes_for_records.py \
      --dataset "$ds" --split "$sp" --records exp/out_test_columns/records_tc_${tag}.jsonl \
      --phrase-source sent --device cuda:0 >> exp/out_test_columns/boxes_${tag}.log 2>&1
  echo "boxes rc=$? :: $(tail -1 exp/out_test_columns/boxes_${tag}.log)" >> $D
  echo "=== [$(date)] eval $ds/$sp" >> $D
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$VIS $PY exp/exp40_final_eval.py \
      --dataset "$ds" --split "$sp" --phrase-source sent \
      --records exp/out_test_columns/records_tc_${tag}.jsonl \
      --output exp/out_test_columns/main_${tag}.json >> exp/out_test_columns/eval_${tag}.log 2>&1
  echo "eval rc=$? :: $(tail -1 exp/out_test_columns/eval_${tag}.log)" >> $D
done
echo "ALLDONE $(date)" >> $D
