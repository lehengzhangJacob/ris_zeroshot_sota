#!/bin/bash
# 2026-10-12 instruction: complete the compared table (testA/testB/test(u)).
# Same frozen components and cache budget; aligned protocol; one pass per split.
cd /home/msj_team/Jacob/research/ris
PY=/home/msj_team/.conda/envs/glod-ris/bin/python
D=exp/out_test_columns/driver.log
mkdir -p exp/out_test_columns
echo "=== [$(date)] test-columns driver start" > $D
DEV=${DEV:-cuda:0}
run() {  # dataset split tag
  local ds=$1 sp=$2 tag=$3
  local rec=exp/out_test_columns/records_${tag}.jsonl
  echo "=== [$(date)] records $ds/$sp" >> $D
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${VIS:-1} $PY exp/exp4_guided_full.py \
      --dataset "$ds" --split "$sp" --tag "tc_${tag}" --output exp/out_test_columns \
      >> exp/out_test_columns/records_${tag}.log 2>&1
  echo "records rc=$? :: $(tail -1 exp/out_test_columns/records_${tag}.log)" >> $D
  echo "=== [$(date)] boxes $ds/$sp" >> $D
  CUDA_VISIBLE_DEVICES=${VIS:-1} $PY exp/boxes_for_records.py \
      --dataset "$ds" --split "$sp" --records exp/out_test_columns/records_tc_${tag}.jsonl \
      --phrase-source sent --device "$DEV" >> exp/out_test_columns/boxes_${tag}.log 2>&1
  echo "boxes rc=$? :: $(tail -1 exp/out_test_columns/boxes_${tag}.log)" >> $D
  echo "=== [$(date)] eval $ds/$sp" >> $D
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${VIS:-1} $PY exp/exp40_final_eval.py \
      --dataset "$ds" --split "$sp" --phrase-source sent \
      --records exp/out_test_columns/records_tc_${tag}.jsonl \
      --output exp/out_test_columns/main_${tag}.json >> exp/out_test_columns/eval_${tag}.log 2>&1
  echo "eval rc=$? :: $(tail -1 exp/out_test_columns/eval_${tag}.log)" >> $D
}
run refcoco testA refcoco_testA
run refcoco testB refcoco_testB
run "refcoco+" testA refcocoplus_testA
run "refcoco+" testB refcocoplus_testB
run refcocog test refcocog_testu
echo "ALLDONE $(date)" >> $D
