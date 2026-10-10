#!/bin/bash
# Wait for the RefCOCOg box workers, then re-run the evaluation chain so the
# RefCOCOg main (and every table derived from it) carries full coverage.
set -u
source /home/anaconda3/etc/profile.d/conda.sh
conda activate glod-ris
cd /home/msj_team/Jacob/research/ris
export PYTHONPATH="/home/msj_team/Jacob/sys/glod_ris_repro:${PYTHONPATH:-}"
export PATH="$HOME/.local/texlive/bin/x86_64-linux:${PATH:-}"

echo "[finish-g] waiting for RefCOCOg workers at $(date)"
while pgrep -f "records_refcocog.jsonl" > /dev/null; do sleep 60; done

echo "[finish-g] exp40 refcocog at $(date)"
python exp/exp40_final_eval.py --dataset refcocog \
  --records exp/out_extreme/records_refcocog.jsonl --phrase-source sent \
  --output exp/out_final/main_refcocog.json

echo "[finish-g] tables"
python exp/exp43_write_tables.py
python exp/exp44_paper_tables.py

echo "[finish-g] compile"
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/fin_g1.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/fin_g2.log 2>&1
echo "[finish-g] pdf: $(ls -la main.pdf | awk '{print $5}') bytes"
echo "[finish-g] DONE at $(date)"
