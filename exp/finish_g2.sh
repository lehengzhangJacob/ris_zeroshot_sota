#!/bin/bash
# Re-run the RefCOCOg evaluation until the box cache is complete, then refresh
# the tables and rebuild the PDF.  Polls COVERAGE, not process names — the
# earlier watcher raced its own workers and fired on a partial cache.
set -u
source /home/anaconda3/etc/profile.d/conda.sh
conda activate glod-ris
cd /home/msj_team/Jacob/research/ris
export PYTHONPATH="/home/msj_team/Jacob/sys/glod_ris_repro:${PYTHONPATH:-}"
export PATH="$HOME/.local/texlive/bin/x86_64-linux:${PATH:-}"

for attempt in $(seq 1 60); do
  echo "[finish-g2] attempt $attempt at $(date)"
  out=$(python exp/exp40_final_eval.py --dataset refcocog \
        --records exp/out_extreme/records_refcocog.jsonl --phrase-source sent \
        --output exp/out_final/main_refcocog.json 2>&1)
  echo "$out" | tail -8
  if echo "$out" | grep -q "REFUSING"; then
    cov=$(echo "$out" | grep -o "coverage [0-9.]*" | tail -1)
    echo "[finish-g2] incomplete ($cov); workers still running: $(pgrep -cf 'records_refcocog.jsonl')"
    if [ "$(pgrep -cf 'records_refcocog.jsonl')" -eq 0 ]; then
      echo "[finish-g2] NO WORKERS AND INCOMPLETE — needs manual attention"
      exit 2
    fi
    sleep 600
    continue
  fi
  echo "[finish-g2] coverage complete; refreshing tables"
  python exp/exp43_write_tables.py
  python exp/exp44_paper_tables.py
  cd paper
  pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/fin2_1.log 2>&1
  pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/fin2_2.log 2>&1
  echo "[finish-g2] pdf: $(ls -la main.pdf | awk '{print $5}') bytes"
  echo "[finish-g2] DONE at $(date)"
  exit 0
done
echo "[finish-g2] gave up after 60 attempts"
exit 1
