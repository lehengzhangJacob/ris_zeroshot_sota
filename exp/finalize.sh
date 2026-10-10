#!/bin/bash
# Wait for all val box extraction to finish, then run the final evaluation
# chain.  Safe to re-run: every step is a deterministic re-analysis.
set -u
source /home/anaconda3/etc/profile.d/conda.sh
conda activate glod-ris
cd /home/msj_team/Jacob/research/ris
export PYTHONPATH="/home/msj_team/Jacob/sys/glod_ris_repro:${PYTHONPATH:-}"
export PATH="$HOME/.local/texlive/bin/x86_64-linux:${PATH:-}"

echo "[finalize] waiting for box extraction..."
while pgrep -f "boxes_for_records.py" > /dev/null; do sleep 60; done
echo "[finalize] extraction done at $(date)"

for spec in "refcoco+ exp/out_extreme/records_refcocoplus.jsonl exp/out_final/main_refcocoplus.json" \
            "refcocog exp/out_extreme/records_refcocog.jsonl exp/out_final/main_refcocog.json" \
            "refcoco exp/out_extreme/records_refcoco.jsonl exp/out_final/main_refcoco.json"; do
  set -- $spec
  echo "[finalize] exp40 $1"
  python exp/exp40_final_eval.py --dataset "$1" --records "$2" --phrase-source sent --output "$3" \
    || echo "[finalize] WARN exp40 failed for $1"
done

echo "[finalize] tables"
python exp/exp43_write_tables.py || echo "[finalize] WARN exp43"
python exp/exp44_paper_tables.py || echo "[finalize] WARN exp44"

echo "[finalize] qualitative figure"
python exp/exp45_qualitative.py || echo "[finalize] WARN exp45"

echo "[finalize] compile"
cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/final_tex1.log 2>&1 \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/final_tex2.log 2>&1
echo "[finalize] pdf: $(ls -la main.pdf 2>/dev/null | awk '{print $5}') bytes"
echo "[finalize] DONE at $(date)"
