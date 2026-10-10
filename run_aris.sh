#!/bin/bash
source /home/anaconda3/etc/profile.d/conda.sh
conda activate glod-ris
unset CUDA_VISIBLE_DEVICES
cd /home/msj_team/Jacob/research/ris
SESSION_FILE=/home/msj_team/Jacob/research/ris/aris-sota.session
LOG=/home/msj_team/Jacob/research/ris/aris-sota.log
STAMP=/home/msj_team/Jacob/research/ris/extreme_started.txt
SOTA_STAMP=/home/msj_team/Jacob/research/ris/sota_bar_started.txt
[ -f "$STAMP" ] || touch "$STAMP"
[ -f "$SOTA_STAMP" ] || touch "$SOTA_STAMP"

START_PROMPT=$(cat <<'EOF'
/research-pipeline "Beat published SOTA on every RIS validation set" — AUTO_PROCEED: true, auto_write: false, venue: CVPR, difficulty: hard, code review: false, render html: false, compact: true, base repo: https://github.com/fhgyuanshen/HybridGL

This run does not end when a reviewer scores 6/10 or says almost, and it does not end at MAX_ROUNDS=4. Those skill defaults are overridden. auto_write is false on purpose. The current paper is rejected: its best RefCOCO-family numbers are 55.58 / 46.03 / 51.00 mIoU against GLOD-RIS 58.42 / 54.01 / 57.85, and gRefCOCO at 23.11 cIoU / 20.67 gIoU is not a published SOTA. Do not polish that draft. Do not call /paper-writing until python3 extreme_bar.py prints PASS.

Read RESEARCH_BRIEF.md. Do not redo the literature survey. The old gated-growth result stays dead. If a one-line control explains a gain, kill that method, record it, and design a different one. Repeat until a method still leads after those controls.

Required before any submission draft, all newer than extreme_started.txt:
- exp/out_final/main_refcoco.json, main_refcocoplus.json, main_refcocog.json, main_grefcoco.json with n >= 10834, 10758, 4896, 2000. These are four full validation sets: RefCOCO, RefCOCO+, RefCOCOg, and gRefCOCO. Download gRefCOCO. Fewer than four datasets does not pass.
- exp/out_final/sota_table.json: {"methods": [...]} length >= 6. Each measured_by_us true row needs an evidence path. At least 3 methods must be run here, not copied from papers.
- exp/out_final/ablation.json: {"rows": [...]} length >= 5, each with an evidence path from a real run.
- refine-logs/METHOD_GRAVEYARD.md with at least two killed methods.
- exp/out_final/protocol_audit.json: cross-paper metric-protocol consistency (operator instruction 2026-10-08). Every shipped number and every cross-paper comparison must use the SAME metric protocol as the published numbers compared against. Audit all four datasets, gRefCOCO first: identify the exact protocol the compared papers use (official benchmark eval code where cited; for gRefCOCO the official evaluation, including the no-target treatment and which records enter each average), re-evaluate the saved predictions under it, and record per dataset: protocol_source, definition, aligned: true, plus the numbers after alignment. If alignment changes any number, update the main jsons, sota_table.json, and the paper tables and shipped lines --- the aligned numbers stand even if weaker. Document the protocol in the paper. Rebuild paper/main.pdf AFTER writing the audit (the gate checks the PDF is newer).
- exp/out_final/sensitivity.json, stability.json, complexity.json (user instruction 2026-10-09): parameter-sensitivity sweeps, stability tests, and model-complexity accounting (parameters, FLOPs per query, peak memory, latency; training_time states explicitly that nothing is trained and reports the selector-fitting cost). The SOTA table must carry both mIoU and oIoU for every RefCOCO-family row. Rebuild paper/main.pdf after these files (the bar gates them).
- paper/figures/architecture.pdf or .svg from /figure-spec, at least 20 KB. Not a matplotlib sketch.
- A CVPR paper with \appendix, then paper/main.pdf.
- Full-line SOTA, written into each main json as "shipped" after sota_bar_started.txt. RefCOCO, RefCOCO+, and RefCOCOg shipped.mIoU must be strictly greater than shipped.best_published.mIoU, and best_published.mIoU must be at least 58.42, 54.01, and 57.85. For gRefCOCO the comparison is same-methodology (operator instruction 2026-10-08): shipped.cIoU and shipped.gIoU must strictly beat shipped.best_published from the same method class, and the same-class published floors are at least 42.0 cIoU and 43.3 gIoU (PSALM zero-shot on gRefCOCO val, ECCV 2024 Table 12). Fully-supervised methods (InstAlign 68.94/74.34, GSVA(ft), ReLA) are reference rows in a separate labelled table, never the floor. A little supervision is allowed for gRefCOCO: a small component (e.g. a selector head, not a full supervised segmenter) may be trained on the gRefCOCO train split, tuned only on a held-out split carved from train, official val/test kept evaluation-only; declare trained-vs-frozen in the json and the paper. If a newer same-class paper is higher, raise best_published and beat that. Beating the reproduced HybridGL row is not SOTA. A method that loses any one dataset is killed.
If the current architecture misses any dataset, run Optuna for at least 300 completed trials before changing the architecture. The search objective is a held-out split, never the official validation or test split used for the SOTA claim. Write the study to exp/out_final/optuna_study.json with n_trials and the best params. If 300 trials still miss any dataset, change the model, record the failed architecture in the graveyard, and start a new study of at least 300 trials. Do not write the paper during a study.
python3 extreme_bar.py must print PASS. Until then, keep experimenting.

Use conda env glod-ris. Do not download PyTorch from download.pytorch.org. Reuse RefCOCO data already on disk. All 4 GPUs are available; check nvidia-smi and do not kill other users or the RVOS job. Do not call Codex. Reviews go to mcp__llm-chat__review with executor_model deepseek-flash. Record the reviewer as glm-5.3.
EOF
)

CONT_PROMPT=$(cat <<'EOF'
The previous turn stopped too early, or the connection dropped. Continue until python3 extreme_bar.py prints PASS. A 6/10 review is not completion. The current paper is rejected because it is not SOTA on every dataset. The teacher requires the baseline to be the best published number in the same method class, then an improvement on that. Stop stacking selectors on HybridGL. For RefCOCO, RefCOCO+, and RefCOCOg, reproduce GLOD-RIS first (published mIoU 58.42 / 54.01 / 57.85). The author repo has no runnable code; the local reimplementation is /home/msj_team/Jacob/sys/glod_ris_repro. Match those published numbers before changing the model, and report the gain over that unmodified reproduction. For gRefCOCO the standard is now same-methodology (operator instruction 2026-10-08, made by the user, not by you): beat the best published number from the same method class. The same-class floor is PSALM zero-shot on gRefCOCO val: cIoU 42.0, gIoU 43.3 (ECCV 2024 Table 12). InstAlign's code is not released (github.com/nero1342/InstAlign is a placeholder); do not chase it. Fully-supervised rows (InstAlign, GSVA(ft), ReLA) stay as labelled reference rows, never floors. A little supervision is allowed: a small component (e.g. a selector head) may be trained on the gRefCOCO train split, tuned on a held-out split carved from train, official val/test remain evaluation-only; declare trained-vs-frozen. The current 50.99 cIoU / 74.73 gIoU already beats the same-class floor: record shipped and best_published honestly under this standard (best_published = PSALM 42.0 / 43.3 with the citation) and keep family labels in the table; if a newer same-class paper is higher, raise best_published and beat that. Beat GLOD-RIS on RefCOCO, RefCOCO+, and RefCOCOg as before. Losing any dataset kills the method. Before changing the architecture, finish at least 300 Optuna trials on a held-out split, recorded in exp/out_final/optuna_study.json. If that study still misses any dataset, change the model and start another 300-trial study. Do not tune on the official validation or test split. Stay in conda env glod-ris. pip install optuna there if it is missing. Use free memory on all 4 GPUs and do not kill other processes. Do not call Codex. Reviews go to mcp__llm-chat__review with executor_model deepseek-flash. Metric-protocol consistency is now a hard requirement (operator instruction 2026-10-08; the gRefCOCO PASS carried a cross-paper comparability caveat). Every shipped number and every cross-paper comparison must be computed under the SAME metric protocol as the published numbers it is compared against --- align to the official evaluation the compared papers cite. Audit all four datasets, gRefCOCO first (its gIoU margin over PSALM/SELF1E was flagged as possibly inside protocol noise): for each, establish the exact protocol (for gRefCOCO the official gRefCOCO evaluation, including the no-target treatment and which records enter each average; for the RefCOCO family the protocol the compared papers and benchmarks use), re-evaluate the saved predictions under that protocol, and write exp/out_final/protocol_audit.json with a datasets{} entry per dataset (refcoco, refcocoplus, refcocog, grefcoco), each carrying non-empty protocol_source and definition strings and aligned: true, plus the numbers after alignment. If alignment changes any number, update the main jsons, sota_table.json, and the paper tables and shipped lines --- the aligned numbers stand even if weaker; never keep a headline number computed under a protocol that differs from the compared papers, and never revert to a mismatched protocol to pass a floor. If a floor then fails, improve the method genuinely (the light-supervision allowance applies). Document the metric protocol in the paper and rebuild paper/main.pdf AFTER writing the audit (the gate requires the PDF to be newer than the audit file). Do not end your turn until the audit file exists and the package reflects it. Fair-comparison and analysis pack (user instruction 2026-10-09; the bar now gates all of it). Same-metric comparison: extend exp/out_final/sota_table.json so EVERY RefCOCO-family row carries both mIoU and oIoU --- published rows take the compared papers' own oIoU numbers (e.g. GLOD-RIS Table I), our rows use the aligned re-evaluations; keep both cIoU and gIoU for gRefCOCO. Add testA/testB (and RefCOCOg test(u)) columns if you can produce them for the same frozen components within the shared-GPU budget; if not, state the val-column scope explicitly in the paper. Analysis pack, as new artifacts reflected in the paper: exp/out_final/sensitivity.json (Parameter Sensitivity Analysis: sweep the shipped rule's key hyperparameters --- box-fill weight and margin, cap, k, abstention threshold --- over meaningful ranges on held-out/val records, recording metric values per setting); exp/out_final/stability.json (Testing Stability: at least three tests, e.g. prompt-template paraphrases re-run on a held-out sample, repeat/determinism checks, box jitter or detector-source swap, recording the deltas); exp/out_final/complexity.json (Model Complexity: per component and totals --- parameters, FLOPs per query, peak memory, latency --- plus a training_time entry that states explicitly that this pipeline trains nothing and reports the selector-fitting cost; measure locally what is measurable, estimate the rest with stated assumptions). All three files must be new, and paper/main.pdf must be rebuilt after them. Prefer deterministic re-analysis of saved records; re-run models only where a measurement truly requires it. Push gRefCOCO cIoU beyond the shipped record (user question 2026-10-09; the bar's cIoU floor is now the previous record --- shipped.cIoU must strictly exceed 50.99 and best_published must be raised to it). The measured headroom in exp/out_final/gref_wall.json (oracle cumulative r 0.883 vs the 0.6894 needed at perfect abstention; realization needed 0.781 vs the current ~0.489) says the route is a stronger trained selector over the existing candidate pool: use the light-supervision allowance (train on gRefCOCO train, tune on a held-out split carved from train, official val/test evaluation-only; declare trained-vs-frozen). Aim to close the gap to the supervised class; if a focused attempt cannot clear a bound, document the measured ceiling with evidence instead of looping on closed axes. API budget discipline (operator note 2026-10-09): the shared API balance is small --- when waiting on long-running jobs, prefer the longest single waits the harness allows (or a background wait that notifies on completion) over frequent polls; never poll or call when nothing is running; batch checks; avoid repeat-verdict cycles. CRITICAL for monitoring: while a detached job you are watching is still running, do NOT end your turn after a status check --- keep the wait inside the turn (e.g. sleep ~590s bash calls, or a background wait) so checks do not cost a full turn each; ending the turn between checks trips the anti-spin guard after three sub-60s cycles. Component-matched GLM test (user instruction 2026-10-10): evaluate the shipped box-prior path with GLOD-RIS's exact MLLM --- GLM-4.1V-Thinking-Flash via the Zhipu API (model id glm-4.1v-thinking-flash, access confirmed working; endpoint https://open.bigmodel.cn/api/paas/v4/chat/completions; Authorization: Bearer <contents of /home/msj_team/Jacob/sys/glm_key>; images as base64 data URIs; it is a thinking model --- parse choices[0].message.content; cache every response). Same prompt template as the local LLaVA path; reuse the existing selection/scoring machinery with only the box source swapped; evaluate under the aligned protocol on the three RefCOCO-family official val splits (full splits preferred; if budget or latency forces sampling, state n and scope). Write exp/out_final/glm_mllm_swap.json: model string + per-dataset n and mIoU/oIoU under the GLM boxes, alongside the LLaVA numbers. The bar gates this file and requires paper/main.pdf rebuilt after it. Report the outcome honestly either way --- with this, the comparison is component-matched to GLOD-RIS: same SAM ViT-H, same CLIP ViT-B/16, same MLLM. GLOD-RIS reproduction, second attempt (user instruction 2026-10-10): complete a full official-val reproduction of GLOD-RIS (published mIoU/oIoU: 58.42/50.85, 54.01/45.96, 57.85/50.03) using the now-working GLM access. The earlier attempt at /home/msj_team/Jacob/sys/glod_ris_repro/ (see its README and glod/glm.py) aborted at 223/10,834 expressions on 'GLM invalid response: Grid must contain exactly n*n binary values'. Make the GLM layer production-grade: tolerant grid parsing (quoted digits, floats, category->cells dicts), a repair-retry prompt that states the parse error, enough max_tokens for a thinking model and handling of finish_reason=length, treat 429/5xx as retryable with backoff, and never abort the whole run on one expression --- cache every response, park persistent failures and continue, retry them at the end; reuse the existing 704MB cache where valid. Run the full splits for all three datasets and write exp/out_final/glodris_repro.json: per dataset n + mIoU + oIoU + published values + delta + GLM-failure statistics + explicit coverage. The bar gates this file and requires paper/main.pdf rebuilt after it. Then update the paper: if the reproduction lands at/near the published numbers, the GLOD-RIS comparison becomes same-environment (state that clearly); if it still cannot complete, record exactly where and why. About 26.5k GLM calls total --- cache everything, modest concurrency.
EOF
)

# Anti-spin guard: a worker that re-concludes the same BAR_FAIL in under 60s
# makes no progress. Three such consecutive cycles means the supervisor would
# keep relaunching into an identical wall -- stop instead of burning API spend.
short_fails=0
while true; do
  t0=$(date +%s)
  if [ -s "$SESSION_FILE" ]; then
    SESSION=$(cat "$SESSION_FILE")
    claude -p --resume "$SESSION" \
      --permission-mode bypassPermissions \
      --dangerously-skip-permissions \
      --output-format stream-json --verbose \
      --add-dir /home/msj_team/Jacob/sys/RIS \
      --effort high \
      "$CONT_PROMPT" >> "$LOG" 2>&1
  else
    claude -p \
      --permission-mode bypassPermissions \
      --dangerously-skip-permissions \
      --output-format stream-json --verbose \
      --add-dir /home/msj_team/Jacob/sys/RIS \
      --effort high \
      "$START_PROMPT" >> "$LOG" 2>&1
    python3 - "$LOG" "$SESSION_FILE" << 'PY'
import json, sys
log, dest = sys.argv[1], sys.argv[2]
sid = None
with open(log, errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") == "system" and obj.get("subtype") == "init" and obj.get("session_id"):
            sid = obj["session_id"]
if sid:
    open(dest, "w").write(sid + "\n")
PY
  fi
  dur=$(( $(date +%s) - t0 ))
  if [ -x ./backup.sh ]; then bash ./backup.sh >> "$LOG" 2>&1 || true; fi
  reason=$(python3 - "$LOG" << 'PY'
import json, sys
last = None
with open(sys.argv[1], errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") == "result":
            last = obj
text = "" if last is None else str(last.get("result", ""))
if last is None:
    print("NO_RESULT")
elif last.get("is_error") and any(s in text for s in ("ECONNRESET", "ETIMEDOUT", "ECONNREFUSED", "Unable to connect")):
    print("RESET")
elif last.get("is_error"):
    print("ERROR")
else:
    print("OK")
PY
)
  if [ "$reason" = "OK" ]; then
    if python3 extreme_bar.py >> "$LOG" 2>&1; then
      echo "{\"type\":\"supervisor\",\"event\":\"claude_exit\",\"reason\":\"BAR_PASS\"}" >> "$LOG"
      exit 0
    fi
    if [ "$dur" -lt 60 ]; then
      short_fails=$((short_fails + 1))
    else
      short_fails=0
    fi
    echo "{\"type\":\"supervisor\",\"event\":\"claude_exit\",\"reason\":\"BAR_FAIL\",\"dur_s\":$dur,\"short_fails\":$short_fails}" >> "$LOG"
    if [ "$short_fails" -ge 3 ]; then
      echo "{\"type\":\"supervisor\",\"event\":\"RUN_STOPPED\",\"reason\":\"degenerate_spin\"}" >> "$LOG"
      echo "stopped $(date -Is): 3 consecutive sub-60s BAR_FAIL cycles with no new work. The worker re-derived the same wall (see exp/out_final/gref_wall.json). Review before relaunching." > /home/msj_team/Jacob/research/ris/aris_stopped.txt
      exit 2
    fi
    sleep 10
    continue
  fi
  echo "{\"type\":\"supervisor\",\"event\":\"claude_exit\",\"reason\":\"$reason\"}" >> "$LOG"
  if [ "$reason" = "RESET" ] || [ "$reason" = "NO_RESULT" ]; then
    sleep 30
    continue
  fi
  exit 1
done
