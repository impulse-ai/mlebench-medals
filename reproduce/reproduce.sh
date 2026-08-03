#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — one command to independently reproduce our MLE-bench Lite-22
#                Class-A (autonomous) medals with OpenAI's official grader.
#
# What this does, per task:
#   1. prepares the task data via `mlebench prepare` (OpenAI's tooling)
#   2. runs OUR autonomous agent:  bench.lite22_controller --agent ees:standalone
#   3. grades the agent's submission with `mlebench grade-sample` (OpenAI's grader)
#   4. collects a medal table (task, official score, medal)
#
# The grader is OpenAI's `mlebench`, UNMODIFIED. We do not compute medals; we
# only run their grader and read its output. See reproduce/VERIFY.md.
#
# HONESTY: this reproduces the CLASS-A set (autonomous, one command). The
# Class-B GPU/assisted medals (histo, birds, jigsaw, leaf) are NOT run here —
# their capture + independent-grade evidence lives in reproduce/gpu/. See
# reproduce/EVIDENCE.md for the full 17-medal A/B ledger.
#
# MANUAL GATE (unavoidable, per OpenAI/Kaggle): you must accept each
# competition's rules on kaggle.com with your own account before `mlebench
# prepare` can download it. See reproduce/VERIFY.md section (a) for the URLs.
#
# Usage:
#   reproduce/reproduce.sh                        # full Class-A set
#   reproduce/reproduce.sh --tasks "spooky-author-identification nomad2018-predict-transparent-conductors"
#   reproduce/reproduce.sh --skip-prepare         # data already prepared
#   reproduce/reproduce.sh --output-root bench/runs/verify-$(date +%s)
#   reproduce/reproduce.sh --force                # re-run even if a result exists
#
# Resumable: tasks with an existing terminal result are skipped unless --force.
# =============================================================================
set -euo pipefail

# ---- Locate repo root (this script lives in <repo>/reproduce/) --------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- Defaults ---------------------------------------------------------------
# Class-A autonomous set (13 medals). See reproduce/EVIDENCE.md for the ledger.
DEFAULT_TASKS=(
  # --- gold (6) ---
  aerial-cactus-identification
  detecting-insults-in-social-commentary
  nomad2018-predict-transparent-conductors
  dogs-vs-cats-redux-kernels-edition
  plant-pathology-2020-fgvc7
  tabular-playground-series-dec-2021
  # --- silver (4) ---
  spooky-author-identification
  denoising-dirty-documents
  leaf-classification
  mlsp-2013-birds
  # --- bronze (5) ---
  the-icml-2013-whale-challenge-right-whale-redux
  text-normalization-challenge-russian-language
  text-normalization-challenge-english-language
  dog-breed-identification
  random-acts-of-pizza
)

CACHE_DIR="${MLEBENCH_DATA_DIR:-}"          # empty => ask mlebench / platform default
OUTPUT_ROOT="bench/runs/verify-$(date +%Y%m%d-%H%M%S)"
TIMEOUT_S=14400                              # 4h/task (matches the controller default)
SKIP_PREPARE=0
FORCE=0
TASKS=()

# ---- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)        read -r -a TASKS <<< "$2"; shift 2 ;;
    --cache-dir)    CACHE_DIR="$2"; shift 2 ;;
    --output-root)  OUTPUT_ROOT="$2"; shift 2 ;;
    --timeout)      TIMEOUT_S="$2"; shift 2 ;;
    --skip-prepare) SKIP_PREPARE=1; shift ;;
    --force)        FORCE=1; shift ;;
    -h|--help)      sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ ${#TASKS[@]} -eq 0 ]] && TASKS=("${DEFAULT_TASKS[@]}")

mkdir -p "$OUTPUT_ROOT"
RUN_LOG="$OUTPUT_ROOT/reproduce.log"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RUN_LOG"; }

# ---- Resolve cache dir (mlebench-aware) -------------------------------------
if [[ -z "$CACHE_DIR" ]]; then
  CACHE_DIR="$(python3 - <<'PY'
try:
    from mlebench.registry import registry
    print(registry.get_data_dir())
except Exception:
    import os
    mac = os.path.expanduser("~/Library/Caches/mle-bench/data")
    print(mac if os.path.isdir(mac) else os.path.expanduser("~/.cache/mle-bench/data"))
PY
)"
fi

log "=============================================================="
log "MLE-bench Lite-22 Class-A reproduction"
log "repo root:   $REPO_ROOT"
log "git commit:  $(git rev-parse HEAD 2>/dev/null || echo '(no git)')"
log "cache dir:   $CACHE_DIR"
log "output root: $OUTPUT_ROOT"
log "tasks (${#TASKS[@]}): ${TASKS[*]}"
log "=============================================================="

# ---- Preflight --------------------------------------------------------------
fail() { log "PREREQ FAILED: $*"; exit 1; }

command -v python3 >/dev/null || fail "python3 not found"
python3 -c "import bench.lite22_controller" 2>/dev/null \
  || fail "cannot import bench.lite22_controller — run from the repo root with deps installed (pip install -r reproduce/requirements.lock -r bench/requirements.txt)"
command -v mlebench >/dev/null \
  || fail "mlebench CLI not found — pip install -r bench/requirements.txt (installs OpenAI's grader from git)"

# pandas pin check (load-bearing — see reproduce/requirements.lock)
PANDAS_V="$(python3 -c 'import pandas; print(pandas.__version__)' 2>/dev/null || echo '?')"
if [[ "$PANDAS_V" != 3.* ]]; then
  log "WARNING: pandas $PANDAS_V detected; medals were graded under pandas 3.0.3."
  log "         Scores/medals may differ. See reproduce/requirements.lock."
fi

# Kaggle creds are required by `mlebench prepare` to download competition data.
if [[ "$SKIP_PREPARE" -eq 0 ]]; then
  if [[ ! -f "$HOME/.kaggle/kaggle.json" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
    fail "no Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME+KAGGLE_KEY). \
Needed for 'mlebench prepare'. Also accept each competition's rules on kaggle.com \
first — see reproduce/VERIFY.md. Or pass --skip-prepare if data is already prepared."
  fi
fi

# ---- Pinned agent environment (verbatim from the graded bundles) ------------
export EES_ENABLE_WEB_RESEARCH=1
export EES_ENABLE_LIVE_WEB_RESEARCH=1
export EES_OPERATOR_BUDGET_SECONDS=10800
export EES_TABULAR_MAX_TRAIN_ROWS=500000
export EES_IMAGE_EMBED_MAX_TRAIN_ROWS=15000
# Grade against the same cache the agent trained on.
export BENCH_GRADER_DATA_DIR="$CACHE_DIR"

# ---- Per-task loop ----------------------------------------------------------
for task in "${TASKS[@]}"; do
  log "--------------------------------------------------------------"
  log "TASK: $task"

  result_json="$OUTPUT_ROOT/$task/result.json"
  if [[ "$FORCE" -eq 0 && -f "$result_json" ]] \
     && python3 -c "import json,sys; sys.exit(0 if json.load(open('$result_json')).get('terminal') else 1)" 2>/dev/null; then
    log "  already have a terminal result — skipping (use --force to re-run)"
    continue
  fi

  # 1) Prepare data (OpenAI's tooling). Rule-acceptance is a manual prerequisite.
  if [[ "$SKIP_PREPARE" -eq 0 ]]; then
    if [[ -d "$CACHE_DIR/$task/prepared/private" ]] \
       && [[ -n "$(ls -A "$CACHE_DIR/$task/prepared/private" 2>/dev/null)" ]]; then
      log "  data already prepared at $CACHE_DIR/$task"
    else
      log "  preparing data: mlebench prepare -c $task"
      if ! mlebench prepare -c "$task" >>"$RUN_LOG" 2>&1; then
        log "  PREPARE FAILED for $task — likely un-accepted competition rules."
        log "  Accept them at the URL in reproduce/VERIFY.md, then re-run (resumable)."
        continue
      fi
    fi
  fi

  # 2) Run OUR autonomous agent + 3) grade with mlebench (the controller does both).
  log "  running agent + official grader (timeout ${TIMEOUT_S}s)"
  force_flag=(); [[ "$FORCE" -eq 1 ]] && force_flag=(--force)
  if python3 -m bench.lite22_controller \
        --agent ees:standalone \
        --cache-dir "$CACHE_DIR" \
        --output-root "$OUTPUT_ROOT" \
        --timeout "$TIMEOUT_S" \
        --task "$task" \
        "${force_flag[@]}" >>"$RUN_LOG" 2>&1; then
    log "  controller finished"
  else
    log "  controller returned non-zero (task may have failed/no-medal) — see $RUN_LOG"
  fi

  # Per-task grade line
  grade_json="$OUTPUT_ROOT/$task/grade.json"
  if [[ -f "$grade_json" ]]; then
    line="$(python3 -c "import json; g=json.load(open('$grade_json')); print(f\"  -> score={g.get('score')} medal={g.get('medal')}\")")"
    log "$line"
  fi
done

# ---- Medal table ------------------------------------------------------------
log "=============================================================="
log "MEDAL TABLE (official mlebench grades)"
MEDAL_TABLE="$OUTPUT_ROOT/medal_table.tsv"
python3 - "$OUTPUT_ROOT" "$MEDAL_TABLE" <<'PY' | tee -a "$RUN_LOG"
import json, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
rows, medals = [], {"gold": 0, "silver": 0, "bronze": 0}
for gj in sorted(root.glob("*/grade.json")):
    task = gj.parent.name
    g = json.load(open(gj))
    medal, score = g.get("medal", "none"), g.get("score")
    if medal in medals:
        medals[medal] += 1
    rows.append((task, score, medal, g.get("gold_threshold"),
                 g.get("silver_threshold"), g.get("bronze_threshold")))
hdr = f"{'task':52s}  {'score':>10}  {'medal':7}  {'gold':>9}  {'silver':>9}  {'bronze':>9}"
print(hdr); print("-" * len(hdr))
lines = [hdr]
for task, score, medal, gt, st, bt in rows:
    s = "None" if score is None else f"{score:.5f}"
    line = f"{task:52s}  {s:>10}  {medal:7}  {str(gt):>9}  {str(st):>9}  {str(bt):>9}"
    print(line); lines.append(line)
total = medals["gold"] + medals["silver"] + medals["bronze"]
summary = f"TOTAL medals: {total}  (gold={medals['gold']} silver={medals['silver']} bronze={medals['bronze']})"
print("-" * len(hdr)); print(summary)
out.write_text("\n".join(lines) + "\n" + summary + "\n")
PY
log "medal table written to $MEDAL_TABLE"
log "full log: $RUN_LOG"
log "reproducibility bundles per task: $OUTPUT_ROOT/<task>/bundle/"
log "=============================================================="
