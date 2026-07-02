#!/bin/bash
# train_all.sh — train a method on all tasks in a task set, sequentially.
#
# Usage:
#   bash scripts/train_all.sh [OPTIONS] [HYDRA_ARGS...]
#
# Options:
#   --method METHOD       Hydra method name (default: bip)
#   --tasks  TASKSET      Task set: subset | rlbench18 (default: subset)
#   --log-dir DIR         Directory for per-task log files (default: logs/train_all)
#   --skip-existing       Skip tasks that already have a checkpoint in exp_local/
#   -h, --help            Show this help message
#
# Any additional arguments are forwarded verbatim to train.py as Hydra overrides.
#
# Examples:
#   bash scripts/train_all.sh
#   bash scripts/train_all.sh --method bip --tasks rlbench18
#   bash scripts/train_all.sh --method dp num_train_steps=30000
#   bash scripts/train_all.sh --skip-existing --method coa

set -euo pipefail

# ── task lists ───────────────────────────────────────────────────────────────

subset_tasks=(
    "reach_target"
    "press_switch"
    "pick_up_cup"
    "open_drawer"
    "stack_wine"
    "open_box"
    "sweep_to_dustpan"
    "turn_tap"
    "push_button"
    "take_lid_off_saucepan"
)

rlbench18_tasks=(
    "open_drawer"
    "close_drawer"
    "open_jar"
    "close_jar"
    "insert_onto_square_peg"
    "stack_blocks"
    "place_cups"
    "place_shape_in_shape_sorter"
    "put_groceries_in_cupboard"
    "put_item_in_drawer"
    "put_money_in_safe"
    "push_buttons"
    "reach_and_drag"
    "slide_block_to_target"
    "sweep_to_dustpan"
    "turn_tap"
    "light_bulb_in"
    "meat_off_grill"
)

# ── defaults ─────────────────────────────────────────────────────────────────

METHOD="bip"
TASKSET="subset"
LOG_DIR="logs/train_all"
SKIP_EXISTING=true
EXTRA_ARGS=()

# ── arg parsing ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)   METHOD="$2"; shift 2 ;;
        --tasks)    TASKSET="$2"; shift 2 ;;
        --log-dir)  LOG_DIR="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "$TASKSET" in
    subset)     TASKS=("${subset_tasks[@]}") ;;
    rlbench18)  TASKS=("${rlbench18_tasks[@]}") ;;
    *)
        echo "[ERROR] Unknown task set: $TASKSET  (choose: subset | rlbench18)"
        exit 1
        ;;
esac

PYTHON=/home/fuze/miniforge3/envs/coa/bin/python

# ── display (Xvfb) ───────────────────────────────────────────────────────────

if [[ -z "${DISPLAY:-}" ]]; then
    DISPLAY_NUM=99
    echo "[INFO] No DISPLAY set — starting Xvfb :${DISPLAY_NUM}"
    Xvfb ":${DISPLAY_NUM}" -screen 0 1024x768x24 >/tmp/xvfb_train_all.log 2>&1 &
    XVFB_PID=$!
    export DISPLAY=":${DISPLAY_NUM}"
    trap 'kill $XVFB_PID 2>/dev/null || true' EXIT
    sleep 1
fi

# ── logging ──────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)

echo "========================================================"
echo "  train_all.sh"
echo "  method   : $METHOD"
echo "  task set : $TASKSET  (${#TASKS[@]} tasks)"
echo "  log dir  : $LOG_DIR"
echo "  extra    : ${EXTRA_ARGS[*]:-<none>}"
echo "  started  : $(date)"
echo "========================================================"

# ── helpers ──────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

has_checkpoint() {
    local task="$1"
    # use find to avoid zsh NULL_GLOB expanding unmatched globs to empty
    local dir
    dir=$(find "exp_local/${METHOD}" -maxdepth 1 -name "rlbench_${task}_*" -type d 2>/dev/null | head -1)
    [[ -n "$dir" && -d "${dir}/checkpoints" ]] && \
        find "${dir}/checkpoints" -maxdepth 1 -name "${METHOD}_*.pt" 2>/dev/null | grep -q .
}

SUCCESS_TASKS=()
FAILED_TASKS=()
SKIPPED_TASKS=()

# ── main loop ────────────────────────────────────────────────────────────────

TOTAL=${#TASKS[@]}
IDX=0
for TASK in "${TASKS[@]}"; do
    IDX=$((IDX + 1))
    LOG_FILE="${LOG_DIR}/${RUN_TS}_${METHOD}_${TASK}.log"

    echo -e "\n${BLUE}[${IDX}/${TOTAL}]${NC} task=${TASK}  method=${METHOD}"

    if [[ "$SKIP_EXISTING" == "true" ]] && has_checkpoint "$TASK"; then
        EXISTING_LOG=$(ls -t "exp_local/${METHOD}/rlbench_${TASK}_"*/train.log 2>/dev/null | head -1 || true)
        echo -e "  ${YELLOW}[SKIP]${NC} checkpoint found — log: ${EXISTING_LOG:-<none>}"
        SKIPPED_TASKS+=("$TASK")
        continue
    fi

    echo "  log → $LOG_FILE"

    CMD=("$PYTHON" scripts/train.py "task=${TASK}" "method=${METHOD}")
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi
    echo "  cmd: ${CMD[*]}"

    START_TS=$(date +%s)
    if "${CMD[@]}" >"$LOG_FILE" 2>&1; then
        END_TS=$(date +%s)
        ELAPSED=$(( END_TS - START_TS ))
        echo -e "  ${GREEN}[OK]${NC} finished in ${ELAPSED}s"
        SUCCESS_TASKS+=("$TASK")
    else
        EXIT_CODE=$?
        END_TS=$(date +%s)
        ELAPSED=$(( END_TS - START_TS ))
        echo -e "  ${RED}[FAIL]${NC} exit=${EXIT_CODE} after ${ELAPSED}s — see $LOG_FILE"
        tail -20 "$LOG_FILE" | sed 's/^/    /'
        FAILED_TASKS+=("$TASK")
    fi
done

# ── summary ──────────────────────────────────────────────────────────────────

# Find the most recent log for a task: prefer train_all log in LOG_DIR (this run),
# fall back to the exp_local train.log (for skipped tasks).
find_log() {
    local task="$1"
    # most recent file matching *_<method>_<task>.log in the log dir
    local f
    f=$(ls -t "${LOG_DIR}/"*"_${METHOD}_${task}.log" 2>/dev/null | head -1)
    [[ -f "$f" ]] && echo "$f" && return
    # fall back to exp_local hydra log
    ls -t "exp_local/${METHOD}/rlbench_${task}_"*/train.log 2>/dev/null | head -1
}

last_success_rate() {
    local log="$1"
    [[ -z "$log" || ! -f "$log" ]] && echo "N/A" && return
    local sr
    sr=$(grep -oE 'success_rate=[0-9]+(\.[0-9]+)?' "$log" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?$')
    echo "${sr:-N/A}"
}

echo ""
echo "========================================================"
echo "  Summary  (method=${METHOD}, tasks=${TASKSET})"
echo "  finished : $(date)"
echo "========================================================"
printf "  %-35s  %-8s  %s\n" "TASK" "STATUS" "success_rate"
printf "  %-35s  %-8s  %s\n" "----" "------" "------------"

for TASK in "${TASKS[@]}"; do
    LOG=$(find_log "$TASK")
    SR=$(last_success_rate "$LOG")

    STATUS="SKIPPED"; COLOR="$YELLOW"
    for t in "${SUCCESS_TASKS[@]:-}"; do
        [[ "$t" == "$TASK" ]] && STATUS="OK"   && COLOR="$GREEN" && break
    done
    for t in "${FAILED_TASKS[@]:-}"; do
        [[ "$t" == "$TASK" ]] && STATUS="FAIL" && COLOR="$RED"   && break
    done

    printf "  %-35s  ${COLOR}%-8s${NC}  %s\n" "$TASK" "$STATUS" "$SR"
done

echo ""
echo -e "  ${GREEN}OK${NC}: ${#SUCCESS_TASKS[@]}   ${RED}FAIL${NC}: ${#FAILED_TASKS[@]}   ${YELLOW}SKIPPED${NC}: ${#SKIPPED_TASKS[@]}"

# Average success rate across tasks with numeric results
SR_SUM=0; SR_COUNT=0
for TASK in "${TASKS[@]}"; do
    VAL=$(last_success_rate "$(find_log "$TASK")")
    if [[ "$VAL" != "N/A" ]]; then
        SR_SUM=$(awk "BEGIN {printf \"%.6f\", $SR_SUM + $VAL}")
        SR_COUNT=$((SR_COUNT + 1))
    fi
done
if [[ $SR_COUNT -gt 0 ]]; then
    AVG_SR=$(awk "BEGIN {printf \"%.4f\", $SR_SUM / $SR_COUNT}")
    echo -e "  Average success_rate (${SR_COUNT}/${#TASKS[@]} tasks): ${GREEN}${AVG_SR}${NC}"
else
    echo "  Average success_rate: N/A"
fi
echo ""

[[ ${#FAILED_TASKS[@]} -eq 0 ]]
