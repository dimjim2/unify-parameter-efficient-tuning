#!/usr/bin/env bash
# Run four selected small-budget Table 4 variants; PA-FFN is omitted.
set -euo pipefail
cd "$(dirname "$0")"

METHODS=(prefix prefix_no_gate pa_attn mh_pa_attn)
IFS=',' read -r -a GPUS <<< "${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
if ((${#GPUS[@]} != ${#METHODS[@]})); then
    echo "Expected four GPU IDs in GPU_IDS (got ${#GPUS[@]}): ${GPU_IDS:-0,1,2,3}" >&2
    exit 2
fi

declare -A seen=()
for gpu in "${GPUS[@]}"; do
    [[ -n "$gpu" && -z "${seen[$gpu]:-}" ]] || { echo 'GPU IDs must be nonempty and distinct' >&2; exit 2; }
    seen[$gpu]=1
done

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_TAG' >&2; exit 2; }
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-50000}"
LOG_DIR="logs/table4_small/$RUN_TAG"
mkdir -p "$(dirname "$LOG_DIR")"
mkdir "$LOG_DIR"

echo "Table 4 small-budget runs; seed=$SEED max_steps=$MAX_STEPS GPUs=${GPUS[*]}"
echo "Outputs: checkpoints/wmt16_table4_small/$RUN_TAG; logs: $LOG_DIR"

pids=()
for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    gpu="${GPUS[$i]}"
    echo "Starting $method on GPU $gpu"
    METHOD="$method" SEED="$SEED" RUN_TAG="$RUN_TAG" MAX_STEPS="$MAX_STEPS" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        bash exps/run_en_ro_table4_small.sh >"$LOG_DIR/$method.log" 2>&1 &
    pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "SUCCESS ${METHODS[$i]}"
    else
        echo "FAILED ${METHODS[$i]} (see $LOG_DIR/${METHODS[$i]}.log)" >&2
        failed=1
    fi
done

if ((failed)); then
    exit 1
fi
echo "All four Table 4 small-budget runs completed."
