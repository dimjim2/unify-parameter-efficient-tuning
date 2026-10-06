#!/bin/bash

# Reproduce the XSum portion of Table 3 from:
# He et al. (ICLR 2022), "Towards a Unified View of
# Parameter-Efficient Transfer Learning"
#
# Runs sequentially on a single GPU:
#   1. Prefix tuning, l=200
#   2. Sequential Adapter at attention, r=200
#   3. Sequential Adapter at FFN, r=200
#   4. Parallel Adapter at attention, r=200
#   5. Parallel Adapter at FFN, r=200
#
# Usage:
#   bash run_table3.sh

cd "$(dirname "$0")" || exit 1

mkdir -p logs/table3

METHODS=(
    prefix
    sa_attn
    sa_ffn
    pa_attn
    pa_ffn
)

SEED=${SEED:-42}
GPU=${GPU:-0}

FAILED=()

echo "============================================================"
echo "TABLE 3 — XSum reproduction"
echo "============================================================"
echo "Start time : $(date)"
echo "GPU        : ${GPU}"
echo "Seed       : ${SEED}"
echo "Methods    : ${METHODS[*]}"
echo "============================================================"
echo

for METHOD in "${METHODS[@]}"; do

    LOG="logs/table3/${METHOD}.log"

    echo "============================================================"
    echo "Starting METHOD=${METHOD}"
    echo "Time: $(date)"
    echo "Log : ${LOG}"
    echo "============================================================"

    if METHOD="${METHOD}" \
       SEED="${SEED}" \
       CUDA_VISIBLE_DEVICES="${GPU}" \
       bash exps/run_xsum_table3.sh \
       > "${LOG}" 2>&1
    then
        echo
        echo "SUCCESS: ${METHOD}"
        echo "Finished: $(date)"
    else
        EXIT_CODE=$?

        echo
        echo "FAILED: ${METHOD}"
        echo "Exit code: ${EXIT_CODE}"
        echo "Check: ${LOG}"

        FAILED+=("${METHOD}")
    fi

    echo
done

echo "============================================================"
echo "TABLE 3 FINISHED"
echo "End time: $(date)"
echo "============================================================"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All 5 runs completed successfully."
    exit 0
else
    echo "Failed methods: ${FAILED[*]}"
    echo "Successful runs have been preserved."
    exit 1
fi