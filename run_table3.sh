#!/bin/bash

cd "$(dirname "$0")" || exit 1

mkdir -p logs/table3

SEED=${SEED:-42}

METHODS=(
    prefix
    sa_attn
    sa_ffn
    pa_attn
    pa_ffn
)

NUM_GPUS=4

FAILED_FILE="logs/table3/failed.txt"
rm -f "${FAILED_FILE}"

run_method () {
    METHOD=$1
    GPU=$2

    LOG="logs/table3/${METHOD}.log"

    echo "============================================================"
    echo "Starting ${METHOD} on GPU ${GPU}"
    echo "Time: $(date)"
    echo "Log: ${LOG}"
    echo "============================================================"

    if METHOD="${METHOD}" \
       SEED="${SEED}" \
       CUDA_VISIBLE_DEVICES="${GPU}" \
       bash exps/run_xsum_table3.sh \
       > "${LOG}" 2>&1
    then
        echo "SUCCESS: ${METHOD} on GPU ${GPU}"
    else
        echo "FAILED: ${METHOD} on GPU ${GPU}"
        echo "${METHOD}" >> "${FAILED_FILE}"
    fi
}


echo "============================================================"
echo "TABLE 3 — XSum reproduction"
echo "Using ${NUM_GPUS} GPUs"
echo "Seed: ${SEED}"
echo "Start: $(date)"
echo "============================================================"

for ((i=0; i<${#METHODS[@]}; i+=NUM_GPUS)); do

    pids=()

    for ((g=0; g<NUM_GPUS; g++)); do
        j=$((i + g))

        if [ $j -lt ${#METHODS[@]} ]; then
            run_method "${METHODS[$j]}" "${g}" &
            pids+=($!)
        fi
    done

    # Wait for all methods in this wave to finish
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done

done


echo
echo "============================================================"
echo "TABLE 3 FINISHED"
echo "End: $(date)"
echo "============================================================"

if [ -f "${FAILED_FILE}" ]; then
    echo "Failed methods:"
    cat "${FAILED_FILE}"
    exit 1
else
    echo "All five methods completed successfully."
fi