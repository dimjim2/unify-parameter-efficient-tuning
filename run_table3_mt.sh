#!/bin/bash

set -u

METHODS=(prefix sa_attn sa_ffn pa_attn pa_ffn)
NUM_GPUS=4

mkdir -p logs/table3_mt
rm -f logs/table3_mt/failed.txt

run_method () {
    METHOD=$1
    GPU=$2

    echo "Starting ${METHOD} on GPU ${GPU}"

    METHOD=${METHOD} \
    SEED=42 \
    CUDA_VISIBLE_DEVICES=${GPU} \
    bash exps/run_en_ro_table3.sh \
        > logs/table3_mt/${METHOD}.log 2>&1

    status=$?

    if [ ${status} -ne 0 ]; then
        echo "${METHOD}" >> logs/table3_mt/failed.txt
        echo "${METHOD} FAILED"
    else
        echo "${METHOD} finished"
    fi
}

# First wave: 4 experiments on 4 GPUs
for i in 0 1 2 3; do
    run_method "${METHODS[$i]}" "$i" &
done

wait

# Second wave: last experiment
run_method "${METHODS[4]}" 0

echo "All Table 3 MT runs completed."
