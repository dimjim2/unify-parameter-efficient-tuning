#!/bin/bash
cd "$(dirname "$0")"
mkdir -p logs/table2
NGPU=$(nvidia-smi -L | wc -l)
jobs=()
for t in sst2 mnli; do for i in 0 1 2 3 4; do for m in mam lora prefix adapter; do
  jobs+=("$m $t $i"); done; done; done
SLOTS=$((NGPU * 3))
echo "$(date +%H:%M) start: ${#jobs[@]} runs on $NGPU GPU(s), $SLOTS at a time"
for ((s=0; s<SLOTS; s++)); do
  gpu=$((s % NGPU))
  ( for ((j=s; j<${#jobs[@]}; j+=SLOTS)); do
      set -- ${jobs[$j]}
      METHOD=$1 TASK_NAME=$2 SLURM_ARRAY_TASK_ID=$3 CUDA_VISIBLE_DEVICES=$gpu \
        bash exps/run_glue.sh > logs/table2/$1.$2.seed$3.log 2>&1
      echo "$(date +%H:%M) done $1 $2 seed$3 (gpu $gpu)"
    done ) &
  sleep 30
done
wait
echo "$(date +%H:%M) ALL DONE"
