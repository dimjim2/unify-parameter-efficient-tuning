#!/bin/bash
# Table 5 (XSum, FFN composition functions): 5 runs, one per GPU (5th shares GPU 0).
# Usage: nohup bash run_table5.sh > logs/table5_progress.log 2>&1 &
#        DEBUG=1 bash run_table5.sh      (smoke test, ~5 min)
#        MAX_STEPS=30000 bash run_table5.sh   (shorter run if time is tight)
cd "$(dirname "$0")"
OUT=logs/table5; [ "${DEBUG:-0}" = 1 ] && OUT=logs/table5_debug
mkdir -p $OUT
NGPU=$(nvidia-smi -L | wc -l)
METHODS="lora_s4 lora_s1 pa spa_s4 spa_learn"
echo "$(date +%H:%M) start: $METHODS on $NGPU GPU(s), DEBUG=${DEBUG:-0}, MAX_STEPS=${MAX_STEPS:-100000}"
i=0
for m in $METHODS; do
  gpu=$((i % NGPU)); i=$((i+1))
  ( METHOD=$m CUDA_VISIBLE_DEVICES=$gpu bash exps/run_xsum.sh > $OUT/$m.log 2>&1
    echo "$(date +%H:%M) done $m (gpu $gpu)" ) &
  sleep 60   # stagger start-up (data loading)
done
wait
echo "$(date +%H:%M) ALL DONE"
