#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -uo pipefail

WORK_DIR=/work/unify-parameter-efficient-tuning
REPO_DIR=$WORK_DIR/unify-parameter-efficient-tuning
ENV_DIR=$REPO_DIR/_setup
MODEL_DIR=$ENV_DIR/models/bart-large
DATA_DIR=$REPO_DIR/data/xsum

OUTPUT_DIR=checkpoints/xsum/20261009/pa_attn_r512_retry
LOG_DIR=logs/fig5_xsum_retry
LOG_FILE=$LOG_DIR/pa_attn_r512_retry.log

mkdir -p "$REPO_DIR/$OUTPUT_DIR" "$REPO_DIR/$LOG_DIR"

cd "$REPO_DIR"
source _setup/activate.sh

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting PA-attention r=512 retry"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path "$MODEL_DIR" \
    --config_name "$MODEL_DIR" \
    --tokenizer_name "$MODEL_DIR" \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/validation.json" \
    --test_file "$DATA_DIR/test.json" \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type polynomial \
    --warmup_ratio 0.1 \
    --max_steps 100000 \
    --save_steps 3000 \
    --eval_steps 3000 \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --metric_for_best_model rouge2 \
    --greater_is_better true \
    --predict_with_generate \
    --num_beams 5 \
    --max_length 60 \
    --min_length 10 \
    --no_repeat_ngram_size 3 \
    --fp16 \
    --seed 42 \
    --attn_mode adapter \
    --attn_option parallel \
    --attn_bn 512 \
    --ffn_mode none \
    --ffn_option none \
    --ffn_bn 0 \
    --ffn_adapter_init_option bert \
    --ffn_adapter_scalar 1 \
    --lora_alpha 0 \
    --lora_dropout 0

status=$?

if [ "$status" -eq 0 ]; then
    echo "[pa_attn_r512_retry] completed successfully"
else
    echo "[pa_attn_r512_retry] failed with exit code $status" >&2
fi

date
exit "$status"
