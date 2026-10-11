#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -Eeuo pipefail

# Full Figure-5-style XSum run:
# parallel adapter applied to the FFN sub-layer, bottleneck dimension 200.
# This script is intentionally separate from exps/run_xsum.sh.

REPO_DIR="${REPO_DIR:-/work/unify-parameter-efficient-tuning/unify-parameter-efficient-tuning}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
DATE="$(date +%Y%m%d)"
SAVE="checkpoints/xsum/${DATE}/xsum.pa_ffn.seed${SEED}"

cd "$REPO_DIR"

if [[ ! -f "_setup/activate.sh" ]]; then
    echo "ERROR: missing _setup/activate.sh; run setup_env.sh first." >&2
    exit 1
fi

if [[ ! -f "examples/pytorch/summarization/run_summarization.py" ]]; then
    echo "ERROR: summarization training script is missing." >&2
    exit 1
fi

if [[ ! -f data/xsum/train.json ]]; then
    echo "ERROR: required file is missing: data/xsum/train.json" >&2
    exit 1
fi
if [[ ! -f data/xsum/validation.json ]]; then
    echo "ERROR: required file is missing: data/xsum/validation.json" >&2
    exit 1
fi
if [[ ! -f data/xsum/test.json ]]; then
    echo "ERROR: required file is missing: data/xsum/test.json" >&2
    exit 1
fi
if [[ ! -f _setup/models/bart-large/config.json ]]; then
    echo "ERROR: required file is missing: _setup/models/bart-large/config.json" >&2
    exit 1
fi
if [[ ! -f _setup/models/bart-large/pytorch_model.bin ]]; then
    echo "ERROR: required file is missing: _setup/models/bart-large/pytorch_model.bin" >&2
    exit 1
fi

source "_setup/activate.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

python -c "import numpy, torch, transformers, datasets, petl; print('Imports OK'); print('torch', torch.__version__); print('GPUs visible', torch.cuda.device_count())"

mkdir -p "$SAVE"
if [[ -d checkpoints/hf_model ]]; then
    find checkpoints/hf_model -maxdepth 2 -type f -name '*.lock' -delete
fi

echo "Starting full XSum Figure-5 FFN parallel-adapter run"
echo "method=pa_ffn seed=$SEED gpu=$GPU"
echo "output=$SAVE"

python -u examples/pytorch/summarization/run_summarization.py \
    --train_file data/xsum/train.json \
    --validation_file data/xsum/validation.json \
    --test_file data/xsum/test.json \
    --text_column document \
    --summary_column summary \
    --model_name_or_path _setup/models/bart-large \
    --cache_dir checkpoints/hf_model \
    --attn_mode none \
    --attn_option parallel \
    --attn_composition add \
    --ffn_mode adapter \
    --ffn_option parallel \
    --ffn_adapter_layernorm_option none \
    --ffn_adapter_init_option bert \
    --ffn_adapter_scalar 1 \
    --mid_dim 800 \
    --attn_bn 200 \
    --ffn_bn 200 \
    --unfreeze_params ef_ \
    --preprocessing_num_workers 2 \
    --max_source_length 512 \
    --max_target_length 128 \
    --val_max_target_length 60 \
    --max_eval_samples 1600 \
    --num_beams 5 \
    --max_length 60 \
    --min_length 10 \
    --no_repeat_ngram_size 3 \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_steps 100000 \
    --num_train_epochs 30 \
    --learning_rate 5e-5 \
    --lr_scheduler_type polynomial \
    --max_grad_norm 0.1 \
    --weight_decay 0.01 \
    --warmup_steps 0 \
    --seed "$SEED" \
    --fp16 \
    --logging_steps 100 \
    --save_total_limit 2 \
    --label_smoothing_factor 0.1 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps 3000 \
    --eval_steps 3000 \
    --load_best_model_at_end \
    --report_to none \
    --run_name "xsum.pa_ffn.seed${SEED}" \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model rouge2 \
    --greater_is_better True \
    --predict_with_generate \
    --output_dir "$SAVE" \
    2>&1 | tee "$SAVE/log.txt"

if ! grep -q "Training completed" "$SAVE/log.txt"; then
    echo "ERROR: training did not complete. See $SAVE/log.txt" >&2
    exit 1
fi

if ! grep -Eq "predict_rouge2|eval_rouge2" "$SAVE/log.txt"; then
    echo "ERROR: no ROUGE metrics found. See $SAVE/log.txt" >&2
    exit 1
fi

echo "XSum Figure-5 run completed successfully."
echo "Results: $REPO_DIR/$SAVE"
grep -E "predict_rouge1|predict_rouge2|predict_rougeL|eval_rouge2" "$SAVE/log.txt" | tail -n 20 || true