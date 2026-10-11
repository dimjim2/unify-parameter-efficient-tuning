#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/work/unify-parameter-efficient-tuning/unify-parameter-efficient-tuning}"
DATE="${DATE:-$(date +%Y%m%d)}"
SEED="${SEED:-42}"
GPU_COUNT="${GPU_COUNT:-2}"
BASE_OUTPUT="checkpoints/xsum/${DATE}/fig5_seed${SEED}"
LOG_OUTPUT="logs/fig5_xsum_seed${SEED}"

cd "$REPO_DIR"

if [[ ! -f "_setup/activate.sh" ]]; then
    echo "ERROR: missing _setup/activate.sh" >&2
    exit 1
fi
if [[ ! -f "examples/pytorch/summarization/run_summarization.py" ]]; then
    echo "ERROR: missing summarization training script" >&2
    exit 1
fi
if [[ ! -f "data/xsum/train.json" || ! -f "data/xsum/validation.json" || ! -f "data/xsum/test.json" ]]; then
    echo "ERROR: XSum JSON files are missing under data/xsum" >&2
    exit 1
fi
if [[ ! -f "_setup/models/bart-large/pytorch_model.bin" ]]; then
    echo "ERROR: local BART model is missing" >&2
    exit 1
fi
if [[ "$GPU_COUNT" -ne 2 ]]; then
    echo "ERROR: this launcher expects GPU_COUNT=2" >&2
    exit 1
fi

source "_setup/activate.sh"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

python -c "import numpy, torch, transformers, datasets, petl; print('Imports OK'); print('torch', torch.__version__); print('GPUs visible', torch.cuda.device_count())"

mkdir -p "$BASE_OUTPUT" "$LOG_OUTPUT"
if [[ -d checkpoints/hf_model ]]; then
    find checkpoints/hf_model -maxdepth 2 -type f -name '*.lock' -delete
fi

run_config() {
    local method="$1"
    local gpu="$2"
    local output="$BASE_OUTPUT/$method"
    local log="$LOG_OUTPUT/$method.log"
    local attn_mode="$3"
    local attn_option="$4"
    local attn_bn="$5"
    local ffn_mode="$6"
    local ffn_option="$7"
    local ffn_bn="$8"
    local ffn_init="$9"
    local ffn_scalar="${10}"
    local lora_alpha="${11}"
    local lora_dropout="${12}"

    mkdir -p "$output"
    export CUDA_VISIBLE_DEVICES="$gpu"

    echo "[$method] GPU $gpu starting; output=$output" | tee "$log"

    args=(
        examples/pytorch/summarization/run_summarization.py
        --train_file data/xsum/train.json
        --validation_file data/xsum/validation.json
        --test_file data/xsum/test.json
        --text_column document
        --summary_column summary
        --model_name_or_path _setup/models/bart-large
        --cache_dir checkpoints/hf_model
        --lora_alpha "$lora_alpha"
        --lora_dropout "$lora_dropout"
        --lora_init lora
        --attn_mode "$attn_mode"
        --attn_option "$attn_option"
        --attn_composition add
        --ffn_mode "$ffn_mode"
        --ffn_option "$ffn_option"
        --ffn_adapter_layernorm_option none
        --ffn_adapter_scalar "$ffn_scalar"
        --ffn_adapter_init_option "$ffn_init"
        --mid_dim 800
        --attn_bn "$attn_bn"
        --ffn_bn "$ffn_bn"
        --unfreeze_params ef_
        --preprocessing_num_workers 2
        --max_source_length 512
        --max_target_length 128
        --val_max_target_length 60
        --max_eval_samples 1600
        --num_beams 5
        --max_length 60
        --min_length 10
        --no_repeat_ngram_size 3
        --do_train
        --do_eval
        --do_predict
        --per_device_train_batch_size 16
        --per_device_eval_batch_size 16
        --gradient_accumulation_steps 4
        --max_steps 100000
        --num_train_epochs 30
        --learning_rate 5e-5
        --lr_scheduler_type polynomial
        --max_grad_norm 0.1
        --weight_decay 0.01
        --warmup_steps 0
        --seed "$SEED"
        --fp16
        --logging_steps 100
        --save_total_limit 2
        --label_smoothing_factor 0.1
        --evaluation_strategy steps
        --save_strategy steps
        --save_steps 3000
        --eval_steps 3000
        --load_best_model_at_end
        --report_to none
        --run_name "xsum.fig5.${method}.seed${SEED}"
        --overwrite_output_dir
        --disable_tqdm True
        --metric_for_best_model rouge2
        --greater_is_better True
        --predict_with_generate
        --output_dir "$output"
    )

    python -u "${args[@]}" 2>&1 | tee -a "$log"
    if ! grep -q "Training completed" "$log"; then
        echo "[$method] ERROR: training did not complete" | tee -a "$log"
        return 1
    fi
    if ! grep -Eq "predict_rouge2|eval_rouge2" "$log"; then
        echo "[$method] ERROR: no ROUGE metrics found" | tee -a "$log"
        return 1
    fi
    echo "[$method] completed successfully" | tee -a "$log"
}

echo "Starting four independent XSum Figure 5 runs with two GPUs"
echo "seed=$SEED output=$BASE_OUTPUT"

# Run two configurations per wave. The requested paper-style settings are:
# prefix attention l=200; parallel adapter attention r=200;
# LoRA attention r=200; LoRA FFN r=102 with scaling s=4.
run_config prefix_attn 0 prefix concat 200 none parallel 200 bert 1 0 0 &
pid0=$!
run_config pa_attn 1 adapter parallel 200 none parallel 200 bert 1 0 0 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [[ "$status" -ne 0 ]]; then
    echo "ERROR: first wave failed; not starting second wave" >&2
    exit 1
fi

run_config lora_attn 0 lora none 200 none none 200 bert 1 32 0.1 &
pid0=$!
run_config lora_ffn 1 none none 200 lora none 102 bert 1 408 0.1 &
pid1=$!

wait "$pid0" || status=1
wait "$pid1" || status=1

if [[ "$status" -ne 0 ]]; then
    echo "ERROR: one or more Figure 5 runs failed" >&2
    exit 1
fi

echo
echo "All four XSum Figure 5 runs completed successfully."
echo "Results: $REPO_DIR/$BASE_OUTPUT"
for method in prefix_attn pa_attn lora_attn lora_ffn; do
    echo "=== $method ==="
    grep -E "predict_rouge1|predict_rouge2|predict_rougeL" "$LOG_OUTPUT/$method.log" | tail -n 3 || true
done
