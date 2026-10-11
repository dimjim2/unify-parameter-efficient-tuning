#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/work/unify-parameter-efficient-tuning/unify-parameter-efficient-tuning}"
DATE="${DATE:-$(date +%Y%m%d)}"
BASE_OUTPUT="checkpoints/xsum/${DATE}/low_budget_seed42"
LOG_OUTPUT="logs/xsum_low_budget_seed42"

cd "$REPO_DIR"

[[ -f "_setup/activate.sh" ]] || { echo "ERROR: missing _setup/activate.sh" >&2; exit 1; }
[[ -f "examples/pytorch/summarization/run_summarization.py" ]] || {
    echo "ERROR: missing summarization script" >&2
    exit 1
}
[[ -f "data/xsum/train.json" && -f "data/xsum/validation.json" && -f "data/xsum/test.json" ]] || {
    echo "ERROR: XSum data files are missing" >&2
    exit 1
}
[[ -f "_setup/models/bart-large/pytorch_model.bin" ]] || {
    echo "ERROR: local BART model is missing" >&2
    exit 1
}

source "_setup/activate.sh"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$BASE_OUTPUT" "$LOG_OUTPUT"

run_config() {
    local method="$1"
    local gpu="$2"
    local seed="$3"
    local attn_mode="$4"
    local attn_option="$5"
    local attn_bn="$6"
    local ffn_mode="$7"
    local ffn_option="$8"
    local ffn_bn="$9"
    local lora_alpha="${10}"

    local output="$BASE_OUTPUT/${method}_seed${seed}"
    local log="$LOG_OUTPUT/${method}_seed${seed}.log"

    mkdir -p "$output"

    {
        echo "[$method seed=$seed] GPU $gpu starting"
        echo "output=$output"
        date
    } | tee "$log"

    export CUDA_VISIBLE_DEVICES="$gpu"

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
        --lora_dropout 0
        --lora_init lora

        --attn_mode "$attn_mode"
        --attn_option "$attn_option"
        --attn_composition add
        --attn_bn "$attn_bn"

        --ffn_mode "$ffn_mode"
        --ffn_option "$ffn_option"
        --ffn_bn "$ffn_bn"
        --ffn_adapter_layernorm_option none
        --ffn_adapter_scalar 1
        --ffn_adapter_init_option bert

        --mid_dim 800
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
        --seed "$seed"
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
        --run_name "xsum.low_budget.${method}.seed${seed}"
        --overwrite_output_dir
        --disable_tqdm True
        --metric_for_best_model rouge2
        --greater_is_better True
        --predict_with_generate
        --output_dir "$output"
    )

    python -u "${args[@]}" 2>&1 | tee -a "$log"

    grep -q "Training completed" "$log" || {
        echo "[$method seed=$seed] ERROR: training did not complete" | tee -a "$log"
        return 1
    }

    grep -Eq "predict_rouge2|eval_rouge2" "$log" || {
        echo "[$method seed=$seed] ERROR: no ROUGE metrics found" | tee -a "$log"
        return 1
    }

    echo "[$method seed=$seed] completed successfully" | tee -a "$log"
}

run_pair() {
    local method_a="$1"
    local gpu_a="$2"
    local seed_a="$3"
    local attn_mode_a="$4"
    local attn_option_a="$5"
    local attn_bn_a="$6"
    local ffn_mode_a="$7"
    local ffn_option_a="$8"
    local ffn_bn_a="$9"
    local alpha_a="${10}"

    local method_b="${11}"
    local gpu_b="${12}"
    local seed_b="${13}"
    local attn_mode_b="${14}"
    local attn_option_b="${15}"
    local attn_bn_b="${16}"
    local ffn_mode_b="${17}"
    local ffn_option_b="${18}"
    local ffn_bn_b="${19}"
    local alpha_b="${20}"

    run_config "$method_a" "$gpu_a" "$seed_a" \
        "$attn_mode_a" "$attn_option_a" "$attn_bn_a" \
        "$ffn_mode_a" "$ffn_option_a" "$ffn_bn_a" "$alpha_a" &
    local pid_a=$!

    run_config "$method_b" "$gpu_b" "$seed_b" \
        "$attn_mode_b" "$attn_option_b" "$attn_bn_b" \
        "$ffn_mode_b" "$ffn_option_b" "$ffn_bn_b" "$alpha_b" &
    local pid_b=$!

    local status=0
    wait "$pid_a" || status=1
    wait "$pid_b" || status=1
    return "$status"
}

echo "Starting PA-FFN seed repeats and low-budget XSum runs"
echo "Output: $REPO_DIR/$BASE_OUTPUT"

# Wave 1: same PA-FFN r=512 capacity, different seeds.
run_pair \
    pa_ffn_r512 0 43 none none 0 adapter parallel 512 0 \
    pa_ffn_r512 1 44 none none 0 adapter parallel 512 0

# Wave 2: low-budget prefix attention and PA attention.
run_pair \
    prefix_attn_l6 0 42 prefix concat 6 none none 0 0 \
    pa_attn_r6 1 42 adapter parallel 6 none none 0 0

# Wave 3: low-budget PA-FFN and LoRA attention.
run_pair \
    pa_ffn_r8 0 42 none none 0 adapter parallel 8 0 \
    lora_attn_r3 1 42 lora none 3 none none 0 12

# Wave 4: low-budget LoRA-FFN.
run_config \
    lora_ffn_r2 0 42 none none 0 lora none 2 8

echo
echo "All PA-FFN seed repeats and low-budget runs completed successfully."
echo "Results: $REPO_DIR/$BASE_OUTPUT"

for log in "$LOG_OUTPUT"/*.log; do
    echo "=== $log ==="
    grep -E "predict_rouge1|predict_rouge2|predict_rougeL|completed successfully" "$log" | tail -n 5 || true
done 