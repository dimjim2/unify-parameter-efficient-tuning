#!/usr/bin/env bash
# One of four selected Table 4 en->ro variants (PA-FFN is omitted).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

METHOD="${METHOD:?Set METHOD to prefix, prefix_no_gate, pa_attn, or mh_pa_attn}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-50000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_TAG' >&2; exit 2; }
SAVE="$ROOT/checkpoints/wmt16_table4_small/$RUN_TAG/$METHOD"

dataset="wmt16"
attn_mode="prefix"
attn_option="cross_attn_noln"
attn_composition="gate_add"
attn_bn=30
ffn_mode="none"
ffn_option="none"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="bert"
ffn_adapter_scalar="1"
ffn_bn=30

case "$METHOD" in
    prefix)
        # Eq. 7: concatenation implements the gate through joint softmax.
        # Avoid the split gate_add path, which assumes an attention mask.
        attn_option="concat"
        attn_composition="add"
        ;;
    prefix_no_gate)
        # Eq. 9 ablation: independently normalized prefix attention + h.
        attn_composition="add"
        ;;
    pa_attn)
        attn_mode="adapter"
        attn_option="parallel"
        attn_composition="add"
        ;;
    mh_pa_attn)
        attn_mode="prefix"
        attn_option="cross_attn_relu"
        attn_composition="add"
        ;;
    *)
        echo "Unknown METHOD: $METHOD" >&2
        exit 2
        ;;
esac

command -v python >/dev/null
command -v sacrebleu >/dev/null
for tool in scripts/tokenizer/replace-unicode-punctuation.perl scripts/tokenizer/normalize-punctuation.perl scripts/tokenizer/remove-non-printing-char.perl scripts/tokenizer/tokenizer.perl wmt16-scripts/preprocess/normalise-romanian.py wmt16-scripts/preprocess/remove-diacritics.py; do
    [[ -x "$ROOT/mosesdecoder/$tool" ]] || { echo "Missing executable: mosesdecoder/$tool" >&2; exit 2; }
done
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_STEPS must be positive' >&2; exit 2; }
EVAL_STEPS=5000
((MAX_STEPS >= EVAL_STEPS)) || EVAL_STEPS="$MAX_STEPS"

mkdir -p "$(dirname "$SAVE")"
mkdir "$SAVE" # Refuse to reuse an existing run.
cp "$0" "$SAVE/runner.sh"
git rev-parse HEAD > "$SAVE/revision.txt"
git diff > "$SAVE/working-tree.diff"
python -m pip freeze > "$SAVE/environment.txt"
printf 'method=%s\nseed=%s\nmax_steps=%s\n' "$METHOD" "$SEED" "$MAX_STEPS" > "$SAVE/run.txt"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT/checkpoints/hf_model}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$TRANSFORMERS_CACHE}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-$TRANSFORMERS_CACHE}"
export TOKENIZERS_PARALLELISM=false

python -u examples/pytorch/translation/run_translation.py \
    --dataset_name "$dataset" \
    --dataset_config_name ro-en \
    --model_name_or_path facebook/mbart-large-cc25 \
    --cache_dir "$TRANSFORMERS_CACHE" \
    --source_lang en_XX \
    --target_lang ro_RO \
    --do_train --do_eval --do_predict \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --max_tokens_per_batch 4096 \
    --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
    --dropout 0.1 --attention_dropout 0.0 \
    --lora_alpha 0 --lora_dropout 0 --lora_init lora \
    --attn_mode "$attn_mode" \
    --attn_option "$attn_option" \
    --attn_composition "$attn_composition" \
    --ffn_mode "$ffn_mode" \
    --ffn_option "$ffn_option" \
    --ffn_adapter_layernorm_option "$ffn_adapter_layernorm_option" \
    --ffn_adapter_scalar "$ffn_adapter_scalar" \
    --ffn_adapter_init_option "$ffn_adapter_init_option" \
    --mid_dim 800 --attn_bn "$attn_bn" --ffn_bn "$ffn_bn" \
    --unfreeze_params ef_ \
    --preprocessing_num_workers 2 \
    --max_source_length 150 --max_target_length 150 --val_max_target_length 150 \
    --max_eval_samples 1999 \
    --num_beams 5 --max_length 200 --min_length 1 --no_repeat_ngram_size 0 \
    --gradient_accumulation_steps 4 \
    --max_steps "$MAX_STEPS" --num_train_epochs 30 \
    --learning_rate 5e-5 --lr_scheduler_type polynomial \
    --max_grad_norm 1 --weight_decay 0.01 --warmup_steps 0 \
    --fp16 --seed "$SEED" \
    --logging_steps 100 --save_total_limit 2 \
    --label_smoothing_factor 0.1 \
    --evaluation_strategy steps --save_strategy steps \
    --save_steps "$EVAL_STEPS" --eval_steps "$EVAL_STEPS" --load_best_model_at_end \
    --report_to none \
    --run_name "${dataset}.${RUN_TAG}.${METHOD}.seed${SEED}" \
    --disable_tqdm True \
    --metric_for_best_model loss --greater_is_better False \
    --ddp_find_unused_parameter False \
    --predict_with_generate \
    --overwrite_output_dir \
    --output_dir "$SAVE" \
    2>&1 | tee "$SAVE/train.log"

# Moses postprocessing writes tokenized files into the current directory;
# use this run's output directory so simultaneous methods cannot collide.
cd "$SAVE"
bash "$ROOT/exps/romanian_postprocess.sh" \
    "$SAVE/test_generated_predictions.txt" \
    "$SAVE/test_gold_labels.txt" | tee -a "$SAVE/train.log"
