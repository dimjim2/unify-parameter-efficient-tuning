#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=xsum
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=2
#SBATCH --time=0
##SBATCH --array=0

export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}

# Make repo modules importable.
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Respect CUDA_VISIBLE_DEVICES if set by an outer launcher.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATE=`date +%Y%m%d`
dataset="xsum"

# ================================================================
# EXPERIMENT PRESETS
#
# Table 3:
#   METHOD=prefix
#   METHOD=sa_attn
#   METHOD=sa_ffn
#   METHOD=pa_attn
#   METHOD=pa_ffn
#
# Table 5 presets already used by the group:
#   METHOD=lora_s4
#   METHOD=lora_s1
#   METHOD=pa
#   METHOD=spa_s4
#   METHOD=spa_learn
#
# Original default:
#   METHOD=mam
# ================================================================

METHOD=${METHOD:-mam}

case ${METHOD} in

# ----------------------------------------------------------------
# Original MAM configuration
# ----------------------------------------------------------------
mam)
    attn_mode="prefix"
    attn_option="concat"
    attn_composition="add"
    attn_bn=30

    ffn_mode="adapter"
    ffn_option="parallel"
    ffn_adapter_layernorm_option="none"
    ffn_adapter_init_option="lora"
    ffn_adapter_scalar="4"
    ffn_bn=512
    ;;


# ================================================================
# TABLE 3
# ================================================================

# Table 3:
# Prefix, l = 200
# Paper: 3.6% trainable parameters
prefix)
    attn_mode="prefix"
    attn_option="concat"
    attn_composition="add"
    attn_bn=200

    ffn_mode="none"
    ffn_option="parallel"
    ffn_adapter_layernorm_option="none"
    ffn_adapter_init_option="bert"
    ffn_adapter_scalar="1"
    ffn_bn=200
    ;;


# Table 3:
# Sequential Adapter at attention, r = 200
# Paper: 3.6% trainable parameters
sa_attn)
    attn_mode="adapter"
    attn_option="sequential"
    attn_composition="add"
    attn_bn=200

    ffn_mode="none"
    ffn_option="parallel"
    ffn_adapter_layernorm_option="none"
    ffn_adapter_init_option="bert"
    ffn_adapter_scalar="1"
    ffn_bn=200
    ;;


# Table 3:
# Sequential Adapter at FFN, r = 200
# Paper: 2.4% trainable parameters
sa_ffn)
    attn_mode="none"
    attn_option="parallel"
    attn_composition="add"
    attn_bn=200

    ffn_mode="adapter"
    ffn_option="sequential"
    ffn_adapter_layernorm_option="none"
    ffn_adapter_init_option="bert"
    ffn_adapter_scalar="1"
    ffn_bn=200
    ;;


# Table 3:
# Parallel Adapter at attention, r = 200
# Paper: 3.6% trainable parameters
pa_attn)
    attn_mode="adapter"
    attn_option="parallel"
    attn_composition="add"
    attn_bn=200

    ffn_mode="none"
    ffn_option="parallel"
    ffn_adapter_layernorm_option="none"
    ffn_adapter_init_option="bert"
    ffn_adapter_scalar="1"
    ffn_bn=200
    ;;


# Table 3:
# Parallel Adapter at FFN, r = 200
# Paper: 2.4% trainable parameters
pa_ffn)
    attn_mode="none"
    attn_option="parallel"
    attn_composition="add"
    attn_bn=200

    ffn_mode="adapter"
    ffn_option="parallel"
    ffn_adapter_layernorm_option="none"
    ffn_adapter_init_option="bert"
    ffn_adapter_scalar="1"
    ffn_bn=200
    ;;


# ================================================================
# TABLE 5
# Keep existing presets available.
# ================================================================

lora_s4|lora_s1)
    # LoRA on FFN, r = 102
    # scaling s = lora_alpha / r
    attn_mode="none"
    attn_option="none"
    attn_composition="add"
    attn_bn=200

    ffn_mode="lora"
    ffn_option="none"
    ffn_adapter_layernorm_option="none"
    ffn_adapter_init_option="bert"
    ffn_adapter_scalar="1"
    ffn_bn=102

    lora_init="lora"
    lora_dropout=0.1

    if [ "${METHOD}" = "lora_s4" ]; then
        lora_alpha=408
    else
        lora_alpha=102
    fi
    ;;


pa|spa_s4|spa_learn)
    # Parallel Adapter on FFN, r = 512
    attn_mode="none"
    attn_option="none"
    attn_composition="add"
    attn_bn=200

    ffn_mode="adapter"
    ffn_option="parallel"
    ffn_adapter_layernorm_option="none"
    ffn_bn=512

    case ${METHOD} in
        pa)
            ffn_adapter_init_option="bert"
            ffn_adapter_scalar="1"
            ;;
        spa_s4)
            ffn_adapter_init_option="lora"
            ffn_adapter_scalar="4"
            ;;
        spa_learn)
            ffn_adapter_init_option="lora"
            ffn_adapter_scalar="learnable_scalar"
            ;;
    esac
    ;;


*)
    echo "ERROR: unknown METHOD=${METHOD}"
    echo
    echo "Table 3:"
    echo "  prefix"
    echo "  sa_attn"
    echo "  sa_ffn"
    echo "  pa_attn"
    echo "  pa_ffn"
    echo
    echo "Table 5:"
    echo "  lora_s4"
    echo "  lora_s1"
    echo "  pa"
    echo "  spa_s4"
    echo "  spa_learn"
    exit 1
    ;;
esac
# ----------------------------------------------------------------
# Beam size
# Table 3 XSum follows the paper: 5 beams.
# Other presets keep the original script value: 6 beams.
# ----------------------------------------------------------------

case ${METHOD} in
    prefix|sa_attn|sa_ffn|pa_attn|pa_ffn)
        num_beams=5
        ;;
    *)
        num_beams=6
        ;;
esac

# ----------------------------------------------------------------
# LoRA defaults for experiments that are not using LoRA
# ----------------------------------------------------------------

if [ -z ${lora_alpha+x} ]; then
    lora_alpha=0
    lora_init="lora"
    lora_dropout=0
fi


# ================================================================
# GENERAL XSUM CONFIGURATION
# ================================================================

debug=${DEBUG:-0}
report_to="none"

# One fixed seed for Table 3.
# Unlike MNLI/SST-2, the paper does not state that Table 3 is
# aggregated over five random seeds.
seed=${SEED:-42}

label_smoothing_factor=0.1
weight_decay=0.01
max_grad_norm=0.1

max_steps=${MAX_STEPS:-100000}
num_train_epochs=30

warmup_updates=0
lr=5e-5
lr_scheduler_type="polynomial"

# Original script:
# per-device batch 16 x gradient accumulation 4 = effective 64
bsz=16
gradient_steps=4

metric=rouge2
unfreeze='ef_'

# Paper uses 1600 XSum examples for validation.
max_eval_samples=1600

logging_steps=100
eval_strategy="steps"
save_steps=3000

extra_cmd=""
debug_str=""


# ================================================================
# DEBUG MODE
#
# Do NOT use this for the final B200 reproduction.
# It is only retained for local/software debugging.
# ================================================================

if [ "${debug}" = 1 ]; then
    label_smoothing_factor=0
    weight_decay=0
    max_grad_norm=1

    max_train_samples=2000

    bsz=24
    gradient_steps=2

    num_train_epochs=3
    max_steps=-1

    eval_strategy="steps"
    save_steps=100

    report_to="none"
    logging_steps=10

    extra_cmd="--max_train_samples ${max_train_samples} \
               --max_eval_samples 50 \
               --max_predict_samples 50"

    debug_str=".debug"
fi


# ================================================================
# EXPERIMENT NAME
# ================================================================

exp_name=xsum.${METHOD}.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}
exp_name+=.fo_${ffn_option}.abn${attn_bn}.fbn${ffn_bn}.ac_${attn_composition}
exp_name+=.fl_${ffn_adapter_layernorm_option}.finit_${ffn_adapter_init_option}
exp_name+=.fs_${ffn_adapter_scalar}
exp_name+=.unfrz_${unfreeze}
exp_name+=.ms${max_steps}
exp_name+=.ls${label_smoothing_factor}
exp_name+=.warm${warmup_updates}
exp_name+=.wd${weight_decay}
exp_name+=.seed${seed}
exp_name+=${debug_str}

SAVE=checkpoints/${dataset}/${DATE}/${exp_name}


# ================================================================
# PRINT CONFIGURATION
# ================================================================

echo
echo "============================================================"
echo "XSum experiment configuration"
echo "============================================================"
echo "METHOD                     : ${METHOD}"
echo "SEED                       : ${seed}"
echo
echo "attn_mode                  : ${attn_mode}"
echo "attn_option                : ${attn_option}"
echo "attn_composition           : ${attn_composition}"
echo "attn_bn                    : ${attn_bn}"
echo
echo "ffn_mode                   : ${ffn_mode}"
echo "ffn_option                 : ${ffn_option}"
echo "ffn_bn                     : ${ffn_bn}"
echo "ffn_adapter_init_option    : ${ffn_adapter_init_option}"
echo "ffn_adapter_scalar         : ${ffn_adapter_scalar}"
echo
echo "lora_alpha                 : ${lora_alpha}"
echo "lora_dropout               : ${lora_dropout}"
echo
echo "learning_rate              : ${lr}"
echo "per_device_batch_size      : ${bsz}"
echo "gradient_accumulation      : ${gradient_steps}"
echo "effective batch size       : $((bsz * gradient_steps))"
echo "max_steps                  : ${max_steps}"
echo "weight_decay               : ${weight_decay}"
echo "label_smoothing            : ${label_smoothing_factor}"
echo "max_grad_norm              : ${max_grad_norm}"
echo
echo "max_eval_samples           : ${max_eval_samples}"
echo "num_beams                  : ${num_beams}"
echo "output                     : ${SAVE}"
echo "============================================================"
echo


# ================================================================
# CONFIG-ONLY MODE
#
# This is the important local test:
#
# METHOD=pa_ffn CONFIG_ONLY=1 bash exps/run_xsum.sh
#
# It checks all shell configuration without loading BART or training.
# ================================================================

if [ "${CONFIG_ONLY:-0}" = "1" ]; then
    echo "CONFIG_ONLY=1 -> configuration is valid."
    echo "No model will be loaded and no training will be started."
    exit 0
fi


# ================================================================
# OUTPUT DIRECTORY
# ================================================================

# rm -rf ${SAVE}
mkdir -p ${SAVE}

rm -f checkpoints/hf_model/downloads/*.lock
rm -f checkpoints/hf_model/*.lock


# ================================================================
# TRAIN / EVALUATE / TEST
# ================================================================

python -u examples/pytorch/summarization/run_summarization.py \
    --train_file data/xsum/train.json \
    --validation_file data/xsum/validation.json \
    --test_file data/xsum/test.json \
    --text_column document \
    --summary_column summary \
    --model_name_or_path '_setup/models/bart-large' \
    --cache_dir ${cache_dir} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_init ${lora_init} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --attn_composition ${attn_composition} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --ffn_adapter_layernorm_option ${ffn_adapter_layernorm_option} \
    --ffn_adapter_scalar ${ffn_adapter_scalar} \
    --ffn_adapter_init_option ${ffn_adapter_init_option} \
    --mid_dim 800 \
    --attn_bn ${attn_bn} \
    --ffn_bn ${ffn_bn} \
    --unfreeze_params ${unfreeze} \
    --preprocessing_num_workers 2 \
    --max_source_length 512 \
    --max_target_length 128 \
    --val_max_target_length 60 \
    --max_eval_samples ${max_eval_samples} \
    --num_beams ${num_beams} \
    --max_length 60 \
    --min_length 10 \
    --no_repeat_ngram_size 3 \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps ${max_steps} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_updates} \
    --seed ${seed} \
    --fp16 \
    --logging_steps ${logging_steps} \
    --save_total_limit 2 \
    --label_smoothing_factor ${label_smoothing_factor} \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --report_to ${report_to} \
    --run_name ${dataset}.${DATE}.${exp_name} \
    --overwrite_output_dir "True" \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --predict_with_generate \
    --output_dir ${SAVE} \
    --resume_from_checkpoint "${SAVE}/checkpoint-63000" \
    ${extra_cmd} \
    2>&1 | tee ${SAVE}/log.txt
