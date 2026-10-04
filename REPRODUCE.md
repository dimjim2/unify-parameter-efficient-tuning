# Reproducing Table 2 (He et al., ICLR 2022) on modern GPUs

Fork of https://github.com/jxhe/unify-parameter-efficient-tuning, updated for PyTorch 2.7 / CUDA 12.8
(tested on NVIDIA B200, Python 3.9).

## Setup (once, ~10 min)
    bash setup_env.sh
    source _setup/activate.sh            # in every new shell

## Run Table 2 (4 methods x MNLI/SST2 x 5 seeds; ~3 h on 4x B200)
    nohup bash run_table2.sh > logs/table2_progress.log 2>&1 &
    python summarize_table2.py           # -> logs/table2_results.txt

Single run:  METHOD=mam|lora|prefix|adapter TASK_NAME=sst2|mnli SLURM_ARRAY_TASK_ID=0..4 bash exps/run_glue.sh
Smoke test:  add DEBUG=1

## Changes vs. the original repo
- exps/run_glue.sh: METHOD/TASK_NAME/seed from the command line, full dev sets, dynamic padding, one GPU per run
- local model copies and XSum from the Hugging Face parquet mirror (original links/redirects no longer work)
- Our results: logs/table2_results.txt
