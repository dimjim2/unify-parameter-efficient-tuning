#!/bin/bash
# UCloud "Batch processing" script for Table 5: smoke test, then full runs, then summary.
R=/work/unify-parameter-efficient-tuning
source $R/_setup/activate.sh || { echo "ERROR: environment not found - is the repo folder attached?"; exit 1; }
cd $R
mkdir -p logs
exec > >(tee -a logs/table5_batch.log) 2>&1
echo "$(date) batch start on $(nvidia-smi -L | wc -l) GPU(s)"
grep -q "Table 5 presets" exps/run_xsum.sh || { echo "ERROR: run_xsum.sh not patched (run patch_xsum_t5.py first)"; exit 1; }
[ -f run_table5.sh ] && [ -f summarize_table5.py ] || { echo "ERROR: run_table5.sh / summarize_table5.py missing"; exit 1; }
DEBUG=1 bash run_table5.sh
python summarize_table5.py logs/table5_debug
n=$(grep -l "predict_rouge2" logs/table5_debug/*.log 2>/dev/null | wc -l)
if [ "$n" -lt 5 ]; then echo "ERROR: smoke test failed for $((5-n)) method(s) - see logs/table5_debug/"; exit 1; fi
echo "$(date) smoke test passed"
bash run_table5.sh
python summarize_table5.py
echo "$(date) batch finished - results in logs/table5_results.txt"
