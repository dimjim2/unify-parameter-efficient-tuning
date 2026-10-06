#!/bin/bash
# Dry run of the Table 5 pipeline on a CPU (~10 s): runs run_table5.sh, batch_table5.sh and dtu_table5.lsf
# in a temporary copy where training is replaced by a stub that writes fake ROUGE scores.
# Checks: each method is launched once, with the right METHOD/GPU/MAX_STEPS, logs land in the right place,
# and summarize_table5.py builds the table. Nothing in your real logs/ is touched.
set -u
REPO=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/exps" "$T/_setup" "$T/bin" "$T/logs"
cp "$REPO"/run_table5.sh "$REPO"/summarize_table5.py "$REPO"/batch_table5.sh "$REPO"/dtu_table5.lsf "$T/"
cat > "$T/exps/run_xsum.sh" <<'STUB'
# Table 5 presets
echo "STUB METHOD=${METHOD:-} GPU=${CUDA_VISIBLE_DEVICES:-} DEBUG=${DEBUG:-0} MAX_STEPS=${MAX_STEPS:-100000}"
case ${METHOD:-} in lora_s4|lora_s1|pa|spa_s4|spa_learn) ;; *) echo "BAD METHOD"; exit 1;; esac
echo "{'eval_rouge2': 20.5}"; echo "  predict_rouge1 = 44.00"; echo "  predict_rouge2 = 21.00"
echo "  predict_rougeL = 36.00"; echo "  predict_rougeLsum = 36.10"
STUB
printf 'cd %s\n' "$T" > "$T/_setup/activate.sh"
printf '#!/bin/bash\nfor i in 0 1 2 3; do echo "GPU $i: FAKE"; done\n' > "$T/bin/nvidia-smi"
printf '#!/bin/bash\n:\n' > "$T/bin/sleep"
chmod +x "$T/bin/"*
export PATH="$T/bin:$PATH"
pass=0; fail=0
ok(){ if eval "$2"; then echo "  PASS  $1"; pass=$((pass+1)); else echo "  FAIL  $1"; fail=$((fail+1)); fi; }

echo "== 1. run_table5.sh (full mode)"
( cd "$T" && MAX_STEPS=30000 bash run_table5.sh > logs/progress.log 2>&1 )
for m in lora_s4 lora_s1 pa spa_s4 spa_learn; do
  ok "$m launched once with MAX_STEPS=30000" "grep -q 'STUB METHOD=$m .*MAX_STEPS=30000' '$T/logs/table5/$m.log'"
done
ok "5 methods spread over 4 GPUs (0,1,2,3,0)" "[ \"\$(grep -ho 'GPU=[0-9]' $T/logs/table5/*.log | sort | uniq -c | awk '{print \$1}' | tr -d '\n')\" = 2111 ]"
ok "ALL DONE printed" "grep -q 'ALL DONE' '$T/logs/progress.log'"

echo "== 2. summarize_table5.py"
( cd "$T" && python summarize_table5.py > logs/sum.txt 2>&1 )
ok "table has 5 result rows" "[ \$(grep -c '44.00/21.00/36.00' '$T/logs/sum.txt') = 5 ]"
ok "results file written" "[ -s '$T/logs/table5_results.txt' ]"

echo "== 3. batch_table5.sh (smoke test + full run)"
sed "s#^R=.*#R=$T#" "$T/batch_table5.sh" > "$T/batch_test.sh"
rm -rf "$T/logs/table5"
( cd "$T" && bash batch_test.sh > /dev/null 2>&1 )
ok "debug runs used DEBUG=1" "[ \$(grep -l 'DEBUG=1' $T/logs/table5_debug/*.log | wc -l) = 5 ]"
ok "full runs used DEBUG=0" "[ \$(grep -l 'DEBUG=0' $T/logs/table5/*.log | wc -l) = 5 ]"
ok "batch finished" "grep -q 'batch finished' '$T/logs/table5_batch.log'"

echo "== 4. dtu_table5.lsf (simulating the 5 LSF array jobs)"
rm -rf "$T/logs/table5"
for i in 1 2 3 4 5; do ( cd "$T" && LSB_JOBINDEX=$i LS_SUBCWD="$T" MAX_STEPS=30000 bash dtu_table5.lsf > /dev/null 2>&1 ); done
ok "each array index runs a different method" "[ \$(ls $T/logs/table5/*.log | wc -l) = 5 ]"
ok "LSF jobs pass MAX_STEPS" "[ \$(grep -l 'MAX_STEPS=30000' $T/logs/table5/*.log | wc -l) = 5 ]"

echo; echo "RESULT: $pass passed, $fail failed"
[ $fail = 0 ] && echo "Pipeline logic OK - safe to commit (real training still needs the GPU smoke test)"
