"""Table 5 (XSum test set, ROUGE-1/2/L) from logs/table5/*.log. Usage: python summarize_table5.py [logs/table5_debug]"""
import re, sys, os
d = sys.argv[1] if len(sys.argv) > 1 else 'logs/table5'
rows = [('lora_s4', 'LoRA (6.1%), s=4', (44.59, 21.31, 36.25)),
        ('lora_s1', 'LoRA (6.1%), s=1', (44.17, 20.83, 35.74)),
        ('pa', 'PA (6.1%)', (44.35, 20.98, 35.98)),
        ('spa_s4', 'Scaled PA (6.1%), s=4', (44.85, 21.54, 36.58)),
        ('spa_learn', 'Scaled PA (6.1%), trainable s', (44.56, 21.31, 36.29))]
def last(txt, key):
    m = re.findall(rf'{key}\s*=\s*([0-9.]+)', txt)
    return float(m[-1]) if m else None
out = [f"{'method':32} {'ours R-1/2/L (Lsum)':30} {'paper R-1/2/L':20} {'ΔR-2':>5}  best dev R-2"]
for key, name, p in rows:
    f = os.path.join(d, key + '.log')
    if not os.path.exists(f):
        out.append(f"{name:32} (not started)"); continue
    t = open(f, errors='ignore').read()
    r1, r2, rl, rls = (last(t, 'predict_rouge' + k) for k in ('1', '2', 'L', 'Lsum'))
    dev = re.findall(r"'eval_rouge2': ([0-9.]+)", t)
    best = f"{max(map(float, dev)):.2f}" if dev else '-'
    if r2 is None:
        out.append(f"{name:32} {'(running / no test score yet)':30} {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}{'':5}{'':6}  {best}"); continue
    out.append(f"{name:32} {f'{r1:.2f}/{r2:.2f}/{rl:.2f} ({rls:.2f})':30} {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}  {r2-p[1]:+5.2f}  {best}")
txt = "\n".join(out); print(txt)
open(os.path.join(d, '..', 'table5_results.txt' if 'debug' not in d else 'table5_debug_results.txt'), 'w').write(txt + "\n")
