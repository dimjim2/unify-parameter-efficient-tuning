import glob, re, statistics
paper = {('mnli','mam'):(87.4,0.3),('mnli','lora'):(87.2,0.4),('mnli','prefix'):(86.3,0.4),('mnli','adapter'):(87.2,0.2),
         ('sst2','mam'):(94.2,0.3),('sst2','lora'):(94.2,0.2),('sst2','prefix'):(94.0,0.1),('sst2','adapter'):(94.2,0.1)}
lines = [f"{'task':5} {'method':8} {'ours (median ± std)':20} {'n':>2}   {'paper':11} {'diff':>5}   seeds"]
for t in ['mnli','sst2']:
    for m in ['prefix','lora','adapter','mam']:
        accs = []
        for f in sorted(glob.glob(f'logs/table2/{m}.{t}.seed*.log')):
            hits = re.findall(r'eval_accuracy\s*=\s*([0-9.]+)', open(f, errors='ignore').read())
            if hits: accs.append(float(hits[0]) * 100)      # first = MNLI matched
        if not accs:
            lines.append(f"{t:5} {m:8} (no finished runs)"); continue
        med = statistics.median(accs); sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        p = paper[(t, m)]
        lines.append(f"{t:5} {m:8} {med:6.1f} ± {sd:.1f}{'':10} {len(accs):>2}   {p[0]:.1f} ± {p[1]:.1f}  {med-p[0]:+5.1f}   "
                     + " ".join(f"{a:.1f}" for a in accs))
out = "\n".join(lines); print(out)
open('logs/table2_results.txt', 'w').write(out + "\n")
