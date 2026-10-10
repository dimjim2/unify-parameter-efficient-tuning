"""Figure 4 (XSum panel): results table + plot of test ROUGE-2 vs % fine-tuned parameters.
Usage (repo root):  python fig4_results.py            -> logs/fig4_results.txt, .csv, logs/fig4_xsum.png/.pdf
                    python fig4_results.py logs/fig4_debug   (smoke-test logs)
% params follows the paper's Appendix B counting (final prefix vectors, not the training-time MLP);
if logs/fig4_params.json exists (from check_fig4_config.py) its BitFit count is used."""
import csv, json, os, re, sys

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else 'logs/fig4'
D, LAYERS, N_BASE = 1024, 12, 406_291_456          # BART-large: d, layers, total params
def paper_params(m):
    if m.startswith('prefix_l'):  return 2 * int(m[8:]) * D * 3 * LAYERS      # 2ld x 3 attn x 12
    if m.startswith('lora_r'):    return 4 * int(m[6:]) * D * 3 * LAYERS      # 4rd x 3 attn x 12
    if m.startswith('adapter_r'): return 2 * int(m[9:]) * D * 5 * LAYERS      # 2rd x (3 attn + 2 ffn) x 12
    if m == 'fullft':             return N_BASE
    return None
FAMILY = {'prefix': 'Prefix tuning', 'lora': 'LoRA', 'adapter': 'Adapter', 'bitfit': 'BitFit', 'fullft': 'Full fine-tuning'}
# reference numbers printed in the paper (Tables 3, 6, 12) for configs that match ours
PAPER_REF = {'prefix_l200': (3.6, 20.46), 'prefix_l512': (9.2, 20.40), 'lora_r200': (7.2, 20.29), 'bitfit': (0.1, 17.32)}
PAPER_FULLFT = 21.94

pjson = {}
if os.path.exists('logs/fig4_params.json'):
    pjson = json.load(open('logs/fig4_params.json'))

def last(t, k):
    v = re.findall(rf'{k}\s*=\s*([0-9.]+)', t); return float(v[-1]) if v else None

rows = []
for f in sorted(os.listdir(LOGDIR)) if os.path.isdir(LOGDIR) else []:
    if not f.endswith('.log'): continue
    m = f[:-4]; fam = m.split('_')[0]
    if fam not in FAMILY: continue
    t = open(os.path.join(LOGDIR, f), errors='ignore').read()
    n = paper_params(m) or pjson.get(m, {}).get('trainable')
    pct = 100 * n / N_BASE if n else (0.1 if m == 'bitfit' else None)
    dev = [float(x) for x in re.findall(r"'eval_rouge2': ([0-9.]+)", t)]
    rows.append(dict(method=m, family=FAMILY[fam], pct_params=pct,
                     rouge1=last(t, 'predict_rouge1'), rouge2=last(t, 'predict_rouge2'), rougeL=last(t, 'predict_rougeL'),
                     best_dev_rouge2=max(dev) if dev else None,
                     paper_rouge2=PAPER_REF.get(m, (None, None))[1]))
rows.sort(key=lambda r: (r['family'], r['pct_params'] or 0))

fmt = lambda v, p=2: '-' if v is None else f'{v:.{p}f}'
lines = [f"{'method':14} {'family':17} {'% params':>8}  {'test R-1/R-2/R-L':22} {'best dev R-2':>12}  {'paper R-2':>9}"]
for r in rows:
    test = '(running / no test score)' if r['rouge2'] is None else f"{fmt(r['rouge1'])}/{fmt(r['rouge2'])}/{fmt(r['rougeL'])}"
    lines.append(f"{r['method']:14} {r['family']:17} {fmt(r['pct_params'])+'%':>8}  {test:22} {fmt(r['best_dev_rouge2']):>12}  {fmt(r['paper_rouge2']):>9}")
lines.append(f"\nPaper full fine-tuning (Table 6, our run in the paper): R-2 {PAPER_FULLFT}")
txt = '\n'.join(lines); print(txt)
base = os.path.join(os.path.dirname(LOGDIR.rstrip('/')) or '.', 'fig4' + ('_debug' if 'debug' in LOGDIR else ''))
open(base + '_results.txt', 'w').write(txt + '\n')
with open(base + '_results.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['method']); w.writeheader(); w.writerows(rows)

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
except ImportError:
    print("\nmatplotlib not installed - table saved, no plot. Install with:  uv pip install 'matplotlib<3.9'"); sys.exit(0)

# fixed categorical order (validated palette, light mode); markers = secondary encoding
STYLE = {'LoRA': ('#2a78d6', 'o'), 'Adapter': ('#eb6834', 'v'), 'Prefix tuning': ('#1baf7a', 's'), 'BitFit': ('#eda100', 'D')}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)

done = [r for r in rows if r['rouge2'] is not None and r['family'] in STYLE]
for fam, (col, mk) in STYLE.items():
    pts = sorted((r['pct_params'], r['rouge2']) for r in done if r['family'] == fam)
    if not pts: continue
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, linewidth=2, marker=mk, markersize=8, markeredgecolor='#fcfcfb',
            markeredgewidth=1.5, label=fam, zorder=3)
    ax.annotate(fam, (xs[-1], ys[-1]), xytext=(6, 0), textcoords='offset points', va='center',
                fontsize=9, color=INK)
ax.axhline(PAPER_FULLFT, color=MUTED, linestyle='--', linewidth=1.5, zorder=1, label='Full fine-tuning')
ft = [r for r in rows if r['method'] == 'fullft' and r['rouge2'] is not None]
if ft:
    ax.axhline(ft[0]['rouge2'], color=INK, linestyle=':', linewidth=1.5, zorder=1,
               label=f"Full fine-tuning, ours ({ft[0]['rouge2']:.2f})")
ax.set_xlabel('Fine-tuned parameters (%)'); ax.set_ylabel('XSum ROUGE-2 (test)')
ax.set_xlim(-0.5, 16)
ax.legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(f'{base}_xsum.{ext}', facecolor=fig.get_facecolor())
print(f"\nplot saved: {base}_xsum.png / .pdf")
