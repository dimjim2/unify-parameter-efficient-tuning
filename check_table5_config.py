"""CPU-only check of the Table 5 setup (no GPU, ~2-4 min, ~6 GB RAM).
For each METHOD it reads the arguments run_xsum.sh would pass, builds BART-large + PETL modules exactly like
run_summarization.py, counts trainable parameters (paper: 6.1%), and runs one forward/backward pass on
2 real XSum examples to check the modules are wired in and receive gradients.
Usage (repo root, env active):  python check_table5_config.py"""
import os, re, shlex, subprocess, sys, json, math
sys.path.insert(0, os.getcwd())
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser
from petl.options import TuneArguments
from petl.petl_encdec_model import PETLEncDecModel

MODEL = '_setup/models/bart-large'
METHODS = ['lora_s4', 'lora_s1', 'pa', 'spa_s4', 'spa_learn']
EXPECT = {'lora_s4': dict(alpha=408, scale=4.0), 'lora_s1': dict(alpha=102, scale=1.0),
          'pa': dict(scale=1.0), 'spa_s4': dict(scale=4.0), 'spa_learn': dict(scale='learnable')}

def script_args(method):
    """Run exps/run_xsum.sh with the python call replaced by echo, return its argument list."""
    src = open('exps/run_xsum.sh').read()
    src = re.sub(r'^python -u examples/pytorch/summarization/run_summarization.py', 'echo __ARGS__', src, flags=re.M)
    src = re.sub(r'^rm ', ': ', src, flags=re.M)
    src = re.sub(r'2>&1 \| tee .*', '; exit 0', src)
    out = subprocess.run(['bash', '-c', src], capture_output=True, text=True,
                         env={**os.environ, 'METHOD': method}).stdout
    line = [l for l in out.splitlines() if l.startswith('__ARGS__')][0]
    return shlex.split(line)[1:]

tok = AutoTokenizer.from_pretrained(MODEL)
ex = [json.loads(l) for _, l in zip(range(2), open('data/xsum/validation.json'))]
batch = tok([e['document'] for e in ex], max_length=128, truncation=True, padding=True, return_tensors='pt')
with tok.as_target_tokenizer():
    labels = tok([e['summary'] for e in ex], max_length=32, truncation=True, padding=True, return_tensors='pt')['input_ids']
labels[labels == tok.pad_token_id] = -100

ok_all = True
print(f"{'method':10} {'trainable':>12} {'% of BART':>9}  {'loss':>6}  checks")
for m in METHODS:
    args = script_args(m)
    targs, = HfArgumentParser(TuneArguments).parse_args_into_dataclasses(args, return_remaining_strings=True)[:1]
    config = AutoConfig.from_pretrained(MODEL)
    for k, v in vars(targs).items():
        if not hasattr(config, k):
            setattr(config, k, v)
    config.max_source_length, config.max_target_length = 512, 128
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL, config=config)
    n_base = sum(p.numel() for n, p in base.named_parameters() if 'ef_' not in n)
    model = PETLEncDecModel(config, targs, base)
    train = {n: p for n, p in model.named_parameters() if p.requires_grad}
    n_train = sum(p.numel() for p in train.values())
    model.train()
    out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels)
    out.loss.backward()
    checks = []
    checks.append(('finite loss', math.isfinite(out.loss.item())))
    checks.append(('only ef_ params trained', all('ef_' in n for n in train)))
    checks.append(('grads reach modules', any(p.grad is not None and p.grad.abs().sum() > 0 for p in train.values())))
    checks.append(('frozen BART has no grads', all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)))
    if m.startswith('lora'):
        checks.append((f"lora_alpha={EXPECT[m]['alpha']}", targs.lora_alpha == EXPECT[m]['alpha']))
        checks.append(('ffn_bn=102', targs.ffn_bn == 102))
        checks.append(('scaling s', any(abs(getattr(mod, 'scaling', -1) - EXPECT[m]['scale']) < 1e-6 for mod in model.modules())))
    else:
        checks.append(('ffn_bn=512 parallel', targs.ffn_bn == 512 and targs.ffn_option == 'parallel'))
        if m == 'spa_learn':
            checks.append(('learnable scale trained', any(n.endswith('.scale') for n in train)))
        else:
            checks.append((f"scale={EXPECT[m]['scale']}", any(getattr(mod, 'scale', None) == EXPECT[m]['scale'] for mod in model.modules())))
    ok = all(c for _, c in checks); ok_all &= ok
    bad = [n for n, c in checks if not c]
    print(f"{m:10} {n_train:>12,} {100*n_train/n_base:>8.2f}%  {out.loss.item():6.2f}  {'ALL OK' if ok else 'FAILED: ' + ', '.join(bad)}")
    del model, base
print("\nPaper Table 5: 6.1% params for every row.")
print("RESULT:", "configuration matches - ready for GPU runs" if ok_all else "PROBLEM - paste this output")
