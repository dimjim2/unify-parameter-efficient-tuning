#!/bin/bash
# One-time setup for reproducing He et al. (ICLR 2022) Table 2 / XSum.
# Usage (from the repo root):  bash setup_env.sh      then in every new shell:  source _setup/activate.sh
set -e
R=$(cd "$(dirname "$0")" && pwd); S=$R/_setup; cd "$R"; mkdir -p "$S"
[ -x "$S/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$S/bin" INSTALLER_NO_MODIFY_PATH=1 sh
export PATH="$S/bin:$PATH" UV_PYTHON_INSTALL_DIR="$S/uv-python" UV_LINK_MODE=copy
[ -x "$S/petl-env/bin/python" ] || uv venv --python 3.9 "$S/petl-env"
source "$S/petl-env/bin/activate"
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
if [ -f requirements-lock.txt ]; then uv pip install -r requirements-lock.txt
else uv pip install "datasets==1.11.0" "numpy<2" "pyarrow<12" rouge_score nltk scipy scikit-learn sentencepiece protobuf; fi
python -c "import nltk; [nltk.download(p, download_dir='$S/petl-env/nltk_data', quiet=True) for p in ('punkt','punkt_tab')]"
for m in roberta-base facebook/bart-large; do
  d="$S/models/$(basename $m)"
  [ -f "$d/pytorch_model.bin" ] || uvx --from huggingface_hub hf download $m \
      config.json vocab.json merges.txt tokenizer.json pytorch_model.bin --local-dir "$d"
done
mkdir -p data/xsum
for s in train validation test; do
  [ -f data/xsum/$s.json ] && continue
  curl -L -o data/xsum/$s.parquet https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/main/data/$s-00000-of-00001.parquet
  python -c "import pandas as pd; pd.read_parquet('data/xsum/$s.parquet').to_json('data/xsum/$s.json', orient='records', lines=True, force_ascii=False)"
done
cat > "$S/activate.sh" <<EOT
export PATH=$S/bin:\$PATH
export UV_PYTHON_INSTALL_DIR=$S/uv-python
source $S/petl-env/bin/activate
cd $R
EOT
python -c "import torch, transformers, datasets, nltk, petl; print('SETUP OK | torch', torch.__version__, '| GPUs:', torch.cuda.device_count())"
echo "Next time:  source $S/activate.sh"
