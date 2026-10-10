#!/bin/bash
# Окружение для экспериментов на T4: Python 3.11, torch 2.5.1 / Triton 3.1 (тензорные ядра для sm_75).
# Повторный запуск только доустанавливает пакеты.
set -e
VENV=${VENV:-/content/venv_old}
pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q "$VENV" --python 3.11
UVP="uv pip install -q --python $VENV/bin/python"
$UVP "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
$UVP "torch==2.5.1" setuptools numpy pandas packaging huggingface_hub safetensors \
     "transformers==4.48.2" datasets accelerate "bitsandbytes==0.50.2"
$UVP --no-deps "liger-kernel==0.5.10"
$UVP --no-deps "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@3de376c106a1916bc5e1b619f9c77c87a461ee1c"
"$VENV/bin/python" -c "import torch, triton, transformers; print('torch', torch.__version__, '| triton', triton.__version__, '| transformers', transformers.__version__, '|', torch.cuda.get_device_name(0))"
