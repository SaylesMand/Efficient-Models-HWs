"""1.6 Отношение |V|/D (таблица A3, раздел 5.1): один словарь Qwen2.5, разные D.

Только инференс бэкбона для эмбеддингов. Бэкбон 3B в fp32 занимает около 12 ГБ,
поэтому он считается в fp16 (в кэше записан флаг nonfinite).
"""

import torch

from . import data
from .common import Journal, exp_dir, save_json, seed_all
from .exp_table1 import KINDS, N, run_grid
from .t4_patch import setup_cce

MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", torch.float32),
    ("Qwen/Qwen2.5-1.5B-Instruct", torch.float32),
    ("Qwen/Qwen2.5-3B-Instruct", torch.float16),
]
METHODS = ["cce", "torch_compile", "baseline"]
REPS = 3


def run():
    setup_cce()
    seed_all(0)
    out = exp_dir("vd")
    j = Journal(out / "measurements.jsonl", ("method", "kind", "N", "rep", "model"))
    info = {}
    for name, dtype in MODELS:
        d = data.extract_embeddings(name, False, N, backbone_dtype=dtype)
        c = data.classifier(name)
        info[name] = {
            "V": c.size(0),
            "D": c.size(1),
            "V_over_D": c.size(0) / c.size(1),
            "backbone_dtype": str(dtype),
            "nonfinite": d["nonfinite"],
        }
        save_json(out / "models.json", info)
        # одно окно на модель; повторы это повторные замеры
        run_grid(
            j, METHODS, [N], REPS, KINDS, model_name=name, n_cache=N, same_window=True
        )
        data._C_CACHE.clear()
        torch.cuda.empty_cache()
