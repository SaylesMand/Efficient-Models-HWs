"""Пути, лог и журнал результатов (jsonl с возобновлением после обрыва)."""

import datetime
import json
import math
import os
import random
from pathlib import Path

HW2_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(os.environ.get("CCE_RESULTS", HW2_DIR / "results"))
CACHE_DIR = RESULTS_ROOT / "cache"
CONFIG_JSON = RESULTS_ROOT / "cce_config.json"
LOG_FILE = RESULTS_ROOT / "log.txt"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "yahma/alpaca-cleaned"
N_CACHE = 40960  # 5 непересекающихся окон по 8192 токена
MiB = 2**20

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def log(*args) -> None:
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now():%H:%M:%S} {msg}\n")  # noqa: DTZ005  (локальное время в логе)


def exp_dir(name: str) -> Path:
    d = RESULTS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "item"):
        return _clean(v.item())
    return str(v)


class Journal:
    """Журнал jsonl; при повторном запуске уже записанные ключи key_fields пропускаются."""

    def __init__(self, path: Path, key_fields: tuple[str, ...]):
        self.path = Path(path)
        self.key_fields = key_fields
        self.done = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.done.add(self.key(json.loads(line)))

    def key(self, row: dict) -> tuple:
        return tuple(row.get(k) for k in self.key_fields)

    def has(self, **kw) -> bool:
        return self.key(kw) in self.done

    def add(self, row: dict) -> None:
        row = _clean(row)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.done.add(self.key(row))


def save_json(path: Path, data) -> None:
    Path(path).write_text(
        json.dumps(_clean(data), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_json(path: Path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def seed_all(seed: int = 0) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
