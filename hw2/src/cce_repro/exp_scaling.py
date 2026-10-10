"""1.5 Масштабирование по числу токенов (рисунки A1-A2): N = 256 ... 8192."""

from .common import Journal, exp_dir, seed_all
from .exp_table1 import KINDS, run_grid
from .t4_patch import setup_cce

NS = [256, 512, 1024, 2048, 4096, 8192]
METHODS = [
    "cce",
    "liger",
    "torchtune_8",
    "torch_compile",
    "baseline",
    "cce_kahan_full_c",
]
REPS = 3


def run():
    setup_cce()
    seed_all(0)
    j = Journal(
        exp_dir("scaling") / "measurements.jsonl", ("method", "kind", "N", "rep")
    )
    run_grid(j, METHODS, NS, REPS, KINDS, n_iter=10, budget=15.0)
