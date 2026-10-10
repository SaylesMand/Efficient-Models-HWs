"""1.4 Удаление игнорируемых токенов (таблица A1, приложение B).

Данные с промптом (-100). baseline, чанки и Liger считаются по всем токенам и с удалением
игнорируемых до логитов (*_prefilter). CCE и torch.compile в библиотеке удаляют их сами.
Числа без игнорируемых токенов той же длины берутся из 1.2 (таблица 1).
"""

from .common import Journal, exp_dir, seed_all
from .exp_table1 import KINDS, N, run_grid
from .t4_patch import setup_cce

METHODS = [
    "baseline",
    "baseline_prefilter",
    "torchtune_8",
    "torchtune_8_prefilter",
    "liger",
    "liger_prefilter",
    "cce",
    "torch_compile",
]
REPS = 3


def run():
    setup_cce()
    seed_all(0)
    j = Journal(
        exp_dir("ignored") / "measurements.jsonl", ("method", "kind", "N", "rep")
    )
    run_grid(j, METHODS, [N], REPS, KINDS, keep_invalids=True)
