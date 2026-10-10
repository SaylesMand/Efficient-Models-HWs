"""python -m cce_repro.run <эксперимент> [...] | zip[=имя архива]

Каждый эксперимент дописывает результаты построчно и при повторном запуске продолжает с места обрыва.
"""

import importlib
import sys
import traceback
import zipfile

from .common import RESULTS_ROOT, log

EXPERIMENTS = {
    "setup": "exp_setup",  # 1.0
    "correctness": "exp_correctness",  # 1.1
    "table1": "exp_table1",  # 1.2
    "sparsity": "exp_sparsity",  # 1.3
    "ignored": "exp_ignored",  # 1.4
    "scaling": "exp_scaling",  # 1.5
    "vd": "exp_vd",  # 1.6
    "finetune_check": "exp_finetune:run_check",  # 2.1, короткая проверка
    "finetune": "exp_finetune",  # 2.1
    "max_batch": "exp_max_batch",  # 2.2
}


def make_zip(name: str = "results_loss_layer") -> str:
    """results/ без кэша эмбеддингов и чекпоинтов."""
    out = RESULTS_ROOT.parent / f"{name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in RESULTS_ROOT.rglob("*"):
            if p.is_file() and not {"cache", "ckpt"} & set(
                p.relative_to(RESULTS_ROOT).parts
            ):
                z.write(p, p.relative_to(RESULTS_ROOT.parent))
    print(f"Архив: {out}")
    return str(out)


def main(args: list[str]) -> None:
    for a in args:
        if a.split("=")[0] == "zip":
            make_zip(*a.split("=")[1:])
            continue
        log(f"===== {a} =====")
        try:
            mod, _, fn = EXPERIMENTS[a].partition(":")
            getattr(importlib.import_module(f"cce_repro.{mod}"), fn or "run")()
            log(f"===== {a}: done")
        except Exception:  # noqa: BLE001
            log(traceback.format_exc())
            log(f"===== {a}: FAILED (перезапуск продолжит с места остановки)")


if __name__ == "__main__":
    main(sys.argv[1:])
