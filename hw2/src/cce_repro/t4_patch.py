"""Адаптация ядер CCE под T4 без правки исходников библиотеки.

Конфигурация блоков в cut_cross_entropy фиксирована (tl_autotune._cce_best_config:
128x128x32, 4 варпа, 4 стадии) и на T4 не помещается в 64 КБ shared memory. Она навешана
на ядра через triton.heuristics, то есть хранится в словаре `Heuristics.values`, который
можно подменить после импорта. Прямой и обратный проходы получают одну конфигурацию:
по комментарию авторов в tl_autotune.py, разные конфигурации портят градиент.
"""

from itertools import product

from .common import CONFIG_JSON, load_json, log, save_json

AUTHORS_CONFIG = {
    "name": "authors",
    "BLOCK_B": 128,
    "BLOCK_V": 128,
    "BLOCK_D": 32,
    "num_warps": 4,
    "num_stages": 4,
}
# выбрана в ноутбуке 1 (раздел 1.0): самая быстрая корректная форма при N = 8192
NOTEBOOK1_CONFIG = {
    "name": "b64_v64_w4",
    "BLOCK_B": 64,
    "BLOCK_V": 64,
    "BLOCK_D": 32,
    "num_warps": 4,
    "num_stages": 2,
}
_KEYS = ("BLOCK_B", "BLOCK_V", "BLOCK_D", "num_warps", "num_stages")


def grid_configs() -> list[dict]:
    """Сетка форм блоков: B x V x варпы, BLOCK_D = 32, 2 стадии."""
    out = []
    for bb, bv, w in product((32, 64, 128), (32, 64, 128, 256), (4, 8)):
        out.append(
            {
                "name": f"b{bb}_v{bv}_w{w}",
                "BLOCK_B": bb,
                "BLOCK_V": bv,
                "BLOCK_D": 32,
                "num_warps": w,
                "num_stages": 2,
            }
        )
    return out


def ensure_bf16_check() -> bool:
    """cce_linear_cross_entropy падает, если torch.cuda.is_bf16_supported() == False.
    На T4 функция возвращает True (эмуляция); если нет, подменяем и сообщаем."""
    import torch

    if torch.cuda.is_bf16_supported():
        return False
    torch.cuda.is_bf16_supported = lambda *a, **k: True  # type: ignore[assignment]
    log("WARNING: is_bf16_supported() == False, проверка CCE обойдена")
    return True


def _kernels():
    from cut_cross_entropy import cce_backward, cce_lse_forward

    return {
        "forward": cce_lse_forward._cce_lse_forward_kernel,
        "backward": cce_backward._cce_backward_kernel,
    }


def apply_config(cfg: dict) -> None:
    """Подменяет конфигурацию для fp16/bf16 входов; для fp32 остаётся авторская."""
    import torch

    if cfg.get("name") == AUTHORS_CONFIG["name"]:
        return
    for kname, kernel in _kernels().items():
        values = getattr(kernel, "values", None)
        if values is None:
            raise RuntimeError(
                f"Ядро {kname} не обёрнуто в Heuristics; CCE_AUTOTUNE=1?"
            )
        for key in _KEYS:
            orig = values[key]
            values[key] = lambda args, _o=orig, _v=cfg[key]: (
                _o(args) if args["E"].dtype == torch.float32 else _v
            )


def setup_cce() -> dict | None:
    """Применяет форму блоков, выбранную в эксперименте setup."""
    ensure_bf16_check()
    cfg = load_json(CONFIG_JSON)
    if cfg is None:
        cfg = NOTEBOOK1_CONFIG
        log(f"нет results/cce_config.json: форма {cfg['name']}, выбранная в ноутбуке 1")
    apply_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    save_json(CONFIG_JSON, cfg)
