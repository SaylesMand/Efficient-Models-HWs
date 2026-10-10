"""Все методы подсчёта linear-cross-entropy с единым интерфейсом fn(e, c, targets) -> loss.

e: (N, D), c: (V, D), targets: (N,) с -100 для игнорируемых токенов. Лосс усреднён по
неигнорируемым токенам, как F.cross_entropy(reduction="mean").

Соответствие строкам таблицы 1 статьи:
  1 cce, 2 liger, 3 torchtune_8, 4 torch_compile, 5 baseline,
  6 cce_no_sort, 7 cce_no_filter, 8 cce_kahan, 9 cce_kahan_full_c, 10 cce_kahan_full_e.
Дополнительно: cce_exact (без фильтрации, fp32-накопление, эталон библиотеки),
cce_eps_bf16 (fp16-входы с порогом bf16 2^-12), *_prefilter (удаление игнорируемых
токенов до подсчёта логитов, таблица A1).
"""

import functools

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100
EPS_BF16 = 2.0**-12
EPS_FP16 = 2.0**-15  # то, что даёт filter_eps="auto" для fp16: finfo(fp16).eps / 32


def baseline(e, c, t):
    """Как benchmark/__main__.py у авторов: полные логиты, upcast до fp32, F.cross_entropy."""
    logits = e @ c.T
    return F.cross_entropy(logits.float(), t, ignore_index=IGNORE_INDEX)


@torch.compile(dynamic=True)
def _ce_chunk_sum(logits, t):
    return F.cross_entropy(
        logits.float(), t, ignore_index=IGNORE_INDEX, reduction="sum"
    )


def torchtune_chunked(e, c, t, n_chunks: int = 8):
    """Как CEWithChunkedOutputLoss в torchtune: выходная проекция считается по чанкам
    токенов, на каждый чанк скомпилированный CE с upcast до fp32. Логиты чанков в fp16
    всё равно живут до backward, экономится только fp32-копия."""
    n_valid = (t != IGNORE_INDEX).sum().clamp(min=1)
    total = e.new_zeros((), dtype=torch.float32)
    for e_i, t_i in zip(e.chunk(n_chunks), t.chunk(n_chunks)):
        total = total + _ce_chunk_sum(e_i @ c.T, t_i)
    return total / n_valid


@functools.cache
def _liger_loss():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    return LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX)


def liger(e, c, t):
    return _liger_loss()(c, e, t)


def _lce(**kw):
    def fn(e, c, t):
        from cut_cross_entropy import linear_cross_entropy

        return linear_cross_entropy(e, c, t, **kw)

    return fn


def _disable_sort():
    from cut_cross_entropy import cce

    if not hasattr(cce, "_orig_sort_logit_avg"):
        cce._orig_sort_logit_avg = cce.sort_logit_avg
    cce.sort_logit_avg = lambda logit_avg: None


def restore_vocab_sort():
    from cut_cross_entropy import cce

    if hasattr(cce, "_orig_sort_logit_avg"):
        cce.sort_logit_avg = cce._orig_sort_logit_avg


class _DisableSortInBackward(torch.autograd.Function):
    """Стоит после узла CCE: его backward выполняется раньше backward ядра CCE."""

    @staticmethod
    def forward(ctx, loss):
        return loss.clone()

    @staticmethod
    def backward(ctx, g):
        _disable_sort()
        return g


class _RestoreSortInBackward(torch.autograd.Function):
    """Стоит перед узлом CCE на входе e: его backward выполняется после ядра CCE."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        restore_vocab_sort()
        return g


def _cce_no_sort(e, c, t):
    """Абляция «без сортировки словаря»: на время backward sort_logit_avg возвращает None, и ядро
    идёт по словарю в исходном порядке (HAS_VOCAB_ORDERING=False). Фильтрация градиента остаётся."""
    from cut_cross_entropy import linear_cross_entropy

    if e.requires_grad:
        e = _RestoreSortInBackward.apply(e)
    loss = linear_cross_entropy(e, c, t, impl="cce")
    return _DisableSortInBackward.apply(loss)


def prefilter(fn):
    """Таблица A1: выбросить токены с -100 до подсчёта логитов."""

    def wrapped(e, c, t):
        keep = (t != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        return fn(e.index_select(0, keep), c, t.index_select(0, keep))

    return wrapped


METHODS = {
    "cce": _lce(impl="cce"),
    "liger": liger,
    "torchtune_8": torchtune_chunked,
    "torch_compile": _lce(impl="torch_compile"),
    "baseline": baseline,
    "cce_no_sort": _cce_no_sort,
    "cce_no_filter": _lce(impl="cce", filter_eps=None),
    "cce_kahan": _lce(impl="cce", accum_e_fp32=True, accum_c_fp32=True),
    "cce_kahan_full_c": _lce(impl="cce_kahan_full_c"),
    "cce_kahan_full_e": _lce(impl="cce_kahan_full_e"),
    "cce_exact": _lce(impl="cce_exact"),
    "cce_eps_bf16": _lce(impl="cce", filter_eps=EPS_BF16),
    "baseline_prefilter": prefilter(baseline),
    "torchtune_8_prefilter": prefilter(torchtune_chunked),
    "liger_prefilter": prefilter(liger),
}

# строки таблицы 1 статьи в её порядке
TABLE1_ROWS = [
    "cce",
    "liger",
    "torchtune_8",
    "torch_compile",
    "baseline",
    "cce_no_sort",
    "cce_no_filter",
    "cce_kahan",
    "cce_kahan_full_c",
    "cce_kahan_full_e",
]
# Liger считает градиент внутри прямого прохода, в статье у него только Loss+Gradient
FWBW_ONLY = {"liger", "liger_prefilter"}


def get(name: str):
    return METHODS[name]


def after_backward(name: str) -> None:
    if name == "cce_no_sort":
        restore_vocab_sort()
