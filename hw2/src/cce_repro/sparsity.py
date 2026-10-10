"""Разреженность softmax на реальных данных: рисунок 3 и доля блоков, пропускаемых фильтром.

Повторяет логику ядра cce_backward.py на PyTorch:
  A = E C^T (накопление в fp32, приведение к dtype входа), LSE по тем же логитам,
  G = exp(A - LSE) - onehot(target), блок BLOCK_B x BLOCK_V пропускается,
  если все |G| < eps (_block_is_filtered). Словарь при сортировке переставлен
  argsort(среднего логита по токенам) по возрастанию, как sort_logit_avg.
"""

import torch

IGNORE_INDEX = -100


@torch.no_grad()
def logits_block(e_blk, c):
    return (
        e_blk @ c.T
    ).float()  # fp16 matmul: fp32-накопление, результат в fp16, как в ядре


@torch.no_grad()
def logit_avg(e, c, chunk: int = 512):
    """Средний логит каждого токена словаря по всем N токенам (буфер LA в прямом проходе)."""
    acc = torch.zeros(c.size(0), dtype=torch.float32, device=c.device)
    for i in range(0, e.size(0), chunk):
        acc += logits_block(e[i : i + chunk], c).sum(0)
    return acc / e.size(0)


@torch.no_grad()
def sorted_prob_curve(e, c, chunk: int = 256) -> torch.Tensor:
    """Средняя вероятность i-го по величине токена (рисунок 3), float64 на CPU, длина V."""
    acc = torch.zeros(c.size(0), dtype=torch.float64, device=c.device)
    for i in range(0, e.size(0), chunk):
        p = torch.softmax(logits_block(e[i : i + chunk], c), dim=-1)
        acc += p.sort(dim=-1, descending=True).values.double().sum(0)
    return (acc / e.size(0)).cpu()


@torch.no_grad()
def block_skip_stats(
    e, c, t, eps_list, block_b: int = 128, block_v: int = 128, sort_vocab: bool = True
) -> dict:
    """Для каждого eps: доля пропускаемых блоков и доля элементов |G| >= eps."""
    V = c.size(0)
    n_vb = (V + block_v - 1) // block_v
    order = torch.argsort(logit_avg(e, c)) if sort_vocab else None
    skipped = {eps: 0 for eps in eps_list}
    nnz = {eps: 0 for eps in eps_list}
    n_blocks = 0
    n_elems = 0
    for i in range(0, e.size(0), block_b):
        a = logits_block(e[i : i + block_b], c)
        tt = t[i : i + block_b]
        g = torch.exp(a - torch.logsumexp(a, dim=-1, keepdim=True))
        valid = tt != IGNORE_INDEX
        g[valid, tt[valid]] -= 1.0
        g[~valid] = 0.0  # игнорируемые токены в ядре вообще не участвуют (valids)
        g = g.abs()
        if order is not None:
            g = g[:, order]
        if V % block_v:
            g = torch.nn.functional.pad(g, (0, n_vb * block_v - V))
        bmax = g.view(g.size(0), n_vb, block_v).amax(dim=(0, 2))
        for eps in eps_list:
            skipped[eps] += int((bmax < eps).sum())
            nnz[eps] += int((g >= eps).sum())
        n_blocks += n_vb
        n_elems += int(valid.sum()) * V
    return {
        str(eps): {
            "frac_blocks_skipped": skipped[eps] / n_blocks,
            "frac_elements_nonzero": nnz[eps] / max(n_elems, 1),
        }
        for eps in eps_list
    }


def rank_below(curve: torch.Tensor, eps: float) -> int | None:
    """Первый ранг (с 1), где средняя вероятность ниже eps."""
    idx = (curve < eps).nonzero()
    return int(idx[0]) + 1 if len(idx) else None
