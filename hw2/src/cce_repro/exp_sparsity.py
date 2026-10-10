"""1.3 Разреженность softmax и фильтрация: рисунок 3, разделы 4.3 и 5.1, приложение E."""

import numpy as np
import torch

from . import data, methods
from .common import Journal, exp_dir, save_json, seed_all
from .methods import EPS_BF16, EPS_FP16
from .sparsity import block_skip_stats, rank_below, sorted_prob_curve
from .t4_patch import setup_cce

N = 8192
FILTER_METHODS = ["cce", "cce_eps_bf16", "cce_no_filter", "cce_no_sort", "cce_exact"]
REPS = 3


def _grads(name, e, c, t):
    e = e.detach().clone().requires_grad_(True)
    c = c.detach().clone().requires_grad_(True)
    methods.get(name)(e, c, t).backward()
    methods.after_backward(name)
    return e.grad.float(), c.grad.float()


def run():
    cfg = setup_cce() or {}
    seed_all(0)
    out = exp_dir("sparsity")
    e, t = data.window(N, 0)
    c = data.classifier()

    if not (out / "curve.npy").exists():
        curve = sorted_prob_curve(e, c)
        np.save(out / "curve.npy", curve.numpy())
        bb, bv = cfg.get("BLOCK_B", 128), cfg.get("BLOCK_V", 128)
        stats = {
            "N": N,
            "block": [bb, bv],
            "rank_below_bf16": rank_below(curve, EPS_BF16),
            "rank_below_fp16": rank_below(curve, EPS_FP16),
            "prob_at_rank": {
                r: float(curve[r - 1])
                for r in (1, 2, 5, 10, 50, 100, 1000, 10000, 100000)
            },
        }
        for sort in (True, False):
            for eps, v in block_skip_stats(
                e, c, t, [EPS_BF16, EPS_FP16], bb, bv, sort_vocab=sort
            ).items():
                tag = f"{'sorted' if sort else 'unsorted'}_{'bf16' if float(eps) == EPS_BF16 else 'fp16'}"
                stats[f"blocks_skipped_{tag}"] = v["frac_blocks_skipped"]
                stats[f"nonzero_{'bf16' if float(eps) == EPS_BF16 else 'fp16'}"] = v[
                    "frac_elements_nonzero"
                ]
        save_json(out / "stats.json", stats)

    from .exp_table1 import run_grid

    jt = Journal(out / "timing.jsonl", ("method", "kind", "N", "rep"))
    run_grid(jt, FILTER_METHODS, [N], REPS, ("bw", "fwbw"))

    jg = Journal(out / "grad_error.jsonl", ("method",))
    ref = None
    for m in FILTER_METHODS[:-1]:
        if jg.has(method=m):
            continue
        ref = ref or _grads("cce_exact", e, c, t)
        got = _grads(m, e, c, t)
        row = {"method": m}
        for lbl, g, r in zip(("dE", "dC"), got, ref):
            row[f"{lbl}_rel_err"] = float((g - r).norm() / r.norm())
            row[f"{lbl}_max_abs"] = float((g - r).abs().max())
            row[f"{lbl}_cos"] = float(
                torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), dim=0)
            )
            row[f"{lbl}_rows_zeroed"] = float(
                ((g.abs().sum(1) == 0) & (r.abs().sum(1) > 0)).float().mean()
            )
        jg.add(row)
