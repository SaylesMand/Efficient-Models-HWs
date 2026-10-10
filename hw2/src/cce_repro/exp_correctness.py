"""1.1 Корректность: критерий тестов авторов (tests/test_cce_loss_backward.py) для всех вариантов.

gt  - эталон в fp32, ref - обычный PyTorch на fp16, x - проверяемый метод на fp16.
Условие авторов для fp16: |gt - x| <= |gt - ref| + 1e-3 поэлементно.
"""

import torch
import torch.nn.functional as F

from . import data, methods
from .common import Journal, exp_dir, seed_all
from .t4_patch import setup_cce

TOL = 1e-3
CHECK = [
    "cce",
    "cce_exact",
    "cce_kahan",
    "cce_kahan_full_c",
    "cce_kahan_full_e",
    "cce_no_filter",
    "cce_no_sort",
    "cce_eps_bf16",
    "torch_compile",
]
SHAPES = [(256, 512, 512), (252, 507, 512), (252, 507, 497)]  # формы из тестов авторов


def _grads(fn, e, c, t, name=None):
    e = e.detach().clone().requires_grad_(True)
    c = c.detach().clone().requires_grad_(True)
    loss = fn(e, c, t)
    loss.backward()
    if name:
        methods.after_backward(name)
    return loss.detach().float().reshape(1), e.grad.float(), c.grad.float()


def _ref(e, c, t):
    return F.cross_entropy((e @ c.T).float(), t, ignore_index=-100)


def compare(j, case, e16, c16, t):
    torch.set_float32_matmul_precision("highest")
    gt = _grads(_ref, e16.float(), c16.float(), t)
    torch.set_float32_matmul_precision("high")
    ref = _grads(_ref, e16, c16, t)
    for m in CHECK:
        if j.has(case=case, method=m, tensor="loss"):
            continue
        got = _grads(methods.get(m), e16, c16, t, m)
        for name, g, r, x in zip(("loss", "dE", "dC"), gt, ref, got):
            err, ref_err = (g - x).abs().flatten(), (g - r).abs().flatten()
            j.add(
                {
                    "case": case,
                    "method": m,
                    "tensor": name,
                    "N": e16.size(0),
                    "max_abs_err": float(err.max()),
                    "mean_abs_err": float(err.mean()),
                    "ref_fp16_max_abs_err": float(ref_err.max()),
                    "frac_violations": float((err > ref_err + TOL).float().mean()),
                    "cosine": float(
                        F.cosine_similarity(g.flatten(), x.flatten(), dim=0)
                    )
                    if g.numel() > 1
                    else None,
                    "rel_err": float((g - x).norm() / g.norm().clamp(min=1e-30)),
                }
            )


def run():
    setup_cce()
    seed_all(0)
    out = exp_dir("correctness")
    j = Journal(out / "checks.jsonl", ("case", "method", "tensor"))

    for N, V, D in SHAPES:
        for invalids in (False, True):
            torch.manual_seed(0)
            e = (torch.randn(N, D, device="cuda") / D**0.5).half()
            c = torch.randn(V, D, device="cuda").half()
            c[: min(N, V) // 2] = e[: min(N, V) // 2]
            t = torch.randint(0, V, (N,), device="cuda")
            if invalids:
                t[torch.randperm(N, device="cuda")[: N // 5]] = -100
            compare(j, f"random_{N}x{V}x{D}{'_ign' if invalids else ''}", e, c, t)

    e, t = data.window(2048, 0, keep_invalids=True)  # реальные данные с промптом (-100)
    compare(j, "qwen_alpaca_2048", e, data.classifier(), t)

    # shift=1 на (B, T, D) против ручного сдвига
    from cut_cross_entropy import linear_cross_entropy

    js = Journal(out / "extra.jsonl", ("check",))
    if not js.has(check="shift"):
        torch.manual_seed(1)
        B, T, D, V = 4, 128, 896, 4096
        e3 = (torch.randn(B, T, D, device="cuda") / D**0.5).half()
        c3 = torch.randn(V, D, device="cuda").half()
        t3 = torch.randint(0, V, (B, T), device="cuda")
        t3[:, :40] = -100
        a = linear_cross_entropy(e3, c3, t3, shift=1, impl="cce")
        b = linear_cross_entropy(
            e3[:, :-1].reshape(-1, D), c3, t3[:, 1:].reshape(-1), impl="cce"
        )
        r = _ref(e3[:, :-1].reshape(-1, D).float(), c3.float(), t3[:, 1:].reshape(-1))
        js.add(
            {
                "check": "shift",
                "shift_vs_manual": abs(float(a) - float(b)),
                "shift_vs_fp32": abs(float(a) - float(r)),
            }
        )
    if not js.has(check="determinism"):
        c = data.classifier()
        runs = [_grads(methods.get("cce"), e, c, t) for _ in range(3)]
        js.add(
            {
                "check": "determinism",
                "loss_equal": all(torch.equal(runs[0][0], x[0]) for x in runs[1:]),
                "dE_equal": all(torch.equal(runs[0][1], x[1]) for x in runs[1:]),
                "dC_equal": all(torch.equal(runs[0][2], x[2]) for x in runs[1:]),
                "dC_max_diff": max(
                    float((runs[0][2] - x[2]).abs().max()) for x in runs[1:]
                ),
                "dC_rel_diff": max(
                    float((runs[0][2] - x[2]).norm() / runs[0][2].norm())
                    for x in runs[1:]
                ),
            }
        )
