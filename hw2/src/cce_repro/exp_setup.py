"""1.0 Окружение, смоук-тест, кэш эмбеддингов, выбор формы блоков CCE по сетке."""

import importlib.metadata as md
import platform
import subprocess
import time

import torch
import torch.nn.functional as F

from . import data, methods
from .common import (
    CONFIG_JSON,
    MODEL_NAME,
    N_CACHE,
    Journal,
    exp_dir,
    log,
    save_json,
    seed_all,
)
from .kernels import KernelWatcher
from .measure import measure
from .t4_patch import apply_config, ensure_bf16_check, grid_configs, save_config

N_TIME = 8192  # время сравниваем при N таблицы 1
N_GRAD = 2048  # градиенты против fp32-эталона (полные логиты fp32 должны влезть)
SMOKE_METHODS = ["cce", "cce_exact", "torch_compile"]
SAFE_CONFIG = {
    "name": "b64_v128_w4",
    "BLOCK_B": 64,
    "BLOCK_V": 128,
    "BLOCK_D": 32,
    "num_warps": 4,
    "num_stages": 2,
}


def _ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None


def env_info() -> dict:
    props = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=20, check=False
        ).stdout
    except Exception as ex:  # noqa: BLE001
        smi = str(ex)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": props.multi_processor_count,
        "total_mib": props.total_memory / 2**20,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        **{
            p: _ver(p)
            for p in [
                "torch",
                "triton",
                "transformers",
                "datasets",
                "bitsandbytes",
                "liger-kernel",
                "cut-cross-entropy",
                "numpy",
            ]
        },
        "nvidia_smi": smi,
    }


def _grads(fn, e, c, t):
    e = e.detach().clone().requires_grad_(True)
    c = c.detach().clone().requires_grad_(True)
    loss = fn(e, c, t)
    loss.backward()
    return float(loss), e.grad.float(), c.grad.float()


def _ref(e, c, t):
    return F.cross_entropy((e @ c.T).float(), t, ignore_index=-100)


@torch.no_grad()
def ref_loss_chunked(e, c, t, chunk=1024) -> float:
    """fp32-эталон лосса без полной матрицы логитов (8192 x V в fp32 это 5 ГБ)."""
    tot, n = 0.0, int((t != -100).sum())
    cf = c.float()
    for i in range(0, e.size(0), chunk):
        tot += float(
            F.cross_entropy(
                e[i : i + chunk].float() @ cf.T,
                t[i : i + chunk],
                ignore_index=-100,
                reduction="sum",
            )
        )
    return tot / n


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def smoke(out_dir) -> None:
    """cce в fp16 против F.cross_entropy на малых тензорах (условие тестов авторов)."""
    j = Journal(out_dir / "smoke.jsonl", ("shape", "method", "tensor"))
    for shape in [(256, 512, 512), (252, 507, 497)]:
        N, V, D = shape
        torch.manual_seed(0)
        e = (torch.randn(N, D, device="cuda") / D**0.5).half()
        c = torch.randn(V, D, device="cuda").half()
        c[: min(N, V) // 2] = e[: min(N, V) // 2]
        t = torch.randint(0, V, (N,), device="cuda")
        t[::5] = -100
        torch.set_float32_matmul_precision("highest")
        gt = _grads(_ref, e.float(), c.float(), t)
        torch.set_float32_matmul_precision("high")
        ref = _grads(_ref, e, c, t)
        for m in SMOKE_METHODS:
            got = _grads(methods.get(m), e, c, t)
            for i, name in enumerate(("loss", "dE", "dC")):
                g, r, x = (
                    torch.tensor(v) if i == 0 else v for v in (gt[i], ref[i], got[i])
                )
                err, ref_err = (g - x).abs(), (g - r).abs()
                j.add(
                    {
                        "shape": str(shape),
                        "method": m,
                        "tensor": name,
                        "max_abs_err": float(err.max()),
                        "ref_fp16_max_abs_err": float(ref_err.max()),
                        "frac_violations": float((err > ref_err + 1e-3).float().mean()),
                        "cosine": cos(g, x) if i else None,
                    }
                )


def grid(out_dir) -> dict:
    j = Journal(out_dir / "grid.jsonl", ("name",))
    e8, t8 = data.window(N_TIME, 0)
    c = data.classifier()
    e2, t2 = e8[:N_GRAD].contiguous(), t8[:N_GRAD].contiguous()
    loss_ref = ref_loss_chunked(e8, c, t8)
    torch.set_float32_matmul_precision("highest")
    _, gE, gC = _grads(_ref, e2.float(), c.float(), t2)
    torch.set_float32_matmul_precision("high")
    watcher = KernelWatcher()
    for cfg in grid_configs():
        if j.has(name=cfg["name"]):
            continue
        row = dict(cfg)
        apply_config(cfg)
        t0 = time.time()
        try:
            _grads(methods.get("cce"), e2, c, t2)  # компиляция обеих веток
            with torch.no_grad():
                methods.get("cce")(e2, c, t2)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001  (не влезает в shared memory и т.п.)
            row.update(ok=False, error=f"{type(ex).__name__}: {str(ex)[:160]}")
            j.add(row)
            log(f"   {cfg['name']}: {row['error'][:90]}")
            torch.cuda.empty_cache()
            continue
        row["compile_s"] = time.time() - t0
        row["kernels"] = watcher.new()
        _, dE, dC = _grads(methods.get("cce"), e2, c, t2)
        row["cos_dE"], row["cos_dC"] = cos(dE, gE), cos(dC, gC)
        m = measure("cce", e8, c, t8, "fwbw", n_iter=10, time_budget_s=20)
        row.update(
            fwbw_ms=m.get("runtime_ms"),
            fwbw_ms_min=m.get("runtime_ms_min"),
            mem_mib=m.get("mem_mib"),
            n_iter=m.get("n_iter"),
            loss=m.get("loss"),
            loss_abs_err=abs(m.get("loss", 0) - loss_ref),
        )
        row["ok"] = (
            row["loss_abs_err"] < 1e-2
            and row["cos_dE"] > 0.999
            and row["cos_dC"] > 0.999
        )
        j.add(row)
        log(f"   {cfg['name']}: {row['fwbw_ms']:.1f} ms, ok={row['ok']}")
    return {}


def choose(out_dir) -> dict:
    import json

    rows = [
        json.loads(x)
        for x in (out_dir / "grid.jsonl").read_text(encoding="utf-8").splitlines()
        if x
    ]
    good = [r for r in rows if r.get("ok")]
    if not good:
        raise RuntimeError("ни одна форма блоков не прошла проверку")
    best = min(good, key=lambda r: r["fwbw_ms"])
    cfg = {
        k: best[k]
        for k in ("name", "BLOCK_B", "BLOCK_V", "BLOCK_D", "num_warps", "num_stages")
    }
    cfg["fwbw_ms_N8192"] = best["fwbw_ms"]
    save_config(cfg)
    log(
        f"Выбрана форма {cfg['name']}: {best['fwbw_ms']:.1f} ms при N={N_TIME} -> {CONFIG_JSON}"
    )
    return cfg


def run():
    seed_all(0)
    out = exp_dir("setup")
    ensure_bf16_check()
    save_json(out / "env.json", env_info())
    apply_config(SAFE_CONFIG)  # заведомо рабочая форма для смоук-теста
    smoke(out)
    info = {}
    for keep in (False, True):
        t0 = time.time()
        d = data.extract_embeddings(MODEL_NAME, keep, N_CACHE)
        info["all" if keep else "valid"] = {
            "seconds": time.time() - t0,
            "n_tokens": d["e"].size(0),
            "n_examples": d["n_examples"],
            "frac_ignored": float((d["t"] == -100).float().mean()),
            "nonfinite": d["nonfinite"],
        }
    save_json(out / "data.json", info)
    grid(out)
    choose(out)
