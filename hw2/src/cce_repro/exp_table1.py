"""1.2 Таблица 1: 10 методов x {Loss, Gradient, Loss+Gradient}, N = 8192, 5 окон токенов."""

from . import data
from .common import MODEL_NAME, N_CACHE, Journal, exp_dir, log, seed_all
from .measure import measure
from .methods import FWBW_ONLY, TABLE1_ROWS
from .t4_patch import setup_cce

N = 8192
REPS = 5
KINDS = ("fw", "bw", "fwbw")


def run_grid(
    j,
    methods_list,
    Ns,
    reps,
    kinds,
    keep_invalids=False,
    n_iter=20,
    budget=30.0,
    model_name=MODEL_NAME,
    n_cache=N_CACHE,
    same_window=False,
):
    """Общий цикл замеров для 1.2-1.6. Повтор rep берёт окно rep из кэша (same_window: всегда
    окно 0, повторы это повторные замеры). При N < 8192 берётся начало окна, чтобы разные N
    видели одни и те же токены."""
    c = data.classifier(model_name)
    for rep in range(reps):
        e_full, t_full = data.window(
            8192, 0 if same_window else rep, keep_invalids, model_name, n_tokens=n_cache
        )
        for n in Ns:
            e, t = e_full[:n].contiguous(), t_full[:n].contiguous()
            for m in methods_list:
                for kind in kinds:
                    if m in FWBW_ONLY and kind != "fwbw":
                        continue
                    key = {
                        "method": m,
                        "kind": kind,
                        "N": n,
                        "rep": rep,
                        "model": model_name,
                    }
                    if j.has(**key):
                        continue
                    r = measure(m, e, c, t, kind, n_iter=n_iter, time_budget_s=budget)
                    r.update(key, frac_ignored=float((t == -100).float().mean()))
                    j.add(r)
                    log(
                        f"   {model_name.split('/')[-1]} rep{rep} N={n} {m:24s} {kind:4s} "
                        f"{r.get('runtime_ms') or float('nan'):9.1f} ms {r.get('mem_mib') or float('nan'):9.1f} MiB"
                        + (" OOM" if r.get("oom") else "")
                    )


def run():
    setup_cce()
    seed_all(0)
    j = Journal(
        exp_dir("table1") / "measurements.jsonl", ("method", "kind", "N", "rep")
    )
    run_grid(j, TABLE1_ROWS, [N], REPS, KINDS)
