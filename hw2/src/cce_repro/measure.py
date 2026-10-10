"""Замеры пиковой памяти и времени по методике авторов (benchmark/memory.py).

Память: разность allocated_bytes.all.peak до и после операции, в MiB. Входы E и C созданы
заранее и в разность не входят; градиенты .grad, созданные в backward, входят (как у авторов,
поэтому нижняя граница для fw+bw равна размеру ∇E + ∇C).
Время: torch.cuda.Event, среднее по n_iter итераций после одного прогревочного вызова.
Число итераций урезается так, чтобы замер укладывался в time_budget_s (на T4 один вызов CCE
может идти десятки секунд); фактическое число пишется в результат как n_iter.
kind: "fw" (только лосс), "bw" (только обратный проход), "fwbw" (лосс + градиент).
"""

import gc
import time

import torch

from . import methods
from .common import MiB


def _cleanup():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def _clear(e, c):
    e.grad = None
    c.grad = None


def _peak():
    return torch.cuda.memory_stats()["allocated_bytes.all.peak"]


def _run(fn, e, c, t, kind, name):
    loss = fn(e, c, t)
    if kind == "fw":
        return loss
    loss.backward()
    methods.after_backward(name)
    return loss


def _fit_iters(n_iter: int, first_call_s: float, time_budget_s: float) -> int:
    return max(1, min(n_iter, int(time_budget_s / max(first_call_s, 1e-6))))


def measure(
    name: str,
    e,
    c,
    t,
    kind: str = "fwbw",
    n_iter: int = 20,
    time_budget_s: float = 30.0,
) -> dict:
    """Один замер памяти (первый вызов, он же прогрев) и среднее время по n_iter вызовам."""
    fn = methods.get(name)
    e.requires_grad_(True)
    c.requires_grad_(True)
    out = {"method": name, "kind": kind, "N": e.size(0), "n_iter": n_iter}
    try:
        _clear(e, c)
        _cleanup()
        torch.cuda.reset_peak_memory_stats()
        p0 = _peak()
        t_first = time.time()
        loss = fn(e, c, t)
        torch.cuda.synchronize()
        if kind == "fw":
            p1 = _peak()
        else:
            if kind == "bw":
                torch.cuda.reset_peak_memory_stats()
                p0 = _peak()
            loss.backward()
            methods.after_backward(name)
            torch.cuda.synchronize()
            p1 = _peak()
        out["first_call_s"] = time.time() - t_first
        n_iter = _fit_iters(n_iter, out["first_call_s"], time_budget_s)
        out["n_iter"] = n_iter
        out["mem_mib"] = (p1 - p0) / MiB
        out["loss"] = float(loss.detach())
        del loss

        events = []
        for _ in range(n_iter):
            _clear(e, c)
            s, f = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if kind == "bw":
                loss = fn(e, c, t)
                s.record()
                loss.backward()
                methods.after_backward(name)
            else:
                s.record()
                _run(fn, e, c, t, kind, name)
            f.record()
            events.append((s, f))
        torch.cuda.synchronize()
        times = [s.elapsed_time(f) for s, f in events]
        out["runtime_ms"] = sum(times) / len(times)
        out["runtime_ms_min"] = min(times)
        out["oom"] = False
    except torch.cuda.OutOfMemoryError as ex:
        out.update(oom=True, error=str(ex).split("\n")[0])
    finally:
        # OOM посреди backward cce_no_sort оставил бы сортировку словаря выключенной
        methods.after_backward("cce_no_sort")
        _clear(e, c)
        _cleanup()
    return out
