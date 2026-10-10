"""Метаданные скомпилированных ядер CCE: регистры, спиллы, shared memory, mma в PTX."""

import re


def _compiled(kernel) -> dict:
    """Все CompiledKernel под обёртками Heuristics (Triton 3.1: .cache, 3.2+: .device_caches)."""
    obj = kernel
    while not (hasattr(obj, "device_caches") or hasattr(obj, "cache")) and hasattr(
        obj, "fn"
    ):
        obj = obj.fn
    caches = getattr(obj, "device_caches", None) or getattr(obj, "cache", {})
    out = {}
    for v in caches.values():
        dicts = [
            x for x in (v if isinstance(v, tuple) else (v,)) if isinstance(x, dict)
        ]
        for dct in dicts[:1]:
            for key, ck in dct.items():
                out[str(key)] = ck
    return out


def stats(ck) -> dict:
    ptx = (getattr(ck, "asm", {}) or {}).get("ptx", "")
    md = getattr(ck, "metadata", None)
    return {
        "n_regs": getattr(ck, "n_regs", None),
        "n_spills": getattr(ck, "n_spills", None),
        "shared": getattr(md, "shared", None),
        "num_warps": getattr(md, "num_warps", None),
        "mma": len(re.findall(r"\bmma\.", ptx)),
        "fma_f32": len(re.findall(r"\bfma\.rn\.f32", ptx)),
    }


class KernelWatcher:
    """Отдаёт статистику ядер, скомпилированных после предыдущего вызова new()."""

    def __init__(self):
        from cut_cross_entropy import cce_backward, cce_lse_forward

        self.kernels = {
            "fw": cce_lse_forward._cce_lse_forward_kernel,
            "bw": cce_backward._cce_backward_kernel,
        }
        self.seen = {k: set(_compiled(v)) for k, v in self.kernels.items()}

    def new(self) -> dict:
        out = {}
        for name, k in self.kernels.items():
            cur = _compiled(k)
            out[name] = [stats(cur[key]) for key in cur if key not in self.seen[name]]
            self.seen[name] |= set(cur)
        return out
