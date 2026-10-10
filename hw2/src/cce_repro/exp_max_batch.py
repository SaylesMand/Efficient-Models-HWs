"""2.2 Максимальный батч (рисунок 1, таблица A4, приложение D) на одном T4.

Полный шаг обучения Qwen2.5-0.5B (fp16 autocast, 8-bit AdamW, gradient checkpointing),
последовательности по 512 токенов, все токены в лоссе (как в формуле приложения D). Для каждого
лосса (hf, torch_compile, cce) максимальный batch size ищется удвоением и бинарным поиском, затем
найденный максимум перепроверяется (при неудаче батч уменьшается) и замеряется пропускная способность:
на максимуме или, если повторные шаги на нём не помещаются, на батче до 20% меньше. Каждый лосс
считается в отдельном процессе, чтобы OOM-пробы одного не фрагментировали память другого.

Расчёт по приложению D (`appendix_d`) делается здесь же и пишется в formula.json.
"""

import gc
import json
import subprocess
import sys
import time

import torch

from .common import Journal, MiB, exp_dir, log, save_json
from .finetune import build, train_step
from .t4_patch import setup_cce

IMPLS = ["hf", "torch_compile", "cce"]
SEQ = 512
MAX_BS = 1024
N_TIMED = 3


def appendix_d(n_params, layers, d, v, tokens=65536, gpus=16, gb_per_gpu=75):
    """Приложение D статьи: память на 16 x 80 ГБ (75 ГБ доступно), bf16, checkpointing.
    Логиты: tokens x |V| x 4 Б; активации: слои x D x tokens x 2 Б; веса+оптимизатор+градиенты: параметры x 4 x 2 Б.
    Макс. батч до: (доступно - веса) / (логиты + активации на токен); после (CCE): / (активации на токен).
    """
    logits = tokens * v * 4 / MiB
    act = layers * d * tokens * 2 / MiB
    w = n_params * 4 * 2 / MiB
    avail = gb_per_gpu * gpus * 1024 - w
    before = avail / ((logits + act) / tokens)
    after = avail / (act / tokens)
    return {
        "logits_mib": logits,
        "activations_mib": act,
        "weights_opt_grad_mib": w,
        "max_tokens_before": before,
        "max_tokens_after": after,
        "increase": after / before,
    }


def _batch(bs, v):
    ids = torch.randint(0, v, (bs, SEQ), device="cuda")
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones_like(ids),
    }


def _probe(model, opt, scaler, bs, v, impl):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        train_step(model, opt, scaler, _batch(bs, v), impl)
        torch.cuda.synchronize()
        return True, torch.cuda.max_memory_allocated() / MiB
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        return False, None


def search(impl: str) -> dict:
    setup_cce()
    model, opt, scaler = build(lr=2e-5, grad_ckpt=True)
    v = model.lm_head.weight.size(0)
    # создаёт состояния оптимизатора, чтобы static их учитывал
    _probe(model, opt, scaler, 1, v, impl)
    static = torch.cuda.memory_allocated() / MiB
    lo, hi, peaks = 1, None, {}
    bs = 2
    while hi is None and bs <= MAX_BS:
        ok, pk = _probe(model, opt, scaler, bs, v, impl)
        if ok:
            lo, peaks[bs] = bs, pk
            bs *= 2
        else:
            hi = bs
    while hi is not None and hi - lo > 1:
        mid = (lo + hi) // 2
        ok, pk = _probe(model, opt, scaler, mid, v, impl)
        if ok:
            lo, peaks[mid] = mid, pk
        else:
            hi = mid
    # после OOM-проб память фрагментирована, и найденный максимум может уже не поместиться
    found = lo
    recheck, pk = _probe(model, opt, scaler, lo, v, impl)
    while not recheck and lo > 1:
        lo -= 1
        recheck, pk = _probe(model, opt, scaler, lo, v, impl)
    # повторные шаги на максимуме могут упереться в OOM, тогда скорость меряется на меньшем батче
    step_s, timed_bs = None, lo
    while recheck and step_s is None and timed_bs >= max(1, int(lo * 0.8)):
        b = _batch(timed_bs, v)
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(N_TIMED):
                train_step(model, opt, scaler, b, impl)
            torch.cuda.synchronize()
            step_s = (time.time() - t0) / N_TIMED
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            b = None
            gc.collect()
            torch.cuda.empty_cache()
            timed_bs -= max(1, lo // 20)
    cfg = model.config
    return {
        "impl": impl,
        "max_batch": lo,
        "max_tokens": lo * SEQ,
        "max_batch_search": found,
        "first_oom_batch": hi,
        "recheck_ok": recheck,
        "peak_at_max_mib": pk,
        "static_mib": static,
        "peaks_by_bs": peaks,
        "timed_batch": timed_bs if step_s else None,
        "step_s_at_timed": step_s,
        "tokens_per_s": timed_bs * SEQ / step_s if step_s else None,
        "n_params": sum(p.numel() for p in model.parameters()),
        "layers": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size,
        "vocab": v,
        "total_mib": torch.cuda.get_device_properties(0).total_memory / MiB,
    }


def run():
    out = exp_dir("max_batch")
    j = Journal(out / "measurements.jsonl", ("impl",))
    for impl in IMPLS:
        if j.has(impl=impl):
            continue
        log(f"   max batch: {impl}")
        p = subprocess.run(
            [sys.executable, "-m", "cce_repro.exp_max_batch", impl],
            capture_output=True,
            text=True,
            check=False,
        )
        line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
        if line is None:
            log(p.stderr[-3000:])
            raise RuntimeError(f"max batch {impl}: процесс завершился без результата")
        r = json.loads(line[len("RESULT ") :])
        log(
            f"   {impl}: max_batch={r['max_batch']} (поиск {r['max_batch_search']}), peak={r['peak_at_max_mib']} MiB"
        )
        j.add(r)
    m = json.loads(
        (out / "measurements.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    save_json(
        out / "formula.json",
        {
            "Qwen2.5-0.5B-Instruct": appendix_d(
                m["n_params"], m["layers"], m["hidden"], m["vocab"]
            ),
            # проверка формулы на строке таблицы A4: веса+оптимизатор+градиенты 19 946 MB даны в статье
            "Gemma 2 (2B), проверка": appendix_d(19946 * MiB / 8, 26, 2304, 256000),
        },
    )


if __name__ == "__main__":
    res = search(sys.argv[1])
    print("RESULT " + json.dumps(res), flush=True)
