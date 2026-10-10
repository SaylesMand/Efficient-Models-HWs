"""2.1 Дообучение (рисунок 4, раздел 5.3): CCE против torch.compile на Alpaca.

Qwen2.5-0.5B-Instruct, полное дообучение, 3 seed x 2 метода, 500 шагов, батч 8 x до 512 токенов,
формат примеров (системный промпт и шаблон Alpaca) как в training/train.py авторов,
gradient checkpointing, fp16 autocast + GradScaler. Оптимизатор и расписание как в training/train.py
авторов (HF Trainer): AdamW с fp32-состояниями, weight_decay 0, прогрев 5% шагов и линейный спад lr до 0,
клиппинг 1.0; пиковый lr выбирает run_check (у авторов 2e-5 при батче 64).
Seed задаёт порядок примеров, у обоих методов при одном seed батчи одинаковые.
Отложенная выборка одна для всех прогонов.

Первый прогон с 8-bit AdamW и постоянным lr 2e-5 без прогрева дал рост лосса (1.37 -> 1.49) у обоих методов.
С AdamW и расписанием авторов при lr 2e-5 лосс на отложенной выборке за 150 шагов тоже растёт
(1.369 -> 1.415): у авторов батч 64, у нас 8. Поэтому lr выбирается короткой
проверкой (run_check) из 2e-5, 7e-6 (масштабирование по корню) и 2.5e-6 (линейное масштабирование 64 -> 8).

Журналы: steps.jsonl (каждый шаг), eval.jsonl (лосс на отложенной выборке), runs.jsonl (итог прогона).
Чекпоинт каждые CKPT_EVERY шагов в results/ckpt; после обрыва прогон продолжается с последнего
чекпоинта, строки журналов после него отбрасываются.
"""

import json
import os
import time

import torch

from . import data
from .common import RESULTS_ROOT, Journal, MiB, exp_dir, load_json, log, save_json
from .finetune import (
    alpaca_split,
    build,
    eval_loss,
    linear_warmup_decay,
    n_tokens,
    train_order,
    train_step,
)
from .t4_patch import setup_cce

METHODS = ["cce", "torch_compile"]
SEEDS = [0, 1, 2]
STEPS = 500
BATCH = 8
MAX_LEN = 512
LR = 2e-5  # у авторов при батче 64
# как у авторов; масштабирование 64 -> 8 по корню; линейное
CHECK_LRS = [2e-5, 7e-6, 2.5e-6]
CHECK_STEPS = 150
LR_JSON = RESULTS_ROOT / "finetune_lr.json"
WARMUP = int(0.05 * STEPS)
OPTIM = "adamw"
N_EVAL = 256
EVAL_EVERY = 100
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "100"))
CKPT_DIR = RESULTS_ROOT / "ckpt"


def _truncate(path, method, seed, after_step):
    """Удаляет из журнала строки прогона (method, seed) с шагом > after_step."""
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not (r["method"] == method and r["seed"] == seed and r["step"] > after_step):
            keep.append(line)
    path.write_text("".join(x + "\n" for x in keep), encoding="utf-8")


def _save_ckpt(path, model, opt, scaler, sched, step):
    # чекпоинт около 6 ГБ (веса и состояния AdamW в fp32): старый удаляется заранее, чтобы на диске Kaggle
    # (20 ГБ вместе с окружением) не лежали два сразу
    path.unlink(missing_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "sched": sched.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
            "step": step,
        },
        tmp,
    )
    os.replace(tmp, path)


def run_one(method, seed, tok, eval_exs, pool, out, stop_at, lr):
    ckpt = CKPT_DIR / f"{out.name}_{method}_s{seed}.pt"
    model, opt, scaler = build(lr, grad_ckpt=True, seed=seed, optim=OPTIM)
    sched = linear_warmup_decay(opt, WARMUP, STEPS)
    start = 0
    if ckpt.exists():
        # чекпоинт создан этим же кодом, ему можно доверять
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng_cpu"])
        torch.cuda.set_rng_state(st["rng_cuda"])
        start = st["step"]
        del st
        log(f"   {method} seed={seed}: продолжение с шага {start}")
    for f in ("steps.jsonl", "eval.jsonl"):
        _truncate(out / f, method, seed, start)
    steps_j = Journal(out / "steps.jsonl", ("method", "seed", "step"))
    eval_j = Journal(out / "eval.jsonl", ("method", "seed", "step"))

    def evaluate(step):
        if not eval_j.has(method=method, seed=seed, step=step):
            eval_j.add(
                {
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "eval_loss": eval_loss(model, tok, eval_exs),  # noqa: F821  (ложное срабатывание из-за del model ниже)
                }
            )

    exs = train_order(pool, seed, STEPS * BATCH)
    evaluate(start)
    for step in range(start + 1, stop_at + 1):
        enc = [
            data.encode_example(tok, ex, MAX_LEN, authors_format=True)
            for ex in exs[(step - 1) * BATCH : step * BATCH]
        ]
        batch = data.collate(tok, enc, "cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        cur_lr = sched.get_last_lr()[0]
        r = train_step(model, opt, scaler, batch, method)
        sched.step()
        torch.cuda.synchronize()
        steps_j.add(
            dict(
                method=method,
                seed=seed,
                step=step,
                step_s=time.time() - t0,
                tokens=n_tokens(batch),
                peak_mib=torch.cuda.max_memory_allocated() / MiB,
                lr=cur_lr,
                loss_tokens=int((batch["labels"][:, 1:] != -100).sum()),
                **r,
            )
        )
        if step % EVAL_EVERY == 0 or step == stop_at:
            evaluate(step)
        if step % CKPT_EVERY == 0 and step < stop_at:
            _save_ckpt(ckpt, model, opt, scaler, sched, step)
        if step % 50 == 0:
            log(f"   {method} seed={seed} step={step} loss={r['loss']:.4f}")
    del model, opt, scaler, sched
    torch.cuda.empty_cache()
    ckpt.unlink(missing_ok=True)
    return {"resumed_from": start}


def run(name="finetune", methods=METHODS, seeds=SEEDS, stop_at=STEPS, lr=None):
    """stop_at < STEPS: обучение по тому же расписанию на STEPS шагов, остановленное раньше (проверка).
    lr=None: выбранный проверкой (results/finetune_lr.json), иначе LR."""
    if lr is None:
        check = load_json(LR_JSON)
        chosen = check.get("chosen_lr") if check else None
        if check and chosen is None:
            raise RuntimeError(
                "проверка не нашла lr, при котором лосс снижается; полный прогон не запускается"
            )
        lr = chosen if chosen is not None else LR
        log(f"   lr = {lr:g}" + (" (выбран проверкой)" if chosen is not None else ""))
    cfg = setup_cce()
    out = exp_dir(name)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        out / "config.json",
        {
            "methods": methods,
            "seeds": seeds,
            "steps": STEPS,
            "stop_at": stop_at,
            "batch": BATCH,
            "max_len": MAX_LEN,
            "lr": lr,
            "warmup": WARMUP,
            "schedule": "linear warmup + linear decay",
            "n_eval": N_EVAL,
            "eval_every": EVAL_EVERY,
            "grad_ckpt": True,
            "optimizer": "torch AdamW, weight_decay=0",
            "grad_clip": 1.0,
            "amp": "fp16 autocast + GradScaler",
            "cce_block": cfg,
        },
    )
    runs = Journal(out / "runs.jsonl", ("method", "seed"))
    tok = data.load_tokenizer()
    eval_exs, pool = alpaca_split(N_EVAL)
    for seed in seeds:
        for method in methods:
            if runs.has(method=method, seed=seed):
                continue
            log(f"   {method} seed={seed}")
            t0 = time.time()
            res = run_one(method, seed, tok, eval_exs, pool, out, stop_at, lr)
            runs.add(dict(method=method, seed=seed, wall_s=time.time() - t0, **res))


def run_check():
    """Проверка перед полным прогоном: torch.compile, seed 0, первые CHECK_STEPS шагов расписания при каждом lr из
    CHECK_LRS. Правило выбора (зафиксировано до прогона): из lr, при которых лосс на отложенной выборке на шаге
    CHECK_STEPS ниже, чем на шаге 0, берётся lr с наименьшим лоссом на шаге CHECK_STEPS."""
    rows = []
    for lr in CHECK_LRS:
        name = f"finetune_check_lr{lr:g}"
        run(name, ["torch_compile"], [0], stop_at=CHECK_STEPS, lr=lr)
        ev = {
            r["step"]: r["eval_loss"]
            for r in map(
                json.loads,
                (RESULTS_ROOT / name / "eval.jsonl").read_text().splitlines(),
            )
        }
        rows.append(
            {
                "lr": lr,
                "eval_0": ev[0],
                "eval_end": ev[CHECK_STEPS],
                "ok": ev[CHECK_STEPS] < ev[0],
            }
        )
        log(f"   lr={lr:g}: отложенная выборка {ev[0]:.4f} -> {ev[CHECK_STEPS]:.4f}")
    ok = [r for r in rows if r["ok"]]
    chosen = min(ok, key=lambda r: r["eval_end"])["lr"] if ok else None
    save_json(
        LR_JSON,
        {
            "rule": "min eval_end среди eval_end < eval_0",
            "check_steps": CHECK_STEPS,
            "rows": rows,
            "chosen_lr": chosen,
        },
    )
    log(
        f"   выбран lr = {chosen:g}"
        if chosen is not None
        else "   ни один lr не снижает лосс: полный прогон не запускать"
    )
