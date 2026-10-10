"""Шаг полного дообучения Qwen2.5-0.5B на Alpaca: fp16 autocast, GradScaler, AdamW (torch или 8-bit).

Лосс считается без патча transformers:
  hidden = decoder(input_ids, attention_mask)            # (B, T, D)
  loss   = linear_cross_entropy(hidden, lm_head.weight, labels, shift=1, impl=...)
Под autocast библиотека сама приводит E и C к fp16 (порог фильтра тогда 2^-15).
У Qwen2.5-0.5B lm_head.weight это та же матрица, что embed_tokens, поэтому ∇C попадает
во входные эмбеддинги.
impl: любой impl библиотеки ("cce", "torch_compile", ...) или "hf" (стандартный лосс transformers).
"""

import random

import torch
import torch.nn.functional as F

from . import data

IGNORE_INDEX = -100


def compute_loss(model, batch, impl: str):
    if impl == "hf":
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
    from cut_cross_entropy import linear_cross_entropy

    hidden = data.backbone(model)(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )[0]
    return linear_cross_entropy(
        hidden, model.lm_head.weight, batch["labels"], shift=1, impl=impl
    )


def alpaca_split(n_eval: int) -> tuple[list[dict], list[dict]]:
    """Фиксированная отложенная выборка (первые n_eval примеров в порядке seed=0) и остальное для обучения."""
    exs = data.load_alpaca(None, seed=0)
    return exs[:n_eval], exs[n_eval:]


def train_order(pool: list[dict], seed: int, n: int) -> list[dict]:
    """Порядок обучающих примеров для seed; у разных методов при одном seed батчи совпадают."""
    idx = list(range(len(pool)))
    random.Random(seed).shuffle(idx)
    return [pool[i] for i in idx[:n]]


def build(lr: float, grad_ckpt: bool = True, seed: int = 0, optim: str = "adamw8bit"):
    """optim: "adamw" (torch, fp32-состояния, как HF Trainer у авторов) или "adamw8bit" (bitsandbytes)."""
    torch.manual_seed(seed)
    random.seed(seed)
    model = data.load_model(torch.float32, "cuda")
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    if optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    else:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    return model, opt, scaler


def linear_warmup_decay(opt, warmup: int, total: int):
    """Как get_linear_schedule_with_warmup в transformers: рост lr до пика за warmup шагов, затем линейно до 0."""
    return torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            s / max(1, warmup)
            if s < warmup
            else max(0.0, (total - s) / max(1, total - warmup))
        ),
    )


def train_step(
    model, opt, scaler, batch, impl: str, max_grad_norm: float = 1.0
) -> dict:
    with torch.autocast("cuda", dtype=torch.float16):
        loss = compute_loss(model, batch, impl)
    scale_before = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach()),
        "grad_norm": float(gnorm),
        "skipped": scaler.get_scale() < scale_before,
        "scale": scaler.get_scale(),
    }


@torch.no_grad()
def eval_loss(
    model,
    tok,
    examples: list[dict],
    batch_size: int = 8,
    chunk: int = 2048,
    max_len: int = 512,
) -> float:
    """Средний по токенам лосс на отложенной выборке. Один и тот же расчёт для всех методов:
    бэкбон под fp16 autocast, логиты и cross-entropy в fp32 по кускам токенов."""
    model.eval()
    w = model.lm_head.weight.float()
    total, count = 0.0, 0
    for i in range(0, len(examples), batch_size):
        enc = [
            data.encode_example(tok, ex, max_len, authors_format=True)
            for ex in examples[i : i + batch_size]
        ]
        batch = data.collate(tok, enc, "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = data.backbone(model)(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )[0]
        h = hidden[:, :-1].reshape(-1, hidden.size(-1)).float()
        t = batch["labels"][:, 1:].reshape(-1)
        keep = t != IGNORE_INDEX
        h, t = h[keep], t[keep]
        for j in range(0, t.numel(), chunk):
            total += float(
                F.cross_entropy(
                    h[j : j + chunk] @ w.T, t[j : j + chunk], reduction="sum"
                )
            )
        count += t.numel()
    model.train()
    return total / max(count, 1)


def n_tokens(batch) -> int:
    return int(batch["attention_mask"].sum())
