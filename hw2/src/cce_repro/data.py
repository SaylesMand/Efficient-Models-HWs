"""Alpaca в чат-шаблоне модели, эмбеддинги E (выход последнего слоя бэкбона) и классификатор C.

Как в benchmark/data/models.py авторов: E из настоящей модели на Alpaca, C = веса выходного
слоя. Отличия: Qwen2.5 вместо Gemma 2, бэкбон в fp32 (где влезает), кэш в results/cache.
"""

import json
import random

import torch

from .common import CACHE_DIR, DATASET_NAME, MODEL_NAME, N_CACHE, log

IGNORE_INDEX = -100


def load_tokenizer(model_name: str = MODEL_NAME):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# формат из training/train.py авторов (дообучение, рисунок 4)
AUTHORS_SYSTEM_PROMPT = "You are a helpful AI assistant."
AUTHORS_PROMPT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
    ),
}


def _prompt_text(ex: dict) -> str:
    return ex["instruction"] + (f"\n\n{ex['input']}" if ex.get("input") else "")


def encode_example(
    tok, ex: dict, max_len: int = 512, authors_format: bool = False
) -> tuple[list[int], list[int]]:
    """Токены диалога и метки; системный промпт и вопрос получают -100.
    authors_format=True: промпт из training/train.py авторов (ноутбук 2), иначе шаблон Qwen по умолчанию (ноутбук 1)."""
    if authors_format:
        tmpl = AUTHORS_PROMPT["prompt_input" if ex.get("input") else "prompt_no_input"]
        msgs = [
            {"role": "system", "content": AUTHORS_SYSTEM_PROMPT},
            {"role": "user", "content": tmpl.format_map(ex)},
        ]
    else:
        msgs = [{"role": "user", "content": _prompt_text(ex)}]
    msgs.append({"role": "assistant", "content": ex["output"]})
    prompt = tok.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tok.apply_chat_template(msgs, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [IGNORE_INDEX] * len(p_ids) + f_ids[len(p_ids) :]
    return f_ids[:max_len], labels[:max_len]


def load_alpaca(n: int | None = None, seed: int = 0, offset: int = 0) -> list[dict]:
    """Примеры Alpaca в фиксированном случайном порядке; offset позволяет брать непересекающиеся части."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[offset:] if n is None else idx[offset : offset + n]
    return [ds[i] for i in idx]


def collate(tok, encoded, device="cpu") -> dict:
    T = max(len(i) for i, _ in encoded)
    B = len(encoded)
    input_ids = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    for b, (ids, lab) in enumerate(encoded):
        input_ids[b, : len(ids)] = torch.tensor(ids)
        labels[b, : len(lab)] = torch.tensor(lab)
        attn[b, : len(ids)] = 1
    return {
        k: v.to(device)
        for k, v in {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
        }.items()
    }


def load_model(dtype=torch.float32, device="cuda", model_name: str = MODEL_NAME):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(
        device
    )


def backbone(model):
    return model.get_decoder() if hasattr(model, "get_decoder") else model.model


def load_classifier(
    dtype=torch.float16, device="cuda", model_name: str = MODEL_NAME
) -> torch.Tensor:
    """C (V, D) из safetensors без загрузки модели.
    У Qwen2.5 веса связаны, C = model.embed_tokens.weight."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    try:
        with open(hf_hub_download(model_name, "model.safetensors.index.json")) as f:
            idx = json.load(f)["weight_map"]
        key = (
            "lm_head.weight" if "lm_head.weight" in idx else "model.embed_tokens.weight"
        )
        path = hf_hub_download(model_name, idx[key])
    except Exception:  # noqa: BLE001  (один файл без индекса)
        path = hf_hub_download(model_name, "model.safetensors")
        key = None
    with safe_open(path, framework="pt") as f:
        if key is None:
            key = (
                "lm_head.weight"
                if "lm_head.weight" in f
                else "model.embed_tokens.weight"
            )
        w = f.get_tensor(key)
    return w.to(device=device, dtype=dtype).contiguous()


def _cache_path(model_name: str, keep_invalids: bool, n_tokens: int):
    tag = model_name.split("/")[-1]
    return CACHE_DIR / f"E_{tag}_{'all' if keep_invalids else 'valid'}_{n_tokens}.pt"


@torch.inference_mode()
def extract_embeddings(
    model_name: str = MODEL_NAME,
    keep_invalids: bool = False,
    n_tokens: int = N_CACHE,
    batch_size: int = 8,
    seed: int = 0,
    backbone_dtype=torch.float32,
) -> dict:
    """n_tokens пар (E_i, target_i) после сдвига на 1.
    keep_invalids=False: только токены ответа (как `gemma2` у авторов);
    True: все не-паддинговые позиции, промпт с меткой -100 (как `gemma2-invalids`, но без паддинга)."""
    path = _cache_path(model_name, keep_invalids, n_tokens)
    if path.exists():
        return torch.load(path, weights_only=True)

    log(
        f"Эмбеддинги: {model_name}, {n_tokens} токенов, keep_invalids={keep_invalids}, бэкбон {backbone_dtype}"
    )
    tok = load_tokenizer(model_name)
    model = load_model(backbone_dtype, "cuda", model_name).eval()
    dec = backbone(model)
    examples = load_alpaca(None, seed)

    es, ts, n, n_seen, n_ign, used, nonfinite = [], [], 0, 0, 0, 0, False
    for i in range(0, len(examples), batch_size):
        enc = [encode_example(tok, ex) for ex in examples[i : i + batch_size]]
        used += len(enc)
        batch = collate(tok, enc, "cuda")
        hidden = dec(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )[0].float()
        h = hidden[:, :-1].reshape(-1, hidden.size(-1))
        lab = batch["labels"][:, 1:].reshape(-1)
        nonpad = batch["attention_mask"][:, 1:].reshape(-1).bool()
        n_seen += int(nonpad.sum())
        n_ign += int((nonpad & (lab == IGNORE_INDEX)).sum())
        keep = nonpad if keep_invalids else (lab != IGNORE_INDEX)
        nonfinite |= bool((~torch.isfinite(h[keep])).any())
        es.append(h[keep].cpu())
        ts.append(lab[keep].cpu())
        n += int(keep.sum())
        if n >= n_tokens:
            break

    out = {
        "e": torch.cat(es)[:n_tokens].contiguous(),
        "t": torch.cat(ts)[:n_tokens].contiguous(),
        "model": model_name,
        "keep_invalids": keep_invalids,
        "frac_ignored_nonpad": n_ign / max(n_seen, 1),
        "n_examples": used,
        "backbone_dtype": str(backbone_dtype),
        "nonfinite": nonfinite,
    }
    torch.save(out, path)
    del model
    torch.cuda.empty_cache()
    return out


def window(
    N: int,
    rep: int = 0,
    keep_invalids: bool = False,
    model_name: str = MODEL_NAME,
    n_tokens: int = N_CACHE,
    dtype=torch.float16,
    device="cuda",
    **kw,
):
    """Окно токенов [rep*N, (rep+1)*N) из кэша, приведённое к dtype."""
    d = extract_embeddings(model_name, keep_invalids, n_tokens, **kw)
    lo, hi = rep * N, (rep + 1) * N
    if hi > d["e"].size(0):
        raise ValueError(f"окно {lo}:{hi} за пределами кэша ({d['e'].size(0)} токенов)")
    e = d["e"][lo:hi].to(device=device, dtype=dtype).contiguous()
    t = d["t"][lo:hi].to(device).contiguous()
    return e, t


_C_CACHE: dict = {}


def classifier(model_name: str = MODEL_NAME, dtype=torch.float16) -> torch.Tensor:
    key = (model_name, dtype)
    if key not in _C_CACHE:
        _C_CACHE.clear()
        _C_CACHE[key] = load_classifier(dtype, "cuda", model_name)
    return _C_CACHE[key]
