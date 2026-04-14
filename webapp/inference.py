# Инференс Mistral-7B: базовая (HuggingFace) и слитая дообученная (merged)
# При первом вызове загружает модель — занимает несколько минут и ~5 ГБ VRAM (4-bit)
# или ~14 ГБ ОЗУ (fp16 без GPU)

from __future__ import annotations

import logging
import os
import threading
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)

BASE_MODEL   = os.getenv("BASE_MODEL",   "mistralai/Mistral-7B-Instruct-v0.2")
MERGED_MODEL = os.getenv("MERGED_MODEL", "/app/models/mistral-merged")
HF_TOKEN     = os.getenv("HF_TOKEN", None) or None   # пустая строка → None

# 4-bit квантизация доступна только при наличии CUDA
_CUDA_OK = torch.cuda.is_available()
USE_4BIT  = os.getenv("USE_4BIT", "1") == "1" and _CUDA_OK

# Синглтоны: дообученная модель и базовая грузятся независимо и лениво
_lock        = threading.Lock()
_ft_model    = None   # merged fine-tuned
_ft_tok      = None
_base_model  = None   # оригинальный Mistral без дообучения
_base_tok    = None
_ft_ready    = False
_base_ready  = False


def _bnb_config() -> BitsAndBytesConfig | None:
    if not USE_4BIT:
        return None
    return BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
    )


def _load_model(path: str, label: str):
    """Загружает токенайзер и модель из пути (HF hub или локальный каталог)."""
    device_info = f"CUDA ({torch.cuda.get_device_name(0)})" if _CUDA_OK else "CPU"
    log.info(f"[inference] Загрузка {label} на {device_info}: {path}")

    tok = AutoTokenizer.from_pretrained(path, token=HF_TOKEN)
    mdl = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config = _bnb_config(),
        device_map          = "auto",
        token               = HF_TOKEN,
        torch_dtype         = torch.float16 if (not USE_4BIT and _CUDA_OK) else None,
    )
    mdl.eval()
    log.info(f"[inference] {label} готова")
    return mdl, tok


def _ensure_ft() -> None:
    global _ft_model, _ft_tok, _ft_ready
    if not _ft_ready:
        with _lock:
            if not _ft_ready:
                _ft_model, _ft_tok = _load_model(MERGED_MODEL, "дообученная (merged)")
                _ft_ready = True


def _ensure_base() -> None:
    global _base_model, _base_tok, _base_ready
    if not _base_ready:
        with _lock:
            if not _base_ready:
                _base_model, _base_tok = _load_model(BASE_MODEL, "базовая (HuggingFace)")
                _base_ready = True


def _run_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device    = next(model.parameters()).device
    messages  = [{"role": "user", "content": prompt}]
    encoded   = tokenizer.apply_chat_template(
        messages,
        tokenize              = True,
        add_generation_prompt = True,
        return_tensors        = "pt",
    )

    # apply_chat_template возвращает тензор или BatchEncoding в зависимости от версии
    input_ids = (encoded["input_ids"] if hasattr(encoded, "__getitem__") and not isinstance(encoded, torch.Tensor)
                 else encoded).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids      = input_ids,
            max_new_tokens = max_new_tokens,
            do_sample      = True,
            temperature    = 0.7,
            top_p          = 0.9,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # Декодируем только новые токены (отрезаем промпт)
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Семафор: GPU не может параллельно обрабатывать два forward-pass
_gen_lock = threading.Semaphore(1)


def generate(
    prompt:         str,
    model_type:     Literal["base", "finetuned"] = "finetuned",
    max_new_tokens: int = 256,
) -> str:
    """Публичный API: генерирует ответ на англоязычный промпт."""
    if model_type == "finetuned":
        _ensure_ft()
        model, tok = _ft_model, _ft_tok
    else:
        _ensure_base()
        model, tok = _base_model, _base_tok

    with _gen_lock:
        return _run_generate(model, tok, prompt, max_new_tokens)


def is_ready() -> bool:
    """Возвращает True, если дообученная модель уже загружена."""
    return _ft_ready
