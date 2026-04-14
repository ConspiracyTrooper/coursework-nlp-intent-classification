# Инференс Mistral-7B: базовая модель и дообученная QLoRA-версия
# При первом вызове загружает модель — занимает несколько минут и ~5 ГБ VRAM (4-bit)
# или ~14 ГБ ОЗУ (fp16 без GPU)

from __future__ import annotations

import logging
import os
import threading
from typing import Literal

import torch
from peft         import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)

BASE_MODEL  = os.getenv("BASE_MODEL",  "mistralai/Mistral-7B-Instruct-v0.2")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "/app/models/mistral-qlora-support")
HF_TOKEN    = os.getenv("HF_TOKEN",    None) or None   # пустая строка → None

# 4-bit квантизация доступна только при наличии CUDA
_CUDA_OK = torch.cuda.is_available()
USE_4BIT  = os.getenv("USE_4BIT", "1") == "1" and _CUDA_OK

# Синглтоны — инициализируются один раз при первом запросе
_init_lock = threading.Lock()
_model:     PeftModel | None  = None
_tokenizer: AutoTokenizer | None = None
_ready      = False


def _load() -> None:
    """Загружает токенайзер и модель в память (вызывается строго один раз)."""
    global _model, _tokenizer, _ready

    device_info = f"CUDA ({torch.cuda.get_device_name(0)})" if _CUDA_OK else "CPU"
    log.info(f"[inference] Загрузка модели на {device_info}")
    log.info(f"[inference] Режим: {'4-bit quantization' if USE_4BIT else 'fp32/fp16'}")

    # Токенайзер
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)

    # Конфигурация квантизации (только при наличии CUDA)
    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.float16,
        )

    # Базовая модель
    log.info(f"[inference] Загрузка базовой модели: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = bnb_config,
        device_map          = "auto",
        token               = HF_TOKEN,
        torch_dtype         = torch.float16 if (not USE_4BIT and _CUDA_OK) else None,
    )

    # LoRA-адаптеры поверх базовой модели
    log.info(f"[inference] Загрузка LoRA-адаптеров: {ADAPTER_DIR}")
    _model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    _model.eval()

    _ready = True
    log.info("[inference] Модель готова к инференсу")


def _ensure_loaded() -> None:
    """Потокобезопасная ленивая инициализация."""
    global _ready
    if not _ready:
        with _init_lock:
            if not _ready:
                _load()


def _generate_tokens(prompt: str, use_adapter: bool, max_new_tokens: int) -> str:
    """Внутренняя генерация; вызывается уже при захваченном _init_lock не нужен."""
    messages  = [{"role": "user", "content": prompt}]
    input_ids = _tokenizer.apply_chat_template(
        messages,
        tokenize              = True,
        add_generation_prompt = True,
        return_tensors        = "pt",
    ).to(next(_model.parameters()).device)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample":      True,
        "temperature":    0.7,
        "top_p":          0.9,
        "pad_token_id":   _tokenizer.eos_token_id,
    }

    with torch.no_grad():
        if use_adapter:
            output_ids = _model.generate(input_ids, **gen_kwargs)
        else:
            # disable_adapter временно отключает LoRA → чистая базовая модель
            with _model.disable_adapter():
                output_ids = _model.generate(input_ids, **gen_kwargs)

    # Декодируем только новые токены (отрезаем промпт)
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Семафор для последовательного инференса (GPU не может параллелить два forward-pass)
_gen_lock = threading.Semaphore(1)


def generate(
    prompt:         str,
    model_type:     Literal["base", "finetuned"] = "finetuned",
    max_new_tokens: int = 256,
) -> str:
    """Публичный API: генерирует ответ на англоязычный промпт.

    Первый вызов блокируется до загрузки модели (~5–15 мин при скачивании).
    """
    _ensure_loaded()

    with _gen_lock:
        return _generate_tokens(
            prompt         = prompt,
            use_adapter    = (model_type == "finetuned"),
            max_new_tokens = max_new_tokens,
        )


def is_ready() -> bool:
    """Возвращает True, если модель уже загружена в память."""
    return _ready
