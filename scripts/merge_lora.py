# Слияние LoRA-адаптеров с базовой моделью Mistral-7B-Instruct-v0.2
# Запускать на Linux-машине с GPU после завершения обучения.
# Результат: models/mistral-merged/ (~14 GB, формат HuggingFace)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft         import PeftModel

BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "models/mistral-qlora-support"
OUTPUT_DIR  = "models/mistral-merged"

print(f"Загрузка базовой модели: {BASE_MODEL}")
print("Загрузка на CPU — fp16 весов 7B модели ~14 GB (не хватит GPU для слияния)")

# Загружаем на CPU: merge_and_unload требует полной точности без разбивки по устройствам
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype        = torch.float16,
    device_map         = "cpu",
    trust_remote_code  = True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print(f"Загрузка LoRA-адаптеров из: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

# Сливаем адаптеры в веса базовой модели и убираем PEFT-обёртку
print("Слияние адаптеров (merge_and_unload)...")
model = model.merge_and_unload()

# Сохраняем в safetensors-формате (безопаснее и быстрее pickle)
print(f"Сохранение слитой модели → {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Готово. Следующий шаг: конвертация в GGUF через llama.cpp")
