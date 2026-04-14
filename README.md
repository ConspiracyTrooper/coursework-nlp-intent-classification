# Чат технической поддержки — Mistral QLoRA Demo

Веб-приложение на FastAPI для демонстрации дообученной модели Mistral-7B (QLoRA).
Запускается в Docker, работает на Windows через Docker Desktop.

---

## Структура проекта (минимально необходимая)

```
Курсовая/
├── docker-compose.yml
├── .dockerignore
├── models/
│   ├── tfidf.joblib
│   ├── clf_Logistic_Regression.joblib
│   ├── le.joblib
│   └── mistral-qlora-support/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── webapp/
    ├── Dockerfile
    ├── main.py
    ├── classifier.py
    ├── translator.py
    ├── inference.py
    ├── requirements.txt
    └── static/
        └── index.html
```

---

## Требования

| Компонент | Требование |
|-----------|------------|
| ОС | Windows 10/11 (64-bit) |
| Docker | Docker Desktop ≥ 4.x (WSL2-бэкенд) |
| GPU | NVIDIA GPU с ≥ 6 ГБ VRAM (рекомендуется) |
| Драйвер | NVIDIA ≥ 525.x + NVIDIA Container Toolkit |
| ОЗУ | ≥ 16 ГБ (при работе без GPU — ≥ 32 ГБ) |
| Диск | ≥ 20 ГБ свободного места (базовая модель ~14 ГБ) |
| Интернет | Нужен для первого запуска (скачивание модели) и Google Translate |

> **Без GPU**: закомментируйте блок `deploy:` в `docker-compose.yml`.
> Инференс на CPU работает, но очень медленно (~10–30 минут на ответ).

---

## Пошаговая инструкция запуска

### 1. Установите Docker Desktop

Скачайте и установите [Docker Desktop для Windows](https://www.docker.com/products/docker-desktop/).
При установке выберите **WSL2-бэкенд** (рекомендуется).

После установки убедитесь, что Docker запущен (иконка в трее):
```
docker --version
```

### 2. Установите NVIDIA Container Toolkit (только при наличии GPU)

Откройте PowerShell от имени администратора и выполните:
```powershell
# Скачайте и запустите установщик NVIDIA Container Toolkit
# Следуйте официальной инструкции:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```

Проверьте, что GPU доступен в Docker:
```powershell
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 3. Получите токен HuggingFace

Модель `mistralai/Mistral-7B-Instruct-v0.2` — закрытая. Нужен токен доступа:

1. Зайдите на [huggingface.co](https://huggingface.co) и создайте аккаунт
2. Откройте страницу модели: **mistralai/Mistral-7B-Instruct-v0.2**
3. Нажмите **Agree and access repository** (принятие лицензии)
4. Перейдите в **Settings → Access Tokens → New token** (тип: Read)
5. Скопируйте токен вида `hf_xxxxxxxxxxxxxxxx`

### 4. Создайте файл `.env`

В папке проекта (рядом с `docker-compose.yml`) создайте файл `.env`:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Замените `hf_xxxxxxxx...` на ваш реальный токен.

### 5. Соберите и запустите

Откройте PowerShell в папке проекта (`Курсовая/`):
```powershell
docker compose up --build
```

**Первый запуск:** Docker соберёт образ (~10 мин), затем при первом запросе к чату
скачает базовую модель Mistral-7B с HuggingFace (~14 ГБ). Это занимает
от 15 минут до 1 часа в зависимости от скорости интернета.

Последующие запуски быстрее — модель кешируется в Docker-томе `hf_cache`.

### 6. Откройте приложение

После строки `Application startup complete` в логах:
```
http://localhost:8000
```

---

## Работа без GPU (CPU-режим)

Откройте `docker-compose.yml` и закомментируйте блок GPU:
```yaml
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

Также установите `USE_4BIT=0` в секции `environment`:
```yaml
    environment:
      - USE_4BIT=0
```

> Без GPU каждый ответ занимает 10–30 минут. Для демонстрации рекомендуется GPU.

---

## Управление контейнером

```powershell
# Запустить в фоне
docker compose up -d

# Посмотреть логи
docker compose logs -f

# Остановить
docker compose down

# Пересобрать образ после изменения кода
docker compose up --build

# Удалить кеш моделей (освободит ~14+ ГБ)
docker compose down -v
```

---

## Устранение неполадок

| Проблема | Решение |
|----------|---------|
| `Error: no space left` | Освободите место на диске (нужно ≥ 20 ГБ) |
| `CUDA out of memory` | Убедитесь, что GPU не занят другими задачами; или используйте CPU-режим |
| `401 Unauthorized` (HuggingFace) | Проверьте токен в `.env` и принятие лицензии на HuggingFace |
| `Could not find adapter` | Убедитесь, что `models/mistral-qlora-support/` содержит `adapter_model.safetensors` |
| Переводчик не работает | Проверьте доступ к интернету из контейнера (Google Translate API) |
| `nvidia-smi` недоступен | Установите NVIDIA Container Toolkit и перезапустите Docker Desktop |

---

## Как работает приложение

```
Пользователь (RU)
      ↓
  Google Translate (RU→EN)
      ↓
  TF-IDF + Logistic Regression — классификация интента
      ↓
  Mistral-7B-Instruct-v0.2
    ├── Базовая модель (без LoRA)
    └── Дообученная QLoRA (LoRA-адаптер из models/mistral-qlora-support/)
      ↓
  Google Translate (EN→RU)
      ↓
Пользователь (RU)
```
