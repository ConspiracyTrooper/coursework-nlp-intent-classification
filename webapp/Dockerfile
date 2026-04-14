# Базовый образ: PyTorch 2.6 + CUDA 12.4 (соответствует обучающей среде)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем Python-зависимости
# torch уже в базовом образе — не переустанавливаем
COPY webapp/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Загружаем NLTK-данные при сборке (не при каждом старте контейнера)
RUN python -c "\
import nltk; \
nltk.download('punkt',                        quiet=True); \
nltk.download('punkt_tab',                    quiet=True); \
nltk.download('stopwords',                    quiet=True); \
nltk.download('wordnet',                      quiet=True); \
nltk.download('averaged_perceptron_tagger_eng', quiet=True); \
"

# Копируем исходный код приложения
COPY webapp/ /app/webapp/

WORKDIR /app/webapp
EXPOSE 8000

# Запускаем uvicorn с одним воркером (GPU нельзя шарить между процессами)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
