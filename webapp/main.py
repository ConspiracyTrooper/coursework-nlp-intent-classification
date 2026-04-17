# FastAPI-бэкенд диалоговой системы технической поддержки
# Пайплайн: сообщение (RU) → перевод RU→EN → классификация интента →
# Mistral-7B (базовая или QLoRA) → перевод EN→RU → ответ пользователю

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi          import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic         import BaseModel

from classifier import le, predict_intent
from inference  import generate as model_generate, is_ready
from translator import en_to_ru, ru_to_en

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Support Chat API", version="2.0")

# Раздача статических файлов (index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message:    str
    model_type: Literal["base", "finetuned"] = "finetuned"


class ChatResponse(BaseModel):
    response:     str
    intent:       str
    confidence:   float
    top3_intents: list
    model_used:   str


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
async def health() -> dict:
    """Проверка готовности сервиса и модели."""
    return {
        "status":       "ok",
        "model_loaded": is_ready(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Основной эндпоинт: перевод → классификация → генерация → перевод."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Пустое сообщение")

    # Шаг 1: перевод сообщения пользователя RU → EN (модель обучена на английском)
    en_message = await asyncio.to_thread(ru_to_en, req.message)

    # Шаг 2: классификация интента (TF-IDF + Logistic Regression)
    intent_result = await asyncio.to_thread(predict_intent, en_message)

    # Шаг 3: генерация ответа через Mistral-7B (блокирующий вызов в отдельном потоке)
    en_response = await asyncio.to_thread(
        model_generate,
        en_message,
        req.model_type,
    )

    # Шаг 4: перевод ответа EN → RU для пользователя
    ru_response = await asyncio.to_thread(en_to_ru, en_response)

    model_label = (
        "mistral-7b-instruct-qlora" if req.model_type == "finetuned"
        else "mistral-7b-instruct-base"
    )

    return ChatResponse(
        response     = ru_response,
        intent       = intent_result["intent"],
        confidence   = intent_result["confidence"],
        top3_intents = intent_result["top3"],
        model_used   = model_label,
    )


@app.get("/intents")
async def get_intents() -> dict:
    """Возвращает список всех интентов из LabelEncoder."""
    return {"intents": list(le.classes_), "count": int(len(le.classes_))}
