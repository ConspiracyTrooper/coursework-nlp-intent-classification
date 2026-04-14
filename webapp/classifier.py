# Классификатор интентов на основе TF-IDF + Logistic Regression
# Повторяет предобработку из Курсовая.ipynb (гл. 2.2–2.3)
# Артефакты: models/tfidf.joblib, models/clf_Logistic_Regression.joblib, models/le.joblib

import os
import re

import joblib
import numpy as np
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Пути к артефактам классификатора (относительно корня проекта)
_BASE = os.path.join(os.path.dirname(__file__), "..", "models")

TFIDF_PATH = os.path.join(_BASE, "tfidf.joblib")
# Logistic Regression — лучший классификатор по F1-macro (~0.97), поддерживает predict_proba
CLF_PATH   = os.path.join(_BASE, "clf_Logistic_Regression.joblib")
LE_PATH    = os.path.join(_BASE, "le.joblib")

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def _preprocess(text: str) -> str:
    """Предобработка текста: повторяет clean_text + tokenize_and_lemmatize из ноутбука."""
    # clean_text: нижний регистр, оставляем буквы/цифры/апострофы
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # tokenize_and_lemmatize: punkt → только буквенные → без стоп-слов → лемматизация
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in _stop_words]
    lemmas = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)


# Загрузка артефактов при старте сервера (один раз)
tfidf = joblib.load(TFIDF_PATH)
clf   = joblib.load(CLF_PATH)
le    = joblib.load(LE_PATH)


def predict_intent(text: str) -> dict:
    """Классифицирует интент и возвращает топ-3 варианта с вероятностями."""
    processed = _preprocess(text)
    vec       = tfidf.transform([processed])
    proba     = clf.predict_proba(vec)[0]           # вероятности по всем классам
    top3_idx  = np.argsort(proba)[::-1][:3]         # индексы топ-3

    return {
        "intent":     le.inverse_transform([top3_idx[0]])[0],
        "confidence": float(proba[top3_idx[0]]),
        "top3": [
            {"intent": le.inverse_transform([i])[0], "prob": float(proba[i])}
            for i in top3_idx
        ],
    }
