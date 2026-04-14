# Обёртка над deep-translator для перевода RU↔EN
# Используется для общения с моделью, обученной на английском датасете

from deep_translator import GoogleTranslator


def ru_to_en(text: str) -> str:
    """Переводит текст с русского на английский."""
    return GoogleTranslator(source="ru", target="en").translate(text)


def en_to_ru(text: str) -> str:
    """Переводит текст с английского на русский."""
    return GoogleTranslator(source="en", target="ru").translate(text)
