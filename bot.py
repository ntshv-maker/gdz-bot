#!/usr/bin/env python3
"""
ИИ ГДЗ — Telegram-бот на aiogram + SQLite + Groq.
Текст: llama-3.3-70b-versatile, фото: Llama 4 Scout (vision).
Запуск: python bot.py
"""

from __future__ import annotations

import asyncio
import base64
import html
import logging
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DB_PATH = Path(os.getenv("DB_PATH", "gdz_bot.db"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
MAX_IMAGE_BYTES = 20 * 1024 * 1024

SYSTEM_PROMPT = """Ты — умный и дружелюбный помощник по домашним заданиям (ГДЗ).
Пользователь присылает задание — ты даёшь понятное решение с объяснением.

Правила:
- Отвечай на языке задания (если задание на русском — отвечай по-русски).
- Сначала краткий ответ, затем пошаговое решение.
- Для задач по математике показывай вычисления построчно обычными числами.
- Для текстовых заданий (сочинения, пересказы) давай готовый качественный текст.
- НЕ используй markdown (**жирный**, `код`) и НЕ используй LaTeX ($, $$, \\frac, \\left, \\right).
- Дроби пиши так: 3/7, (5 + 3/7), смешанные: 5 целых 3/7.
- Если в условии только числа и знак «=» в конце — это ВЫЧИСЛЕНИЕ, найди числовой ответ.
- Не придумывай переменные (x, m, n), если их явно нет в условии.
- Если задание неполное или непонятное — вежливо попроси уточнить."""

VISION_OCR_PROMPT = """Ты — распознаватель текста с фото учебника.
Посмотри на фото и перепиши задание ДОСЛОВНО обычным текстом.

Правила:
- Цифры на фото — это ЧИСЛА (7, 3, 5), НЕ буквы-переменные (m, n, x).
- Дроби: 3/7, (5 + 3/7), смешанные дроби: 5 целых 3/7.
- Сохраняй знаки: +, -, ×, ÷, =, скобки — как на фото.
- НЕ решай задачу — только перепиши условие одной строкой или несколькими.
- Если в конце стоит «=» без правой части — это вычисление, не уравнение «= 0».
- Если текст нечитаем — ответь ровно: НЕЧИТАЕМО

Выведи только текст задания, без пояснений и без LaTeX."""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("gdz_bot")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with closing(get_db()) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER NOT NULL UNIQUE,
                username    TEXT,
                first_name  TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
            """
        )
        conn.commit()
    log.info("БД готова: %s", DB_PATH.resolve())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_user(telegram_id: int, username: str | None, first_name: str | None) -> int:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO users (telegram_id, username, first_name, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(telegram_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name
            """,
            (telegram_id, username, first_name, utc_now()),
        )
        row = conn.execute(
            "SELECT id FROM users WHERE telegram_id = ?", (telegram_id,)
        ).fetchone()
        conn.commit()
        return int(row["id"])


def add_message(user_id: int, role: str, content: str) -> None:
    with closing(get_db()) as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, utc_now()),
        )
        conn.commit()


def get_history(user_id: int, limit: int = MAX_HISTORY) -> list[dict[str, str]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE user_id = ? AND role IN ('user', 'assistant')
            ORDER BY id DESC LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def clear_history(user_id: int) -> None:
    with closing(get_db()) as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.commit()


def escape_html(text: str) -> str:
    return html.escape(text, quote=False)


def strip_latex(text: str) -> str:
    """Убирает LaTeX-разметку, которую Telegram HTML не рендерит."""
    text = re.sub(r"\$\$([^$]+)\$\$", r"\1", text)
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    for _ in range(6):
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\left[\(\[\{]", "(", text)
    text = re.sub(r"\\right[\)\]\}]", ")", text)
    text = re.sub(r"\\(?:quad|,|;|:|!)\s*", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def markdown_to_html(text: str) -> str:
    text = escape_html(text)

    def repl_pre(match: re.Match[str]) -> str:
        return f"<pre>{match.group(1)}</pre>"

    text = re.sub(r"```(?:\w*\n)?([\s\S]*?)```", repl_pre, text)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", text)
    return text


def format_for_telegram(text: str) -> str:
    return markdown_to_html(strip_latex(text))


def split_message(text: str, limit: int = 4096) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


async def groq_chat(
    messages: list[dict], model: str, *, temperature: float = 0.4
) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
    }

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(GROQ_API_URL, headers=headers, json=payload) as resp:
            body = await resp.json()
            if resp.status != 200:
                err = body.get("error", {})
                msg = err.get("message", str(body))
                raise RuntimeError(f"Groq API {resp.status}: {msg}")

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Groq вернул пустой ответ")

    content = choices[0].get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Groq вернул пустой текст")
    return content


async def ask_groq(history: list[dict[str, str]], user_text: str) -> str:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return await groq_chat(messages, GROQ_MODEL)


def build_photo_task_text(extracted: str, caption: str) -> str:
    parts = [f"Задание с фото:\n{extracted}"]
    if caption:
        parts.append(f"Комментарий ученика: {caption}")
    parts.append("Реши это задание. Если это вычисление — дай числовой ответ.")
    return "\n\n".join(parts)


async def extract_task_from_photo(
    image_bytes: bytes,
    mime_type: str,
    caption: str,
) -> str:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Фото слишком большое (максимум 20 МБ)")

    ocr_prompt = VISION_OCR_PROMPT
    if caption:
        ocr_prompt += f"\n\nКомментарий ученика (учти при распознавании): {caption}"

    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ocr_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    extracted = await groq_chat(messages, GROQ_VISION_MODEL, temperature=0.1)
    extracted = strip_latex(extracted).strip()

    if not extracted or extracted.upper() == "НЕЧИТАЕМО":
        raise ValueError(
            "Не удалось прочитать задание с фото. Пришли более чёткое фото."
        )
    return extracted


async def download_telegram_photo(message: Message) -> tuple[bytes, str]:
    if not message.photo:
        raise ValueError("Фото не найдено")

    photo = message.photo[-1]
    if photo.file_size and photo.file_size > MAX_IMAGE_BYTES:
        raise ValueError("Фото слишком большое (максимум 20 МБ)")

    bot = message.bot
    if not bot:
        raise RuntimeError("Bot instance недоступен")

    tg_file = await bot.get_file(photo.file_id)
    if not tg_file.file_path:
        raise RuntimeError("Не удалось получить файл из Telegram")

    buffer = await bot.download_file(tg_file.file_path)
    data = buffer.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Фото слишком большое (максимум 20 МБ)")

    return data, "image/jpeg"


dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    user = message.from_user
    if not user:
        return

    upsert_user(user.id, user.username, user.first_name)

    text = (
        "<b>📚 ИИ ГДЗ</b>\n\n"
        "Пришли задание <b>текстом</b> или <b>фото</b> — "
        "я прочитаю и помогу с решением.\n\n"
        "<b>Команды:</b>\n"
        "• /help — справка\n"
        "• /clear — очистить историю диалога"
    )
    await message.answer(text)


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    text = (
        "<b>Как пользоваться ботом</b>\n\n"
        "1. Отправь текст задания — например:\n"
        "<code>Реши: 2x + 5 = 15</code>\n\n"
        "2. Или отправь <b>фото</b> задания из учебника.\n"
        "   Подпись к фото необязательна.\n\n"
        "3. Бот запоминает последние сообщения для контекста.\n"
        "   Чтобы начать заново — /clear\n\n"
        f"<i>Текст: {escape_html(GROQ_MODEL)}</i>\n"
        f"<i>Фото: {escape_html(GROQ_VISION_MODEL)}</i>"
    )
    await message.answer(text)


@dp.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    user = message.from_user
    if not user:
        return

    db_user_id = upsert_user(user.id, user.username, user.first_name)
    clear_history(db_user_id)
    await message.answer("🗑 История диалога очищена. Можешь прислать новое задание.")


@dp.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message) -> None:
    await process_task(message, message.text or "")


@dp.message(F.photo)
async def handle_photo(message: Message) -> None:
    caption = (message.caption or "").strip()
    await process_photo_task(message, caption)


async def process_photo_task(message: Message, caption: str) -> None:
    user = message.from_user
    if not user:
        return

    db_user_id = upsert_user(user.id, user.username, user.first_name)
    history = get_history(db_user_id)

    status = await message.answer("📷 <i>Читаю задание с фото...</i>")

    try:
        image_bytes, mime_type = await download_telegram_photo(message)
        extracted = await extract_task_from_photo(image_bytes, mime_type, caption)
        await status.edit_text("⏳ <i>Решаю задание...</i>")
        task_text = build_photo_task_text(extracted, caption)
        answer = await ask_groq(history, task_text)
    except Exception as exc:
        log.exception("Ошибка обработки фото")
        await status.edit_text(
            f"❌ <b>Ошибка</b>\n\n<code>{escape_html(str(exc))}</code>"
        )
        return

    db_record = f"[Фото] {extracted}"
    if caption:
        db_record += f" ({caption})"
    add_message(db_user_id, "user", db_record)
    add_message(db_user_id, "assistant", answer)

    header = f"📝 <b>Распознано:</b> <code>{escape_html(extracted)}</code>\n\n"
    formatted = header + format_for_telegram(answer)
    parts = split_message(formatted)

    await status.edit_text(parts[0])
    for part in parts[1:]:
        await message.answer(part)


async def process_task(message: Message, task_text: str) -> None:
    user = message.from_user
    if not user:
        return

    task_text = task_text.strip()
    if not task_text:
        await message.answer("Пришли текст задания.")
        return

    db_user_id = upsert_user(user.id, user.username, user.first_name)
    history = get_history(db_user_id)

    status = await message.answer("⏳ <i>Думаю над решением...</i>")

    try:
        answer = await ask_groq(history, task_text)
    except Exception as exc:
        log.exception("Ошибка Groq")
        await status.edit_text(
            f"❌ <b>Ошибка</b>\n\n<code>{escape_html(str(exc))}</code>"
        )
        return

    add_message(db_user_id, "user", task_text)
    add_message(db_user_id, "assistant", answer)

    formatted = format_for_telegram(answer)
    parts = split_message(formatted)

    await status.edit_text(parts[0])
    for part in parts[1:]:
        await message.answer(part)


def validate_config() -> None:
    missing = []
    if not BOT_TOKEN:
        missing.append("BOT_TOKEN")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise SystemExit(
            f"Не заданы переменные окружения: {', '.join(missing)}\n"
            "Создай файл .env на основе .env.example"
        )


async def main() -> None:
    validate_config()
    init_db()

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    log.info(
        "Бот запущен (текст: %s, фото: %s)", GROQ_MODEL, GROQ_VISION_MODEL
    )
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
