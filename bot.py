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
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LabeledPrice,
    Message,
    PreCheckoutQuery,
)
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
DAILY_FREE_LIMIT = int(os.getenv("DAILY_FREE_LIMIT", os.getenv("FREE_REQUEST_LIMIT", "5")))
DAILY_TIMEZONE = ZoneInfo(os.getenv("DAILY_TIMEZONE", "Europe/Moscow"))
SUBSCRIPTION_PRICE_RUB = int(os.getenv("SUBSCRIPTION_PRICE_RUB", "99"))
SUBSCRIPTION_DAYS = int(os.getenv("SUBSCRIPTION_DAYS", "30"))
PAYMENT_PROVIDER_TOKEN = os.getenv("PAYMENT_PROVIDER_TOKEN", "")
SUBSCRIPTION_PAYLOAD_PREFIX = "gdz_sub"

SYSTEM_PROMPT = """Ты — умный и дружелюбный помощник по домашним заданиям (ГДЗ).
Пользователь присылает задание — ты даёшь понятное решение с объяснением.

Правила:
- Отвечай на языке задания (если задание на русском — отвечай по-русски).
- Сначала пошаговое решение с вычислениями.
- НЕ пиши «Краткий ответ» в начале — итог только одной строкой в самом конце: «Ответ: [число]».
- Для задач по математике показывай вычисления построчно обычными числами.
- Для текстовых заданий (сочинения, пересказы) давай готовый качественный текст.
- НЕ используй markdown (**жирный**, `код`) и НЕ используй LaTeX ($, $$, \\frac, \\left, \\right).
- Дроби пиши так: 3/7, (5 + 3/7), смешанные: 5 целых 3/7.
- Если в условии только числа и знак «=» в конце — это ВЫЧИСЛЕНИЕ, обязательно дай числовой ответ.
- Не говори «нет числового ответа», если в условии только числа — посчитай.
- Не придумывай переменные (x, m, n), если их явно нет в условии.
- Если задание неполное или непонятное — вежливо попроси уточнить."""

VISION_OCR_PROMPT = """Ты — распознаватель текста с фото школьного учебника по математике.
Посмотри на фото и перепиши задание ДОСЛОВНО обычным текстом.

КРИТИЧЕСКИ ВАЖНО:
- Это задача из учебника для вычисления. Там почти всегда только ЦИФРЫ, не переменные.
- Буквы m, n, l, o часто ошибочно принимают за цифры 7, 3, 1, 0 — смотри внимательно на форму символа.
- В знаменателе дроби почти всегда ЦИФРА: 3/7, а не 3/m.
- Во второй скобке перед + почти всегда ЦИФРА: (3 + 1/2), а не (n + 1/2).
- Пример правильного распознавания: (5 + 3/7) × (3 + 1/2) =

Правила:
- Дроби: 3/7, (5 + 3/7), смешанные: 5 целых 3/7.
- Знаки: +, -, ×, ÷, =, скобки — как на фото.
- НЕ решай — только перепиши условие.
- Если в конце «=» без правой части — это вычисление.
- Если нечитаемо — ответь: НЕЧИТАЕМО

Выведи только текст задания, без LaTeX и пояснений."""

OCR_FIX_PROMPT = """Текст ниже — OCR с фото домашнего задания по математике (5–9 класс).
OCR часто путает цифры с буквами. Исправь только ошибки распознавания:

- m→7, n→3, l→1, o→0 в дробях и скобках (если это явно вычисление, а не уравнение с переменными)
- Если задача на вычисление (= в конце, нет «найти x/m/n») — все одиночные буквы в числовых выражениях замени на цифры
- Не меняй, если это реальная алгебра (уравнение, «найти m», «при x=»)

OCR: {text}

Выведи ТОЛЬКО исправленный текст задания одной строкой."""

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

            CREATE TABLE IF NOT EXISTS payments (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id                     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                telegram_payment_charge_id  TEXT NOT NULL UNIQUE,
                amount                      INTEGER NOT NULL,
                currency                    TEXT NOT NULL,
                payload                     TEXT NOT NULL,
                created_at                  TEXT NOT NULL
            );
            """
        )
        migrate_db(conn)
        conn.commit()
    log.info("БД готова: %s", DB_PATH.resolve())


def migrate_db(conn: sqlite3.Connection) -> None:
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    if "free_requests_used" not in columns:
        conn.execute(
            "ALTER TABLE users ADD COLUMN free_requests_used INTEGER NOT NULL DEFAULT 0"
        )
    if "subscription_until" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN subscription_until TEXT")
    if "free_requests_date" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN free_requests_date TEXT")


def today_key() -> str:
    return datetime.now(DAILY_TIMEZONE).date().isoformat()


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


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def has_active_subscription(db_user_id: int) -> bool:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT subscription_until FROM users WHERE id = ?", (db_user_id,)
        ).fetchone()
    until = parse_dt(row["subscription_until"] if row else None)
    return until is not None and until > datetime.now(timezone.utc)


def get_free_requests_used(db_user_id: int) -> int:
    with closing(get_db()) as conn:
        row = conn.execute(
            """
            SELECT free_requests_used, free_requests_date
            FROM users WHERE id = ?
            """,
            (db_user_id,),
        ).fetchone()
    if not row or row["free_requests_date"] != today_key():
        return 0
    return int(row["free_requests_used"])


def get_access_info(db_user_id: int) -> dict[str, int | str | bool | None]:
    subscribed = has_active_subscription(db_user_id)
    used = get_free_requests_used(db_user_id)
    remaining = max(DAILY_FREE_LIMIT - used, 0)
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT subscription_until FROM users WHERE id = ?", (db_user_id,)
        ).fetchone()
    until = row["subscription_until"] if row else None
    return {
        "subscribed": subscribed,
        "free_used": used,
        "free_remaining": remaining if not subscribed else None,
        "subscription_until": until,
    }


def can_make_request(db_user_id: int) -> bool:
    if has_active_subscription(db_user_id):
        return True
    return get_free_requests_used(db_user_id) < DAILY_FREE_LIMIT


def consume_request(db_user_id: int) -> None:
    if has_active_subscription(db_user_id):
        return
    today = today_key()
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT free_requests_date FROM users WHERE id = ?", (db_user_id,)
        ).fetchone()
        if row and row["free_requests_date"] == today:
            conn.execute(
                """
                UPDATE users
                SET free_requests_used = free_requests_used + 1
                WHERE id = ?
                """,
                (db_user_id,),
            )
        else:
            conn.execute(
                """
                UPDATE users
                SET free_requests_used = 1, free_requests_date = ?
                WHERE id = ?
                """,
                (today, db_user_id),
            )
        conn.commit()


def activate_subscription(db_user_id: int, days: int = SUBSCRIPTION_DAYS) -> datetime:
    now = datetime.now(timezone.utc)
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT subscription_until FROM users WHERE id = ?", (db_user_id,)
        ).fetchone()
        current_until = parse_dt(row["subscription_until"] if row else None)
        base = current_until if current_until and current_until > now else now
        new_until = base + timedelta(days=days)
        conn.execute(
            """
            UPDATE users
            SET subscription_until = ?
            WHERE id = ?
            """,
            (new_until.isoformat(), db_user_id),
        )
        conn.commit()
    return new_until


def record_payment(
    db_user_id: int,
    charge_id: str,
    amount: int,
    currency: str,
    payload: str,
) -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO payments (
                user_id, telegram_payment_charge_id, amount, currency, payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (db_user_id, charge_id, amount, currency, payload, utc_now()),
        )
        conn.commit()


def subscribe_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"💎 Подписка {SUBSCRIPTION_PRICE_RUB} ₽/мес",
                    callback_data="subscribe",
                )
            ]
        ]
    )


def format_access_status(db_user_id: int) -> str:
    info = get_access_info(db_user_id)
    if info["subscribed"]:
        until = parse_dt(str(info["subscription_until"]))
        until_text = until.strftime("%d.%m.%Y") if until else "—"
        return (
            f"💎 <b>Подписка активна</b> до {until_text}\n"
            "Запросы: <b>безлимит</b>"
        )
    remaining = int(info["free_remaining"] or 0)
    used = int(info["free_used"])
    return (
        f"🆓 Сегодня: <b>{remaining}</b> из {DAILY_FREE_LIMIT} бесплатных запросов\n"
        f"Использовано сегодня: {used}\n"
        "<i>Лимит обновляется каждый день в 00:00 (МСК)</i>\n\n"
        f"Безлимит — подписка <b>{SUBSCRIPTION_PRICE_RUB} ₽/мес</b>: /subscribe"
    )


def paywall_text(db_user_id: int) -> str:
    return (
        "🚫 <b>Лимит на сегодня исчерпан</b>\n\n"
        f"Ты использовал {DAILY_FREE_LIMIT} из {DAILY_FREE_LIMIT} бесплатных запросов за сегодня.\n"
        "Завтра лимит обновится.\n\n"
        f"Или оформи подписку за <b>{SUBSCRIPTION_PRICE_RUB} ₽/мес</b> — "
        "и решай задания без ограничений."
    )


def usage_footer(db_user_id: int) -> str:
    if has_active_subscription(db_user_id):
        return ""
    remaining = int(get_access_info(db_user_id)["free_remaining"] or 0)
    if remaining <= 0:
        return ""
    return f"\n\n<i>Осталось бесплатных запросов сегодня: {remaining}</i>"


async def send_subscription_invoice(message: Message, db_user_id: int) -> None:
    if not PAYMENT_PROVIDER_TOKEN:
        await message.answer(
            "💳 Оплата пока не настроена.\n"
            "Администратор должен подключить платёжный провайдер в BotFather "
            "и указать PAYMENT_PROVIDER_TOKEN в .env",
            reply_markup=subscribe_keyboard(),
        )
        return

    bot = message.bot
    if not bot:
        return

    payload = f"{SUBSCRIPTION_PAYLOAD_PREFIX}:{db_user_id}:{int(datetime.now(timezone.utc).timestamp())}"
    await bot.send_invoice(
        chat_id=message.chat.id,
        title="ИИ ГДЗ — подписка",
        description=(
            f"Безлимитные запросы на {SUBSCRIPTION_DAYS} дней. "
            f"Автопродление через Telegram не включено — продли вручную."
        ),
        payload=payload,
        provider_token=PAYMENT_PROVIDER_TOKEN,
        currency="RUB",
        prices=[
            LabeledPrice(
                label=f"Подписка {SUBSCRIPTION_DAYS} дней",
                amount=SUBSCRIPTION_PRICE_RUB * 100,
            )
        ],
    )


def looks_like_numeric_calculation(text: str) -> bool:
    lower = text.lower()
    algebra_markers = (
        "найти", "найдите", "уравнен", "переменн", "при x", "при m", "при n",
        "неизвестн", "корень", "решите уравнение", "значение x", "значение m",
        "значение n", "выразите через",
    )
    if any(marker in lower for marker in algebra_markers):
        return False
    stripped = text.strip()
    if re.search(r"=\s*$", stripped):
        return True
    return bool(re.search(r"\d", text)) and not re.search(r"\b[xyz]\b", lower)


_OCR_LETTER_TO_DIGIT = {
    "m": "7",
    "n": "3",
    "l": "1",
    "o": "0",
    "O": "0",
    "s": "5",
    "b": "6",
    "g": "9",
}


def fix_ocr_heuristics(text: str) -> str:
    if not looks_like_numeric_calculation(text):
        return text

    def fix_denominator(match: re.Match[str]) -> str:
        num, letter = match.group(1), match.group(2)
        digit = _OCR_LETTER_TO_DIGIT.get(letter)
        return f"{num}/{digit}" if digit else match.group(0)

    text = re.sub(r"(\d+)/([a-zA-Z])\b", fix_denominator, text)

    def fix_leading_letter(match: re.Match[str]) -> str:
        letter = match.group(1)
        digit = _OCR_LETTER_TO_DIGIT.get(letter)
        return f"({digit} +" if digit else match.group(0)

    text = re.sub(r"\(\s*([a-zA-Z])\s+\+", fix_leading_letter, text)
    return text


async def fix_ocr_with_llm(text: str) -> str:
    if not re.search(r"[a-zA-Z]", text):
        return text

    prompt = OCR_FIX_PROMPT.format(text=text)
    fixed = await groq_chat(
        [{"role": "user", "content": prompt}],
        GROQ_MODEL,
        temperature=0.0,
    )
    return strip_latex(fixed).strip() or text


async def normalize_ocr_text(raw: str) -> str:
    text = fix_ocr_heuristics(raw)
    if re.search(r"[a-zA-Z]", text) and looks_like_numeric_calculation(text):
        text = await fix_ocr_with_llm(text)
        text = fix_ocr_heuristics(text)
    return text


def escape_html(text: str) -> str:
    return html.escape(text, quote=False)


def strip_latex(text: str) -> str:
    """Убирает LaTeX-разметку, которую Telegram HTML не рендерит."""
    text = re.sub(r"\$\$([^$]+)\$\$", r"\1", text)
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    for _ in range(6):
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\{(\d+)\}\{(\d+)\}", r"\1/\2", text)
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


def normalize_math_expression(text: str) -> str:
    text = re.sub(r"\{(\d+)\}\{(\d+)\}", r"\1/\2", text)
    text = re.sub(
        r"(\d+)\s+цел(?:ых|ая|ое|)\s+(\d+)\s*/\s*(\d+)",
        r"(\1 + \2/\3)",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("×", "*").replace("·", "*").replace("÷", "/")
    return text.strip()


def extract_math_expression(task_text: str) -> str:
    for line in task_text.splitlines():
        line = line.strip()
        if not line or any(
            skip in line.lower()
            for skip in ("комментарий", "реши", "пошагов", "задание с фото")
        ):
            if "задание с фото" in line.lower():
                line = re.sub(r"^.*?:\s*", "", line, flags=re.IGNORECASE).strip()
                if line:
                    return normalize_math_expression(line)
            continue
        if re.search(r"\d", line) and re.search(r"[+\-*/=()]", line):
            return normalize_math_expression(line)
    return normalize_math_expression(task_text)


def _format_number(value: Fraction | float | int) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, float):
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))
        return str(value)
    return str(value)


def try_compute_expression(task_text: str) -> str | None:
    expr = extract_math_expression(task_text)
    expr = re.sub(r"=\s*$", "", expr).strip()
    if not expr or not looks_like_numeric_calculation(expr + "="):
        return None

    expr = re.sub(r"\)\s*\(", ") * (", expr)
    converted = re.sub(r"(\d+)/(\d+)", r"Fraction(\1,\2)", expr)
    allowed = set("0123456789+-*/().,Fraction ")
    if not all(ch in allowed for ch in converted.replace(" ", "")):
        return None

    try:
        result = eval(  # noqa: S307 — выражение ограничено цифрами и Fraction
            converted,
            {"__builtins__": {}},
            {"Fraction": Fraction},
        )
    except (SyntaxError, TypeError, ZeroDivisionError, ValueError):
        return None

    if isinstance(result, (Fraction, int, float)):
        return _format_number(result)
    return None


def finalize_math_answer(llm_text: str, task_text: str) -> str:
    computed = try_compute_expression(task_text)
    text = llm_text.strip()

    text = re.sub(
        r"(?im)^\s*краткий\s+ответ\s*:\s*.+\n?",
        "",
        text,
    )

    if computed:
        text = re.sub(
            r"(?im)\n*(?:✅\s*)?(?:ответ|итог|результат)\s*:\s*.+$",
            "",
            text,
        ).rstrip()
        text = f"{text}\n\n✅ Ответ: {computed}"

    return text.strip()


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
    return await groq_chat(messages, GROQ_MODEL, temperature=0.2)


def build_photo_task_text(extracted: str, caption: str) -> str:
    parts = [f"Задание с фото:\n{extracted}"]
    if caption:
        parts.append(f"Комментарий ученика: {caption}")
    parts.append(
        "Реши это задание. Это вычисление — дай конкретный числовой ответ и пошаговое решение."
    )
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
    extracted = await groq_chat(messages, GROQ_VISION_MODEL, temperature=0.0)
    extracted = strip_latex(extracted).strip()
    extracted = await normalize_ocr_text(extracted)

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

    db_user_id = upsert_user(user.id, user.username, user.first_name)

    text = (
        "<b>📚 ИИ ГДЗ</b>\n\n"
        "Пришли задание <b>текстом</b> или <b>фото</b> — "
        "я прочитаю и помогу с решением.\n\n"
        f"{format_access_status(db_user_id)}\n\n"
        "<b>Команды:</b>\n"
        "• /help — справка\n"
        "• /status — баланс запросов\n"
        "• /subscribe — оформить подписку\n"
        "• /clear — очистить историю диалога"
    )
    await message.answer(text, reply_markup=subscribe_keyboard())


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    user = message.from_user
    db_user_id = upsert_user(
        user.id, user.username, user.first_name
    ) if user else 0

    text = (
        "<b>Как пользоваться ботом</b>\n\n"
        "1. Отправь текст задания — например:\n"
        "<code>Реши: 2x + 5 = 15</code>\n\n"
        "2. Или отправь <b>фото</b> задания из учебника.\n"
        "   Подпись к фото необязательна.\n\n"
        "3. Бот запоминает последние сообщения для контекста.\n"
        "   Чтобы начать заново — /clear\n\n"
        f"🆓 Бесплатно: <b>{DAILY_FREE_LIMIT}</b> запросов в сутки.\n"
        f"💎 Подписка: <b>{SUBSCRIPTION_PRICE_RUB} ₽/мес</b> — безлимит (/subscribe)\n\n"
        + (format_access_status(db_user_id) + "\n\n" if db_user_id else "")
        + f"<i>Текст: {escape_html(GROQ_MODEL)}</i>\n"
        f"<i>Фото: {escape_html(GROQ_VISION_MODEL)}</i>"
    )
    await message.answer(text, reply_markup=subscribe_keyboard())


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    user = message.from_user
    if not user:
        return
    db_user_id = upsert_user(user.id, user.username, user.first_name)
    await message.answer(
        format_access_status(db_user_id),
        reply_markup=subscribe_keyboard(),
    )


@dp.message(Command("subscribe"))
async def cmd_subscribe(message: Message) -> None:
    user = message.from_user
    if not user:
        return
    db_user_id = upsert_user(user.id, user.username, user.first_name)
    await send_subscription_invoice(message, db_user_id)


@dp.callback_query(F.data == "subscribe")
async def callback_subscribe(callback: CallbackQuery) -> None:
    user = callback.from_user
    message = callback.message
    if not user or not message:
        return
    await callback.answer()
    db_user_id = upsert_user(user.id, user.username, user.first_name)
    await send_subscription_invoice(message, db_user_id)


@dp.pre_checkout_query()
async def pre_checkout_handler(query: PreCheckoutQuery) -> None:
    payload = query.invoice_payload or ""
    if not payload.startswith(f"{SUBSCRIPTION_PAYLOAD_PREFIX}:"):
        await query.answer(ok=False, error_message="Неизвестный тип оплаты")
        return
    await query.answer(ok=True)


@dp.message(F.successful_payment)
async def successful_payment_handler(message: Message) -> None:
    user = message.from_user
    payment = message.successful_payment
    if not user or not payment:
        return

    db_user_id = upsert_user(user.id, user.username, user.first_name)
    until = activate_subscription(db_user_id, SUBSCRIPTION_DAYS)
    record_payment(
        db_user_id,
        payment.telegram_payment_charge_id,
        payment.total_amount,
        payment.currency,
        payment.invoice_payload,
    )

    await message.answer(
        "✅ <b>Оплата прошла успешно!</b>\n\n"
        f"Подписка активна до <b>{until.strftime('%d.%m.%Y')}</b>.\n"
        "Теперь запросы без ограничений — присылай задания.",
        reply_markup=subscribe_keyboard(),
    )


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

    if not can_make_request(db_user_id):
        await message.answer(
            paywall_text(db_user_id),
            reply_markup=subscribe_keyboard(),
        )
        return

    history = get_history(db_user_id)

    status = await message.answer("📷 <i>Читаю задание с фото...</i>")

    try:
        image_bytes, mime_type = await download_telegram_photo(message)
        extracted = await extract_task_from_photo(image_bytes, mime_type, caption)
        await status.edit_text("⏳ <i>Решаю задание...</i>")
        task_text = build_photo_task_text(extracted, caption)
        answer = await ask_groq(history, task_text)
        answer = finalize_math_answer(answer, extracted)
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
    consume_request(db_user_id)

    header = f"📝 <b>Распознано:</b> <code>{escape_html(extracted)}</code>\n\n"
    formatted = header + format_for_telegram(answer)
    formatted += usage_footer(db_user_id)
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

    if not can_make_request(db_user_id):
        await message.answer(
            paywall_text(db_user_id),
            reply_markup=subscribe_keyboard(),
        )
        return

    history = get_history(db_user_id)

    status = await message.answer("⏳ <i>Думаю над решением...</i>")

    try:
        answer = await ask_groq(history, task_text)
        answer = finalize_math_answer(answer, task_text)
    except Exception as exc:
        log.exception("Ошибка Groq")
        await status.edit_text(
            f"❌ <b>Ошибка</b>\n\n<code>{escape_html(str(exc))}</code>"
        )
        return

    add_message(db_user_id, "user", task_text)
    add_message(db_user_id, "assistant", answer)
    consume_request(db_user_id)

    formatted = format_for_telegram(answer) + usage_footer(db_user_id)
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
    if not PAYMENT_PROVIDER_TOKEN:
        log.warning(
            "PAYMENT_PROVIDER_TOKEN не задан — оплата подписки недоступна"
        )
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
