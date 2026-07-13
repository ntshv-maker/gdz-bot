# ИИ ГДЗ — Telegram бот

Бот помогает решать домашние задания через Groq: текст — `llama-3.3-70b-versatile`, фото — `llama-4-scout` (vision).

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # заполните токены
python bot.py
```

## Docker

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f gdz_bot
```

## Команды бота

- `/start` — приветствие
- `/help` — справка
- `/status` — баланс бесплатных запросов / подписка
- `/subscribe` — оформить подписку 99 ₽/мес
- `/clear` — очистить историю диалога

## Тарифы

- **Бесплатно:** 5 запросов в сутки (сброс в 00:00 МСК)
- **Подписка:** 99 ₽/мес — безлимитные запросы

Для оплаты подключи провайдера в [@BotFather](https://t.me/BotFather) → Payments и укажи `PAYMENT_PROVIDER_TOKEN` в `.env`.
