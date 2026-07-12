# ИИ ГДЗ — Telegram бот

Бот помогает решать домашние задания через Groq (llama-3.3-70b-versatile).

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
- `/clear` — очистить историю диалога
