FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt bot.py ./

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 \
    DB_PATH=/data/gdz_bot.db

CMD ["python", "bot.py"]
