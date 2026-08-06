FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LLM_PROVIDER=local
ENV DEEPSEEK_API_KEY=
ENV DEEPSEEK_API_BASE=https://api.deepseek.com/v1
ENV DEEPSEEK_MODEL=deepseek-chat
ENV DEEPSEEK_EMBED_MODEL=deepseek-embed
ENV QWEN_API_KEY=
ENV ERNIE_API_KEY=
ENV ERNIE_SECRET_KEY=
ENV OPENAI_API_KEY=
ENV FLASK_ENV=production
ENV ENV=production

EXPOSE 5000

CMD ["python", "app.py"]
