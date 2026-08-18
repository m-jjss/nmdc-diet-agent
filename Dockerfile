FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DEEPSEEK_API_KEY=
ENV DEEPSEEK_API_BASE=https://api.deepseek.com/v1
ENV DEEPSEEK_MODEL=deepseek-v4-flash
ENV QWEN_API_KEY=
ENV ERNIE_API_KEY=
ENV ERNIE_SECRET_KEY=
ENV OPENAI_API_KEY=
ENV FLASK_ENV=production
ENV ENV=production

EXPOSE 5000

# 单进程多线程：gunicorn -w 1 保证多轮对话状态（进程内 DialogManager）共享，
# --threads 8 提供并发处理能力，避免多 worker 间状态丢失
CMD ["gunicorn", "-w", "1", "--threads", "8", "-b", "0.0.0.0:5000", "app:app", "--timeout", "120"]
