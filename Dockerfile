FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --target=/opt/python-site -r requirements.txt

COPY . .

RUN PYTHONPATH=/opt/python-site python -m src.train


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /opt/python-site /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/data ./data
COPY --from=builder /app/models ./models
COPY --from=builder /app/src ./src
COPY --from=builder /app/requirements.txt ./requirements.txt
COPY --from=builder /app/README.md ./README.md
COPY --from=builder /app/scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
