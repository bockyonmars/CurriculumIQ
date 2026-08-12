# CurriculumIQ Python AI service (FastAPI). Build context = repo root.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CHROMA_PERSIST_DIRECTORY=/app/.curriculumiq_data/chroma

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY requirements.txt ./requirements.txt
COPY python_service/requirements.txt ./python_service/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r python_service/requirements.txt

# Application code (domain logic is reused, not duplicated).
COPY src ./src
COPY python_service ./python_service

# Persistent Chroma location + non-root user.
RUN mkdir -p /app/.curriculumiq_data \
    && useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "python_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
