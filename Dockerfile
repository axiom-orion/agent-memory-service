# Cloud Run-ready image for the agent memory service.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SENTENCE_TRANSFORMERS_HOME=/models \
    HF_HOME=/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY serve ./serve

RUN pip install ".[serve]"

# Bake the local embedding model into the image so cold starts don't download it.
# (Skipped automatically when EMBEDDINGS_BACKEND=vertex is used at runtime.)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV PORT=8080
EXPOSE 8080
# Cloud Run injects $PORT; bind to it.
CMD exec uvicorn serve.app:app --host 0.0.0.0 --port ${PORT}
