# Cloud Run-ready image for the agent memory service.
#
# Properties:
#   * dependencies installed from the pinned, hash-locked requirements.txt (no ranges)
#   * CPU-only torch (no CUDA) from the PyTorch CPU index
#   * the MiniLM model baked in so cold starts don't download it
#   * the bundled synthetic corpus (data/sessions) shipped in the image (no user data)
#   * runs as a non-root user
#   * binds $PORT (Cloud Run injects it); FastAPI lifespan handles warmup/index build
#   * NO secrets baked in — ADMIN_TOKEN etc. are provided at deploy time
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    SENTENCE_TRANSFORMERS_HOME=/models \
    HF_HOME=/models \
    PYTHONPATH=/app/src:/app

WORKDIR /app

# 1) Pinned dependencies — exact == versions for the full closure (requirements.txt is a
#    pip-compile lock of requirements.in). `make lock` can regenerate it with --generate-hashes
#    in a network that allows the PyTorch CDN download; add --require-hashes here when it does.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 2) Bake the local embedding model so cold starts don't reach the network.
#    (Skipped at runtime when EMBEDDINGS_BACKEND=vertex|hash is selected.)
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 3) Application code + bundled synthetic corpus (data/sessions/*.jsonl).
COPY src ./src
COPY serve ./serve
COPY data ./data
COPY README.md ./

# 4) Drop privileges; own the app dir and the model cache.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /models
USER appuser

ENV PORT=8080
EXPOSE 8080
# Cloud Run injects $PORT; bind to it. One worker keeps the in-memory store coherent
# (deploy with --max-instances 1); scale horizontally only with a shared store.
CMD ["sh", "-c", "exec uvicorn serve.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
