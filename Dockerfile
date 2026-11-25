# syntax=docker/dockerfile:1
# API image. The corpus and index are not baked in: build them with `make all` (or
# `python -m visarag.refresh --apply`) and mount ./data. GPU via the NVIDIA container toolkit;
# the PyTorch wheels bring their own CUDA runtime, so a slim base image is enough.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/models \
    VISARAG_HOST=0.0.0.0 \
    VISARAG_OLLAMA_URL=http://ollama:11434

WORKDIR /app
# Dependencies first, so code changes don't reinstall PyTorch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
 && useradd --create-home --uid 1000 app \
 && mkdir -p /app/data /models && chown app:app /app/data /models

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["/app/.venv/bin/python", "-m", "visarag.api"]
