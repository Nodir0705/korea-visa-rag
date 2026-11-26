"""Thin client for Ollama's native API (/api/embed, /api/chat).

The native API is used on purpose instead of the OpenAI-compatible /v1 one:
it honours ``options.num_ctx`` per request, accepts a JSON schema in
``format`` (constrained decoding), and takes ``think: false`` for reasoning
models. ``prompt_eval_count`` from the response is checked against the prompt
so that silent context truncation is caught instead of trusted.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from visarag.config import settings


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def embed(self, model: str, texts: list[str], batch: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            r = self.client.post(
                "/api/embed",
                json={"model": model, "input": texts[i : i + batch], "keep_alive": settings.keep_alive},
            )
            if r.status_code != 200:
                raise OllamaError(f"embed {model}: {r.status_code} {r.text[:200]}")
            out.extend(r.json()["embeddings"])
        return out

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        num_ctx: int | None = None,
        temperature: float = 0.0,
        num_predict: int | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        body = self._body(model, messages, schema, num_ctx, temperature, num_predict, stream=False, think=think)
        t0 = time.perf_counter()
        r = self.client.post("/api/chat", json=body)
        if r.status_code != 200:
            raise OllamaError(f"chat {model}: {r.status_code} {r.text[:300]}")
        data = r.json()
        data["_latency_s"] = time.perf_counter() - t0
        return data

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        num_ctx: int | None = None,
        temperature: float = 0.0,
    ) -> Iterator[dict[str, Any]]:
        body = self._body(model, messages, schema, num_ctx, temperature, None, stream=True)
        with self.client.stream("POST", "/api/chat", json=body) as r:
            if r.status_code != 200:
                raise OllamaError(f"chat {model}: {r.status_code} {r.read()[:300]!r}")
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)

    def _body(self, model, messages, schema, num_ctx, temperature, num_predict, stream, think=False) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": num_ctx or settings.num_ctx, "seed": 7}
        if num_predict:
            options["num_predict"] = num_predict
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": think,
            "keep_alive": settings.keep_alive,
            "options": options,
        }
        if schema:
            body["format"] = schema
        return body

    def loaded(self) -> list[dict[str, Any]]:
        return self.client.get("/api/ps").json().get("models", [])
