"""HTTP service: ``POST /v1/ask``, ``GET /v1/documents``, the cited PDFs, and the web UI.

Service basics, all without extra dependencies:

* one JSON log line per request (request id, status, latency; for /v1/ask also the
  outcome and stage timings). Question text is not logged unless ``log_questions``
  is on, because people ask about their own immigration situation;
* optional API key (``X-API-Key``) on ``/v1/*``;
* a per-client rate limit and a bounded queue in front of the single GPU, so a
  burst gets a fast 429/503 instead of piling up threads;
* ``/health`` checks the index, Ollama and the generator model, and answers 503
  when one is missing.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import re
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Self

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from visarag.config import settings
from visarag.retrieve import MODES

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("visarag.api")
state: dict = {}


# ---------------------------------------------------------------- logging

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname, "msg": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        return json.dumps(out, ensure_ascii=False)


def _setup_logging() -> None:
    if any(getattr(h, "_visarag", False) for h in log.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler._visarag = True  # type: ignore[attr-defined]
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


# ---------------------------------------------------------------- limits

class RateLimiter:
    """Sliding one-minute window per client key, kept in memory (one process)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str, now: float | None = None) -> int:
        """0 if the request may go ahead (and records it), else seconds to wait."""
        if self.per_minute <= 0:
            return 0
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(1, int(60 - (now - q[0])) + 1)
            q.append(now)
            if len(self._hits) > 10_000:  # drop idle clients
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= 60]:
                    del self._hits[k]
            return 0


class GpuQueue:
    """One generator on one GPU: run requests one at a time, and refuse new ones
    when ``limit`` are already waiting or running."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._gpu = threading.Lock()
        self._count = 0
        self._count_lock = threading.Lock()

    def __enter__(self) -> Self:
        with self._count_lock:
            if self._count >= self.limit:
                raise HTTPException(503, "busy, try again shortly", headers={"Retry-After": "5"})
            self._count += 1
        self._gpu.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self._gpu.release()
        with self._count_lock:
            self._count -= 1


limiter = RateLimiter(settings.rate_limit_per_minute)
gpu = GpuQueue(settings.max_queue)


def require_key(request: Request) -> None:
    if settings.api_key and not hmac.compare_digest(
        request.headers.get("x-api-key", "").encode(), settings.api_key.encode()
    ):
        raise HTTPException(401, "missing or wrong X-API-Key")


def _client(request: Request) -> str:
    # Behind a reverse proxy this is the proxy's address; run uvicorn with
    # --proxy-headers / forwarded_allow_ips there so it becomes the real client.
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------- health

def _build_info() -> dict:
    path = settings.index_dir / "build.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_health() -> tuple[bool, dict]:
    checks: dict[str, dict] = {"answerer": {"ok": state.get("answerer") is not None}}
    try:
        from visarag.index import table_name
        from visarag.retrieve import _table

        checks["index"] = {"ok": True, "chunks": _table(table_name()).count_rows()}
    except Exception as exc:  # noqa: BLE001 - report, don't raise
        checks["index"] = {"ok": False, "error": type(exc).__name__}
    try:
        r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=2.0)
        r.raise_for_status()
        names = {m.get("name") for m in r.json().get("models", [])}
        checks["ollama"] = {"ok": True}
        checks["generator"] = {"ok": settings.gen_model in names, "model": settings.gen_model}
    except (httpx.HTTPError, ValueError) as exc:
        checks["ollama"] = {"ok": False, "error": type(exc).__name__}
        checks["generator"] = {"ok": False, "model": settings.gen_model}
    return all(c["ok"] for c in checks.values()), checks


# ---------------------------------------------------------------- app

def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from visarag.answer import Answerer

    _setup_logging()
    if not settings.api_key and not _is_loopback(settings.host):
        log.warning("bound to %s without an API key; set VISARAG_API_KEY", settings.host)
    t0 = time.perf_counter()
    ans = Answerer()
    ans.retriever.warm()
    ans.llm.chat(ans.model, [{"role": "user", "content": "Reply with OK."}], num_predict=2)
    state["answerer"] = ans
    state["warmup_s"] = round(time.perf_counter() - t0, 2)
    log.info("ready", extra={"fields": {"warmup_s": state["warmup_s"], "model": settings.gen_model}})
    yield
    state.clear()


app = FastAPI(
    title="Korea Visa Portal RAG",
    version="1.2.0",
    description="English answers with citations over visa.go.kr: its English PDFs and per-visa guide pages.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_log(request: Request, call_next):
    rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:12]
    request.state.rid = rid
    request.state.log = {}
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        if request.url.path != "/health":
            fields = {
                "rid": rid,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "ms": round((time.perf_counter() - t0) * 1000),
                "client": _client(request),
                **request.state.log,
            }
            log.info("request", extra={"fields": fields})


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What documents do I need for a tuberculosis test?"])
    mode: str = Field(default="hybrid+rerank", description=f"one of {MODES}")


@app.get("/health")
def health() -> JSONResponse:
    ok, checks = check_health()
    body = {
        "status": "ok" if ok else "degraded",
        "checks": checks,
        "generator": settings.gen_model,
        "embedder": settings.embed_model,
        "reranker": settings.reranker,
        "warmup_s": state.get("warmup_s"),
        "index_built_at": _build_info().get("built_at"),
    }
    return JSONResponse(body, status_code=200 if ok else 503)


@app.post("/v1/ask", dependencies=[Depends(require_key)])
def ask(req: AskRequest, request: Request) -> dict:
    if req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    wait = limiter.retry_after(_client(request))
    if wait:
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(wait)})
    question = req.question.strip()
    meta = request.state.log
    meta.update(q_len=len(question), q_sha=hashlib.sha256(question.encode()).hexdigest()[:10], mode=req.mode)
    if settings.log_questions:
        meta["question"] = question
    with gpu:
        out = state["answerer"].ask(question, mode=req.mode)
    meta.update(
        refused=out.refused,
        citations=len(out.citations),
        flags=out.flags,
        timings=out.timings,
    )
    return out.to_dict()


@app.get("/v1/documents", dependencies=[Depends(require_key)])
def documents() -> list[dict]:
    rows = [json.loads(x) for x in (settings.parsed_dir / "catalog.jsonl").open(encoding="utf-8")]
    keep = ("sha12", "post_title", "file_name", "post_date", "pages", "language", "english", "reason", "board")
    return [{k: r[k] for k in keep} for r in rows]


@app.get("/pdf/{doc_id}.pdf")
def pdf(doc_id: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{12}", doc_id):
        raise HTTPException(404)
    path = settings.raw_dir / "pdf" / f"{doc_id}.pdf"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/pdf", headers={"Content-Disposition": "inline"})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def main() -> None:
    import uvicorn

    uvicorn.run("visarag.api:app", host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
