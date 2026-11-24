"""HTTP service: ``POST /v1/ask``, ``GET /v1/documents``, the cited PDFs, and the web UI."""
from __future__ import annotations

import json
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from visarag.answer import Answerer
from visarag.config import settings
from visarag.retrieve import MODES

STATIC = Path(__file__).parent / "static"
_lock = threading.Lock()  # one GPU, one generator: serialize model calls
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    ans = Answerer()
    ans.retriever.warm()
    ans.llm.chat(ans.model, [{"role": "user", "content": "Reply with OK."}], num_predict=2)
    state["answerer"] = ans
    state["warmup_s"] = round(time.perf_counter() - t0, 2)
    yield
    state.clear()


app = FastAPI(
    title="Korea Visa Portal RAG",
    version="1.1.0",
    description="English answers with citations over visa.go.kr: its English PDFs and per-visa guide pages.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What documents do I need for a tuberculosis test?"])
    mode: str = Field(default="hybrid+rerank", description=f"one of {MODES}")


@app.get("/health")
def health() -> dict:
    ans: Answerer | None = state.get("answerer")
    return {
        "ok": ans is not None,
        "generator": settings.gen_model,
        "embedder": settings.embed_model,
        "reranker": settings.reranker,
        "warmup_s": state.get("warmup_s"),
    }


@app.post("/v1/ask")
def ask(req: AskRequest) -> dict:
    if req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    with _lock:
        return state["answerer"].ask(req.question.strip(), mode=req.mode).to_dict()


@app.get("/v1/documents")
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
