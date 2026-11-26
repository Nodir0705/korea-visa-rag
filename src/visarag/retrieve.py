"""Hybrid retrieval: dense (LanceDB vectors) + sparse (LanceDB BM25 full-text),
fused with reciprocal rank fusion, then re-scored by a cross-encoder.

Every stage is switchable (``mode``) so the evaluation can report each arm on
its own: ``dense``, ``sparse``, ``hybrid`` and ``hybrid+rerank``.
"""
from __future__ import annotations

import functools
import re
import time
from dataclasses import dataclass, field

import lancedb

from visarag.config import settings
from visarag.embed import Embedder
from visarag.index import table_name

MODES = ("dense", "sparse", "hybrid", "hybrid+rerank")
_FTS_SAFE = re.compile(r"[^\w\s-]")


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    file_name: str
    page: int
    pages: list[int]
    headings: str
    body: str
    text: str
    source_url: str
    pdf_path: str
    post_date: str
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    source: str = "pdf"  # "pdf" or "page" (visa.go.kr per-visa guide page)
    note: str = ""


@functools.lru_cache(maxsize=4)
def _table(name: str):
    return lancedb.connect(settings.index_dir / "lancedb").open_table(name)


@functools.lru_cache(maxsize=2)
def _reranker(name: str, device: str):
    import torch
    from sentence_transformers import CrossEncoder

    kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    return CrossEncoder(name, device=device, max_length=512, model_kwargs=kwargs)


def _to_hit(row: dict) -> Hit:
    return Hit(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        title=row["title"],
        file_name=row["file_name"],
        page=int(row["page"]),
        pages=list(row["pages"]),
        headings=row["headings"],
        body=row["body"],
        text=row["text"],
        source_url=row["source_url"],
        pdf_path=row["pdf_path"],
        post_date=row["post_date"],
        source=row.get("source", "pdf"),
        note=row.get("note", ""),
    )


class Retriever:
    def __init__(self, embed_model: str | None = None, reranker: str | None = None) -> None:
        self.embed_model = embed_model or settings.embed_model
        self.reranker_name = reranker or settings.reranker
        self.table = _table(table_name(self.embed_model))
        self.embedder = Embedder(self.embed_model)
        self.timings: dict[str, float] = {}

    def dense(self, q: str, k: int) -> list[Hit]:
        vec = self.embedder.query(q)
        rows = self.table.search(vec, query_type="vector").metric("cosine").limit(k).to_list()
        return [_to_hit(r) for r in rows]

    def sparse(self, q: str, k: int) -> list[Hit]:
        safe = _FTS_SAFE.sub(" ", q).strip()
        if not safe:
            return []
        rows = self.table.search(safe, query_type="fts", fts_columns="text").limit(k).to_list()
        return [_to_hit(r) for r in rows]

    @staticmethod
    def rrf(arms: dict[str, list[Hit]], k: int) -> list[Hit]:
        fused: dict[str, Hit] = {}
        for arm, hits in arms.items():
            for rank, h in enumerate(hits, 1):
                cur = fused.setdefault(h.chunk_id, h)
                cur.ranks[arm] = rank
                cur.score += 1.0 / (k + rank)
        return sorted(fused.values(), key=lambda h: h.score, reverse=True)

    def rerank(self, q: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        model = _reranker(self.reranker_name, settings.reranker_device)
        scores = model.predict([(q, h.text) for h in hits], batch_size=16, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h.ranks["fused"] = h.ranks.get("fused", 0)
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def score_texts(self, q: str, texts: list[str]) -> list[float]:
        """Cross-encoder relevance of each text to ``q`` (used to pick sub-sections of a guide page)."""
        if not texts:
            return []
        model = _reranker(self.reranker_name, settings.reranker_device)
        return [float(s) for s in model.predict([(q, t) for t in texts], batch_size=16, show_progress_bar=False)]

    def page_hit(self, code: str) -> Hit | None:
        """A chunk of the visa guide page for ``code``, or None if there is no such page."""
        if not re.fullmatch(r"[A-H]-\d{1,2}(-\d{1,2})?", code):
            return None
        rows = self.table.search().where(f"doc_id = 'page-{code}'").limit(1).to_list()
        return _to_hit(rows[0]) if rows else None

    def search(self, q: str, k: int | None = None, mode: str = "hybrid+rerank") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        k = k or settings.context_k
        n = settings.candidates
        self.timings = {}
        t = time.perf_counter()
        if mode == "dense":
            out = self.dense(q, k)
        elif mode == "sparse":
            out = self.sparse(q, k)
        else:
            d = self.dense(q, n)
            self.timings["dense_s"] = time.perf_counter() - t
            t2 = time.perf_counter()
            s = self.sparse(q, n)
            self.timings["sparse_s"] = time.perf_counter() - t2
            fused = self.rrf({"dense": d, "sparse": s}, settings.rrf_k)
            for i, h in enumerate(fused, 1):
                h.ranks["fused"] = i
            if mode == "hybrid":
                out = fused[:k]
            else:
                t3 = time.perf_counter()
                out = self.rerank(q, fused[: settings.rerank_top_n])[:k]
                self.timings["rerank_s"] = time.perf_counter() - t3
        self.timings["retrieve_s"] = time.perf_counter() - t
        return out

    def warm(self) -> None:
        """Load the embedder and the cross-encoder so the first user query is not the slow one."""
        self.search("visa application documents", mode="hybrid+rerank")
