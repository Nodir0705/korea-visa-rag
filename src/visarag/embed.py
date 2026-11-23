"""Embedding backends.

``st:<hf-model>`` runs the model in-process with sentence-transformers. This is
the default because of the 8 GB GPU: the generator (Ollama) and the
cross-encoder reranker already fill it, and when Ollama also had to hold an
embedding model it evicted part of the generator to the CPU (30% CPU / 70% GPU,
measured), roughly tripling generation time. Query embedding is one short
string, so it runs on the CPU (~tens of ms); bulk indexing runs on the GPU.

``ollama:<tag>`` embeds through Ollama (e.g. ``ollama:qwen3-embedding:0.6b``).
"""
from __future__ import annotations

import functools

from visarag.config import EMBED_QUERY_PREFIX, settings


@functools.lru_cache(maxsize=4)
def _st_model(name: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    return SentenceTransformer(name, device=device, model_kwargs=kwargs)


class Embedder:
    def __init__(self, spec: str | None = None) -> None:
        self.spec = spec or settings.embed_model
        self.backend, _, self.name = self.spec.partition(":")
        if self.backend not in ("st", "ollama"):
            raise ValueError(f"embed model must start with 'st:' or 'ollama:', got {self.spec!r}")
        self.prefix = EMBED_QUERY_PREFIX.get(self.spec, "")

    def documents(self, texts: list[str], device: str | None = None) -> list[list[float]]:
        if self.backend == "ollama":
            from visarag.ollama import Ollama

            return Ollama().embed(self.name, texts)
        model = _st_model(self.name, device or settings.index_device)
        vecs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def query(self, text: str) -> list[float]:
        return list(_query_cached(self.spec, self.prefix + text))


@functools.lru_cache(maxsize=1024)
def _query_cached(spec: str, text: str) -> tuple[float, ...]:
    backend, _, name = spec.partition(":")
    if backend == "ollama":
        from visarag.ollama import Ollama

        return tuple(Ollama().embed(name, [text])[0])
    model = _st_model(name, settings.query_device)
    return tuple(model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist())
