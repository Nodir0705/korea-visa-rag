"""All settings in one place, overridable by environment variables (VISARAG_*) or .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

# Embedding models and the instruction each expects on the query side.
# Documents are embedded without an instruction in both cases.
EMBED_QUERY_PREFIX = {
    "ollama:qwen3-embedding:0.6b": (
        "Instruct: Given a question about Korean visas, retrieve passages from official "
        "visa documents that answer it\nQuery:"
    ),
    "st:BAAI/bge-m3": "",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISARAG_", env_file=ROOT / ".env", extra="ignore")

    root: Path = ROOT
    data_dir: Path = ROOT / "data"

    # Models (all local, via Ollama, except the cross-encoder reranker which runs in-process)
    ollama_url: str = "http://localhost:11434"
    host: str = "127.0.0.1"  # API bind address; no auth, so keep it off public interfaces
    port: int = 8000
    embed_model: str = "st:BAAI/bge-m3"  # "st:<hf model>" in-process, or "ollama:<tag>"
    embed_tokenizer: str = "BAAI/bge-m3"  # HF tokenizer used to size chunks
    index_device: str = "cuda"
    query_device: str = "cpu"
    gen_model: str = "qwen3.5:4b"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    keep_alive: str = "30m"
    num_ctx: int = 6144  # prompts are ~1.3-2k tokens; 6k keeps the 4B model 100% on the GPU

    # Chunking
    chunk_max_tokens: int = 384

    # Retrieval
    candidates: int = 30  # per arm, before fusion
    rrf_k: int = 60
    rerank_top_n: int = 30  # fused candidates sent to the cross-encoder
    context_k: int = 6  # chunks given to the generator
    min_rerank_score: float = 0.02  # below this for every chunk -> refuse without calling the LLM

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


settings = Settings()
