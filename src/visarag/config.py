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
    host: str = "127.0.0.1"  # API bind address; set api_key before binding anything but loopback
    port: int = 8000

    # Service limits and logging (api.py)
    api_key: str = ""  # if set, /v1/* requires the header X-API-Key
    rate_limit_per_minute: int = 20  # /v1/ask requests per client per minute; 0 = off
    max_queue: int = 4  # /v1/ask requests waiting for the single GPU; beyond this -> 503
    log_questions: bool = False  # question text can be personal; by default only its length and hash are logged
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
    include_pages: bool = True  # also index the per-visa guide pages (data/pages/en)

    # Retrieval
    candidates: int = 30  # per arm, before fusion
    rrf_k: int = 60
    rerank_top_n: int = 30  # fused candidates sent to the cross-encoder
    context_k: int = 6  # chunks given to the generator
    min_rerank_score: float = 0.02  # below this for every chunk -> refuse without calling the LLM

    # Generation
    expand_pages: bool = True  # small-to-big: give the model the whole (short) PDF page around a hit
    page_expand_max_chars: int = 3000  # only pages up to this size are expanded
    page_expand_max_doc_pages: int = 20  # long reference lists (e.g. the 89-page apostille directory) stay chunked
    context_max_chars: int = 16000  # cap on all passages together (num_ctx is 6144 tokens)
    expand_guide_pages: bool = True  # small-to-big for guide pages: overview + the whole route
    guide_expand_max_chars: int = 8000  # guide pages up to this size are given whole
    scope_filter: bool = True  # drop guide passages for other visa codes / routes than the question names
    think: bool = False  # let the generator reason before answering (slower)
    verify: bool = False  # second pass that drops unsupported sentences; measured worse (56/61 vs 58/61), so off

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
