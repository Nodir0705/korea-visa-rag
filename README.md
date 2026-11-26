# korea-visa-rag

[![CI](https://github.com/Nodir0705/korea-visa-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Nodir0705/korea-visa-rag/actions/workflows/ci.yml)

Q&A over the Korea Visa Portal ([visa.go.kr](https://www.visa.go.kr)) in English. Answers come only from the portal's own documents, and every sentence links to its source. Everything runs locally: no hosted LLM.

![Answer with citations](docs/demo-pdf.png)

## How it works

```
visa.go.kr ─► crawl ─► Docling parse ─► chunk + embed ─► LanceDB (vectors + BM25)
                                                              │
question ─► hybrid search (bge-m3 + BM25, RRF) ─► bge-reranker-v2-m3 ─► context ─► LLM (Ollama) ─► checks ─► answer + [n] citations
```

- **Corpus.** The portal's English PDFs (notices, forms, regulations) plus the guide pages for 110 visa codes.
- **Context building.** Short PDF pages are passed whole. Tables are passed row by row. Guide pages are cut down to the visa code and application route (embassy / CCVI / e-visa) the question asks about.
- **Generation.** Each sentence must cite a passage. If the documents don't cover the question, the model says so.
- **Checks in code.** Invalid citations trigger a regeneration. A number that doesn't appear in the cited passage is removed.

Stack: Python 3.12, uv, Docling, LanceDB, sentence-transformers, Ollama, FastAPI.

## Results

Measured on 58 held-out questions. They were written from the documents and committed before the system was run on them. Two independent LLM graders checked every answer against the source text. *Main point* means the core answer is right. *Fully right* also means no wrong extra detail and nothing missing.

| Profile | Model | Main point | Fully right | Generation (p50) |
|---|---|---:|---:|---:|
| 16 GB GPU | gemma4:12b | 98.3% | 93.1% | 1.4 s |
| 8 GB GPU | qwen3.5:4b | 96.6% | 84.5% | 0.7 s |

The answers, the grades and the scoring script are in [`results/`](results/README.md).

The questions and the grading were both done by LLMs, not by people. The guide pages show requirements as displayed for one nationality (mostly US), and answers say so.

## Run it

Needs Linux, an NVIDIA GPU, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
uv sync
ollama pull qwen3.5:4b                      # 8 GB profile (default)
make crawl parse                            # download + parse the PDFs
uv run python -m visarag.visa_pages         # guide pages
make index
make serve                                  # http://localhost:8000
```

For a 16 GB GPU, run `ollama pull gemma4:12b`, then `set -a; . profiles/gpu16.env; set +a` before `make serve`.

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question": "Who must submit a tuberculosis test certificate?"}' | jq .answer
```

## Service

- `VISARAG_API_KEY` requires an `X-API-Key` header.
- There is a per-client rate limit, and a small queue in front of the GPU; a full queue returns 503.
- Logs are JSON, one line per request. Question text is not logged by default.
- `/health` checks the index, Ollama and the model.
- `make refresh` checks the portal for new or changed documents; `make refresh-apply` re-indexes.
- `Dockerfile`, `docker-compose.yml` and systemd units in `deploy/`.
- CI runs ruff and the tests.

## Evaluation

```bash
make test
uv run python eval/run.py retrieval
uv run python eval/run.py answers --golden eval/heldout3.yaml
```

The grading rubric is in [`eval/RUBRIC.md`](eval/RUBRIC.md).

## Limits

- English documents only. The Korean visa manuals hold more detail, but they are not indexed.
- Most English notices date from 2014–2022. Answers show the source date.
- No documents are stored in this repo. They belong to the Ministry of Justice (Korea Visa Portal) and are downloaded by the crawler.
