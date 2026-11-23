# Korea Visa Portal RAG

English question answering over **every PDF published on visa.go.kr**, with a citation to the exact page for each claim. Runs fully locally on one 8 GB GPU. The correct page is ranked first for **87%** of evaluation questions and is in the top 5 for **98%**. Answers take **2.3 s** at the median.

## Results

All numbers come from `eval/run.py` on `eval/golden.yaml`: 45 answerable questions and 7 the system must refuse. They were measured on an RTX 4060 8 GB.

**Retrieval** (`results/*-retrieval-bge-m3-v2.json`). A hit is a retrieved chunk from the right document that covers the right page.

| Mode | Correct page ranked 1st | Correct page in top 5 | Correct page in top 10 | MRR@10 | nDCG@10 | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense (bge-m3) | 0.733 | 0.933 | 0.956 | 0.819 | 0.808 | 93 ms |
| Sparse (BM25) | 0.756 | 0.978 | 0.978 | 0.848 | 0.840 | 3 ms |
| Hybrid (RRF) | 0.778 | 0.956 | 0.978 | 0.863 | 0.842 | 94 ms |
| **Hybrid + cross-encoder rerank** | **0.867** | **0.978** | **1.000** | **0.926** | **0.893** | 349 ms |

**Answers** (`results/20261124-0248-answers-qwen35-4b-v2*`, `results/20261124-0228-answers-qwen35-9b-v2*`)

| Metric | **qwen3.5:4b** (default) | qwen3.5:9b |
|---|---:|---:|
| Answered, of 45 answerable | 45 | 45 |
| Refused, of 7 unanswerable | 7 | 7 |
| Correct, LLM judge (κ = 0.80 against reference labels) | **0.978** (44/45) | 0.956 (43/45) |
| Every claim stated in the cited passage, LLM judge | **0.956** | 0.844 |
| Answers citing at least one gold page | 0.956 | 0.978 |
| Citation precision (cited passages that are on a gold page) | **0.855** | 0.848 |
| Invented citations (marker pointing to no passage) | 0 | 0 |
| End-to-end latency p50 / p95 | **2.25 s / 3.30 s** | 20.9 s / 27.8 s ¹ |

¹ At a 6k context, qwen3.5:9b needs 8.9 GB, so it cannot share the 8 GB GPU with the cross-encoder. In this run the reranker ran on the CPU (about 16 s), and 28% of the model still spilled to the CPU.

The judge is qwen3.5:9b, chosen by calibration (below). Reading every answer by hand gave the same verdict for the 4B run: 44/45, and the one wrong answer is VF02. It reads a two-table country list backwards (it says the UK lost visa-free entry; the source lists the UK as still eligible), even though the correct passage is ranked 2nd in its context. For the 9B run, the manual check found 42/45: the judge missed PR01, which gives Vietnamese city residents a 10-year visa instead of 5 years.

**Corpus**

| | |
|---|---:|
| Attachment links found on visa.go.kr (English and Korean sides, 3 boards and 34 static menus) | 536 |
| Unique PDFs downloaded (the rest are HWP, DOCX or XLSX) | 107 |
| PDFs whose text is English, so they are indexed | 40 (253 pages) |
| Indexed chunks (English text only, about 137 tokens on average) | 814 |
| Unit and data tests | `uv run pytest`: 13 passed |

> **Caveat: read before trusting the answer scores.** The 52 golden questions and the judge-calibration labels were written by an LLM (Claude) while it read the cited pages. A human did not write them, and no human has reviewed them yet. Retrieval scores use page labels that can be checked mechanically against the PDFs. Answer-correctness scores depend on those LLM-written reference answers.

## How it works

```mermaid
flowchart LR
    A["crawl.py<br/>visa.go.kr boards + menus<br/>EN and KO sides, 1 req/s"] --> B["parse.py<br/>Docling layout + TableFormer<br/>OCR only for scanned pages<br/>language tagging"]
    B --> C["index.py<br/>HybridChunker (384 tokens)<br/>heading trail + orphan recovery<br/>bge-m3 embeddings"]
    C --> D[("LanceDB<br/>vectors + BM25 FTS")]
    Q["Question (English)"] --> R["retrieve.py<br/>dense top 30 + BM25 top 30<br/>RRF fusion<br/>bge-reranker-v2-m3"]
    D --> R
    R -->|"top score < 0.02"| X["Refuse without calling the LLM"]
    R --> G["answer.py<br/>qwen3.5:4b on Ollama<br/>numbered passages, [n] after each claim<br/>NOT_FOUND contract"]
    G --> V["Citation check in code<br/>invalid [n] → regenerate once<br/>coverage, old-source flag"]
    V --> API["FastAPI /v1/ask + web UI<br/>citations open the PDF at the page"]
```

1. **Crawl.** The portal is a JSP application. Its boards only return a detail page when the form post carries `TRAN_TYPE=ComSubmit`. Without that field the server silently returns the list page again. Attachments come from `POST /cmm/downloadFile.do`. The crawler covers both language sides and every public menu. It stores PDFs by content hash, so the same file linked from 20 posts is downloaded once. Every link, PDF or not, is recorded in `data/raw/manifest.jsonl`.
2. **Parse.** Docling runs layout analysis and TableFormer table structure, and keeps the page number and bounding box of every item. OCR runs only on documents whose text layer is mostly empty. Language is decided per document: Hangul share for Korean, and `langdetect` on the non-Hangul text to tell English from Vietnamese or Russian, which are also written in Latin or Cyrillic script. Of 107 PDFs, 40 are English or bilingual Korean/English.
3. **Chunk.** Docling's HybridChunker splits by document structure first and by the embedding model's own token budget second. Two fixes were needed on these PDFs:
   - **Orphan recovery.** A heading with no body, such as *"5. Effective Date: March 2, 2016"*, appeared in no chunk at all, so the fact was unretrievable.
   - **Heading trail.** Many of these PDFs have flat heading levels, so a chunk kept only *"Contact details:"* and lost *"Bulgaria – Competent Authority"*. Each chunk now carries the heading before its own.

   Korean and Chinese halves of bilingual forms are dropped at chunk level. Documents posted only with Korean titles get a short English title written once by the local LLM.
4. **Retrieve.** Dense retrieval (bge-m3, run in-process) and BM25 (LanceDB native full-text search) each take the top 30, fused by reciprocal rank fusion. Then the bge-reranker-v2-m3 cross-encoder re-scores the fused list. If even the best passage scores below 0.02, the system refuses without calling the LLM: 5 of the 7 unanswerable questions are refused in about 0.3 s this way.
5. **Generate.** qwen3.5:4b gets 6 numbered passages, each with title, page and date. It must put a `[n]` after every factual sentence, or reply exactly `NOT_FOUND`. Code then checks every marker. A marker outside `1..k` is an invented citation, so the answer is regenerated once, and any invalid marker still left is removed and flagged. The response reports citation coverage, a flag for sources older than 3 years, and per-stage timings.

## Design decisions

**Hybrid + rerank instead of dense only.** These documents are full of literal tokens: visa codes (`F-2-5`), amounts (`KRW 500,000`), country lists and article numbers. On this corpus BM25 alone beats dense retrieval at every cutoff (top 5: 0.978 vs 0.933). Fusion alone ranks slightly better than BM25 (MRR 0.863 vs 0.848) but finds the right page in the top 5 a little less often. The cross-encoder is what pays: it puts the right page first for 0.867 of questions, against 0.778 for fusion, and costs about 250 ms on the GPU.

**The embedder runs in-process on the CPU.** The first version embedded queries through Ollama. Ollama then had to hold the embedder next to the generator, and the 8 GB GPU overflowed: `ollama ps` showed the generator split **30% CPU / 70% GPU**, and generation took 8–11 s. Embedding one short query on the CPU costs about 90 ms. Moving it there frees the GPU for the generator and the reranker, and generation dropped to about 2 s.

**qwen3.5:4b, not 9b.** Both ran on the same 52 questions with the same retrieval, and the larger model was not better. Correctness was 0.956 vs 0.978, every-claim-supported was 0.844 vs 0.956, and the 9B is about 9× slower on this card because it does not fit next to the reranker. A bigger generator is the wrong lever here. The errors that remain are table-reading errors, and both models make them.

**Plain-text citations, not JSON.** See "What did not work".

**What is not indexed.** Korean-only PDFs (66) and 235 HWP/DOCX/XLSX attachments are downloaded and listed in the manifest but not indexed, because the system is English-only by design. Several of them, such as the Korean visa manuals, hold substantive rules that are missing from the English side of the portal.

## What did not work

- **JSON-schema output with thinking off.** Ollama 0.17.7 with qwen3.5 *ignores* `format` (constrained decoding) when `think: false`. The model returned `{"message": "Hello!"}` for a schema that required `ok` and `text`. With thinking on, it follows the schema but spends 800–3,400 thinking tokens first. So answers are plain text with `[n]` markers and a `NOT_FOUND` sentinel, and citations are validated in code. The same issue first made qwen3.5:9b look like a useless judge: it answered `{"is_correct": true}` instead of the schema's `verdict`, so every verdict parsed as an error until the parser accepted that shape.
- **An uncalibrated local judge.** mistral-nemo marked **45/45** answers correct, including one that reversed a fact (it said UK citizens could *not* enter visa-free, while the source lists the UK as still eligible). See `eval/judge_cases.yaml` and `eval/calibrate.py`: on 55 labelled cases (45 real answers plus 10 answers altered by hand to contain one wrong number, date, list entry or yes/no reversal), mistral-nemo with a step-by-step prompt caught 9/12 wrong answers and kept 38/43 correct ones (κ = 0.598). **qwen3.5:9b caught 11/12 and kept 40/43 (κ = 0.799)**, so it is the judge. It belongs to the same family as the generator, which risks self-preference bias. The calibration measures that risk directly on real 4B answers.
- **A verbose prompt invites errors.** The first prompt asked the model to "mention the date of dated notices". The result was filler (*"The announcement was published on…"*), one comment about the passages themselves, and an answer that opened with "Yes" to a question whose answer was "in principle, no". The second prompt is shorter and answer-first. It moved dates into code-generated flags and UI badges. Citation precision rose from 0.772 to 0.855, and the yes/no error went away (`results/20261124-0217-*` vs `results/20261124-0248-*`).
- **Bugs that looked like model failures.** Two retrieval misses came from the chunker, not the model: a dropped heading and a lost country name. The citation-support judge failed answers because it was shown 600-character snippets instead of the full cited passage.

## Run it

Requirements: Linux, an NVIDIA GPU with 8 GB, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) (tested on 0.17.7) with `qwen3.5:4b`. The judge also needs `qwen3.5:9b`.

```bash
uv sync
ollama pull qwen3.5:4b
make crawl     # ~10 min at 1 request/s -> data/raw/pdf + manifest.jsonl
make parse     # Docling -> data/parsed + catalog.jsonl
make index     # chunks + bge-m3 + LanceDB (vector + BM25)
make serve     # http://localhost:8000  (API docs at /docs)
```

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question": "Who must submit a tuberculosis test certificate?"}' | jq .answer
```

Evaluation:

```bash
make test
uv run python eval/run.py retrieval                 # 4 retrieval arms
uv run python eval/run.py answers                   # end-to-end answers + latency
uv run python eval/calibrate.py --models mistral-nemo qwen3.5:9b
uv run python eval/run.py judge results/<run>-answers.jsonl --judge-model qwen3.5:9b
```

## Layout

```
src/visarag/   crawl, parse, index, embed, retrieve, answer, ollama, api, config, static/index.html
eval/          golden.yaml (52 Q), run.py, judge_cases.yaml (55 cases), calibrate.py
results/       every measured run, as JSON/JSONL
tests/         unit tests (parsers, language tagging, citations, RRF) + golden-set integrity
```

## Open items

1. **A human review of `eval/golden.yaml` and `eval/judge_cases.yaml`.** This is the largest methodological gap.
2. **Korean-only documents.** The richest rules (visa manuals) are Korean PDFs and HWP files. Indexing them needs cross-lingual retrieval and translation, which this English-only version leaves out by design.
3. **The corpus is old.** Most English notices are from 2014 to 2022. Every answer shows its source date, and the API flags sources older than three years.
