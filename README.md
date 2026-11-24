# Korea Visa Portal RAG

English question answering over the Korea Visa Portal (**visa.go.kr**). It uses two sources: the site's **English PDFs** (40 of the 107 PDFs it publishes) and its **110 English per-visa guide pages** (eligibility, required documents, visa validity and period of stay for each visa code). Every claim is cited, either to a PDF page or to a visa guide page. It runs fully locally on one 8 GB GPU and answers in **about 2 s**. Overall accuracy is **98.6%** on the 70-question development set, and **94–96%** on 50 held-out questions that were never tuned on (judge score and manual review, respectively).

## Results

All numbers come from `eval/run.py` on `eval/golden.yaml`: 61 answerable questions (45 on PDFs, 16 on visa pages, 2 of them combining both) and 9 the system must refuse. They were measured on an RTX 4060 8 GB.

**Retrieval** (`results/*-retrieval-pages.json`). A hit is a retrieved chunk from the right document that covers the right page (for visa pages: the right visa code).

| Mode | Correct source ranked 1st | In top 5 | In top 10 | MRR@10 | p50 latency |
|---|---:|---:|---:|---:|---:|
| Dense (bge-m3) | 0.754 | 0.951 | 0.967 | 0.836 | 94 ms |
| Sparse (BM25) | 0.738 | 0.951 | 0.967 | 0.828 | 3 ms |
| Hybrid (RRF) | 0.787 | 0.984 | 0.984 | 0.873 | 94 ms |
| **Hybrid + cross-encoder rerank** | **0.885** | **0.967** | **0.984** | **0.920** | 336 ms |

**Answers.** Overall accuracy counts correct answers to answerable questions plus unanswerable questions that are refused or answered "the documents do not state this". The final system has small-to-big page context and number grounding; files are `results/*answers-v2-*` and `results/*answers-v3-heldout2*`.

| Set | Questions | Correct, LLM judge | Unanswerable handled | **Overall (judge)** | Overall (manual review) | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Development (`golden.yaml`) | 61 + 9 | 60/61 | 9/9 | **98.6%** | 96–97% ¹ | 2.20 s / 3.74 s |
| Held-out 1 (`heldout.yaml`) | 22 + 3 | 20/22 | 3/3 | **92%** | 96% ² | 1.92 s / 3.16 s |
| Held-out 2 (`heldout2.yaml`) | 21 + 4 | 20/21 | 4/4 | **96%** | 96% | 1.86 s / 3.44 s |
| Both held-out sets | 50 | | | **94%** | **96%** | |

Other measures:
- **Invented citations:** 0 in every run.
- **Unanswerable questions answered with an invented fact:** 0 in the final runs.
- **"Every claim stated in the cited passage"** (judge, not calibrated): 0.87 / 0.91 / 0.86 on the three sets.

¹ The judge passed FM04 (it gives the paper form's "block letters" rule instead of the e-form's "uppercase only" rule), and PR01 is borderline. Both are counted as wrong in the manual column.
² The judge failed H04, whose answer (valid 3 years, first stay up to 1 year) is correct; it only omits the words "multiple-entry".

**How the held-out sets were used.** Each set was committed before the system it measures was run on it (git history: `36a96d1`, `fffe16c`), so first scores are not tuned. The first scoring of each one exposed failure classes that were then fixed:
- **Held-out 1, first scored at 88% (judge):**
  - The model quoted a minimum investment from memory, and number grounding was added.
  - Whole-page context hurt the 89-page apostille directory, so small-to-big is now limited to documents of 20 pages or fewer.
  - One label was wrong: H22 does have an answer in the source (`b3df5e2`).
- **Held-out 2, first scored at 92% (judge):**
  - K24 applied a D-3 TOPIK rule to D-4-1. Number grounding, added after held-out 1, already fixes it.
  - Then a false catch had to be fixed: a year repeated from the question.

So the honest figure for unseen questions is **about 92–96%**. The numbers in the table come after those fixes.

**Remaining errors.**
- **K05:** the model picks the wrong case on the E-3 page.
- **VF04:** the model misreads which countries a notice covers.
- **H18:** the Macao entry of the apostille directory has no country name in its chunk, so the model refuses.

**Generator choice** (PDF-only index, 45 questions, `results/20261124-0248-*-4b-v2*` vs `results/20261124-0228-*-9b-v2*`): qwen3.5:4b scored 0.978 correct, 2.25 s p50. qwen3.5:9b scored 0.956 correct, 20.9 s p50, because the reranker had to run on the CPU (the 9B model needs 8.9 GB).

**Corpus**

| | |
|---|---:|
| Attachment links found on visa.go.kr (English and Korean sides, 3 boards and 34 static menus) | 536 |
| Unique PDFs downloaded (the other 235 links are HWP, XLSX, DOC, images and archives) | 107 |
| English PDFs, indexed | 40 (253 pages, 810 chunks) |
| English per-visa guide pages, indexed | 110 visa codes (731 chunks) |
| Unit and data tests | `uv run pytest`: 17 passed |

> **Caveat: read before trusting the answer scores.** The golden questions and the judge-calibration labels were written by an LLM (Claude) while it read the cited sources. A human did not write them, and no human has reviewed them yet. The chunking fixes, the prompt rewrite, the amount guard and one reference-answer correction (FM05) were all made after looking at these questions, so every number is in-sample. A fresh held-out set has not been scored yet. The visa guide pages show requirements as the portal displays them for one nationality, mostly US applicants. Every page chunk says so, and the answers carry that note.

## How it works

```mermaid
flowchart LR
    P["visa_pages.py<br/>per-visa guide pages<br/>110 visa codes (EN)"] --> PC["pages.py<br/>leaf sections, full heading path,<br/>nationality note"]
    PC --> D
    A["crawl.py<br/>visa.go.kr boards + menus<br/>EN and KO sides, 1 req/s"] --> B["parse.py<br/>Docling layout + TableFormer<br/>OCR only for scanned pages<br/>language tagging"]
    B --> C["index.py<br/>HybridChunker (384 tokens)<br/>heading trail + orphan recovery<br/>bge-m3 embeddings"]
    C --> D[("LanceDB<br/>vectors + BM25 FTS")]
    Q["Question (English)"] --> R["retrieve.py<br/>dense top 30 + BM25 top 30<br/>RRF fusion<br/>bge-reranker-v2-m3"]
    D --> R
    R -->|"top score < 0.02"| X["Refuse without calling the LLM"]
    R --> G["answer.py<br/>qwen3.5:4b on Ollama<br/>numbered passages, [n] after each claim<br/>NOT_FOUND contract"]
    G --> V["Checks in code<br/>invalid [n] → regenerate once<br/>amount guard, old-source flag"]
    V --> API["FastAPI /v1/ask + web UI<br/>citations open the PDF at the page"]
```

1. **Crawl.** The portal is a JSP application. Its boards only return a detail page when the form post carries `TRAN_TYPE=ComSubmit`. Without that field the server silently returns the list page again. Attachments come from `POST /cmm/downloadFile.do`. The crawler covers both language sides and every public menu. It stores PDFs by content hash, so the same file linked from 20 posts is downloaded once. Every link, PDF or not, is recorded in `data/raw/manifest.jsonl`.
2. **Parse.** Docling runs layout analysis and TableFormer table structure, and keeps the page number and bounding box of every item. OCR runs only on documents whose text layer is mostly empty. Language is decided per document: Hangul share for Korean, and `langdetect` on the non-Hangul text to tell English from Vietnamese or Russian, which are also written in Latin or Cyrillic script. Of 107 PDFs, 40 are English or bilingual Korean/English.
3. **Chunk.** Docling's HybridChunker splits by document structure first and by the embedding model's own token budget second. Two fixes were needed on these PDFs:
   - **Orphan recovery.** A heading with no body, such as *"5. Effective Date: March 2, 2016"*, appeared in no chunk at all, so the fact was unretrievable.
   - **Heading trail.** Many of these PDFs have flat heading levels, so a chunk kept only *"Contact details:"* and lost *"Bulgaria – Competent Authority"*. Each chunk now carries the heading before its own.

   Korean and Chinese halves of bilingual forms are dropped at chunk level. Documents posted only with Korean titles get a short English title written once by the local LLM.
4. **Retrieve.** Dense retrieval (bge-m3, run in-process) and BM25 (LanceDB native full-text search) each take the top 30, fused by reciprocal rank fusion. Then the bge-reranker-v2-m3 cross-encoder re-scores the fused list. If even the best passage scores below 0.02, the system refuses without calling the LLM: 4 of the 7 unanswerable questions are refused in about 0.3 s this way. The LLM itself refuses the other 3.
5. **Visa guide pages.** `visa_pages.py` (ported from the author's earlier project) collects the portal's per-visa pages through its visa-wizard endpoints. `pages.py` turns each leaf section ("Required documents", "Visa validity…") into a chunk. Each chunk carries its full heading path, such as *Embassy visa application — Case 2 > Required documents*, the visa code and title, and a note on which nationality the page was shown for.
6. **Generate.** qwen3.5:4b gets 6 numbered passages, each with title, page and date. It must put a `[n]` after every factual sentence, or reply exactly `NOT_FOUND`. Code then checks every marker. A marker outside `1..k` is an invented citation, so the answer is regenerated once, and any invalid marker still left is removed and flagged. The response reports citation coverage, a flag for sources older than 3 years, and per-stage timings.

## Design decisions

**Hybrid + rerank instead of dense only.** These documents are full of literal tokens: visa codes (`F-2-5`), amounts (`KRW 500,000`), country lists and article numbers. On this corpus BM25 alone beats dense retrieval at every cutoff (top 5: 0.978 vs 0.933). Fusion alone ranks slightly better than BM25 (MRR 0.863 vs 0.848) but finds the right page in the top 5 a little less often. The cross-encoder is what pays: it puts the right page first for 0.867 of questions, against 0.778 for fusion, and costs about 250 ms on the GPU.

**The embedder runs in-process on the CPU.** The first version embedded queries through Ollama. Ollama then had to hold the embedder next to the generator, and the 8 GB GPU overflowed: `ollama ps` showed the generator split **30% CPU / 70% GPU**, and generation took 8–11 s. Embedding one short query on the CPU costs about 90 ms. Moving it there frees the GPU for the generator and the reranker, and generation dropped to about 2 s.

**qwen3.5:4b, not 9b.** Both ran on the same 52 questions with the same retrieval, and the larger model was not better. Correctness was 0.956 vs 0.978, every-claim-supported was 0.844 vs 0.956, and the 9B is about 9× slower on this card because it does not fit next to the reranker. A bigger generator is the wrong lever here. The errors that remain are table-reading errors, and both models make them.

**Plain-text citations, not JSON.** See "What did not work".

**Embedder and reranker models.** bge-m3 and bge-reranker-v2-m3 were chosen because they were already on the machine, and the disk had only a few GB free. Newer models (Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, which their model cards report as stronger) were not A/B tested. That is the next retrieval experiment.

**What is not indexed.** 67 non-English PDFs (Korean, plus Vietnamese, Russian, Thai and Chinese versions of notices) and 235 non-PDF attachments (98 HWP, 79 XLSX, 31 DOC, 23 images, 4 other) are downloaded and listed in the manifest but not indexed, because the system is English-only by design. Several of them, such as the Korean visa manuals, hold substantive rules that are missing from the English side of the portal.

## What did not work
- **A self-verification pass.** After answering, the same 4B model checked each sentence against its cited passages and dropped the unsupported ones. This dropped correct content too: 56/61 correct with it, against 58/61 without. It is kept as an option (`VISARAG_VERIFY=true`) and is off by default. The two changes that did help are deterministic: small-to-big page context, which fixed the scrambled-table and wrapped-line errors, and number grounding, under which every number in a sentence must appear in the passages it cites.
- **Prompt rules against borrowing a number from another document.** With the visa pages added, "How much does a D-2 visa cost?" retrieved D-2 pages that list "fee" as a required item but give no amount. The 4B model filled the gap with the 50,000 KRW re-entry permit fee from a notice that mentions D-2 only in its exemption list. A stronger prompt rule did not stop it, and it made another answer worse (FM04) and lowered citation precision (0.832 → 0.777), so it was reverted. What works is a check in code (`amount_guard`). When the question names a visa code, every money amount must come from a cited passage about that code: its title or section names the code, or the amount appears on the same line as the code. Otherwise the answer becomes "the documents do not state this amount for D-2". On the evaluation it fired once, on that question, and never on an answerable one.

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
uv run python -m visarag.visa_pages   # per-visa guide pages -> data/pages/en (cached raw responses are reused)
make index     # chunks (PDFs + visa pages) + bge-m3 + LanceDB (vector + BM25)
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
src/visarag/   crawl, parse, visa_pages, pages, index, embed, retrieve, answer, ollama, api, config, static/index.html
eval/          golden.yaml (70 Q), run.py, judge_cases.yaml (55 cases), calibrate.py, resummarize.py
results/       every measured run, as JSON/JSONL
tests/         unit tests (parsers, language tagging, citations, RRF) + golden-set integrity
```

## Open items

1. **A human review of `eval/golden.yaml` and `eval/judge_cases.yaml`.** This is the largest methodological gap.
2. **Korean-only documents.** The richest rules (visa manuals) are Korean PDFs and HWP files. Indexing them needs cross-lingual retrieval and translation, which this English-only version leaves out by design.
3. **The corpus is old.** Most English notices are from 2014 to 2022. Every answer shows its source date, and the API flags sources older than three years.
