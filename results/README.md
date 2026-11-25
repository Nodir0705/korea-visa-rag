# Results

Every measured run is kept. A run is a `*.jsonl` file of answers (or a `*-retrieval-*.json`
table), plus `*.summary.json` (the numbers `eval/run.py` prints) and, where a judge ran,
`*.judged-qwen3.5_9b.jsonl` with one verdict per answer and its own `.summary.json`.
File names are `YYYYMMDD-HHMM-<what>-<tag>`.

## Runs behind the README tables (final system)

| What | Files |
|---|---|
| Development set, answers + judge | `20261124-1745-answers-v2-golden*` |
| Held-out 1, answers + judge | `20261124-1748-answers-v2-heldout*` |
| Held-out 2, answers + judge | `20261124-2012-answers-v3-heldout2*` |
| Retrieval, development set (four retrieval modes) | `20261124-1342-retrieval-pages.json` |
| Retrieval, held-out sets (hybrid + rerank) | `20261124-1745-retrieval-v2-heldout.json`, `20261124-1713-retrieval-final-heldout2.json` |
| Source check of all 120 answers against the corpus (LLM reviewers, evidence per item) | `20261125-source-check.json` |
| Judge calibration (qwen3.5:9b vs mistral-nemo) | `judge-calibration-20261124-0329.json` (qwen3.5:9b) and `-0259.json` (mistral-nemo row) |

## First scoring of the held-out sets (before the fixes they exposed)

| Set | Files |
|---|---|
| Held-out 1 | `20261124-1658-answers-heldout*`, `20261124-1658-retrieval-heldout.json` |
| Held-out 2 | `20261124-1713-answers-final-heldout2*` |

## How the system got there (development set, in order)

| Step | Files |
|---|---|
| PDF-only corpus, first prompt | `20261124-0213-retrieval-bge-m3.json`, `20261124-0217-answers-qwen35-4b*` |
| Chunking fixes (orphan headings, heading trail) | `20261124-0216-retrieval-bge-m3-v2.json` |
| Shorter prompt; 4B vs 9B generator | `20261124-0248-answers-qwen35-4b-v2*`, `20261124-0228-answers-qwen35-9b-v2*` |
| Table captions, scoped heading trail | `20261124-0405-*-v3*` |
| Per-visa guide pages added | `20261124-1343-answers-qwen35-4b-pages*` |
| Prompt rule against borrowed amounts (reverted) | `20261124-1404-answers-qwen35-4b-pages-v2*` |
| Amount guard in code | `20261124-1419-answers-qwen35-4b-pages-v3*` |
| Small-to-big context (A) vs + self-verification (B) | `20261124-1554-answers-A-expand*`, `20261124-1556-answers-B-expand-verify*` |
| Line unwrapping, two-page expansion | `20261124-1637-answers-A2-final*` |
| All three sets before number grounding | `20261124-17*-final-*` |

The qwen3.5:9b rows in `judge-calibration-20261124-0259.json` and `-0323.json` are failed
runs: qwen3.5 ignored the JSON schema and every verdict parsed as an error (0/12, 0/43). The
parser was fixed and the calibration re-run in `-0329.json`. The mistral-nemo row in `-0259`
is valid.
