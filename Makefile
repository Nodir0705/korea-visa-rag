.PHONY: crawl parse index all serve test eval-retrieval eval-answers lint

UV := uv run

crawl:            ## download every attachment linked from visa.go.kr (EN + KO sides), keep PDFs
	$(UV) python -m visarag.crawl

parse:            ## Docling parse + language tagging
	$(UV) python -m visarag.parse

index:            ## chunk, embed, build LanceDB vector + BM25 index
	$(UV) python -m visarag.index

all: crawl parse index

serve:            ## API + web UI on http://localhost:8000
	$(UV) python -m visarag.api

test:
	$(UV) pytest -q

eval-retrieval:
	$(UV) python eval/run.py retrieval

eval-answers:
	$(UV) python eval/run.py answers

lint:
	$(UV) ruff check src eval tests
