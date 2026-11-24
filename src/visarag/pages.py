"""Chunk the visa.go.kr per-visa guide pages (second source, next to the PDFs).

The pages come from ``visa_pages.py`` as markdown with YAML front matter, one
file per visa code (``data/pages/en/D-2-1.md``). Each page is a strict outline:

    # D-2-1 Associate Degree
    ## Embassy visa application
    ### Embassy visa application — Case 1      (only when there are several cases)
    #### Eligibility / Required documents / Visa validity and period of stay

A chunk is one leaf section. Long sections (e.g. the F-6-1 document list) are
packed line by line up to the token budget, and every piece repeats the full
heading path, so "Required documents" is never separated from the visa code and
the case it belongs to.

The portal answers per (nationality, purpose, duration). The page records which
nationality it was fetched for; that is written into every chunk, because
requirements can differ by nationality and by embassy.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from visarag.config import settings

PAGES_DIR = settings.data_dir / "pages" / "en"
NATIONALITY = {"275": "the United States", "112": "China", "185": "Vietnam", "404": "Australia"}
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def read_page(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        _, front, body = text.split("---", 2)
        return yaml.safe_load(front) or {}, body
    return {}, text


def sections(body: str) -> list[tuple[list[str], list[str]]]:
    """(heading path below the page title, body lines) for every section that has text."""
    stack: list[tuple[int, str]] = []
    out: list[tuple[list[str], list[str]]] = []
    lines: list[str] = []

    def flush() -> None:
        content = [ln for ln in lines if ln.strip()]
        if content:
            out.append(([h for lvl, h in stack if lvl >= 2], content))
        lines.clear()

    for line in body.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, m.group(2).strip())]
        else:
            lines.append(line.rstrip())
    flush()
    return out


def pack(lines: list[str], count_tokens, budget: int) -> list[str]:
    pieces, cur = [], []
    for ln in lines:
        if cur and count_tokens("\n".join([*cur, ln])) > budget:
            pieces.append("\n".join(cur))
            cur = []
        cur.append(ln)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def page_chunks(count_tokens, budget: int | None = None) -> list[dict]:
    budget = budget or settings.chunk_max_tokens
    chunks: list[dict] = []
    for path in sorted(PAGES_DIR.glob("*.md")):
        front, body = read_page(path)
        code = front.get("visa_code") or path.stem
        title = f"Visa guide: {front.get('title') or code}"
        nat = NATIONALITY.get(str(front.get("nat_cd", "")), "")
        note = (
            f"visa.go.kr visa guide page, requirements as shown for applicants from {nat}; "
            "they can differ by nationality and embassy"
            if nat else "visa.go.kr visa guide page; requirements can differ by nationality and embassy"
        )
        i = 0
        for path_h, lines in sections(body):
            trail = " > ".join(path_h)
            # Room for the title, note and heading path that are repeated in every piece.
            room = budget - count_tokens(f"{title}\n{note}\n{trail}")
            for piece in pack(lines, count_tokens, max(room, 64)):
                context = "\n".join(x for x in (title, f"({note})", trail, piece) if x)
                chunks.append(
                    {
                        "chunk_id": f"page-{code}-{i:03d}",
                        "doc_id": f"page-{code}",
                        "title": title,
                        "file_name": path.name,
                        "post_date": "",
                        "source_url": front.get("url", ""),
                        "pdf_path": "",
                        "pages": [],
                        "page": 0,
                        "headings": trail,
                        "text": context,
                        "body": piece,
                        "n_tokens": count_tokens(context),
                        "source": "page",
                        "note": f"{note}; fetched {front.get('fetched_at', '')}",
                    }
                )
                i += 1
    return chunks
