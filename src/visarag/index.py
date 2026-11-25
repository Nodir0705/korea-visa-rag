"""Chunk the parsed English documents and build the hybrid index.

Chunking uses Docling's HybridChunker: it starts from the document structure
(sections, list items, tables), splits only what exceeds the token budget of the
embedding model's own tokenizer, and merges undersized neighbours that share a
heading. Each chunk is embedded *with* its heading path and document title
prefixed (``contextualize``), which a 2026 study found matches per-chunk LLM
enrichment at no cost; the bare text is kept separately for display.

Every chunk carries the page numbers its items came from, so an answer can cite
"<document>, p. N" and link to that page of the original PDF.

Index: one LanceDB table (embedded, no server) with the dense vector column and
a native full-text (BM25) index over the contextualized text. Dense and sparse
results are fused with reciprocal rank fusion at query time (retrieve.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path

import lancedb
from lancedb.index import FTS

from visarag.config import settings
from visarag.embed import Embedder

log = logging.getLogger("visarag.index")
TABLE = "chunks"


def _chunker():
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    tok = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(settings.embed_tokenizer),
        max_tokens=settings.chunk_max_tokens,
    )
    return HybridChunker(tokenizer=tok, merge_peers=True)


def english_docs() -> list[dict]:
    rows = [json.loads(x) for x in (settings.parsed_dir / "catalog.jsonl").open(encoding="utf-8")]
    return [r for r in rows if r["english"] and r["parsed_path"]]


LATIN = re.compile(r"[A-Za-z]")
HANGUL = re.compile(r"[가-힣]")


def _not_english(text: str) -> bool:
    """Korean or Chinese half of a bilingual form: Latin letters are under 40% of the text.

    A CJK character carries roughly a word's worth of meaning, so each one is weighted
    x3 against a single Latin letter before comparing.
    """
    latin = len(LATIN.findall(text))
    other = sum(c.isalpha() for c in text) - latin
    return latin == 0 or latin / (latin + 3 * other) < 0.4


def _english_title(d: dict, doc) -> str:
    """Post titles from the Korean side of the portal are Korean; prefer an English heading."""
    for cand in (d["post_title"], Path(d["file_name"]).stem):
        if len(cand) > 8 and not HANGUL.search(cand):
            return cand
    for item, _ in doc.iterate_items():
        text = (getattr(item, "text", "") or "").strip()
        if 12 <= len(text) <= 120 and not HANGUL.search(text) and len(LATIN.findall(text)) > 8:
            return text
    return Path(d["file_name"]).stem


def build_chunks() -> list[dict]:
    from docling_core.types.doc import DoclingDocument

    chunker = _chunker()
    titles = TitleCache(settings.parsed_dir / "titles.json")
    chunks: list[dict] = []
    for d in english_docs():
        doc = DoclingDocument.load_from_json(settings.root / d["parsed_path"])
        title = titles.get(d, doc)
        order, headers = _reading_order(doc)
        covered: set[str] = set()
        pieces: list[tuple[list, list[str], str]] = []  # (doc_items, headings, body)
        for ch in chunker.chunk(doc):
            covered |= {it.self_ref for it in ch.meta.doc_items}
            first = min((order.get(it.self_ref, 0) for it in ch.meta.doc_items), default=0)
            pieces.append((ch.meta.doc_items, _heading_trail(headers, first, ch.meta.headings), ch.text.strip()))
        pieces += _orphans(doc, order, headers, covered)
        pieces.sort(key=lambda p: min((order.get(it.self_ref, 0) for it in p[0]), default=0))
        pieces = _attach_table_captions(pieces, order)
        for i, (items, headings, body) in enumerate(pieces):
            if len(body) < 20 or _not_english(body):
                continue  # Korean/Chinese half of a bilingual form: not indexed in the English system
            pages = sorted({p.page_no for it in items for p in it.prov})
            context = "\n".join(x for x in (title, " > ".join(headings), body) if x)
            chunks.append(
                {
                    "chunk_id": f"{d['sha12']}-{i:04d}",
                    "doc_id": d["sha12"],
                    "title": title,
                    "file_name": d["file_name"],
                    "post_date": d["post_date"],
                    "source_url": d["source_url"],
                    "pdf_path": f"data/raw/pdf/{d['sha12']}.pdf",
                    "pages": pages,
                    "page": pages[0] if pages else 0,
                    "headings": " > ".join(headings),
                    "text": context,  # what is embedded and BM25-indexed
                    "body": body,  # what is shown and cited
                    "n_tokens": chunker.tokenizer.count_tokens(context),
                    "source": "pdf",
                    "note": "",
                }
            )
    if settings.include_pages:
        from visarag.pages import page_chunks

        chunks += page_chunks(chunker.tokenizer.count_tokens)
    return chunks


HEADER_LABELS = {"section_header", "title"}


def _reading_order(doc) -> tuple[dict[str, int], list[tuple[int, str]]]:
    """Position of every item in reading order, and every heading with its position."""
    order: dict[str, int] = {}
    headers: list[tuple[int, str]] = []
    for pos, (item, _) in enumerate(doc.iterate_items()):
        order[item.self_ref] = pos
        if str(getattr(item, "label", "")).split(".")[-1].lower() in HEADER_LABELS:
            text = (getattr(item, "text", "") or "").strip()
            if text and not _not_english(text):
                headers.append((pos, text))
    return order, headers


def _heading_trail(headers: list[tuple[int, str]], pos: int, own: list[str] | None) -> list[str]:
    """The chunk's headings, plus the subject heading when the nearest one is only a field label.

    Many of these PDFs have flat heading levels, so Docling keeps only the nearest
    heading. When that heading is a field label ending in a colon ("Contact details:"),
    the subject ("Bulgaria - Competent Authority") is lost, so the last heading that is
    not a label is put in front of it. Headings that name a subject are left alone:
    carrying the previous heading unconditionally put the TB country list under
    "[Note] Exceptions (... exempt from submission)", and the model then read the
    countries as exempt.
    """
    own = list(own or [])
    before = [t for p, t in headers if p < pos]
    if not own and before:
        own = [before[-1]]
    if own and own[-1].rstrip().endswith(":"):
        subject = next((t for t in reversed(before) if not t.rstrip().endswith(":") and t not in own), None)
        if subject:
            own = [subject, *own]
    return own[-3:]


def _is_table(item) -> bool:
    return str(getattr(item, "label", "")).split(".")[-1].lower() == "table"


def _attach_table_captions(pieces: list[tuple[list, list[str], str]], order: dict[str, int]) -> list:
    """Move a short lead-in line ("The countries still eligible ... are as follows.") from the
    end of one chunk to the start of the next when the next chunk starts with a table.
    Without this, the table's chunk has no caption, and the caption sits in the
    previous chunk, attached to the wrong list."""
    out = [list(p) for p in pieces]
    for i in range(len(out) - 1):
        items, _, body = out[i]
        nxt_items = out[i + 1][0]
        if not items or not nxt_items:
            continue
        first_next = min(nxt_items, key=lambda it: order.get(it.self_ref, 0))
        last = max(items, key=lambda it: order.get(it.self_ref, 0))
        text = (getattr(last, "text", "") or "").strip()
        if _is_table(first_next) and not _is_table(last) and 0 < len(text) <= 200 and body.rstrip().endswith(text):
            out[i][2] = body.rstrip()[: -len(text)].rstrip()
            out[i][0] = [it for it in items if it is not last]
            out[i + 1][2] = f"{text}\n{out[i + 1][2]}"
            out[i + 1][0] = [last, *nxt_items]
    return [tuple(p) for p in out]


def _orphans(doc, order, headers, covered) -> list[tuple[list, list[str], str]]:
    """Text the chunker emitted in no chunk (e.g. a heading with no body under it,
    like "5. Effective Date: March 2, 2016"), grouped per page."""
    groups: dict[int, list] = {}
    for item, _ in doc.iterate_items():
        text = (getattr(item, "text", "") or "").strip()
        if item.self_ref in covered or len(text) < 10 or not getattr(item, "prov", None):
            continue
        groups.setdefault(item.prov[0].page_no, []).append(item)
    out = []
    for _, items in sorted(groups.items()):
        pos = order.get(items[0].self_ref, 0)
        out.append((items, _heading_trail(headers, pos, None), "\n".join(i.text.strip() for i in items)))
    return out


class TitleCache:
    """English display titles. Posts from the Korean side of the portal carry Korean
    titles; for those, the local LLM writes a short English title from the first page.
    Cached in data/parsed/titles.json so it runs once per document."""

    PROMPT = (
        "Write a short English title (at most 12 words) for this official Korean immigration "
        "document, based on its text below. Reply with the title only.\n\nFile name: {name}\n\n{text}"
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, d: dict, doc) -> str:
        if len(d["post_title"]) > 8 and not HANGUL.search(d["post_title"]):
            return d["post_title"]
        if d["sha12"] not in self.data:
            from visarag.ollama import Ollama

            text = "\n".join(
                t for it, _ in doc.iterate_items() if (t := (getattr(it, "text", "") or "").strip())
                and not _not_english(t)
            )[:2000]
            reply = Ollama().chat(
                settings.gen_model,
                [{"role": "user", "content": self.PROMPT.format(name=d["file_name"], text=text)}],
                num_predict=40,
            )["message"]["content"]
            title = reply.strip().strip('"').splitlines()[0][:120] if reply.strip() else ""
            self.data[d["sha12"]] = title or _english_title(d, doc)
            self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        return self.data[d["sha12"]]


class EmbeddingCache:
    """SQLite cache keyed by (model, sha256(text)): re-indexing only embeds new text."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (model TEXT, h TEXT, v TEXT, PRIMARY KEY (model, h))")

    def get_many(self, model: str, texts: list[str], embed_fn) -> list[list[float]]:
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        found = {
            h: json.loads(v)
            for h, v in self.db.execute(
                f"SELECT h, v FROM emb WHERE model=? AND h IN ({','.join('?' * len(hashes))})",
                [model, *hashes],
            )
        } if hashes else {}
        missing = [(h, t) for h, t in zip(hashes, texts) if h not in found]
        if missing:
            vecs = embed_fn([t for _, t in missing])
            self.db.executemany(
                "INSERT OR REPLACE INTO emb VALUES (?,?,?)",
                [(model, h, json.dumps(v)) for (h, _), v in zip(missing, vecs)],
            )
            self.db.commit()
            found.update({h: v for (h, _), v in zip(missing, vecs)})
        return [found[h] for h in hashes]


def table_name(model: str | None = None) -> str:
    model = model or settings.embed_model
    return f"{TABLE}_{re.sub(r'[^a-z0-9]+', '_', model.lower()).strip('_')}"


def build(model: str | None = None) -> dict:
    model = model or settings.embed_model
    t0 = time.perf_counter()
    chunks = build_chunks()
    t_chunk = time.perf_counter() - t0
    cache = EmbeddingCache(settings.index_dir / "embeddings.sqlite")
    embedder = Embedder(model)
    t1 = time.perf_counter()
    vecs = cache.get_many(model, [c["text"] for c in chunks], embedder.documents)
    t_embed = time.perf_counter() - t1
    for c, v in zip(chunks, vecs):
        c["vector"] = v
    db = lancedb.connect(settings.index_dir / "lancedb")
    name = table_name(model)
    tbl = db.create_table(name, data=chunks, mode="overwrite")
    tbl.create_index("text", config=FTS(with_position=False, stem=True, remove_stop_words=True), replace=True)
    stats = {
        "table": name,
        "embed_model": model,
        "documents": len({c["doc_id"] for c in chunks}),
        "pdf_documents": len({c["doc_id"] for c in chunks if c["source"] == "pdf"}),
        "visa_pages": len({c["doc_id"] for c in chunks if c["source"] == "page"}),
        "chunks": len(chunks),
        "pdf_chunks": sum(c["source"] == "pdf" for c in chunks),
        "dim": len(vecs[0]) if vecs else 0,
        "mean_tokens": round(sum(c["n_tokens"] for c in chunks) / max(len(chunks), 1), 1),
        "max_tokens": max((c["n_tokens"] for c in chunks), default=0),
        "chunk_seconds": round(t_chunk, 1),
        "embed_seconds": round(t_embed, 1),
    }
    (settings.index_dir / f"{name}.stats.json").write_text(json.dumps(stats, indent=2))
    built_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (settings.index_dir / "build.json").write_text(json.dumps({"built_at": built_at, **stats}, indent=2))
    write_page_texts()
    return stats


ITEM_START = re.compile(r"^\s*([○□■◎❍※•\-\*<|]|[①-⑳]|\(?\d{1,2}[.)]|\([A-Za-z][^)]{0,30}\)|[A-Za-z][.)]\s|Article\s+\d)")


def unwrap(text: str) -> str:
    """Join the hard line wraps of a PDF text layer ("Ho Chi Minh City, Da" / "Nang") while
    keeping real breaks: a line break stays when the next line starts a list item or
    the current line ends a sentence or a label."""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out: list[str] = []
    for ln in lines:
        if not ln:
            continue
        if out and not ITEM_START.match(ln) and not re.search(r"[.:;!?)\]>]$", out[-1]):
            out[-1] = f"{out[-1]} {ln}"
        else:
            out.append(ln)
    return "\n".join(out)


def write_page_texts() -> None:
    """Text layer of every page of every indexed PDF, in the PDF's own reading order.

    Used for small-to-big context (answer.py): Docling's layout model sometimes
    reorders a page (on one notice it attached the caption "still eligible" to the
    wrong country table), while the text layer of these born-digital PDFs keeps
    the order the author wrote. Scanned pages have an empty text layer and fall
    back to the chunk.
    """
    import pypdfium2 as pdfium

    out: dict[str, dict[int, str]] = {}
    for d in english_docs():
        pdf = pdfium.PdfDocument(settings.raw_dir / "pdf" / f"{d['sha12']}.pdf")
        try:
            out[d["sha12"]] = {i + 1: unwrap(page.get_textpage().get_text_range()) for i, page in enumerate(pdf)}
        finally:
            pdf.close()
    (settings.index_dir / "page_text.json").write_text(json.dumps(out, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk parsed documents and build the LanceDB hybrid index")
    ap.add_argument("--embed-model", default=None, help=f"st:<hf model> or ollama:<tag> (default {settings.embed_model})")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(build(args.embed_model), indent=2))


if __name__ == "__main__":
    main()
