"""Parse downloaded PDFs with Docling and decide which ones are English.

For every unique PDF in ``data/raw/manifest.jsonl``:

1. Probe the text layer with pypdfium2 (characters per page). Pages with almost no
   text are scanned images; only documents that need it get OCR, because OCR is
   slow and the layout model alone is enough for born-digital government PDFs.
2. Convert with Docling (layout analysis + TableFormer table structure). The
   result keeps, for every text item and table, the page number and bounding box
   it came from, which is what the citations point at later.
3. Detect the language of the document's text. Korean is measured by Hangul
   share; Latin-script languages (English vs Vietnamese, for example) by
   langdetect over the non-Hangul text. Bilingual Korean/English forms count as
   English when their English text alone is substantial.

Output:
    data/parsed/<sha12>.json         DoclingDocument (lossless, re-chunkable)
    data/parsed/catalog.jsonl        one row per PDF: title, pages, language, decision
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pypdfium2 as pdfium
from langdetect import DetectorFactory, LangDetectException, detect_langs

from visarag.config import settings

DetectorFactory.seed = 0  # langdetect is randomized; make it reproducible
log = logging.getLogger("visarag.parse")

HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
LATIN = re.compile(r"[A-Za-z]")
SCANNED_CHARS_PER_PAGE = 30
MIN_ENGLISH_LETTERS = 300  # a bilingual form needs at least this much English to be indexed


@dataclass
class CatalogRow:
    sha12: str
    file_name: str
    post_title: str
    post_date: str
    board: str
    source_url: str
    pages: int
    chars_per_page: float
    scanned_pages: int
    hangul_share: float
    latin_letters: int
    language: str
    english: bool
    reason: str
    parsed_path: str | None = None
    parse_seconds: float | None = None


def probe_text(pdf: Path) -> tuple[int, list[str]]:
    doc = pdfium.PdfDocument(pdf)
    try:
        texts = [page.get_textpage().get_text_range() for page in doc]
    finally:
        doc.close()
    return len(texts), texts


def classify_language(text: str) -> tuple[str, float, int, bool, str]:
    """(language, hangul share of letters, latin letters, is_english, reason)."""
    hangul = len(HANGUL.findall(text))
    latin = len(LATIN.findall(text))
    letters = hangul + latin
    share = hangul / letters if letters else 0.0
    non_korean = HANGUL.sub(" ", text)
    try:
        langs = detect_langs(non_korean) if latin >= 20 else []
    except LangDetectException:
        langs = []
    top = langs[0].lang if langs else ("ko" if hangul else "unknown")
    if latin < MIN_ENGLISH_LETTERS:
        return (top if latin else "ko"), share, latin, False, f"too little Latin text ({latin} letters)"
    if top != "en":
        return top, share, latin, False, f"Latin text detected as '{top}'"
    if share > 0.5:
        return "ko+en", share, latin, True, "bilingual Korean/English, English part indexed"
    return "en", share, latin, True, "English"


def converter(ocr: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def unique_pdfs() -> list[dict]:
    """First manifest row per PDF, preferring the English side of the portal."""
    rows = [json.loads(line) for line in (settings.raw_dir / "manifest.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r.get("sha256")]
    rows.sort(key=lambda r: (r.get("site_lang", "EN") != "EN", r["post_date"] == ""))
    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(r["sha256"], r)
    return list(seen.values())


def run(force: bool = False, only: str | None = None) -> list[CatalogRow]:
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    plain, with_ocr = None, None
    catalog: list[CatalogRow] = []
    for r in unique_pdfs():
        sha12 = r["sha256"][:12]
        if only and sha12 != only:
            continue
        pdf = settings.root / r["local_path"]
        n_pages, texts = probe_text(pdf)
        joined = "\n".join(texts)
        scanned = sum(len(t.strip()) < SCANNED_CHARS_PER_PAGE for t in texts)
        lang, share, latin, english, reason = classify_language(joined)
        row = CatalogRow(
            sha12=sha12,
            file_name=r["file_name"],
            post_title=r["post_title"],
            post_date=r["post_date"],
            board=r["board"],
            source_url=r["post_url"],
            pages=n_pages,
            chars_per_page=round(len(joined) / max(n_pages, 1), 1),
            scanned_pages=scanned,
            hangul_share=round(share, 3),
            latin_letters=latin,
            language=lang,
            english=english,
            reason=reason,
        )
        # A scanned document has no text layer to classify; OCR it first, then decide.
        needs_ocr = scanned > n_pages / 2
        if english or needs_ocr:
            out = settings.parsed_dir / f"{sha12}.json"
            if force or not out.exists():
                t0 = time.perf_counter()
                if needs_ocr:
                    with_ocr = with_ocr or converter(ocr=True)
                    conv = with_ocr
                else:
                    plain = plain or converter(ocr=False)
                    conv = plain
                doc = conv.convert(pdf).document
                doc.save_as_json(out)
                row.parse_seconds = round(time.perf_counter() - t0, 2)
                if needs_ocr:
                    lang, share, latin, english, reason = classify_language(doc.export_to_text())
                    row.language, row.hangul_share, row.latin_letters = lang, round(share, 3), latin
                    row.english, row.reason = english, f"OCR: {reason}"
            row.parsed_path = str(out.relative_to(settings.root))
        catalog.append(row)
        log.info("%-5s %-6s %3dp %s", "EN" if row.english else "skip", row.language, n_pages, row.file_name[:70])
    with (settings.parsed_dir / "catalog.jsonl").open("w", encoding="utf-8") as fh:
        for row in catalog:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return catalog


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse PDFs with Docling and tag their language")
    ap.add_argument("--force", action="store_true", help="re-parse PDFs already parsed")
    ap.add_argument("--only", help="parse one PDF by its sha12")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("docling", "httpx", "RapidOCR", "easyocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cat = run(force=args.force, only=args.only)
    eng = [c for c in cat if c.english]
    print(json.dumps({
        "pdfs": len(cat),
        "english_indexed": len(eng),
        "english_pages": sum(c.pages for c in eng),
        "skipped": len(cat) - len(eng),
    }, indent=2))


if __name__ == "__main__":
    main()
