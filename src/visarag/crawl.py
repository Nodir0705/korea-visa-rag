"""Download every PDF published on visa.go.kr (English and Korean sides of the portal).

The portal (Korea Visa Portal) is a JSP app whose boards are reached by form
POSTs to ``/openPage.do?MENU_ID=<id>`` with ``TRAN_TYPE=ComSubmit``; without
that field the server silently returns the list page instead of the detail.
Attachments are streamed by ``POST /cmm/downloadFile.do`` with
``ELEMENT_ID=<APND_FILE_SEQ>``. No login or cookie is needed.

Sources crawled (each with ``LANG_TYPE=EN`` and again with ``LANG_TYPE=KO``):

* Forms & Guides (data room)   list 10503   -> detail 1050301 (DATA_ID_DETAIL)
* Notices                      list 10501   -> detail 1050101 (NOTICE_ID_DETAIL)
* Investment-immigration notices list 10406 -> detail 1040601 (NOTICE_ID_DETAIL)
* Every public static menu (34), for attachments and ``/downfile/*.pdf`` links

Output:
    data/raw/pdf/<sha12>.pdf        one file per unique PDF (content-addressed)
    data/raw/manifest.jsonl         one row per attachment seen, PDF or not

Idempotent: a PDF already on disk is not downloaded again. Polite: one
request per second, a descriptive User-Agent, retries with backoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import unquote, unquote_plus

import httpx
from bs4 import BeautifulSoup

from visarag.config import settings

log = logging.getLogger("visarag.crawl")

BASE = "https://www.visa.go.kr"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) visa-rag-research (polite crawler, 1 req/s)"
SLEEP = 1.0
TIMEOUT = 60.0
RETRIES = 3

BOARDS = {
    # name: (list menu, detail menu, row id input, detail id field)
    "forms_guides": ("10503", "1050301", "DATA_ID", "DATA_ID_DETAIL"),
    "notices": ("10501", "1050101", "NOTICE_ID", "NOTICE_ID_DETAIL"),
    "investment_notices": ("10406", "1040601", "NOTICE_ID", "NOTICE_ID_DETAIL"),
}
DOWNFILE_RE = re.compile(r"""["'](/downfile/[^"']+?\.pdf)["']""", re.IGNORECASE)


@dataclass
class Attachment:
    board: str
    post_id: str
    post_title: str
    post_date: str
    category: str
    post_url: str
    file_name: str
    element_id: str | None = None  # None for static /downfile/ links
    static_path: str | None = None
    is_pdf: bool = False
    sha256: str | None = None
    local_path: str | None = None
    bytes: int | None = None
    site_lang: str = "EN"  # which language side of the portal linked it
    status: str = "pending"  # downloaded | cached | duplicate_link | skipped_non_pdf | failed
    error: str | None = None
    extra: dict = field(default_factory=dict)


class Portal:
    def __init__(self, lang: str = "EN") -> None:
        self.lang = lang
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
            follow_redirects=True,
        )
        self._last = 0.0

    def _wait(self) -> None:
        delta = time.monotonic() - self._last
        if delta < SLEEP:
            time.sleep(SLEEP - delta)
        self._last = time.monotonic()

    def post(self, url: str, data: dict) -> httpx.Response:
        payload = {"TRAN_TYPE": "ComSubmit", "LANG_TYPE": self.lang, "SE_FLAG_YN": "N", **data}
        for attempt in range(RETRIES + 1):
            self._wait()
            try:
                r = self.client.post(url, data=payload)
                r.raise_for_status()
                return r
            except httpx.HTTPError as exc:
                if attempt == RETRIES:
                    raise
                log.warning("retry %s after %s", url, exc)
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def get(self, url: str) -> httpx.Response:
        for attempt in range(RETRIES + 1):
            self._wait()
            try:
                r = self.client.get(url)
                r.raise_for_status()
                return r
            except httpx.HTTPError as exc:
                if attempt == RETRIES:
                    raise
                log.warning("retry %s after %s", url, exc)
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def page(self, menu_id: str, **data: str) -> str:
        return self.post(f"{BASE}/openPage.do?MENU_ID={menu_id}", data).text


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_list(html: str, id_input: str) -> list[dict]:
    """Rows of a board list page: id, category, title, date."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.select("table.board_list01 tbody tr"):
        hidden = tr.find("input", attrs={"name": id_input})
        title_a = tr.select_one("td.title a")
        if not hidden or not title_a:
            continue
        tds = [_clean(td.get_text(" ")) for td in tr.find_all("td")]
        date = next((t for t in tds if re.match(r"\d{4}-\d{2}-\d{2}", t)), "")
        rows.append(
            {
                "id": hidden["value"],
                "title": _clean(title_a.get_text(" ")),
                "category": tds[1] if len(tds) > 1 and id_input == "DATA_ID" else "Notice",
                "date": date[:10],
            }
        )
    return rows


def parse_attachments(html: str) -> list[tuple[str, str]]:
    """(element_id, file name) pairs on a detail page."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for inp in soup.find_all("input", attrs={"name": "APND_FILE_SEQ"}):
        a = inp.find_next("a", attrs={"name": "fileNm"})
        name = unquote_plus(_clean(a.get_text())) if a else ""
        out.append((inp["value"], name))
    return out


def crawl_board(portal: Portal, board: str, max_pages: int = 50) -> list[Attachment]:
    list_menu, detail_menu, id_input, detail_field = BOARDS[board]
    found: list[Attachment] = []
    seen_ids: set[str] = set()
    for page_no in range(1, max_pages + 1):
        rows = parse_list(portal.page(list_menu, currentPageNo=str(page_no)), id_input)
        new = [r for r in rows if r["id"] not in seen_ids]
        if not new:
            break
        for row in new:
            seen_ids.add(row["id"])
            html = portal.page(detail_menu, **{detail_field: row["id"], "currentPageNo": "1"})
            post_url = f"{BASE}/openPage.do?MENU_ID={detail_menu}#{detail_field}={row['id']}"
            for element_id, name in parse_attachments(html):
                found.append(
                    Attachment(
                        board=board,
                        post_id=row["id"],
                        post_title=row["title"],
                        post_date=row["date"],
                        category=row["category"],
                        post_url=post_url,
                        file_name=name,
                        element_id=element_id,
                        site_lang=portal.lang,
                    )
                )
            for path in sorted(set(DOWNFILE_RE.findall(html))):
                found.append(_static(board, row, post_url, path, portal.lang))
        log.info("%s page %d: %d posts, %d attachments so far", board, page_no, len(new), len(found))
    return found


def _static(board: str, row: dict, post_url: str, path: str, lang: str) -> Attachment:
    return Attachment(
        board=board,
        post_id=row["id"],
        post_title=row["title"],
        post_date=row["date"],
        category=row["category"],
        post_url=post_url,
        file_name=Path(path).name,
        static_path=path,
        site_lang=lang,
    )


# Every public, non-board menu (PERMISSION_YN=Y in the site's own menu JSON). Swept once per
# language for attachments and /downfile/ links, so nothing linked from a static page is missed.
STATIC_MENUS = (
    "10101", "10102", "10105", "10109", "10108", "10103", "10107", "10106",
    "10205", "10201", "10204", "10207", "10208", "10302",
    "10401", "10402", "10403", "10404", "10407",
    "10502", "10504",
    "11200", "11201", "11202", "11203", "11204", "11205", "11206", "11207", "11208", "11209",
    "112010", "112011", "112012",
)


def crawl_static_pages(portal: Portal) -> list[Attachment]:
    found: list[Attachment] = []
    seen: set[str] = set()
    for menu in STATIC_MENUS:
        try:
            html = portal.page(menu)
        except httpx.HTTPError as exc:
            log.warning("menu %s failed: %s", menu, exc)
            continue
        title = BeautifulSoup(html, "lxml").title
        row = {"id": menu, "title": _clean(title.get_text()) if title else menu, "date": "", "category": "Page"}
        url = f"{BASE}/openPage.do?MENU_ID={menu}"
        for element_id, name in parse_attachments(html):
            if element_id not in seen:
                seen.add(element_id)
                found.append(
                    Attachment(
                        board="static_page", post_id=menu, post_title=row["title"], post_date="",
                        category="Page", post_url=url, file_name=name, element_id=element_id,
                        site_lang=portal.lang,
                    )
                )
        for path in sorted(set(DOWNFILE_RE.findall(html))):
            if path not in seen:
                seen.add(path)
                found.append(_static("static_page", row, url, path, portal.lang))
    log.info("static pages (%s): %d attachments", portal.lang, len(found))
    return found


def _filename_from_headers(resp: httpx.Response) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r"filename\*?=\"?([^\";]+)", cd, re.IGNORECASE)
    return unquote_plus(unquote(m.group(1))) if m else ""


def download(portal: Portal, att: Attachment, pdf_dir: Path) -> None:
    try:
        if att.static_path:
            resp = portal.get(BASE + att.static_path)
        else:
            resp = portal.post(f"{BASE}/cmm/downloadFile.do", {"ELEMENT_ID": att.element_id or ""})
    except httpx.HTTPError as exc:
        att.status, att.error = "failed", str(exc)
        return
    body = resp.content
    header_name = _filename_from_headers(resp)
    if header_name and not att.file_name:
        att.file_name = header_name
    att.bytes = len(body)
    att.is_pdf = body[:5] == b"%PDF-"
    if not att.is_pdf:
        att.status = "skipped_non_pdf"
        att.extra["ext"] = Path(att.file_name or header_name).suffix.lower()
        return
    att.sha256 = hashlib.sha256(body).hexdigest()
    out = pdf_dir / f"{att.sha256[:12]}.pdf"
    att.local_path = str(out.relative_to(settings.root))
    if out.exists():
        att.status = "cached"
        return
    out.write_bytes(body)
    att.status = "downloaded"


def run(boards: list[str], langs: list[str], limit: int | None = None) -> list[Attachment]:
    pdf_dir = settings.raw_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    atts: list[Attachment] = []
    for lang in langs:
        portal = Portal(lang)
        for board in boards:
            atts += crawl_board(portal, board)
        atts += crawl_static_pages(portal)
    if limit:
        atts = atts[:limit]
    # The same file is linked from several posts and from both language sides;
    # fetch each element once.
    portal = Portal(langs[0])
    by_key: dict[str, Attachment] = {}
    for att in atts:
        key = att.element_id or att.static_path or ""
        if key in by_key:
            src = by_key[key]
            att.is_pdf, att.sha256, att.local_path, att.bytes = src.is_pdf, src.sha256, src.local_path, src.bytes
            att.status = "failed" if src.status == "failed" else "duplicate_link"
            continue
        by_key[key] = att
        download(portal, att, pdf_dir)
        log.info("%-15s %s  %s", att.status, att.file_name[:70], att.local_path or "")
    manifest = settings.raw_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for att in atts:
            fh.write(json.dumps(asdict(att), ensure_ascii=False) + "\n")
    return atts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--boards", nargs="+", default=list(BOARDS), choices=list(BOARDS))
    ap.add_argument("--langs", nargs="+", default=["EN", "KO"], choices=["EN", "KO"])
    ap.add_argument("--limit", type=int, help="stop after N attachments (smoke test)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    atts = run(args.boards, args.langs, args.limit)
    pdfs = {a.sha256 for a in atts if a.sha256}
    counts: dict[str, int] = {}
    for a in atts:
        counts[a.status] = counts.get(a.status, 0) + 1
    print(json.dumps({"attachments": len(atts), "unique_pdfs": len(pdfs), "by_status": counts}, indent=2))


if __name__ == "__main__":
    main()
