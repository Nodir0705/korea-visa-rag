"""Scrape per-visa detail pages ("visa guide") from visa.go.kr into data/pages/.

Ported from the author's earlier HiKorea RAG project (visa_scrape.py); only the
paths, the User-Agent and the default locale (en) changed.

Every visa code has up to four sources, all plain form-urlencoded POSTs
(no cookies/session needed):

  openPage.do?MENU_ID=1010201   static summary page; content in div#contents
                                (h4 category + table.board_wright04)
  detailSojQual.do              embassy channel: RES_TARGET (eligibility),
                                RES_ATTACH_FILE (documents), RES_SOJ_DUR_DESC
  ccviApplyList.do              visa-issuance-confirmation channel (CCVI_*);
                                the only channel with data for E-9-x
  eVisaApplyList.do             e-visa channel

The three AJAX endpoints only answer when handed a (IN_NAT_CD,
IN_REP_ENTRY_PURP_CD, IN_SOJ_DUR) combo the wizard considers valid for the
code, so discovery first sweeps selectVisaWizardList.do over 4 nationalities
x 15 purposes x 3 durations and records one working combo per code.  Codes
that never appear in the wizard (22 of them) still have a static page; the
full static code list comes from hidden inputs on openPage.do?MENU_ID=10102.
Discovery results are cached in data/raw/visa/_codes.json.

Raw bytes land in data/raw/visa/{locale}/ (idempotent: existing files are
not re-fetched), corpus docs in data/corpus/visa/{locale}/{code}.md.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "visa_pages"
CORPUS_DIR = DATA_DIR / "pages"

BASE = "https://www.visa.go.kr"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) visa-rag-research (polite crawler, 1 req/s)"
DISALLOWED = ("/search/", "/bbs/", "/oidc/")  # per robots.txt
TIMEOUT = 20.0
RETRIES = 2
SLEEP = 1.0  # politeness: 1 req/s

STATIC_URL = f"{BASE}/openPage.do?MENU_ID=1010201"
CODE_LIST_URL = f"{BASE}/openPage.do?MENU_ID=10102"
WIZARD_URL = f"{BASE}/biz/vg/vw/selectVisaWizardList.do"
CHANNELS = {  # suffix -> (endpoint, include IN_SOJ_DUR)
    "dsq": (f"{BASE}/biz/vg/vw/detailSojQual.do", True),
    "ccvi": (f"{BASE}/biz/vg/vw/ccviApplyList.do", False),
    "evisa": (f"{BASE}/biz/vg/vw/eVisaApplyList.do", False),
}

LOCALES = {"ko": "KO", "en": "EN", "zh": "CH"}  # locale -> LANG_TYPE
NATS = ["275", "112", "185", "404"]  # US, China, Vietnam, Australia
PURPS = [f"VC{i:03d}" for i in range(1, 16)]
DURS = ["90UNDER", "91OVER", "FOREVER"]

# Codes whose static page content is the visa-waiver guide block.
WAIVER_CODES = {"B-1", "B-2-1", "B-2-2"}
WAIVER_DIV = {"ko": "noVisaGuide", "en": "noVisaGuideEn", "zh": "noVisaGuideCh"}

# No minimum-length gate: a sub-code's whole rule is often one sentence
# ("FTA전근: 자유무역협정(FTA) 내용에 따른 기업 내 전근자"), and that sentence is
# exactly what retrieval should return. A flat character budget also penalized
# CJK, which says in 108 chars what English spends 608 on (F-4-13), so it cut
# 24 valid ko / 26 zh pages against 3 en. Emptiness is already caught below by
# the "no content in any channel" guard, which is the real failure mode.

HEADINGS = {
    "ko": {
        "overview": "개요 (신청대상 또는 활동범위)",
        "dsq": "재외공관 비자 신청",
        "ccvi": "사증발급인정서 신청",
        "evisa": "전자비자 신청",
        "target": "자격요건",
        "attach": "제출서류",
        "dur": "비자 유효기간 및 체류기간",
        "case": "사례",
        "waiver": "무비자 입국 안내",
    },
    "en": {
        "overview": "Overview (eligible applicants / activities)",
        "dsq": "Embassy visa application",
        "ccvi": "Visa Issuance Confirmation (CCVI) application",
        "evisa": "E-visa application",
        "target": "Eligibility",
        "attach": "Required documents",
        "dur": "Visa validity and period of stay",
        "case": "Case",
        "waiver": "Visa-free entry guide",
    },
    "zh": {
        "overview": "概要（申请对象或活动范围）",
        "dsq": "驻外使领馆签证申请",
        "ccvi": "签证发给认定书申请",
        "evisa": "电子签证申请",
        "target": "资格条件",
        "attach": "提交材料",
        "dur": "签证有效期及停留期间",
        "case": "类别",
        "waiver": "免签入境指南",
    },
}


# ---------------------------------------------------------------- fetching

def _polite_post(client: httpx.Client, url: str, data: dict[str, str]) -> bytes | None:
    if any(p in url for p in DISALLOWED):
        return None
    for attempt in range(RETRIES + 1):
        try:
            resp = client.post(url, data=data)
            resp.raise_for_status()
            time.sleep(SLEEP)
            return resp.content
        except httpx.HTTPError as exc:
            if attempt == RETRIES:
                print(f"[visa] giving up on {url} {data.get('SOJ_QUAL_GRP', '')}: {exc}")
                return None
            time.sleep(2 ** (attempt + 1))
    return None


def _polite_get(client: httpx.Client, url: str) -> bytes | None:
    if any(p in url for p in DISALLOWED):
        return None
    for attempt in range(RETRIES + 1):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            time.sleep(SLEEP)
            return resp.content
        except httpx.HTTPError as exc:
            if attempt == RETRIES:
                print(f"[visa] giving up on {url}: {exc}")
                return None
            time.sleep(2 ** (attempt + 1))
    return None


def _common(lang: str) -> dict[str, str]:
    return {"TRAN_TYPE": "ComAjax", "SE_FLAG_YN": "", "LANG_TYPE": lang}


def _is_error_page(content: bytes) -> bool:
    head = content[:600].decode("utf-8", errors="replace")
    return "<title>에러</title>" in head


def _valid_json(content: bytes) -> bool:
    try:
        return "resultList" in json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return False


# ---------------------------------------------------------------- discovery

def discover(client: httpx.Client, force: bool = False) -> dict[str, dict[str, Any]]:
    """Map every visa code to its wizard combo (or static_only)."""
    cache = RAW_DIR / "_codes.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    codes: dict[str, dict[str, Any]] = {}
    n_calls = 0
    for nat in NATS:
        for purp in PURPS:
            for dur in DURS:
                body = _common("KO") | {
                    "IN_REP_ENTRY_PURP_CD": purp,
                    "IN_NAT_CD": nat,
                    "IN_SOJ_DUR": dur,
                }
                content = _polite_post(client, WIZARD_URL, body)
                n_calls += 1
                if content is None or not _valid_json(content):
                    continue
                for row in json.loads(content)["resultList"]:
                    code = row.get("SOJ_QUAL_GRP")
                    if not code or code in codes:
                        continue
                    # Prefer a duration the row itself flags as valid.
                    best_dur = dur
                    for d, flag in (
                        ("90UNDER", "MAIN_SOJ_DUR_90UNDER_GB"),
                        ("91OVER", "MAIN_SOJ_DUR_91OVER_GB"),
                        ("FOREVER", "MAIN_SOJ_DUR_FOREVER_GB"),
                    ):
                        if row.get(flag) == "Y":
                            best_dur = d
                            break
                    codes[code] = {
                        "name": row.get("SOJ_QUAL_NM", ""),
                        "nat": nat,
                        "purp": purp,
                        "dur": best_dur,
                    }
        print(f"[visa:discover] nat {nat} done, {len(codes)} codes so far ({n_calls} calls)")

    # Static-only codes from the hidden inputs on the full code list page.
    listing = _polite_get(client, CODE_LIST_URL)
    if listing is not None and not _is_error_page(listing):
        soup = BeautifulSoup(listing, "lxml")
        for inp in soup.find_all("input", attrs={"name": "SOJ_QUAL_GRP"}):
            code = (inp.get("value") or "").strip()
            if code and code not in codes:
                codes[code] = {"name": "", "static_only": True}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(codes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[visa:discover] {len(codes)} codes "
          f"({sum(1 for c in codes.values() if c.get('static_only'))} static-only)")
    return codes


# ---------------------------------------------------------------- fetch per code

def fetch_code(client: httpx.Client, code: str, info: dict[str, Any],
               locale: str, force: bool = False) -> int:
    """Fetch static page + AJAX channels for one code/locale. Returns new fetches."""
    lang = LOCALES[locale]
    raw_dir = RAW_DIR / locale
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0

    static_path = raw_dir / f"{code}_static.html"
    if force or not static_path.exists():
        body = {
            "CMM_TEST_VAL": "test",
            "TRAN_TYPE": "ComSubmit",
            "SE_FLAG_YN": "",
            "LANG_TYPE": lang,
            "IN_SOJ_QUAL_GRP": code,
        }
        content = _polite_post(client, STATIC_URL, body)
        if content is not None and not _is_error_page(content):
            static_path.write_bytes(content)
            fetched += 1
        elif content is not None:
            print(f"[visa:{locale}] {code} static returned error page")

    if info.get("static_only"):
        return fetched

    for suffix, (endpoint, with_dur) in CHANNELS.items():
        path = raw_dir / f"{code}_{suffix}.json"
        if not force and path.exists():
            continue
        body = _common(lang) | {
            "SOJ_QUAL_GRP": code,
            "IN_REP_ENTRY_PURP_CD": info["purp"],
            "IN_NAT_CD": info["nat"],
        }
        if with_dur:
            body["IN_SOJ_DUR"] = info["dur"]
        content = _polite_post(client, endpoint, body)
        if content is not None and _valid_json(content):
            path.write_bytes(content)
            fetched += 1
        elif content is not None:
            print(f"[visa:{locale}] {code} {suffix} returned non-JSON")
    return fetched


# ---------------------------------------------------------------- extraction

def _collapse(text: str) -> str:
    return re.sub(r"[ \t\u00a0\u200b]+", " ", text).strip()


def _table_to_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_collapse(td.get_text(" ", strip=True)) for td in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def frag_to_text(html_frag: str | None) -> str:
    """AJAX field value (HTML fragment with &nbsp;, <br/>, occasional tables) -> text."""
    if not html_frag:
        return ""
    soup = BeautifulSoup(html_frag, "lxml")
    for table in soup.find_all("table"):
        table.replace_with(NavigableString("\n" + _table_to_text(table) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for tag in soup.find_all(["p", "div", "li", "tr"]):
        tag.append(NavigableString("\n"))
    lines = [_collapse(line) for line in soup.get_text().splitlines()]
    out: list[str] = []
    for line in lines:
        if line:
            out.append(line)
        elif out and out[-1]:
            out.append("")
    return "\n".join(out).strip()


def _row_field(row: dict[str, Any], suffix: str) -> str:
    """First row value whose key ends with suffix (channel prefixes vary)."""
    for key, val in row.items():
        if key.endswith(suffix) and isinstance(val, str):
            return frag_to_text(val)
    return ""


def _container_to_markdown(container: Tag) -> str:
    """Waiver-guide div -> markdown (h4/h5 headings, lists, tables)."""
    for tag in container.find_all(["script", "style", "noscript", "iframe", "img", "button", "a"]):
        tag.decompose()
    for table in container.find_all("table"):
        table.replace_with(NavigableString("\n" + _table_to_text(table) + "\n"))
    for li in container.find_all("li"):
        li.insert(0, NavigableString("- "))
    for name, mark in (("h4", "### "), ("h5", "#### ")):
        for tag in container.find_all(name):
            tag.insert(0, NavigableString(mark))
    for br in container.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for tag in container.find_all(["div", "p", "li", "ul", "h4", "h5"]):
        tag.append(NavigableString("\n"))
    lines = [_collapse(line) for line in container.get_text().splitlines()]
    out: list[str] = []
    for line in lines:
        if not line:
            continue
        if line.startswith("#") and out:
            out.append("")
        out.append(line)
    return "\n".join(out)


def parse_static(content: bytes, locale: str, code: str) -> tuple[str, str]:
    """Static page -> (localized visa-category title, overview markdown)."""
    soup = BeautifulSoup(content, "lxml")
    contents = soup.select_one("div#contents")
    if contents is None:
        return "", ""

    title = ""
    parts: list[str] = []
    for table in contents.select("table.board_wright04"):
        h4 = table.find_previous_sibling("h4")
        if h4 is not None and not title:
            title = _collapse(h4.get_text())
        for caption in table.find_all("caption"):
            caption.decompose()  # generic template text, not content
        for tr in table.find_all("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            label = _collapse(th.get_text(" ", strip=True)) if th else ""
            body = "\n".join(
                frag_to_text(td.decode_contents()) for td in tds if td
            ).strip()
            if body:
                parts.append(f"{label}\n{body}" if label else body)

    overview = "\n\n".join(parts)

    if code in WAIVER_CODES:
        block = contents.select_one(f"div#{WAIVER_DIV[locale]}")
        if block is not None:
            guide = _container_to_markdown(block)
            if guide:
                overview = (overview + "\n\n" + guide).strip()
    return title, overview


def _channel_section(raw_path: Path, heading: str, h: dict[str, str]) -> str:
    if not raw_path.exists():
        return ""
    try:
        rows = json.loads(raw_path.read_bytes()).get("resultList", [])
    except ValueError:
        return ""
    cases: list[str] = []
    for i, row in enumerate(rows, 1):
        target = _row_field(row, "_TARGET")
        attach = _row_field(row, "ATTACH_FILE")
        dur_desc = _row_field(row, "SOJ_DUR_DESC")
        if not (target or attach or dur_desc):
            continue
        sub: list[str] = []
        if len(rows) > 1:
            sub.append(f"### {heading} — {h['case']} {i}")
        if target:
            sub.append(f"#### {h['target']}\n{target}" if len(rows) > 1
                       else f"### {h['target']}\n{target}")
        if attach:
            sub.append(f"#### {h['attach']}\n{attach}" if len(rows) > 1
                       else f"### {h['attach']}\n{attach}")
        if dur_desc:
            sub.append(f"#### {h['dur']}\n{dur_desc}" if len(rows) > 1
                       else f"### {h['dur']}\n{dur_desc}")
        cases.append("\n\n".join(sub))
    if not cases:
        return ""
    return f"## {heading}\n\n" + "\n\n".join(cases)


def extract_code(code: str, info: dict[str, Any], locale: str) -> tuple[bool, str]:
    """Build corpus doc for one code/locale from raw files. Returns (written, reason)."""
    raw_dir = RAW_DIR / locale
    h = HEADINGS[locale]

    title_name = ""
    sections: list[str] = []

    static_path = raw_dir / f"{code}_static.html"
    if static_path.exists():
        title_name, overview = parse_static(static_path.read_bytes(), locale, code)
        if overview:
            sections.append(f"## {h['overview']}\n\n{overview}")

    for suffix in CHANNELS:
        section = _channel_section(raw_dir / f"{code}_{suffix}.json", h[suffix], h)
        if section:
            sections.append(section)

    if not sections:
        return False, "no content in any channel"

    name = title_name or info.get("name", "")
    title = f"{code} {name}".strip()
    body = f"# {title}\n\n" + "\n\n".join(sections)
    mtimes = [p.stat().st_mtime for p in raw_dir.glob(f"{code}_*") if p.is_file()]
    fetched_at = datetime.date.fromtimestamp(max(mtimes)).isoformat() if mtimes else \
        datetime.date.today().isoformat()

    meta: dict[str, Any] = {
        "url": STATIC_URL,
        "visa_code": code,
        "title": title,
        "locale": locale,
        "fetched_at": fetched_at,
        "source": "visa.go.kr",
        "soj_qual_grp": code,
    }
    if not info.get("static_only"):
        meta["nat_cd"] = info["nat"]
        meta["purp_cd"] = info["purp"]
        meta["dur"] = info["dur"]

    frontmatter = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    corpus_dir = CORPUS_DIR / locale
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out_path = corpus_dir / f"{code}.md"
    out_path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return True, str(out_path)


# ---------------------------------------------------------------- driver

def run(locales: list[str], limit: int | None = None, force: bool = False,
        extract_only: bool = False) -> None:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
                      follow_redirects=True) as client:
        codes = discover(client, force=False)
        ordered = sorted(codes)
        if limit is not None:
            ordered = ordered[:limit]

        for locale in locales:
            fetched = 0
            written = 0
            skipped: list[str] = []
            for code in ordered:
                info = codes[code]
                if not extract_only:
                    fetched += fetch_code(client, code, info, locale, force=force)
                ok, reason = extract_code(code, info, locale)
                if ok:
                    written += 1
                else:
                    skipped.append(f"{code}\t{reason}")
            corpus_dir = CORPUS_DIR / locale
            corpus_dir.mkdir(parents=True, exist_ok=True)
            (corpus_dir / "_skipped.txt").write_text(
                "\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8")
            print(f"[visa:{locale}] {fetched} raw fetches, {written} docs written, "
                  f"{len(skipped)} skipped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape visa.go.kr per-visa detail pages")
    parser.add_argument("--locale", nargs="+", default=["en"],
                        choices=list(LOCALES))
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N codes (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch raw files even if they exist")
    parser.add_argument("--extract-only", action="store_true",
                        help="skip fetching; rebuild corpus docs from raw files")
    args = parser.parse_args()
    run(args.locale, limit=args.limit, force=args.force, extract_only=args.extract_only)


if __name__ == "__main__":
    main()
