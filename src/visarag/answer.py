"""Grounded answer generation with verified citations.

Flow for one question:

1. Retrieve (hybrid + rerank) the top passages.
2. If no passage clears the reranker's relevance floor, refuse without calling
   the LLM: fast, and the model never gets the chance to improvise.
3. Otherwise the generator receives numbered passages ``[1]..[k]`` (title,
   page, date, text) and answers in plain text with ``[n]`` markers after each
   claim, or exactly ``NOT_FOUND`` when the passages do not answer the question.
   (Plain text, not a JSON schema: on Ollama 0.17.7, qwen3.5 ignores ``format``
   when ``think`` is false, and with thinking on it spends 800-3,400 tokens
   before answering. Measured; see README "What did not work".)
4. Code then checks every marker: a marker that points outside ``1..k`` is an
   invented citation; the answer is regenerated once with a correction, and if
   it is still wrong the invalid markers are removed and the answer flagged.
   Sentences without any marker are counted (citation coverage) and reported.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import re
import time
from dataclasses import asdict, dataclass, field

from visarag.config import settings
from visarag.index import _not_english
from visarag.ollama import Ollama
from visarag.retrieve import Hit, Retriever

REFUSAL = (
    "I could not find this in the visa.go.kr documents I have. "
    "Please check the Korea Visa Portal or ask the Korean embassy or consulate that handles your application."
)

SYSTEM = """You answer questions about Korean visas using ONLY the numbered passages provided.
The passages come from the Korea Visa Portal (visa.go.kr): official PDF documents, and the portal's
per-visa guide pages (eligibility, required documents, visa validity and period of stay for each visa code).

Rules:
- Start with the direct answer.
- When the answer is a list (documents, countries, conditions, options, fields), give every item the
  passage lists for the case asked, in the passage's wording. Never give a selection of the items.
- When the question asks who is eligible, include every condition the passage states (status,
  nationality, age, relationship, period).
- When the question asks who must or who may do something, also give the exemptions or exclusions
  the passage states for that same rule.
- If a passage says a value is the same as for another person or visa (for example "the same as the
  principal applicant") and another passage states that value for that visa and route, give the value
  and cite both passages.
- If the question asks for a figure in a particular form (for example an amount in won) and the
  passages state it only in another form, give what they state and say that the figure in the form
  asked is not stated.
- Otherwise be brief: 1-4 sentences. Answer only what was asked: do not add related rules,
  exceptions, conditions or background the question did not ask about.
- End every sentence that states a fact with the number of the passage that states it, like [2].
  Cite only passages that actually state that fact.
- Use only facts written in the passages. No outside knowledge, no guessing, no filling gaps.
- Keep numbers, amounts, dates, durations, country names and visa codes exactly as written.
  Before saying a country or visa type is on a list, check that it is really in that list.
- Use a passage only if it is about the same visa, permit, form or procedure the question asks about.
  A fee, deadline or rule written for something else (for example a re-entry permit) is not the answer
  to a question about a visa. If no passage is about the thing asked, reply NOT_FOUND.
- A visa guide passage is about exactly one visa code, given in its title. Do not apply it to a different code.
- When the question names an application route (embassy or diplomatic mission, Visa Issuance
  Confirmation (CCVI), or e-visa), answer for that route only. A rule written for visa applications
  filed at a diplomatic mission does not apply to a CCVI application, and the other way round.
- If the passages answer only part of the question, answer that part and say in one short sentence
  what is not covered.
- If the passages do not contain the answer, reply with exactly: NOT_FOUND
- Do not mention publication dates, passage numbers outside the [n] markers, or the passages themselves."""

NOT_FOUND = "NOT_FOUND"
YES_NO = re.compile(
    r"^\W*(is|are|am|can|could|do|does|did|will|would|was|were|should|shall|may|must|have|has|had)\b", re.IGNORECASE
)
YES_NO_RULE = (
    "This is a yes/no question: begin with Yes or No, and make it agree with the rest of your answer "
    "(if the passage exempts the asker or says it is not possible, the answer is No)."
)
NOT_YES_NO_RULE = "This is not a yes/no question: do not begin with Yes or No."


def is_yes_no(question: str) -> bool:
    """Questions that open with an auxiliary verb ("Can I ...", "Is a ... required") take Yes/No;
    "which", "what", "how" questions do not. Casual questions often put the verb later
    ("im australian, do i need ...") so the last clause is checked too. "Is it single or multiple
    entry?" is a choice, not yes/no."""
    first = re.split(r",\s*and\b|[;?]", question.strip())[0]
    last = re.split(r"[,;]\s*", question.strip())[-1]
    clause = first if YES_NO.match(first) else last if YES_NO.match(last) else ""
    return bool(clause) and not re.search(r"\bor\b", clause, re.IGNORECASE)
VISA_CODE = re.compile(r"\b([A-H])-(\d{1,2})(?:-(\d{1,2}))?\b")
AMOUNT = re.compile(
    r"(?:KRW|USD|BGN|won|₩|\$|€)\s?\d[\d,.]*(?:\s(?:million|billion))?"
    r"|\d[\d,.]*(?:\s(?:million|billion))?\s?(?:KRW|USD|BGN|won|€|dollars?)",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
AMOUNT_CORE = re.compile(r"\d[\d,.]*(?:\s(?:million|billion))?", re.IGNORECASE)
NOT_STATED = re.compile(
    r"\b(not (specified|stated|listed|mentioned|given|provided)|do(es)? not (specify|state|list|mention|say|give|contain|include)"
    r"|no (specific|minimum|explicit|exact) .{0,80}?\b(is|are) (listed|stated|specified|given|mentioned))\b",
    re.IGNORECASE,
)
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
HOW_MANY = re.compile(r"\b(how many|number of)\b", re.IGNORECASE)
OLD_SOURCE_BEFORE = "2023-11-24"  # sources older than ~3 years are flagged for the reader
MARKER = re.compile(r"\[(\d+)\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\-•*])|\n+")


ROUTES = {
    "embassy": (re.compile(r"\b(embass(y|ies)|consulates?|diplomatic missions?)\b", re.IGNORECASE), "Embassy visa application"),
    "ccvi": (re.compile(r"\b(ccvi|confirmation of visa issuance|visa issuance confirmation)\b", re.IGNORECASE),
             "Visa Issuance Confirmation (CCVI) application"),
    "evisa": (re.compile(r"\be-?visa\b", re.IGNORECASE), "E-visa application"),
}


def _codes(text: str) -> set[str]:
    return {m.group(0).upper() for m in VISA_CODE.finditer(text)}


def _code_fits(page_code: str, asked: set[str]) -> bool:
    """A guide page fits a question about ``asked`` if it is that code, a sub-type of it
    (C-3 -> C-3-9), or its parent (D-8-1 -> D-8)."""
    return any(page_code == q or page_code.startswith(q + "-") or q.startswith(page_code + "-") for q in asked)


def _has_route(h: Hit, route: str) -> bool:
    return any(route in head for head, _ in _guide_sections(h.doc_id.removeprefix("page-")))


def scope_filter(question: str, hits: list[Hit]) -> tuple[list[Hit], list[str]]:
    """Keep the generator on the visa and application route the question names.

    Guide-page passages for a different visa code (D-8-3 for a D-8-1 question) or a
    different route (CCVI for an embassy question) are dropped before generation, so the
    model cannot borrow a neighbouring case's rule. PDFs are never dropped, and nothing is
    dropped unless a passage of the named visa or route is retrieved.
    """
    flags: list[str] = []
    asked = _codes(question)
    pages = [h for h in hits if h.source == "page"]
    if asked and any(_code_fits(h.doc_id.removeprefix("page-"), asked) for h in pages):
        kept = [h for h in hits if h.source != "page" or _code_fits(h.doc_id.removeprefix("page-"), asked)]
        if len(kept) < len(hits):
            flags.append("scoped_to_visa_code")
        hits = kept
    named = [heading for rx, heading in ROUTES.values() if rx.search(question)]
    if len(named) == 1:
        other = [heading for _, heading in ROUTES.values() if heading != named[0]]
        on_route = [h for h in hits if h.source == "page" and named[0] in h.headings]
        if on_route:
            # A chunk from another route of a page that also has the named route stays: the guide
            # expansion turns it into that page's overview plus the named route.
            kept = [h for h in hits if not (h.source == "page" and any(o in h.headings for o in other)
                                            and not _has_route(h, named[0]))]
            if len(kept) < len(hits):
                flags.append("scoped_to_route")
            hits = kept
    return hits, flags


@dataclass
class Citation:
    n: int
    title: str
    page: int
    pages: list[int]
    file_name: str
    pdf_url: str
    post_date: str
    snippet: str
    score: float
    doc_id: str = ""
    source: str = "pdf"
    note: str = ""


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    citations: list[Citation]
    passages: list[Citation]
    confidence: dict[str, float]
    timings: dict[str, float]
    flags: list[str] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def format_passages(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        if h.source == "page":
            head = f" | section: {h.headings}" if h.headings else ""
            blocks.append(f"[{i}] {h.title} ({h.note.split('; fetched')[0]}){head}\n{h.body}")
            continue
        pages = f"pages {h.pages[0]}-{h.pages[-1]}" if len(h.pages) > 1 else f"page {h.page}"
        date = f", published {h.post_date}" if h.post_date else ""
        head = f" | section: {h.headings}" if h.headings else ""
        blocks.append(f"[{i}] {h.title} ({pages}{date}){head}\n{h.body}")
    return "\n\n".join(blocks)


def _citation(n: int, h: Hit) -> Citation:
    return Citation(
        n=n,
        title=h.title,
        page=h.page,
        pages=h.pages,
        file_name=h.file_name,
        pdf_url=h.source_url if h.source == "page" else f"/pdf/{h.doc_id}.pdf#page={h.page}",
        post_date=h.post_date,
        snippet=h.body,
        score=round(h.score, 4),
        doc_id=h.doc_id,
        source=h.source,
        note=h.note,
    )


def _codes(text: str) -> set[str]:
    """Visa codes at the precision written: "D-2" -> {"D-2"}, "D-2-1" -> {"D-2-1"}."""
    return {m.group(0).upper() for m in VISA_CODE.finditer(text)}


def _about_code(h: Hit, codes: set[str], amount: str) -> bool:
    head = f"{h.title} {h.headings}".upper()
    if any(re.search(rf"\b{re.escape(c)}\b", head) for c in codes):
        return True
    return any(amount in line and _codes(line) & codes for line in h.body.splitlines())


def _is_count(n: str, question: str, sources: str) -> bool:
    """A count the model made by counting a numbered list ("18 countries" for a list that ends at
    ⑱) is not written as a number in the passages, but it is grounded: accept it when the question
    asks how many and a passage has a list item with that number."""
    if not HOW_MANY.search(question) or not n.isdigit() or not 1 <= int(n) <= 99:
        return False
    i = int(n)
    return (i <= len(CIRCLED) and CIRCLED[i - 1] in sources) or bool(
        re.search(rf"(?m)^\s*\(?{i}[.)]\s", sources)
    )


def amount_guard(question: str, text: str, hits: list[Hit]) -> tuple[str, list[str]]:
    """Drop money amounts that a cited passage gives for a different subject than the question's visa.

    The failure it targets: asked for the D-2 visa fee, the model quoted the 50,000 KRW
    re-entry permit fee from a notice that mentions D-2 only in its fee-exemption list.
    Rule: when the question names a visa code, each sentence that states an amount must
    cite a passage whose title/section names that code, or that has the amount and the
    code on the same line. A failing first sentence turns the answer into a not-stated
    answer; a failing later sentence is removed.
    """
    codes = _codes(question)
    sentences = [s for s in SENTENCE.split(text) if s.strip()]
    kept, flags = [], []
    for i, s in enumerate(sentences):
        amounts = [m.group(0) for m in AMOUNT.finditer(s)]
        cited = [hits[int(n) - 1] for n in MARKER.findall(s) if 1 <= int(n) <= len(hits)] or hits
        problem = ""
        for a in amounts:
            core = AMOUNT_CORE.search(a).group(0).lower()
            # 1. The amount must be written in a cited passage at all (catches amounts from model memory).
            if not any(core in f"{h.body}\n{h.text}".lower() for h in cited):
                problem = "amount_not_in_sources"
            # 2. When the question names a visa code, the passage must be about that code.
            elif codes and not any(_about_code(h, codes, re.sub(r"[^\d,.]", "", core)) for h in cited):
                problem = "amount_subject_mismatch"
        # 3. Every number in the sentence must be written in the passages it cites. Catches a rule
        #    copied from a neighbouring visa page (TOPIK "50 out of 200" from D-3 given for D-4-1).
        if not problem:
            bare = MARKER.sub(" ", VISA_CODE.sub(" ", s))
            # Numbers repeated from the question, or from a passage's title or date, are not new claims.
            sources = " ".join([question, *(f"{h.body}\n{h.text}\n{h.title} {h.post_date}" for h in cited)])
            if any(n.rstrip(".,") not in sources and not _is_count(n.rstrip(".,"), question, sources)
                   for n in NUMBER.findall(bare)):
                problem = "number_not_in_sources"
        if problem:
            flags.append(problem)
            if i == 0:
                subject = f" for {min(codes)}" if codes else ""
                what = "this amount" if problem.startswith("amount") else "this"
                return (f"The visa.go.kr documents I have do not state {what}{subject}.", flags)
            continue
        kept.append(s)
    return " ".join(kept), flags


def check_citations(text: str, k: int) -> tuple[set[int], set[int], float]:
    """(valid markers, invalid markers, share of factual sentences carrying a marker)."""
    found = {int(m) for m in MARKER.findall(text)}
    valid = {n for n in found if 1 <= n <= k}
    sentences = [s for s in SENTENCE.split(text) if len(re.sub(r"\W", "", s)) > 15]
    cited = sum(bool(MARKER.search(s)) for s in sentences)
    return valid, found - valid, (cited / len(sentences)) if sentences else 0.0


VERIFY_PROMPT = """Question: {question}

Passages:
{passages}

Answer to check, one sentence per line:
{sentences}

For each numbered sentence decide KEEP or DROP.
DROP a sentence if any of these is true:
- what it states is not written in the passage(s) it cites;
- it applies a rule, number or list to a different visa, person, case or country than the passage does;
- it adds information the question did not ask for.
Otherwise KEEP. Reply with one line per sentence, exactly like "1: KEEP" or "2: DROP", and nothing else."""
VERDICT_LINE = re.compile(r"^\s*(\d+)\s*[:.)-]\s*(KEEP|DROP)", re.IGNORECASE | re.MULTILINE)


@functools.lru_cache(maxsize=1)
def _page_texts() -> dict:
    path = settings.index_dir / "page_text.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def expand_to_pages(hits: list[Hit]) -> list[Hit]:
    """Small-to-big: replace a PDF chunk by its whole page when the page is short and English.

    Several hits on the same page become one passage (best score kept). Visa-guide
    chunks, multi-page chunks, long pages and non-English pages stay as chunks.
    """
    texts = _page_texts()
    out: list[Hit] = []
    seen: dict[tuple[str, int], Hit] = {}
    for h in hits:
        short_doc = len(texts.get(h.doc_id, {})) <= settings.page_expand_max_doc_pages
        pages = h.pages if h.source == "pdf" and short_doc and 1 <= len(h.pages) <= 2 else []
        page = "\n".join(texts.get(h.doc_id, {}).get(str(p), "") for p in pages).strip()
        if page and len(page) <= settings.page_expand_max_chars * len(pages) and not _not_english(page):
            key = (h.doc_id, tuple(pages))
            if key in seen:
                seen[key].score = max(seen[key].score, h.score)
                continue
            seen[key] = dataclasses.replace(h, body=page, headings="", ranks=dict(h.ranks))
            out.append(seen[key])
        else:
            out.append(h)
    return out


_HANGUL = re.compile(r"[\uac00-\ud7a3]")


@functools.lru_cache(maxsize=64)
def _doc_tables(doc_id: str) -> dict[int, tuple[str, ...]]:
    """Docling's table grids for one PDF, written out row by row ("header: value; ...").

    The page text reads a table in layout order, which can put a row's label next to the
    neighbouring row's values (a sanctions table read "Suspension (6 months) 9 ~ 10 ...
    Suspension (3 months) 7 ~ 8"). Docling's cell grid keeps the rows straight, and a
    header-per-value line is a form small models read reliably.
    """
    path = settings.parsed_dir / f"{doc_id}.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, list[str]] = {}
    for t in doc.get("tables", []):
        data, prov = t.get("data") or {}, t.get("prov") or []
        nr, nc = data.get("num_rows", 0), data.get("num_cols", 0)
        if not prov or nr < 2 or nc < 2:
            continue
        grid = [[""] * nc for _ in range(nr)]
        header_rows: set[int] = set()
        for c in data.get("table_cells", []):
            for r in range(c["start_row_offset_idx"], min(c["end_row_offset_idx"], nr)):
                for k in range(c["start_col_offset_idx"], min(c["end_col_offset_idx"], nc)):
                    grid[r][k] = " ".join(c.get("text", "").split())
            if c.get("column_header"):
                header_rows.add(c["start_row_offset_idx"])
        n_head = max(header_rows) + 1 if header_rows and max(header_rows) < nr - 1 else 1
        heads = [" / ".join(dict.fromkeys(grid[r][k] for r in range(n_head) if grid[r][k])) or f"column {k + 1}"
                 for k in range(nc)]
        # Only a real data table has a header row of short, distinct column names. Forms (a title
        # cell spanning the row) and two-column lists (a region next to its countries) come out of
        # Docling with a data row taken as the header, and "header: value" lines would then pair
        # values with the wrong labels. Those are left to the page text.
        if (nc < 3 or nr - n_head < 3 or len(set(heads)) < nc
                or any(len(h) > 60 or h.endswith(":") or h.startswith("column ") or _HANGUL.search(h) for h in heads)
                or any(grid[r][k] == heads[k] for r in range(n_head, nr) for k in range(nc))):
            continue
        rows = []
        for r in range(n_head, nr):
            cells = [f"{heads[k]}: {grid[r][k]}" for k in range(nc) if grid[r][k] and (k == 0 or grid[r][k] != grid[r][k - 1])]
            if cells:
                rows.append("- " + "; ".join(cells))
        text = "Table on this page, row by row:\n" + "\n".join(rows)
        if rows and len(text) <= settings.table_rows_max_chars and not _not_english(text):
            out.setdefault(int(prov[0]["page_no"]), []).append(text)
    return {p: tuple(v) for p, v in out.items()}


def add_table_rows(hits: list[Hit]) -> list[Hit]:
    """Append the row-by-row form of the tables on each PDF passage's pages (once per page)."""
    seen: set[tuple[str, int]] = set()
    out = []
    for h in hits:
        extra = []
        if h.source == "pdf":
            for p in h.pages or [h.page]:
                if (h.doc_id, p) not in seen:
                    seen.add((h.doc_id, p))
                    extra += _doc_tables(h.doc_id).get(p, ())
        out.append(dataclasses.replace(h, body=h.body + "\n\n" + "\n\n".join(extra), ranks=dict(h.ranks)) if extra else h)
    return out


def add_named_pages(question: str, hits: list[Hit], retriever: Retriever) -> tuple[list[Hit], list[str]]:
    """Make sure the guide page of every visa code the question names is in the context.

    Retrieval can miss a short page the question names outright (a D-4-2 question with
    typos retrieved four other visas' pages), and a two-visa question (a dependent's F-3-1
    and the principal's E-1) needs both pages. The page goes in second, after the top hit.
    """
    have = {h.doc_id for h in hits}
    added = []
    for code in sorted(_codes(question))[:3]:
        if f"page-{code}" in have:
            continue
        h = retriever.page_hit(code)
        if h is not None:
            h.score = round(settings.min_rerank_score, 4)
            added.append(h)
    if not added:
        return hits, []
    return hits[:1] + added + hits[1:], ["added_named_visa_page"]


def cap_context(hits: list[Hit]) -> list[Hit]:
    """Keep the passages inside the context budget; hits are in rank order."""
    total, kept = 0, []
    for h in hits:
        total += len(h.body)
        if kept and total > settings.context_max_chars:
            break
        kept.append(h)
    return kept


@functools.lru_cache(maxsize=256)
def _guide_sections(code: str) -> tuple[tuple[str, str], ...]:
    """The top-level sections ("## ...") of a visa guide page as (heading, text)."""
    path = settings.data_dir / "pages" / "en" / f"{code}.md"
    if not path.exists():
        return ()
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n") and "\n---\n" in text:
        text = text.split("\n---\n", 1)[1]
    sections: list[tuple[str, str]] = []
    for block in re.split(r"\n(?=## )", text):
        block = block.strip()
        if block.startswith("## "):
            sections.append((block[3:].split("\n", 1)[0].strip(), block))
    return tuple(sections)


def _overview_and_case(sections: tuple[tuple[str, str], ...], h: Hit, question: str = "",
                       score=None, route: str = "") -> list[str]:
    """The overview's own text (without its sub-sections) plus the sub-section the hit came from,
    and, when a scorer is given, the ``guide_best_cases`` sub-sections (of the named route, if
    any) that the cross-encoder rates closest to the question. The retrieved chunk is often a
    sibling of the right case (Case 1 "event or training" for a diplomat's parent, Case 6)."""
    overview = ""
    subs_all: list[tuple[str, str, str]] = []  # (section head, sub title, text)
    for head, text in sections:
        subs = re.split(r"\n(?=### )", text)
        if head.startswith("Overview"):
            overview = subs[0].strip()
        for sub in subs[1:]:
            subs_all.append((head, sub.split("\n", 1)[0].lstrip("# ").strip(), sub.strip()))
    picked = {i for i, (_, title, _) in enumerate(subs_all) if title and title in h.headings}
    pool = [i for i, (head, _, _) in enumerate(subs_all) if not head.startswith("Overview") and (not route or route in head)]
    if score is not None and question and pool and settings.guide_best_cases > 0:
        scores = score(question, [subs_all[i][2] for i in pool])
        picked |= {i for _, i in sorted(zip(scores, pool), reverse=True)[: settings.guide_best_cases]}
    out = [overview] if overview else []
    out += [f"## {subs_all[i][0]}\n{subs_all[i][2]}" for i in sorted(picked)]
    if len(out) == (1 if overview else 0):  # nothing matched: fall back to the hit itself
        out.append(h.body)
    return out


def expand_guide_pages(question: str, hits: list[Hit], score=None) -> list[Hit]:
    """Small-to-big for visa guide pages: a retrieved section becomes its page's overview plus
    the whole application route (every case) the question names or the hit came from, or the
    whole page when it is short. Eligibility conditions often sit in the overview and the
    right case is often a sibling of the retrieved one, so a single section is not enough.
    Several hits on one page become one passage."""
    named = [heading for rx, heading in ROUTES.values() if rx.search(question)]
    out: list[Hit] = []
    seen: dict[str, Hit] = {}
    for h in hits:
        if h.source != "page":
            out.append(h)
            continue
        if h.doc_id in seen:
            seen[h.doc_id].score = max(seen[h.doc_id].score, h.score)
            continue
        sections = _guide_sections(h.doc_id.removeprefix("page-"))
        if not sections:
            out.append(h)
            continue
        whole = "\n\n".join(t for _, t in sections)
        route = named[0] if len(named) == 1 and any(named[0] in head for head, _ in sections) else ""
        limit = settings.guide_expand_max_chars
        if route:
            keep = [t for head, t in sections if head.startswith("Overview") or route in head]
            limit = max(limit, settings.guide_route_max_chars)
        elif len(whole) <= settings.guide_expand_max_chars:
            keep = [whole]
        else:
            keep = [t for head, t in sections if head.startswith("Overview") or head.split(" — ")[0] in h.headings]
            if len(keep) <= 1:  # the hit's route is not a top-level section of this page
                keep = _overview_and_case(sections, h, question, score)
        body = "\n\n".join(keep) or h.body
        if len(body) > limit:
            # A long route (B-2-1 lists a dozen visa-free schemes) would crowd out the other
            # passages: keep the overview, the case the hit came from and the closest cases.
            body = "\n\n".join(_overview_and_case(sections, h, question, score, route)) or h.body
        seen[h.doc_id] = dataclasses.replace(h, body=body, headings="", ranks=dict(h.ranks))
        out.append(seen[h.doc_id])
    return out


class Answerer:
    def __init__(self, retriever: Retriever | None = None, model: str | None = None) -> None:
        self.retriever = retriever or Retriever()
        self.model = model or settings.gen_model
        self.llm = Ollama()

    def _generate(self, question: str, hits: list[Hit], correction: str = "") -> tuple[str, dict]:
        user = (
            f"Passages:\n\n{format_passages(hits)}\n\n"
            f"Question: {question}\n\n"
            f"{YES_NO_RULE if is_yes_no(question) else NOT_YES_NO_RULE}\n"
            f"Answer from passages [1]-[{len(hits)}] only, with a [n] marker after every factual sentence, "
            f"or reply {NOT_FOUND} if they do not contain the answer."
            + (f"\n\n{correction}" if correction else "")
        )
        data = self.llm.chat(
            self.model,
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            num_predict=4096 if settings.think else 600,
            think=settings.think,
        )
        return data["message"]["content"].strip(), data

    def _verify(self, question: str, text: str, hits: list[Hit]) -> tuple[str, list[str], float]:
        """Second pass over the answer: keep only sentences their cited passages state."""
        sentences = [s.strip() for s in SENTENCE.split(text) if s.strip()]
        if len(sentences) < 2:
            return text, [], 0.0
        cited = sorted({int(n) for n in MARKER.findall(text) if 1 <= int(n) <= len(hits)})
        passages = "\n\n".join(f"[{n}] {hits[n - 1].title}\n{hits[n - 1].body}" for n in cited)
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
        data = self.llm.chat(
            self.model,
            [{"role": "user", "content": VERIFY_PROMPT.format(question=question, passages=passages, sentences=numbered)}],
            num_predict=10 * len(sentences) + 20,
        )
        verdicts = {int(n): v.upper() for n, v in VERDICT_LINE.findall(data["message"]["content"])}
        kept = [s for i, s in enumerate(sentences, 1) if verdicts.get(i, "KEEP") == "KEEP"]
        dropped = len(sentences) - len(kept)
        if not kept:  # never answer with nothing; fall back to the original answer, flagged
            return text, ["verify_dropped_all"], data.get("_latency_s", 0.0)
        flags = [f"verify_dropped_{dropped}"] if dropped else []
        return "\n".join(kept) if any(s.startswith(("-", "*", "•")) for s in kept) else " ".join(kept), flags, data.get(
            "_latency_s", 0.0
        )

    def ask(self, question: str, mode: str = "hybrid+rerank") -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, mode=mode)
        if settings.expand_pages and hits:
            hits = expand_to_pages(hits)
        if settings.table_rows and hits:
            hits = add_table_rows(hits)
        scope_flags: list[str] = []
        if settings.named_pages and hits:
            hits, scope_flags = add_named_pages(question, hits, self.retriever)
        if settings.scope_filter and hits:
            hits, more = scope_filter(question, hits)
            scope_flags += more
        if settings.expand_guide_pages and hits:
            hits = expand_guide_pages(question, hits, self.retriever.score_texts)
        hits = cap_context(hits)
        timings = {k: round(v, 3) for k, v in self.retriever.timings.items()}
        passages = [_citation(i, h) for i, h in enumerate(hits, 1)]
        top = max((h.score for h in hits), default=0.0)
        confidence = {"retrieval_top_score": round(top, 4)}
        flags: list[str] = list(scope_flags)

        if not hits or (mode == "hybrid+rerank" and top < settings.min_rerank_score):
            timings["total_s"] = round(time.perf_counter() - t0, 3)
            return Answer(question, REFUSAL, True, [], passages, confidence, timings, ["low_retrieval_score"], self.model)

        text, raw = self._generate(question, hits)
        valid, invalid, coverage = check_citations(text, len(hits))
        if invalid and NOT_FOUND not in text:
            flags.append("regenerated_invalid_citation")
            text, raw = self._generate(
                question,
                hits,
                correction=f"Your previous answer cited passages that do not exist: {sorted(invalid)}. "
                f"Only [1]-[{len(hits)}] exist.",
            )
            valid, invalid, coverage = check_citations(text, len(hits))
            if invalid:
                text = MARKER.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
                flags.append("removed_invalid_citation")
        prompt_tokens = raw.get("prompt_eval_count", 0)
        timings["generate_s"] = round(raw.get("_latency_s", 0.0), 3)
        timings["prompt_tokens"] = prompt_tokens
        timings["output_tokens"] = raw.get("eval_count", 0)
        timings["total_s"] = round(time.perf_counter() - t0, 3)
        if prompt_tokens >= settings.num_ctx - 16:
            flags.append("context_possibly_truncated")

        if text.startswith(NOT_FOUND) or not text or (NOT_FOUND in text and not valid):
            return Answer(question, REFUSAL, True, [], passages, confidence, timings, flags + ["model_refused"], self.model)
        text = text.replace(NOT_FOUND, "").strip()
        if settings.verify:
            text, vflags, vlat = self._verify(question, text, hits)
            flags += vflags
            timings["verify_s"] = round(vlat, 3)
            timings["total_s"] = round(time.perf_counter() - t0, 3)
        text, guard_flags = amount_guard(question, text, hits)
        flags += guard_flags
        valid, _, coverage = check_citations(text, len(hits))
        if not valid and not any(f.startswith("amount_") for f in flags):
            flags.append("no_citations")
        citations = [passages[n - 1] for n in sorted(valid)]
        oldest = min((c.post_date for c in citations if c.post_date), default="")
        if oldest and oldest < OLD_SOURCE_BEFORE:
            flags.append(f"source_dated_{oldest}")
        confidence["citation_coverage"] = round(coverage, 3)
        confidence["cited_top_score"] = round(max((c.score for c in citations), default=0.0), 4)
        first = SENTENCE.split(text)[0] if text else ""
        if NOT_STATED.search(first) or (not citations and any(f.startswith("amount_") for f in flags)):
            flags.append("states_not_covered")  # answered by saying the documents do not state it
        return Answer(question, text, False, citations, passages, confidence, timings, flags, self.model)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Ask one question")
    ap.add_argument("question")
    ap.add_argument("--mode", default="hybrid+rerank")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    a = Answerer().ask(args.question, mode=args.mode)
    if args.json:
        print(json.dumps(a.to_dict(), indent=2, ensure_ascii=False))
        return
    print(a.answer, "\n")
    for c in a.citations:
        print(f"  [{c.n}] {c.title} — p. {c.page} ({c.file_name})")
    print("\n", a.timings, a.confidence, a.flags)


if __name__ == "__main__":
    main()
