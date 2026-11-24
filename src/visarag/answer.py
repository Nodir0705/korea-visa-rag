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
- Start with the direct answer. If the question is yes/no, the first word is Yes or No.
- Be brief: 1-4 sentences, or a short list when the answer is a list. Answer only what was asked:
  do not add related rules, exceptions, conditions or background the question did not ask about.
- End every sentence that states a fact with the number of the passage that states it, like [2].
  Cite only passages that actually state that fact.
- Use only facts written in the passages. No outside knowledge, no guessing, no filling gaps.
- Keep numbers, amounts, dates, durations, country names and visa codes exactly as written.
  Before saying a country or visa type is on a list, check that it is really in that list.
- Use a passage only if it is about the same visa, permit, form or procedure the question asks about.
  A fee, deadline or rule written for something else (for example a re-entry permit) is not the answer
  to a question about a visa. If no passage is about the thing asked, reply NOT_FOUND.
- A visa guide passage is about exactly one visa code, given in its title. Do not apply it to a different code.
- If the passages answer only part of the question, answer that part and say in one short sentence
  what is not covered.
- If the passages do not contain the answer, reply with exactly: NOT_FOUND
- Do not mention publication dates, passage numbers outside the [n] markers, or the passages themselves."""

NOT_FOUND = "NOT_FOUND"
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
OLD_SOURCE_BEFORE = "2023-11-24"  # sources older than ~3 years are flagged for the reader
MARKER = re.compile(r"\[(\d+)\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\-•*])|\n+")


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
            if any(n.rstrip(".,") not in sources for n in NUMBER.findall(bare)):
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
    total, kept = 0, []
    for h in out:  # keep the context inside num_ctx; hits are in rank order
        total += len(h.body)
        if kept and total > settings.context_max_chars:
            break
        kept.append(h)
    return kept


class Answerer:
    def __init__(self, retriever: Retriever | None = None, model: str | None = None) -> None:
        self.retriever = retriever or Retriever()
        self.model = model or settings.gen_model
        self.llm = Ollama()

    def _generate(self, question: str, hits: list[Hit], correction: str = "") -> tuple[str, dict]:
        user = (
            f"Passages:\n\n{format_passages(hits)}\n\n"
            f"Question: {question}\n\n"
            f"Answer from passages [1]-[{len(hits)}] only, with a [n] marker after every factual sentence, "
            f"or reply {NOT_FOUND} if they do not contain the answer."
            + (f"\n\n{correction}" if correction else "")
        )
        data = self.llm.chat(
            self.model,
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            num_predict=600,
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
        timings = {k: round(v, 3) for k, v in self.retriever.timings.items()}
        passages = [_citation(i, h) for i, h in enumerate(hits, 1)]
        top = max((h.score for h in hits), default=0.0)
        confidence = {"retrieval_top_score": round(top, 4)}
        flags: list[str] = []

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
