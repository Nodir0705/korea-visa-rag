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

import json
import re
import time
from dataclasses import asdict, dataclass, field

from visarag.config import settings
from visarag.ollama import Ollama
from visarag.retrieve import Hit, Retriever

REFUSAL = (
    "I could not find this in the visa.go.kr documents I have. "
    "Please check the Korea Visa Portal or ask the Korean embassy or consulate that handles your application."
)

SYSTEM = """You answer questions about Korean visas using ONLY the numbered passages provided.
The passages come from official PDF documents published on the Korea Visa Portal (visa.go.kr).

Rules:
- Start with the direct answer. If the question is yes/no, the first word is Yes or No.
- Be brief: 1-4 sentences, or a short list when the answer is a list. Answer only what was asked.
- End every sentence that states a fact with the number of the passage that states it, like [2].
  Cite only passages that actually state that fact.
- Use only facts written in the passages. No outside knowledge, no guessing, no filling gaps.
- Keep numbers, amounts, dates, durations, country names and visa codes exactly as written.
  Before saying a country or visa type is on a list, check that it is really in that list.
- Use a passage only if it is about the same visa, permit, form or procedure the question asks about.
  A fee, deadline or rule written for something else (for example a re-entry permit) is not the answer
  to a question about a visa. If no passage is about the thing asked, reply NOT_FOUND.
- If the passages answer only part of the question, answer that part and say in one short sentence
  what is not covered.
- If the passages do not contain the answer, reply with exactly: NOT_FOUND
- Do not mention publication dates, passage numbers outside the [n] markers, or the passages themselves."""

NOT_FOUND = "NOT_FOUND"
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
        pdf_url=f"/pdf/{h.doc_id}.pdf#page={h.page}",
        post_date=h.post_date,
        snippet=h.body,
        score=round(h.score, 4),
    )


def check_citations(text: str, k: int) -> tuple[set[int], set[int], float]:
    """(valid markers, invalid markers, share of factual sentences carrying a marker)."""
    found = {int(m) for m in MARKER.findall(text)}
    valid = {n for n in found if 1 <= n <= k}
    sentences = [s for s in SENTENCE.split(text) if len(re.sub(r"\W", "", s)) > 15]
    cited = sum(bool(MARKER.search(s)) for s in sentences)
    return valid, found - valid, (cited / len(sentences)) if sentences else 0.0


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

    def ask(self, question: str, mode: str = "hybrid+rerank") -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, mode=mode)
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
        if not valid:
            flags.append("no_citations")
        citations = [passages[n - 1] for n in sorted(valid)]
        oldest = min((c.post_date for c in citations if c.post_date), default="")
        if oldest and oldest < OLD_SOURCE_BEFORE:
            flags.append(f"source_dated_{oldest}")
        confidence["citation_coverage"] = round(coverage, 3)
        confidence["cited_top_score"] = round(max((c.score for c in citations), default=0.0), 4)
        text = text.replace(NOT_FOUND, "").strip()
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
