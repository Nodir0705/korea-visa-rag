"""Evaluate retrieval and answers against eval/golden.yaml.

    uv run python eval/run.py retrieval            # 4 retrieval arms, deterministic
    uv run python eval/run.py answers              # end-to-end answers + latency
    uv run python eval/run.py judge  results/<run>.answers.jsonl   # LLM judge pass

Relevance is judged at page level: a retrieved chunk is relevant when it comes
from a gold document AND covers at least one gold page. Document-level hit rate
is reported alongside, since a chunk from the right document but a
neighbouring page is often still useful.

The judge runs as a separate pass so the judge model never shares VRAM with
the generator, and so one set of answers can be re-judged by several judges.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from visarag.ollama import Ollama

RESULTS = ROOT / "results"
KS = (1, 3, 5, 10)


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["questions"]


def _relevant(hit, gold: list[dict]) -> bool:
    for g in gold:
        if hit.doc_id == g["doc"] and (not g.get("pages") or set(hit.pages) & set(g["pages"])):
            return True
    return False


def _doc_relevant(hit, gold: list[dict]) -> bool:
    return any(hit.doc_id == g["doc"] for g in gold)


def retrieval(golden: list[dict], modes: list[str], embed_model: str | None, reranker: str | None) -> dict:
    from visarag.retrieve import Retriever

    r = Retriever(embed_model=embed_model, reranker=reranker)
    qs = [q for q in golden if q.get("gold")]
    table = {}
    per_q = []
    from visarag.embed import _query_cached

    for mode in modes:
        _query_cached.cache_clear()  # every arm pays for its own query embedding
        hits_at = {k: 0 for k in KS}
        doc_hits_at5 = 0
        rr, ndcg, lat = [], [], []
        for q in qs:
            t = time.perf_counter()
            hits = r.search(q["question"], k=10, mode=mode)
            lat.append(time.perf_counter() - t)
            rel = [_relevant(h, q["gold"]) for h in hits]
            for k in KS:
                hits_at[k] += any(rel[:k])
            doc_hits_at5 += any(_doc_relevant(h, q["gold"]) for h in hits[:5])
            first = next((i for i, x in enumerate(rel, 1) if x), None)
            rr.append(1 / first if first else 0.0)
            dcg = sum(1 / math.log2(i + 1) for i, x in enumerate(rel, 1) if x)
            ideal = sum(1 / math.log2(i + 1) for i in range(1, min(sum(rel), 10) + 1)) or 1.0
            ndcg.append(dcg / ideal if any(rel) else 0.0)
            per_q.append({"mode": mode, "id": q["id"], "first_relevant_rank": first})
        n = len(qs)
        table[mode] = {
            **{f"page_recall@{k}": round(hits_at[k] / n, 3) for k in KS},
            "doc_recall@5": round(doc_hits_at5 / n, 3),
            "mrr@10": round(statistics.mean(rr), 3),
            "ndcg@10": round(statistics.mean(ndcg), 3),
            "p50_ms": round(statistics.median(lat) * 1000),
            "p95_ms": round(_p95(lat) * 1000),
        }
    return {"questions": len(qs), "embed_model": r.embed_model, "reranker": r.reranker_name, "table": table, "per_question": per_q}


def _p95(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)] if xs else 0.0


def answers(golden: list[dict], gen_model: str | None, out: Path) -> dict:
    from visarag.answer import Answerer

    a = Answerer(model=gen_model)
    a.retriever.warm()
    a.llm.chat(a.model, [{"role": "user", "content": "OK"}], num_predict=1)
    rows = []
    for q in golden:
        res = a.ask(q["question"])
        cited_rel = [
            any(c.n == p.n for p in res.passages) and _relevant_c(c, q.get("gold") or [])
            for c in res.citations
        ]
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "reference": q.get("answer", ""),
                "gold": q.get("gold") or [],
                "answer": res.answer,
                "refused": res.refused,
                "citations": [c.__dict__ for c in res.citations],
                "passages": [p.__dict__ for p in res.passages],
                "cited_on_gold_page": sum(cited_rel),
                "n_citations": len(res.citations),
                "timings": res.timings,
                "flags": res.flags,
                "confidence": res.confidence,
            }
        )
        print(f"{q['id']:>5} {'REFUSED' if res.refused else 'ok':8} {res.timings.get('total_s')}s  {res.answer[:90]!r}")
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summarize_answers(rows, a.model)


def _relevant_c(c, gold: list[dict]) -> bool:
    doc = c.doc_id
    return any(doc == g["doc"] and (not g.get("pages") or set(c.pages) & set(g["pages"])) for g in gold)


def summarize_answers(rows: list[dict], model: str) -> dict:
    ans = [r for r in rows if r["type"] != "no_answer"]
    noans = [r for r in rows if r["type"] == "no_answer"]
    tot = [r["timings"]["total_s"] for r in rows]
    gen = [r["timings"]["generate_s"] for r in rows if "generate_s" in r["timings"]]
    cited = [r for r in ans if not r["refused"]]
    return {
        "generator": model,
        "questions": len(rows),
        "answered_when_answerable": round(sum(not r["refused"] for r in ans) / max(len(ans), 1), 3),
        "refused_when_unanswerable": round(sum(r["refused"] for r in noans) / max(len(noans), 1), 3),
        "refused_or_said_not_stated_when_unanswerable": round(
            sum(r["refused"] or "states_not_covered" in r["flags"] for r in noans) / max(len(noans), 1), 3
        ),
        "unanswerable_answered_with_claims": [
            r["id"] for r in noans if not r["refused"] and "states_not_covered" not in r["flags"]
        ],
        "answers_citing_a_gold_page": round(sum(r["cited_on_gold_page"] > 0 for r in cited) / max(len(cited), 1), 3),
        "citation_precision_gold_page": round(
            sum(r["cited_on_gold_page"] for r in cited) / max(sum(r["n_citations"] for r in cited), 1), 3
        ),
        "invalid_citation_flags": sum(any("invalid" in f for f in r["flags"]) for r in rows),
        "latency_p50_s": round(statistics.median(tot), 2),
        "latency_p95_s": round(_p95(tot), 2),
        "generation_p50_s": round(statistics.median(gen), 2) if gen else None,
        "mean_prompt_tokens": round(statistics.mean(r["timings"].get("prompt_tokens", 0) for r in rows if r["timings"].get("prompt_tokens"))) if gen else None,
    }


SUPPORT_PROMPT = """Question: {question}

System answer: {answer}

Passages the system cited:
{cited}

Is every factual claim in the system answer stated in the cited passages? Go claim by claim.
A claim that may be true but is not written in the cited passages is unsupported.
Reply with a short check of each claim, then the verdict PASS (all supported) or FAIL."""


def judge(answers_path: Path, judge_model: str, think: bool = False) -> dict:
    """Correctness uses the exact prompt calibrated in eval/calibrate.py; support is a second call."""
    from calibrate import SYSTEM, judge_one, parse_verdict

    llm = Ollama()
    rows = [json.loads(x) for x in answers_path.open(encoding="utf-8")]
    graded = []
    for r in rows:
        if r["type"] == "no_answer" or r["refused"]:
            graded.append({**r, "judge": None})
            continue
        correct = judge_one(llm, judge_model, r, think)
        cited = "\n\n".join(f"[{c['n']}] {c['title']} p.{c['page']}:\n{c['snippet']}" for c in r["citations"]) or "(none)"
        data = llm.chat(
            judge_model,
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": SUPPORT_PROMPT.format(question=r["question"], answer=r["answer"], cited=cited)}],
            num_ctx=8192,
        )
        verdict = {"correct": correct, "supported": parse_verdict(data["message"]["content"])}
        graded.append({**r, "judge": verdict})
        print(f"{r['id']:>5} correct={verdict['correct']} supported={verdict['supported']}", flush=True)
    answers_path = answers_path.resolve()
    tag = judge_model.replace(":", "_").replace("/", "_") + ("-think" if think else "")
    out = answers_path.with_suffix("").with_suffix(f".judged-{tag}.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for g in graded:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    answerable = [g for g in graded if g["type"] != "no_answer"]
    judged = [g for g in answerable if g["judge"]]
    return {
        "judge": judge_model + (" (think)" if think else ""),
        "answerable": len(answerable),
        "correct_over_answerable": round(sum(g["judge"]["correct"] == "PASS" for g in judged) / max(len(answerable), 1), 3),
        "supported_over_answered": round(sum(g["judge"]["supported"] == "PASS" for g in judged) / max(len(judged), 1), 3),
        "judge_errors": sum("ERROR" in g["judge"].values() for g in judged),
        "failed_correctness": [g["id"] for g in judged if g["judge"]["correct"] != "PASS"],
        "file": str(out.relative_to(ROOT)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--modes", nargs="+", default=["dense", "sparse", "hybrid", "hybrid+rerank"])
    r.add_argument("--embed-model")
    r.add_argument("--reranker")
    a = sub.add_parser("answers")
    a.add_argument("--gen-model")
    j = sub.add_parser("judge")
    j.add_argument("answers_file", type=Path)
    j.add_argument("--judge-model", default="mistral-nemo")
    j.add_argument("--think", action="store_true")
    for p in (r, a):
        p.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.yaml")
        p.add_argument("--tag", default="")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    if args.cmd == "retrieval":
        res = retrieval(load_golden(args.golden), args.modes, args.embed_model, args.reranker)
        out = RESULTS / f"{stamp}-retrieval{('-' + args.tag) if args.tag else ''}.json"
        out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res["table"], indent=2))
    elif args.cmd == "answers":
        out = RESULTS / f"{stamp}-answers{('-' + args.tag) if args.tag else ''}.jsonl"
        summary = answers(load_golden(args.golden), args.gen_model, out)
        out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    else:
        summary = judge(args.answers_file, args.judge_model, args.think)
        (ROOT / summary["file"]).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
