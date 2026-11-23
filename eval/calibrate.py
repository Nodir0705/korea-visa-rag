"""Calibrate candidate judge models against eval/judge_cases.yaml.

    uv run python eval/calibrate.py --models mistral-nemo qwen3.5:9b

Reports, per judge: how many wrong answers it catches (TPR on FAIL), how many
correct answers it keeps (TNR on PASS), accuracy, and Cohen's kappa. A judge
that passes everything scores kappa 0 however high its accuracy looks, which is
exactly the failure seen with an uncalibrated judge on this set.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from visarag.ollama import Ollama

SYSTEM = """You check answers produced by a question-answering system about Korean visas against a reference answer.
You are strict about facts and lenient about wording. Return JSON only."""

PROMPT = """Question: {question}

Reference answer (from the source document): {reference}

System answer: {answer}

Is the system answer correct? Work through it:
1. Find every number, amount, duration, date, country, city and visa code in the system answer that also
   appears in, or contradicts, the reference. Compare them one by one.
2. Check yes/no direction: if the reference says yes and the system says no (or the reverse), it is wrong.
3. Check that the system answer does not start with a claim that the reference contradicts.
Verdict FAIL if any compared value differs, any direction is reversed, or a central fact of the reference is
missing. Extra detail that does not contradict the reference is fine. Otherwise PASS."""

SCHEMA = {
    "type": "object",
    "properties": {"checks": {"type": "string"}, "verdict": {"type": "string", "enum": ["PASS", "FAIL"]}},
    "required": ["checks", "verdict"],
}


def judge_one(llm: Ollama, model: str, case: dict, think: bool) -> str:
    body_msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(**{k: case[k] for k in ("question", "reference", "answer")})},
    ]
    if think:
        r = llm.client.post("/api/chat", json={
            "model": model, "messages": body_msgs, "stream": False, "think": True,
            "options": {"temperature": 0, "num_ctx": 8192, "seed": 7},
        }).json()
        return parse_verdict(r["message"]["content"])
    data = llm.chat(model, body_msgs, schema=SCHEMA, num_ctx=8192)
    return parse_verdict(data["message"]["content"])


def parse_verdict(text: str) -> str:
    """Schema output when the model honours ``format``; otherwise the last PASS/FAIL in the text
    (qwen3.5 on Ollama 0.17.7 ignores ``format`` when thinking is off)."""
    try:
        obj = json.loads(text)
        v = obj.get("verdict")
        if isinstance(v, str) and v.upper() in ("PASS", "FAIL"):
            return v.upper()
        # Models that ignore the schema invent their own key, e.g. {"is_correct": true}.
        for key, val in obj.items():
            if isinstance(val, bool) and any(w in key.lower() for w in ("correct", "pass", "support")):
                return "PASS" if val else "FAIL"
    except (json.JSONDecodeError, AttributeError):
        pass
    found = re.findall(r"\b(PASS|FAIL)\b", text.upper())
    return found[-1] if found else "ERROR"


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a == "PASS" for a, _ in pairs) / n
    pb = sum(b == "PASS" for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mistral-nemo"])
    ap.add_argument("--think", action="store_true", help="let reasoning models think before the verdict")
    args = ap.parse_args()
    cases = yaml.safe_load((ROOT / "eval" / "judge_cases.yaml").read_text(encoding="utf-8"))["cases"]
    llm = Ollama()
    report = {}
    for model in args.models:
        t0 = time.perf_counter()
        pairs, misses = [], []
        for c in cases:
            v = judge_one(llm, model, c, args.think)
            pairs.append((c["label"], v))
            if v != c["label"]:
                misses.append({"id": c["id"], "label": c["label"], "judge": v})
        fails = [(a, b) for a, b in pairs if a == "FAIL"]
        passes = [(a, b) for a, b in pairs if a == "PASS"]
        report[model + (" (think)" if args.think else "")] = {
            "cases": len(cases),
            "caught_wrong_answers": f"{sum(b == 'FAIL' for _, b in fails)}/{len(fails)}",
            "kept_correct_answers": f"{sum(b == 'PASS' for _, b in passes)}/{len(passes)}",
            "accuracy": round(sum(a == b for a, b in pairs) / len(pairs), 3),
            "cohen_kappa": round(kappa(pairs), 3),
            "seconds": round(time.perf_counter() - t0, 1),
            "disagreements": misses,
        }
        print(json.dumps({k: v for k, v in report[list(report)[-1]].items() if k != "disagreements"}), model)
    out = ROOT / "results" / f"judge-calibration-{time.strftime('%Y%m%d-%H%M')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(out)


if __name__ == "__main__":
    main()
