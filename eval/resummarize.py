"""Recompute an answers summary after a change to deterministic post-processing.

    uv run python eval/resummarize.py results/<run>-answers-<tag>.jsonl

Re-applies the "states not covered" detector to saved answers (no model calls)
and writes <run>.summary.json next to the file.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from run import summarize_answers

from visarag.answer import NOT_STATED, SENTENCE

path = Path(sys.argv[1])
rows = [json.loads(x) for x in path.open(encoding="utf-8")]
for r in rows:
    first = SENTENCE.split(r["answer"])[0] if r["answer"] else ""
    if not r["refused"] and NOT_STATED.search(first) and "states_not_covered" not in r["flags"]:
        r["flags"].append("states_not_covered")
with path.open("w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
summary = summarize_answers(rows, "qwen3.5:4b")
path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
