"""Score a sealed set from its two blind graders.

    uv run python eval/score_sealed.py heldout4 gpu8 gpu16

Fully right = both graders say yes. Main point right = both say yes or partly. A question is
dropped only if both graders call it invalid (q_valid "no") for every answer graded.
"""
import collections
import glob
import json
import sys
from pathlib import Path

name, runs = sys.argv[1], sys.argv[2:]
D = f"results/sealed/{name}"
A = {v["gid"]: v for f in sorted(glob.glob(f"{D}-grade-A-*.json")) for v in json.loads(Path(f).read_text())}
B = {v["gid"]: v for f in sorted(glob.glob(f"{D}-grade-B-*.json")) for v in json.loads(Path(f).read_text())}
mapping = json.loads(Path(f"{D}-mapping.json").read_text())
gids = {m["h"] for m in mapping}
agree = sum(A[g]["ans_ok"] == B[g]["ans_ok"] for g in gids)
print(f"unique answers {len(gids)}, grader agreement {agree / len(gids):.1%}")
by_q = collections.defaultdict(list)
for m in mapping:
    by_q[m["id"]].append(m["h"])
invalid = sorted(q for q, hs in by_q.items() if all(A[h]["q_valid"] == "no" and B[h]["q_valid"] == "no" for h in hs))
print("dropped:", invalid or "none")
for run in runs:
    rows = [m for m in mapping if m["run"] == run and m["id"] not in invalid]
    full = sum(A[m["h"]]["ans_ok"] == "yes" and B[m["h"]]["ans_ok"] == "yes" for m in rows)
    core = sum(A[m["h"]]["ans_ok"] != "no" and B[m["h"]]["ans_ok"] != "no" for m in rows)
    n = len(rows)
    print(f"{run:9} n={n}  main point {core}/{n} = {core / n:.1%}   fully right {full}/{n} = {full / n:.1%}")
