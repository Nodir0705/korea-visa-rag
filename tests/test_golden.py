"""The golden set is well-formed and every gold document exists in the parsed corpus."""
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = yaml.safe_load((ROOT / "eval" / "golden.yaml").read_text(encoding="utf-8"))["questions"]
TYPES = {"lookup", "multi_hop", "table", "paraphrase", "no_answer"}


def test_ids_unique_and_types_known():
    ids = [q["id"] for q in GOLDEN]
    assert len(ids) == len(set(ids))
    assert {q["type"] for q in GOLDEN} <= TYPES


def test_answerable_questions_have_gold_and_unanswerable_do_not():
    for q in GOLDEN:
        if q["type"] == "no_answer":
            assert not q.get("gold"), q["id"]
        else:
            assert q.get("gold") and q.get("answer"), q["id"]


def test_gold_documents_are_indexed_english_documents():
    catalog = ROOT / "data" / "parsed" / "catalog.jsonl"
    if not catalog.exists():
        pytest.skip("corpus not built")
    english = {json.loads(x)["sha12"] for x in catalog.open() if json.loads(x)["english"]}
    for q in GOLDEN:
        for g in q.get("gold") or []:
            assert g["doc"] in english, (q["id"], g["doc"])
