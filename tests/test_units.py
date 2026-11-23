"""Fast tests: no network, no GPU, no Ollama."""
from visarag.answer import check_citations, format_passages
from visarag.crawl import parse_attachments, parse_list
from visarag.parse import classify_language
from visarag.retrieve import Hit, Retriever

LIST_HTML = """
<table class="board_list01"><tbody>
<tr><td>40<input type="hidden" name="DATA_ID" value="40"></td><td>Manual</td>
<td class="title"><a href="javascript:void(0)" name="TITLE">User Manual for Electronic Form</a></td>
<td><img alt="file"></td><td>2016-04-12 10:00:00.0</td><td>99</td></tr>
<tr><td>header row without id</td></tr>
</tbody></table>"""

DETAIL_HTML = """
<div class="file_list02"><ul><li>
<input type="hidden" name="APND_FILE_SEQ" value="AAHdL6sQBqt0qqb3jZdyY1YG">
<a href="javascript:void(0)" name="fileNm">User+Manual+for+Electronic+Form.pdf</a><span> (599 kb)</span>
</li></ul></div>"""


def test_parse_list_reads_id_title_category_date():
    rows = parse_list(LIST_HTML, "DATA_ID")
    assert rows == [
        {"id": "40", "title": "User Manual for Electronic Form", "category": "Manual", "date": "2016-04-12"}
    ]


def test_parse_attachments_decodes_plus_encoded_names():
    assert parse_attachments(DETAIL_HTML) == [("AAHdL6sQBqt0qqb3jZdyY1YG", "User Manual for Electronic Form.pdf")]


def test_language_english():
    text = "Applicants for long-term visas must submit a tuberculosis test certificate. " * 10
    lang, _, _, english, _ = classify_language(text)
    assert (lang, english) == ("en", True)


def test_language_korean_only_is_skipped():
    text = "장기비자 신청 시 결핵진단서를 제출하여야 합니다. " * 30
    *_, english, reason = classify_language(text)
    assert not english and "Latin" in reason


def test_language_vietnamese_is_not_english():
    text = "Người nộp đơn xin thị thực du lịch phải nộp hồ sơ đầy đủ tại trung tâm tiếp nhận. " * 10
    lang, _, _, english, _ = classify_language(text)
    assert lang == "vi" and not english


def test_language_bilingual_form_counts_as_english():
    text = ("사증발급신청서 성명 생년월일 국적 여권번호 " * 40) + (
        "VISA APPLICATION FORM. Family name, given names, date of birth, nationality, passport number. " * 6
    )
    lang, _, _, english, _ = classify_language(text)
    assert english and lang in {"en", "ko+en"}


def test_check_citations_finds_invalid_and_coverage():
    valid, invalid, cov = check_citations(
        "Long-term visa applicants must submit a TB certificate [1]. It is valid for three months [7]. "
        "Some countries are exempt.",
        k=5,
    )
    assert valid == {1} and invalid == {7}
    assert round(cov, 2) == 0.67


def _hit(cid: str) -> Hit:
    return Hit(cid, "d", "t", "f.pdf", 1, [1], "", "body", "text", "u", "p", "")


def test_rrf_rewards_agreement_between_arms():
    fused = Retriever.rrf({"dense": [_hit("a"), _hit("b")], "sparse": [_hit("b"), _hit("c")]}, k=60)
    assert fused[0].chunk_id == "b"
    assert fused[0].ranks == {"dense": 2, "sparse": 1}


def test_format_passages_numbers_and_pages():
    h = Hit("x", "d", "TB notice", "tb.pdf", 2, [2, 3], "Scope", "Submit the certificate.", "", "u", "p", "2016-02-11")
    out = format_passages([h])
    assert out.startswith("[1] TB notice (pages 2-3, published 2016-02-11) | section: Scope")


def test_parse_verdict_handles_schema_and_model_invented_shapes():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from calibrate import parse_verdict

    assert parse_verdict('{"checks": "ok", "verdict": "PASS"}') == "PASS"
    assert parse_verdict('{"is_correct": false, "reasoning": "x"}') == "FAIL"
    assert parse_verdict("Claim 1 is stated. Claim 2 is not.\nVerdict: FAIL") == "FAIL"
    assert parse_verdict("no verdict here") == "ERROR"
