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


def test_heading_trail_adds_subject_only_for_field_labels():
    from visarag.index import _heading_trail

    headers = [(1, "Bulgaria - Competent Authority (Art. 6)"), (5, "Contact details:")]
    assert _heading_trail(headers, 6, ["Contact details:"]) == ["Bulgaria - Competent Authority (Art. 6)", "Contact details:"]
    tb = [(1, "[Note] Exceptions (the following persons are exempt from submission)"), (5, "(High TB-burden countries)")]
    assert _heading_trail(tb, 6, ["(High TB-burden countries)"]) == ["(High TB-burden countries)"]


def test_page_sections_keep_full_heading_path_and_pack_long_lists():
    from visarag.pages import pack, sections

    body = "# E-7-1 Professionals\n## Embassy visa application\n### Case 1\n#### Required documents\n① form\n② contract\n#### Visa validity\nsingle-entry, 3 months\n"
    secs = sections(body)
    assert secs[0] == (["Embassy visa application", "Case 1", "Required documents"], ["① form", "② contract"])
    assert secs[1][0][-1] == "Visa validity"
    assert pack(["a b", "c d", "e f"], lambda t: len(t.split()), 4) == ["a b\nc d", "e f"]


def test_amount_guard_rejects_fee_from_another_procedure():
    from visarag.answer import amount_guard

    reentry = Hit("r", "a0ab", "Announcement Regarding Re-Entry Permit", "r.pdf", 3, [3], "Fees",
                  "(Fees) 50,000 KRW\nPersons exempted: Student (D-2) scholarship holders", "", "u", "p", "2022-03-31")
    d21 = Hit("g", "page-D-2-1", "Visa guide: D-2-1 Associate Degree", "D-2-1.md", 0, [], "Required documents",
              "Visa Application Form, passport, photo, fee", "", "u", "", "", source="page")
    text, flags = amount_guard("How much does a D-2 visa cost?", "The fee for a D-2 visa is 50,000 KRW [1]. You also need a passport [2].", [reentry, d21])
    assert flags == ["amount_subject_mismatch"] and "do not state this amount for D-2" in text
    same, flags = amount_guard("What is the re-entry permit fee?", "It is 50,000 KRW [1].", [reentry])
    assert flags == [] and same == "It is 50,000 KRW [1]."


def test_not_stated_detector():
    from visarag.answer import NOT_STATED

    assert NOT_STATED.search("No specific minimum TOPIK level is listed as a requirement for the F-6-1 visa.")
    assert NOT_STATED.search("The documents do not state the age limit.")
    assert not NOT_STATED.search("The period of stay is up to 2 years [1].")


def test_expand_to_pages_merges_hits_on_one_short_page(monkeypatch):
    from visarag import answer

    text = "The countries still eligible for visa-free entry after April 13, 2020 are as follows. Malta, Ireland, United Kingdom"
    monkeypatch.setattr(answer, "_page_texts", lambda: {"b481": {"2": text}})
    h1 = Hit("c1", "b481", "Suspension notice", "s.pdf", 2, [2], "", "Malta, Ireland", "", "u", "p", "2020-11-05", score=0.9)
    h2 = Hit("c2", "b481", "Suspension notice", "s.pdf", 2, [2], "", "<10 Visa-waiver>", "", "u", "p", "2020-11-05", score=0.95)
    h3 = Hit("c3", "page-H-1", "Visa guide: H-1", "H-1.md", 0, [], "Fees", "Fee exempted", "", "u", "", "", source="page")
    out = answer.expand_to_pages([h1, h2, h3])
    assert [h.chunk_id for h in out] == ["c1", "c3"]
    assert out[0].body == text and out[0].score == 0.95


def test_verify_verdict_lines_parse():
    from visarag.answer import VERDICT_LINE

    assert VERDICT_LINE.findall("1: KEEP\n2: DROP\n3. keep") == [("1", "KEEP"), ("2", "DROP"), ("3", "keep")]


def test_unwrap_joins_wrapped_names_but_keeps_items():
    from visarag.index import unwrap

    raw = "entry for residents of Hanoi, Ho Chi Minh City, Da \r\nNang in Vietnam.\r\n○ Next item\r\n(Fees) 50,000 KRW\r\n※ Persons exempted"
    assert unwrap(raw) == "entry for residents of Hanoi, Ho Chi Minh City, Da Nang in Vietnam.\n○ Next item\n(Fees) 50,000 KRW\n※ Persons exempted"


def test_amount_guard_rejects_amount_missing_from_sources():
    from visarag.answer import amount_guard

    d81 = Hit("d", "page-D-8-1", "Visa guide: D-8-1 Incorporated Enterprise", "D-8-1.md", 0, [], "Required documents",
              "Additional documents for private investors who has invested less than 300 million KRW", "", "u", "", "",
              source="page")
    text, flags = amount_guard("What is the minimum investment for a D-8-1 visa?", "The minimum is 100 million KRW [1].", [d81])
    assert flags == ["amount_not_in_sources"] and "do not state this amount for D-8-1" in text
    _, flags = amount_guard("Which D-8-1 investors need extra documents?", "Those who invested less than 300 million KRW [1].", [d81])
    assert flags == []


def test_number_grounding_drops_numbers_from_another_page():
    from visarag.answer import amount_guard

    d41 = Hit("d", "page-D-4-1", "Visa guide: D-4-1 Korean Language Trainee", "D-4-1.md", 0, [], "Eligibility",
              "A person with at least high school diploma", "", "u", "", "", source="page")
    text, flags = amount_guard("What TOPIK level is needed for D-4-1?", "The required TOPIK score is 50 out of 200 [1].", [d41])
    assert flags == ["number_not_in_sources"] and text.startswith("The visa.go.kr documents I have do not state this")
    ok_text, flags = amount_guard("What is needed for D-4-1?", "At least a high school diploma [1].", [d41])
    assert flags == [] and ok_text == "At least a high school diploma [1]."


def test_number_grounding_allows_numbers_from_the_question_and_dates():
    from visarag.answer import amount_guard

    form = Hit("f", "80df", "Health Condition Report Form", "f.pdf", 1, [1], "",
               "Please list all cities you have stayed within 30 days prior to application.", "", "u", "p", "2020-02-24")
    text, flags = amount_guard("On the 2020 Health Condition Report Form, which places must applicants list?",
                               "On the 2020 form, applicants list all cities stayed in within 30 days [1].", [form])
    assert flags == []
