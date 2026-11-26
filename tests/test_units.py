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
    _, flags = amount_guard("On the 2020 Health Condition Report Form, which places must applicants list?",
                               "On the 2020 form, applicants list all cities stayed in within 30 days [1].", [form])
    assert flags == []


def _page(code: str, headings: str = "") -> Hit:
    return Hit(chunk_id=f"page-{code}#1", doc_id=f"page-{code}", title=f"Visa guide: {code}", file_name="",
               page=0, pages=[], headings=headings, body="b", text="t", source_url="", pdf_path="",
               post_date="", source="page")


def test_scope_filter_drops_other_visa_codes_and_routes_but_keeps_pdfs(tmp_path, monkeypatch):
    from visarag import answer
    from visarag.answer import scope_filter
    from visarag.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)  # no page files: route chunks are judged by heading only
    answer._guide_sections.cache_clear()

    hits = [_page("D-8-1", "Embassy visa application — Case 1 > Required documents"),
            _page("D-8-3", "Embassy visa application — Case 1 > Required documents"),
            _hit("pdf1")]
    kept, flags = scope_filter("What must a D-8-1 investor submit?", hits)
    assert [h.doc_id for h in kept] == ["page-D-8-1", hits[2].doc_id] and flags == ["scoped_to_visa_code"]

    hits = [_page("E-3", "Embassy visa application — Case 1 > Required documents"),
            _page("E-3", "Visa Issuance Confirmation (CCVI) application — Case 3 > Required documents"),
            _page("E-3", "Overview (eligible applicants / activities)")]
    kept, flags = scope_filter("What does a researcher submit for an E-3 visa at an embassy?", hits)
    assert [h.headings.split()[0] for h in kept] == ["Embassy", "Overview"] and flags == ["scoped_to_route"]
    # A CCVI chunk of a page that also has an embassy route stays; the guide expansion swaps in that route.
    (tmp_path / "pages" / "en").mkdir(parents=True)
    (tmp_path / "pages" / "en" / "E-3.md").write_text("## Embassy visa application\nx\n", encoding="utf-8")
    answer._guide_sections.cache_clear()
    kept, _ = scope_filter("What does a researcher submit for an E-3 visa at an embassy?", hits)
    assert len(kept) == 3
    answer._guide_sections.cache_clear()


def test_scope_filter_keeps_everything_without_a_match_and_accepts_sub_types():
    from visarag.answer import scope_filter

    hits = [_page("C-3-9"), _page("B-1")]
    assert scope_filter("What is the D-2 work limit?", hits)[0] == hits  # no D-2 page retrieved
    assert scope_filter("Who needs a visa?", hits)[0] == hits  # no code or route named
    assert [h.doc_id for h in scope_filter("What does a C-3 visa allow?", hits)[0]] == ["page-C-3-9"]  # sub-type


def test_expand_guide_pages_gives_overview_and_the_named_route(tmp_path, monkeypatch):
    from visarag import answer
    from visarag.config import settings

    (tmp_path / "pages" / "en").mkdir(parents=True)
    (tmp_path / "pages" / "en" / "E-3.md").write_text(
        "---\nnat_cd: 275\n---\n\n# E-3 Researcher\n\n## Overview (eligible applicants / activities)\nAged 25 or older.\n\n"
        "## Embassy visa application\n### Embassy visa application — Case 1\nDegree certificate.\n\n"
        "## Visa Issuance Confirmation (CCVI) application\n### Visa Issuance Confirmation (CCVI) application — Case 3\n"
        "Recommendation letter.\n", encoding="utf-8")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    answer._guide_sections.cache_clear()
    hits = [_page("E-3", "Embassy visa application — Case 1 > Required documents"),
            _page("E-3", "Overview (eligible applicants / activities)")]
    out = answer.expand_guide_pages("What does an E-3 applicant submit at an embassy?", hits)
    assert len(out) == 1  # one passage per page
    assert "Aged 25" in out[0].body and "Degree certificate" in out[0].body and "Recommendation" not in out[0].body
    out = answer.expand_guide_pages("Who can get an E-3 visa?", hits[:1])
    assert "Recommendation letter" in out[0].body  # short page: given whole
    answer._guide_sections.cache_clear()


def test_is_yes_no():
    from visarag.answer import is_yes_no

    assert is_yes_no("Can I apply for a multiple re-entry permit at the airport?")
    assert is_yes_no("im australian doing a working holiday in korea, do i need to show medical insurance?")
    assert not is_yes_no("Which short-term visa types are covered by the accommodation declaration rule?")
    assert not is_yes_no("How much does an apostille cost in Macao?")
    assert not is_yes_no("Is the H-1 working holiday visa single or multiple entry, and how long is it valid?")
    assert is_yes_no("Can the minor child of a permanent resident get an F-2-3 visa, and what must they submit?")


def test_expand_guide_pages_keeps_only_the_case_for_long_pages(tmp_path, monkeypatch):
    from visarag import answer
    from visarag.config import settings

    (tmp_path / "pages" / "en").mkdir(parents=True)
    filler = "x" * 5000
    (tmp_path / "pages" / "en" / "B-2-1.md").write_text(
        "# B-2-1\n\n## Overview (eligible applicants / activities)\nTourists.\n\n## Visa-free entry\n"
        f"### APEC Business Travel Card Holders (ABTC)\nABTC rule.\n\n### Visa Free Entry to the Jeju Area\n{filler}\n\n"
        f"### Re-Entry Permit Holders\n{filler}\n", encoding="utf-8")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    answer._guide_sections.cache_clear()
    hit = _page("B-2-1", "APEC Business Travel Card Holders (ABTC) > Eligibility")
    body = answer.expand_guide_pages("Can ABTC holders enter without a visa?", [hit])[0].body
    assert "Tourists." in body and "ABTC rule." in body and filler not in body
    answer._guide_sections.cache_clear()


def test_long_page_also_gets_the_cases_closest_to_the_question(tmp_path, monkeypatch):
    from visarag import answer
    from visarag.config import settings

    (tmp_path / "pages" / "en").mkdir(parents=True)
    filler = "x" * 5000
    (tmp_path / "pages" / "en" / "C-3-1.md").write_text(
        "# C-3-1\n\n## Overview (eligible applicants / activities)\nShort visits.\n\n## Embassy visa application\n"
        f"### Embassy visa application — Case 1\nEvent or training. {filler}\n\n"
        f"### Embassy visa application — Case 6\nParent of a diplomat. Foreign Minister letter.\n\n"
        f"## Visa Issuance Confirmation (CCVI) application\n### Visa Issuance Confirmation (CCVI) application — Case 1\n{filler}\n",
        encoding="utf-8")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    answer._guide_sections.cache_clear()
    hit = _page("C-3-1", "Embassy visa application — Case 1 > Required documents")

    def score(q, texts):
        return [1.0 if "diplomat" in t else 0.0 for t in texts]

    body = answer.expand_guide_pages("A diplomat's mother applies at the embassy; documents?", [hit], score)[0].body
    assert "Short visits." in body and "Foreign Minister letter" in body and "Event or training" in body
    assert body.count(filler) == 1  # the CCVI route is left out
    answer._guide_sections.cache_clear()


def test_named_visa_page_is_added_second():
    from visarag.answer import add_named_pages

    class FakeRetriever:
        def page_hit(self, code):
            return _page(code) if code in ("D-4-2", "E-1") else None

    hits = [_page("F-1-23"), _page("D-8-1")]
    out, flags = add_named_pages("d-4-2 internship, bank acount? D-4-2 and E-1 and Z-9", hits, FakeRetriever())
    assert [h.doc_id for h in out] == ["page-F-1-23", "page-D-4-2", "page-E-1", "page-D-8-1"]
    assert flags == ["added_named_visa_page"]
    assert add_named_pages("no code here", hits, FakeRetriever()) == (hits, [])


def test_table_rows_from_docling_grid(tmp_path, monkeypatch):
    import json

    from visarag import answer
    from visarag.config import settings

    cells = [(0, 0, "Type", True), (0, 1, "Illegal stayers", True), (0, 2, "Rate", True),
             (1, 0, "Suspension (6 months)", False), (1, 1, "9 ~ 10", False), (1, 2, "2.0%", False),
             (2, 0, "Suspension (3 months)", False), (2, 1, "7 ~ 8", False), (2, 2, "1.5%", False),
             (3, 0, "Warning", False), (3, 1, "1 ~ 2", False), (3, 2, "0.2%", False)]
    doc = {"tables": [{"prov": [{"page_no": 12}], "data": {"num_rows": 4, "num_cols": 3, "table_cells": [
        {"start_row_offset_idx": r, "end_row_offset_idx": r + 1, "start_col_offset_idx": c,
         "end_col_offset_idx": c + 1, "text": t, "column_header": h} for r, c, t, h in cells]}}]}
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.parsed_dir.mkdir(parents=True)
    (settings.parsed_dir / "abc.json").write_text(json.dumps(doc), encoding="utf-8")
    answer._doc_tables.cache_clear()
    hit = Hit("c", "abc", "Regulation", "f.pdf", 12, [12], "", "page text", "page text", "", "", "")
    out = answer.add_table_rows([hit, hit])
    assert "- Type: Suspension (3 months); Illegal stayers: 7 ~ 8; Rate: 1.5%" in out[0].body
    assert out[1].body == "page text"  # once per page
    answer._doc_tables.cache_clear()


def test_count_of_a_numbered_list_is_grounded():
    from visarag.answer import amount_guard

    hit = Hit("c", "tb", "TB notice", "f.pdf", 2, [2], "", "(High TB-burden countries) ① Nepal ② Russia ③ China",
              "", "", "", "")
    text, flags = amount_guard("How many countries are on the list?", "There are 3 countries [1].", [hit])
    assert flags == [] and text.startswith("There are 3")
    text, flags = amount_guard("Which countries are on the list?", "There are 3 countries [1].", [hit])
    assert flags == ["number_not_in_sources"]


def test_table_rows_skip_tables_whose_header_is_a_data_row(tmp_path, monkeypatch):
    import json

    from visarag import answer
    from visarag.config import settings

    long = "Nauru, Marshall Islands, Micronesia, Kiribati, Australia, Solomon Islands, Tuvalu, Fiji"
    cells = [(0, 0, "Asia-Pacific (14)", True), (0, 1, long, True),
             (1, 0, "Americas (5)", False), (1, 1, "Argentina, Canada", False),
             (2, 0, "Europe (5)", False), (2, 1, "Cyprus, Serbia", False)]
    doc = {"tables": [{"prov": [{"page_no": 2}], "data": {"num_rows": 3, "num_cols": 2, "table_cells": [
        {"start_row_offset_idx": r, "end_row_offset_idx": r + 1, "start_col_offset_idx": c,
         "end_col_offset_idx": c + 1, "text": t, "column_header": h} for r, c, t, h in cells]}}]}
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.parsed_dir.mkdir(parents=True)
    (settings.parsed_dir / "vf.json").write_text(json.dumps(doc), encoding="utf-8")
    answer._doc_tables.cache_clear()
    assert answer._doc_tables("vf") == {}
    answer._doc_tables.cache_clear()
