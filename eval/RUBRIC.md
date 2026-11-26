# Strict grading rubric (fixed; reuse verbatim for every grading call)

You grade one system answer to one question about Korean visas. Judge against the corpus
text (the PDFs and guide pages), using the reference answer as a guide but not as the truth:
if the reference is wrong or incomplete, say so and grade against the corpus.

Verdict `ans_ok`, one of:

- **yes**: the main answer is correct AND every other claim in the answer is supported by
  the corpus for the thing asked AND nothing the question explicitly asks for is missing.
  Harmless extra context is fine only if it is true for the visa/case/procedure asked.
  A caveat that the guide page was shown for one nationality is fine.
- **partly**: the main answer is correct, but at least one of:
  - an extra claim is wrong, unsupported, or belongs to a different visa code, case,
    route (embassy / CCVI / e-visa) or section than the one asked;
  - a list the question asks for is missing an item, or a required condition is left out;
  - the answer opens with the wrong word (e.g. "Yes" when the answer is no) or is
    misleading even though later sentences are right;
  - a range or number is looser than the source (e.g. "1 to 6 months" when the source
    allows 1, 2, 3 or 6 months).
- **no**: the main answer is wrong; or it refuses / says "not stated" although the corpus
  answers it; or, for an unanswerable question, it states substantive facts as the answer
  instead of saying the documents do not cover it.

For an unanswerable (`no_answer`) question: "yes" if the answer declines or says the
documents do not state it; mentioning a nearby true fact is fine if it is not presented as
the answer; "partly" if it declines but adds a wrong or misleading claim; "no" if it answers.

Examples from earlier grading:
- Q: Can I apply for a multiple re-entry permit at the airport? A: "Yes, you can apply for a
  single Re-entry Permit at airports. However, only D-8 and F-6 holders can apply for a
  multiple one." -> partly (misleading opening).
- Q: Working holiday, Australian, do I need medical insurance? A: "Yes, you need to show ...
  This requirement is exempted for citizens of Australia." -> no (main answer wrong).
- Q: D-8-1 investor documents. A: correct list + a note copied from the D-8-3 page -> partly.
- Q: Which documents prove Korean ability for F-6-1? A: lists 4 of the 5 options -> partly.
- Q: Who is eligible for H-2-2? A: gives the parent/spouse-of-D-2 condition but omits
  "overseas Korean with foreign nationality, aged 25 or older" -> partly.
- Q: Unanswerable "How many E-9 workers can one company hire?" A: "not stated; employers
  must have fewer than 300 workers" (page says fewer than 300 OR capital under 8 billion
  KRW) -> partly.

Output per item: key, ans_ok, note (one or two sentences for any non-yes verdict: what is
wrong and what the corpus says), evidence (doc id + page or guide file + short exact quote).

Settled rules (added 2026-11-26, before any scoring-set answer was graded):
- Answer only what is asked: if the question asks for one fact (an amount, a duration), an
  answer that gives it correctly is "yes" even if it omits conditions the question did not
  ask about. If the question asks "who is eligible" or "what must they submit", every
  condition or item for the case asked is required.
- Opening word: a "Yes"/"No" opening is judged by whether a reader who stops after the first
  sentence would get the right answer. "No, in principle only a single permit ..., except D-8
  and F-6" is "yes". "Yes, you can apply for a single permit" to a question about a multiple
  permit is "partly". A stray "Yes"/"No" on a which/what question that does not make the first
  sentence false is "yes"; one that makes it false ("No, the rule covers none ...") is "partly".
- Redundant repetition of a correct statement does not lower the verdict.
- Also report `q_valid`: "no" only if the question itself is ambiguous or its reference is wrong
  according to the corpus (say why); otherwise "yes".
