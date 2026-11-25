"""Check visa.go.kr for new or changed documents, and optionally rebuild the index.

    python -m visarag.refresh            # dry run: fetch into a staging folder and report
    python -m visarag.refresh --apply    # also swap the new files in, parse, and re-index

The fetch always goes to ``data/refresh/staging`` first, so a failed or partial
crawl never touches the live corpus. ``--apply`` refuses to continue when a
download failed or the number of PDFs or guide pages drops by more than
``--max-drop`` (default 10%): a site outage must not empty the index.

Work done is proportional to the change: PDFs are content-addressed (an
unchanged file keeps its name), Docling only parses PDFs it has not seen, and the
embedding cache only embeds text it has not seen.

After ``--apply`` restart the API so it loads the new index. The evaluation
numbers in the README belong to the corpus snapshot they were measured on; a
refresh that changes the corpus needs a new evaluation run.

Exit codes: 0 no change or applied, 1 changes found (dry run), 2 refused (unsafe).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from visarag import crawl, visa_pages
from visarag.config import settings

log = logging.getLogger("visarag.refresh")


def pdf_set(manifest: Path) -> dict[str, str]:
    """sha12 -> file name, for every PDF in a crawl manifest."""
    if not manifest.exists():
        return {}
    out: dict[str, str] = {}
    for line in manifest.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("sha256"):
            out.setdefault(r["sha256"][:12], r.get("file_name", ""))
    return out


def page_set(pages_dir: Path) -> dict[str, str]:
    """visa code -> hash of the page body (front matter, which holds the fetch date, excluded)."""
    out: dict[str, str] = {}
    for p in sorted(pages_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        body = text.split("\n---\n", 1)[1] if text.startswith("---\n") and "\n---\n" in text else text
        out[p.stem] = hashlib.sha256(body.encode()).hexdigest()[:16]
    return out


def diff(old: dict[str, str], new: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
    }


def fetch_to_staging(staging: Path, rediscover: bool = False) -> list[crawl.Attachment]:
    """Run both crawlers with every output path pointed at ``staging``.

    Guide pages are shown per nationality, and discovery picks one nationality per visa
    code. Unless ``rediscover`` is set, the live code list is reused so each page is
    fetched for the same nationality as before and only real edits show up as changes.
    """
    live_data = settings.data_dir
    saved = (visa_pages.RAW_DIR, visa_pages.CORPUS_DIR)
    codes = saved[0] / "_codes.json"
    if codes.exists() and not rediscover:
        (staging / "raw" / "visa_pages").mkdir(parents=True, exist_ok=True)
        shutil.copy2(codes, staging / "raw" / "visa_pages" / "_codes.json")
    settings.data_dir = staging
    visa_pages.RAW_DIR, visa_pages.CORPUS_DIR = staging / "raw" / "visa_pages", staging / "pages"
    try:
        atts = crawl.run(list(crawl.BOARDS), ["EN", "KO"])
        visa_pages.run(["en"], force=True)
    finally:
        settings.data_dir = live_data
        visa_pages.RAW_DIR, visa_pages.CORPUS_DIR = saved
    return atts


def safety_problems(atts: list[crawl.Attachment], old_pdfs: dict, new_pdfs: dict,
                    old_pages: dict, new_pages: dict, max_drop: float) -> list[str]:
    problems = []
    failed = [a.file_name or a.element_id or "?" for a in atts if a.status == "failed"]
    if failed:
        problems.append(f"{len(failed)} downloads failed, e.g. {failed[:3]}")
    if len(new_pdfs) < (1 - max_drop) * len(old_pdfs):
        problems.append(f"PDF count fell from {len(old_pdfs)} to {len(new_pdfs)}")
    if len(new_pages) < (1 - max_drop) * len(old_pages):
        problems.append(f"guide page count fell from {len(old_pages)} to {len(new_pages)}")
    return problems


def apply(staging: Path, index_device: str) -> dict:
    from visarag import index, parse

    live = settings.data_dir
    (live / "raw" / "pdf").mkdir(parents=True, exist_ok=True)
    for pdf in (staging / "raw" / "pdf").glob("*.pdf"):
        dest = live / "raw" / "pdf" / pdf.name
        if not dest.exists():
            shutil.copy2(pdf, dest)
    # The crawl ran with data_dir = staging, so its manifest points into staging: repoint it.
    live_pdf = (live / "raw" / "pdf").resolve()
    if live_pdf.is_relative_to(settings.root.resolve()):
        live_pdf = live_pdf.relative_to(settings.root.resolve())  # same form crawl.py writes
    with (staging / "raw" / "manifest.jsonl").open(encoding="utf-8") as src, \
            (live / "raw" / "manifest.jsonl").open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            if row.get("local_path"):
                row["local_path"] = str(live_pdf / Path(row["local_path"]).name)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    for sub in ("pages", "raw/visa_pages"):
        if (live / sub).exists():
            shutil.rmtree(live / sub)
        shutil.copytree(staging / sub, live / sub)
    parse.run()
    settings.index_device = index_device
    return index.build()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="swap the new files in and re-index")
    ap.add_argument("--max-drop", type=float, default=0.10, help="largest allowed drop in PDFs or pages")
    ap.add_argument("--index-device", default="cpu",
                    help="device for embedding new chunks (cpu leaves the GPU to the running API)")
    ap.add_argument("--rediscover", action="store_true",
                    help="rediscover visa codes instead of reusing the saved list (finds new codes, but a "
                         "page may then be fetched for another nationality and show up as changed)")
    ap.add_argument("--keep-staging", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    live = settings.data_dir
    staging = live / "refresh" / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    t0 = time.perf_counter()
    atts = fetch_to_staging(staging, args.rediscover)
    old_pdfs, new_pdfs = pdf_set(live / "raw" / "manifest.jsonl"), pdf_set(staging / "raw" / "manifest.jsonl")
    old_pages, new_pages = page_set(live / "pages" / "en"), page_set(staging / "pages" / "en")
    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fetch_seconds": round(time.perf_counter() - t0, 1),
        "pdfs": {"before": len(old_pdfs), "after": len(new_pdfs), **diff(old_pdfs, new_pdfs)},
        "guide_pages": {"before": len(old_pages), "after": len(new_pages), **diff(old_pages, new_pages)},
    }
    changed = any(report[k][c] for k in ("pdfs", "guide_pages") for c in ("added", "removed", "changed"))
    problems = safety_problems(atts, old_pdfs, new_pdfs, old_pages, new_pages, args.max_drop)
    report.update(changed=changed, problems=problems, applied=False)

    code = 0
    if problems:
        code = 2
    elif changed and args.apply:
        report["index"] = apply(staging, args.index_device)
        report["applied"] = True
    elif changed:
        code = 1

    (live / "refresh" / "last_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["applied"]:
        print("Index rebuilt: restart the API to load it, and re-run the evaluation.", file=sys.stderr)
    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
