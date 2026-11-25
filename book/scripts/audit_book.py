#!/usr/bin/env python3
"""Automated quality & technical accuracy auditor for the book source and HTML/PDF/EPUB outputs."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = BOOK_DIR / "dist"
OUTPUT_BASENAME = "从零到一大模型底层与后训练实战"


def check_svgs() -> list[str]:
    errors = []
    svgs = sorted((BOOK_DIR / "assets").rglob("*.svg"))
    if len(svgs) < 19:
        errors.append(f"Expected >= 19 SVGs (cover + 18 diagrams), found {len(svgs)}")
    for p in svgs:
        try:
            ET.parse(p)
        except Exception as exc:
            errors.append(f"Invalid SVG XML {p.name}: {exc}")
    return errors


def check_markdown_and_code() -> tuple[list[str], dict[str, int]]:
    errors = []
    stats = {"chapters": 0, "chars": 0, "python_blocks": 0, "images": 0, "callouts": 0}
    chapters = sorted((BOOK_DIR / "chapters").glob("*.md"))
    stats["chapters"] = len(chapters)
    referenced_svgs: set[str] = set()

    for ch in chapters:
        text = ch.read_text(encoding="utf-8")
        stats["chars"] += len(text)

        # 1. Check image references exist
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
            rel = m.group(1)
            target = (BOOK_DIR / rel).resolve()
            stats["images"] += 1
            if not target.is_file():
                errors.append(f"{ch.name}: missing image {rel}")
            else:
                referenced_svgs.add(target.name)

        # 2. Check callouts are balanced
        opens = len(re.findall(r"^:::\s+(?:tip|warning|example|note)", text, re.M))
        closes = len(re.findall(r"^:::\s*$", text, re.M))
        stats["callouts"] += opens
        if opens != closes:
            errors.append(f"{ch.name}: unbalanced callouts (open={opens}, close={closes})")

        # 3. Check Python code blocks compile cleanly with ast.parse
        for idx, m in enumerate(re.finditer(r"```python\n(.*?)```", text, re.S), start=1):
            code = m.group(1)
            stats["python_blocks"] += 1
            try:
                ast.parse(code)
            except SyntaxError as exc:
                errors.append(f"{ch.name} python block #{idx} SyntaxError: {exc}")

        # 4. Check display math $$ balance (outside code blocks)
        no_code = re.sub(r"```.*?```", "", text, flags=re.S)
        dd_count = len(re.findall(r"\$\$", no_code))
        if dd_count % 2 != 0:
            errors.append(f"{ch.name}: odd number of $$ delimiters ({dd_count})")

    # Verify all 18 diagrams are referenced in chapters
    all_diags = {p.name for p in (BOOK_DIR / "assets" / "diagrams").glob("*.svg")}
    unused = sorted(all_diags - referenced_svgs)
    if unused:
        errors.append(f"Unused diagram SVGs not referenced in any chapter: {unused}")

    return errors, stats


def check_outputs() -> list[str]:
    errors = []
    html_p = DIST_DIR / f"{OUTPUT_BASENAME}.html"
    pdf_p = DIST_DIR / f"{OUTPUT_BASENAME}.pdf"
    epub_p = DIST_DIR / f"{OUTPUT_BASENAME}.epub"

    for p in (html_p, pdf_p, epub_p):
        if not p.is_file() or p.stat().st_size < 10_000:
            errors.append(f"Output missing or suspiciously small: {p}")

    # Inspect HTML for leaked raw $$ or raw ::: or unrendered math errors
    if html_p.is_file():
        h = html_p.read_text(encoding="utf-8")
        if "<math" not in h:
            errors.append("HTML contains no <math> MathML elements!")
        # Remove <script>, <pre>, <style>, <code> before checking for leaked ::: or $
        h_clean = re.sub(r"<script.*?</script>|<pre.*?</pre>|<style.*?</style>|<code>.*?</code>", "", h, flags=re.S)
        leaked_dd = re.findall(r"\$\$", h_clean)
        if leaked_dd:
            errors.append(f"HTML has {len(leaked_dd)} leaked raw $$ tokens!")
        leaked_dollar = re.findall(r"\$", h_clean)
        if leaked_dollar:
            errors.append(f"HTML has {len(leaked_dollar)} leaked raw $ inline math delimiters!")
        leaked_div = re.findall(r":::", h_clean)
        if leaked_div:
            errors.append(f"HTML has {len(leaked_div)} leaked raw ::: callout markers!")
        # Check no collapsed bullet list inside a paragraph (e.g. "： - ")
        collapsed = re.findall(r"<p>[^<]*[：。！] - [^<]*</p>", h_clean)
        if collapsed:
            errors.append(f"HTML has {len(collapsed)} collapsed bullet lists inside <p>: {collapsed[:3]}")
        img_count = h.count("data:image/svg+xml;base64,")
        if img_count < 18:
            errors.append(f"HTML expected >= 18 embedded SVG diagrams, found {img_count}")

    # Inspect PDF with pdfinfo
    if pdf_p.is_file():
        res = subprocess.run(["pdfinfo", str(pdf_p)], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            errors.append(f"pdfinfo failed on PDF: {res.stderr}")
        else:
            m = re.search(r"Pages:\s+(\d+)", res.stdout)
            pages = int(m.group(1)) if m else 0
            print(f"[INFO] PDF page count: {pages}")
            if pages < 20:
                errors.append(f"PDF page count unexpectedly low: {pages}")

    # Inspect EPUB3 ZIP integrity & contents
    if epub_p.is_file():
        try:
            with zipfile.ZipFile(epub_p, "r") as zf:
                names = zf.namelist()
                if "mimetype" not in names:
                    errors.append("EPUB missing mimetype")
                elif zf.read("mimetype") != b"application/epub+zip":
                    errors.append("EPUB invalid mimetype content")
                svgs_in_epub = [n for n in names if n.endswith(".svg")]
                xhtmls = [n for n in names if n.endswith(".xhtml")]
                print(f"[INFO] EPUB entries: {len(names)} total, {len(xhtmls)} xhtml, {len(svgs_in_epub)} svgs")
                if len(svgs_in_epub) < 18:
                    errors.append(f"EPUB expected >= 18 SVGs, found {len(svgs_in_epub)}")
                # Validate XML well-formedness of every xhtml in EPUB and check no raw $$ leaks
                for x in xhtmls:
                    data = zf.read(x).decode("utf-8")
                    ET.fromstring(data)
                    cleaned = re.sub(r"<pre.*?</pre>", "", data, flags=re.S)
                    if "$$" in cleaned or ":::" in cleaned:
                        errors.append(f"EPUB {x} has leaked raw $$ or :::")
        except Exception as exc:
            errors.append(f"EPUB validation failed: {exc}")

    return errors


def check_math_ledger() -> list[str]:
    """Cross-verify core numerical claims in the book via exact Python arithmetic."""
    errors = []
    # Qwen3-ASR-0.6B exact parameter check
    C, Da, Fa, L_enc, H = 480, 896, 3584, 18, 1024
    conv_stem = (1 * C * 9 + C) + 2 * (C * C * 9 + C) + (C * 16) * Da
    enc_layer = 4 * (Da * Da + Da) + (Da * Fa + Fa) + (Fa * Da + Da) + 4 * Da
    projector = 2 * Da + (Da * Da + Da) + (Da * H + H)
    audio_total = conv_stem + L_enc * enc_layer + projector

    V, L_dec, Hq, Hkv, D, I = 151936, 28, 16, 8, 128, 3072
    emb = V * H
    dec_layer = H * (Hq * D) + H * (Hkv * D) + H * (Hkv * D) + (Hq * D) * H + 2 * D + 3 * H * I + 2 * H
    decoder_total = emb + L_dec * dec_layer + H
    lora_total = L_dec * ((16 * H + (Hq * D) * 16) * 2 + (16 * H + (Hkv * D) * 16) * 2)

    if audio_total != 186_376_192:
        errors.append(f"audio_total mismatch: {audio_total}")
    if decoder_total != 596_049_920:
        errors.append(f"decoder_total mismatch: {decoder_total}")
    if audio_total + decoder_total != 782_426_112:
        errors.append(f"base_total mismatch: {audio_total + decoder_total}")
    if lora_total != 4_587_520:
        errors.append(f"lora_total mismatch: {lora_total}")
    if audio_total + decoder_total + lora_total != 787_013_632:
        errors.append("total_with_lora mismatch")

    # KV cache per token for Qwen3-0.6B BF16 = 112 KiB
    kv_per_tok = 2 * L_dec * Hkv * D * 2
    if kv_per_tok != 112 * 1024:
        errors.append(f"kv_per_tok mismatch: {kv_per_tok}")

    return errors


def main() -> int:
    all_errors: list[str] = []
    all_errors.extend(check_svgs())
    md_errs, stats = check_markdown_and_code()
    all_errors.extend(md_errs)
    all_errors.extend(check_math_ledger())
    all_errors.extend(check_outputs())

    print(f"[STATS] Chapters={stats['chapters']} | Chars={stats['chars']:,} | "
          f"PythonBlocks={stats['python_blocks']} | Diagrams={stats['images']} | Callouts={stats['callouts']}")

    if all_errors:
        print("\n[AUDIT FAILED] Found issues:")
        for e in all_errors:
            print("  -", e)
        return 1
    print("\n[AUDIT PASSED] Zero errors across SVGs, Markdown, Python AST, exact math ledger, HTML, PDF, and EPUB3!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
