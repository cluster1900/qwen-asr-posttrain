#!/usr/bin/env python3
"""Assemble a GitHub Pages static site directly from a book Release HTML asset.

Supports two sources:
  1. --from-release <tag>  (used by GitHub Actions: downloads the HTML asset & SHA256SUMS.txt
                            directly from the GitHub Release and verifies integrity)
  2. --html-file <path>    (local build preview without hitting the network)

Outputs a clean `_site/` directory ready for `actions/upload-pages-artifact` or `gh-pages`:
  - index.html      : the full interactive HTML book (opens straight into reading mode)
  - 404.html        : friendly redirect back to ./
  - release.json    : release tag, source asset filename, SHA-256, and download URLs
  - .nojekyll       : disables Jekyll so GitHub Pages serves static HTML verbatim
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "cluster1900/qwen-asr-posttrain"
DEFAULT_TAG = "book-v1.0.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_against_checksums(html_path: Path, sums_path: Path) -> str:
    actual = sha256_file(html_path)
    expected = None
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and Path(parts[-1]).name == html_path.name:
            expected = parts[0].lower()
            break
    if expected is None:
        raise ValueError(f"{html_path.name} not found in {sums_path.name}")
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {html_path.name}: expected {expected}, got {actual}")
    return actual


def fetch_from_release(repo: str, tag: str, work_dir: Path) -> tuple[Path, str]:
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            tag,
            "-R",
            repo,
            "--pattern",
            "*.html",
            "--pattern",
            "*SHA256SUMS.txt",
            "-D",
            str(work_dir),
        ],
        check=True,
    )
    html_files = sorted(work_dir.glob("*.html"))
    if len(html_files) != 1:
        raise RuntimeError(f"Expected exactly 1 HTML asset in release {tag}, found: {[p.name for p in html_files]}")
    html_path = html_files[0]
    sums_files = sorted(work_dir.glob("*SHA256SUMS.txt"))
    if sums_files:
        digest = verify_against_checksums(html_path, sums_files[0])
        print(f"[OK] Verified Release asset {html_path.name} SHA-256: {digest}")
    else:
        digest = sha256_file(html_path)
        print(f"[WARN] No SHA256SUMS.txt in {tag}; computed SHA-256: {digest}")
    return html_path, digest


def assemble_site(html_source: Path, digest: str, out_dir: Path, repo: str, tag: str) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Serve the Release HTML book directly at /index.html
    index_path = out_dir / "index.html"
    shutil.copyfile(html_source, index_path)

    # 2. Disable Jekyll processing
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    # 3. Clean 404 redirect page
    (out_dir / "404.html").write_text(
        """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="0; url=./" />
  <title>正在跳转到电子书首页…</title>
</head>
<body style="font-family:sans-serif;text-align:center;padding:64px;">
  <p>正在跳转到《从零到一：大模型底层原理、GPU 算子优化与任意模型后训练实战》…</p>
  <p><a href="./">如果没有自动跳转，请点击这里开始阅读</a></p>
</body>
</html>
""",
        encoding="utf-8",
    )

    # 4. Machine-readable provenance metadata
    stem = html_source.stem
    release_base = f"https://github.com/{repo}/releases"
    meta = {
        "repo": repo,
        "release_tag": tag,
        "source_html_asset": html_source.name,
        "sha256": digest,
        "bytes": index_path.stat().st_size,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "downloads": {
            "release_page": f"{release_base}/tag/{tag}",
            "html": f"{release_base}/download/{tag}/{html_source.name}",
            "pdf": f"{release_base}/download/{tag}/{stem}.pdf",
            "epub": f"{release_base}/download/{tag}/{stem}.epub",
        },
    }
    (out_dir / "release.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] GitHub Pages site assembled at {out_dir} (index.html = {index_path.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GitHub Pages site directly from a book Release HTML asset.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-release", metavar="TAG", help="Download HTML asset from this GitHub Release tag (e.g. book-v1.0.0)")
    src.add_argument("--html-file", type=Path, help="Use a local HTML book file instead of downloading from GitHub")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repository owner/name (default: {DEFAULT_REPO})")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag metadata when using --html-file (default: {DEFAULT_TAG})")
    parser.add_argument("--out-dir", type=Path, default=BOOK_DIR / "_site", help="Output static site directory")
    parser.add_argument("--serve", type=int, default=0, metavar="PORT", help="Optional local HTTP preview port (e.g. 8080)")
    args = parser.parse_args()

    if args.from_release:
        with tempfile.TemporaryDirectory() as tmp:
            html_path, digest = fetch_from_release(args.repo, args.from_release, Path(tmp))
            assemble_site(html_path, digest, args.out_dir, args.repo, args.from_release)
    else:
        html_path = args.html_file.resolve()
        if not html_path.is_file():
            raise FileNotFoundError(f"HTML file not found: {html_path}")
        digest = sha256_file(html_path)
        assemble_site(html_path, digest, args.out_dir, args.repo, args.tag)

    if args.serve > 0:
        os.chdir(args.out_dir)
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("127.0.0.1", args.serve), handler) as httpd:
            print(f"[SERVE] Local preview ready at http://127.0.0.1:{args.serve}/ (Ctrl+C to stop)")
            httpd.serve_forever()


if __name__ == "__main__":
    main()
