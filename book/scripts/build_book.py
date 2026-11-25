#!/usr/bin/env python3
"""Build HTML, PDF, and EPUB3 editions of the book from Markdown chapters and SVG diagrams."""

from __future__ import annotations

import base64
import html as html_lib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parents[1]
CHROME_BIN = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CALLOUT_ICONS = {
    "tip": "💡",
    "warning": "⚠️",
    "example": "🧮",
    "note": "📌",
}

CHAPTER_FILES = [
    "chapters/00_preface.md",
    "chapters/01_math_and_neural_intuition.md",
    "chapters/02_pytorch_internals.md",
    "chapters/03_gpu_architecture_optimization.md",
    "chapters/04_operators_and_kernels.md",
    "chapters/05_vram_calculation_gpu_choice.md",
    "chapters/06_model_anatomy_overview.md",
    "chapters/07_audio_encoder_dissection.md",
    "chapters/08_decoder_and_generalization.md",
    "chapters/09_posttraining_lora_and_sft.md",
    "chapters/10_dpo_preference_alignment.md",
    "chapters/11_rl_grpo_and_evaluation_gym.md",
    "chapters/12_zero_to_one_any_model_playbook.md",
    "chapters/13_appendices.md",
]

BOOK_TITLE = "从零到一：大模型底层原理、GPU 算子优化与任意模型后训练实战"
BOOK_SUBTITLE = "PyTorch 内核 · GPU 与 Roofline · 核心算子 · 显存精算 · 逐层解剖 · SFT/DPO/RL 全闭环"
BOOK_AUTHOR = "Qwen3-ASR Post-Training Engineering Team"
OUTPUT_BASENAME = "从零到一大模型底层与后训练实战"


def preprocess_markdown(raw_md: str) -> str:
    """Convert custom `::: kind title` blocks and part banners into Pandoc-safe divs/classes."""
    lines = raw_md.splitlines()
    out: list[str] = []
    in_code = False
    in_callout = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue

        # Mark Part Banners (# 第X篇：...) with .part-banner class
        if re.match(r"^# 第[一二三四]篇：", line):
            out.append(f"{line} {{.part-banner}}")
            continue

        m_open = re.match(r"^:::\s+(tip|warning|example|note)(?:\s+(.*))?$", line)
        if m_open and not in_callout:
            kind = m_open.group(1)
            title = (m_open.group(2) or kind.upper()).strip()
            icon = CALLOUT_ICONS[kind]
            out.append(f"::: {{.callout .callout-{kind}}}")
            out.append(f"::: {{.callout-title}}")
            out.append(f"{icon} {title}")
            out.append(":::")
            out.append("")
            in_callout = True
            continue

        if stripped == ":::" and in_callout:
            out.append(":::")
            in_callout = False
            continue

        out.append(line)

    if in_callout:
        raise ValueError("Unclosed callout block detected during preprocessing!")
    return "\n".join(out) + "\n"


def render_cover_png(cover_svg: Path, cover_png: Path) -> None:
    """Render vector cover.svg into a crisp 1200x1600 PNG for EPUB3 cover embedding."""
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "cover.html"
        svg_text = cover_svg.read_text(encoding="utf-8")
        html_path.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>html,body{margin:0;padding:0;width:1200px;height:1600px;overflow:hidden;background:#0b1120;}"
            "svg{display:block;width:1200px;height:1600px;}</style></head><body>"
            f"{svg_text}</body></html>",
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(CHROME_BIN),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--window-size=1200,1600",
                f"--screenshot={cover_png}",
                f"file://{html_path}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def embed_svg_images_as_data_uri(html_text: str, base_dir: Path) -> str:
    """Inline local SVG <img> src as base64 data URIs so HTML & print HTML are 100% self-contained."""
    def _replace(match: re.Match[str]) -> str:
        prefix, rel_path, suffix = match.group(1), match.group(2), match.group(3)
        asset = (base_dir / rel_path).resolve()
        if not asset.is_file() or asset.suffix.lower() != ".svg":
            return match.group(0)
        encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
        return f'{prefix}data:image/svg+xml;base64,{encoded}{suffix}'

    return re.sub(r'(<img[^>]+src=")([^"]+\.svg)(")', _replace, html_text)


PANDOC_FROM = (
    "markdown+pipe_tables+fenced_code_blocks+fenced_divs+tex_math_dollars"
    "+implicit_figures+lists_without_preceding_blankline+task_lists"
)


def pandoc_to_fragment_and_toc(combined_md: Path) -> tuple[str, str]:
    """Convert preprocessed Markdown to HTML body fragment + clean nested TOC using Pandoc MathML."""
    with tempfile.TemporaryDirectory() as tmp:
        tpl = Path(tmp) / "frag.html"
        tpl.write_text(
            "<!--TOC_START-->$toc$<!--TOC_END-->\n<!--BODY_START-->$body$<!--BODY_END-->\n",
            encoding="utf-8",
        )
        out = Path(tmp) / "out.html"
        subprocess.run(
            [
                "pandoc",
                str(combined_md),
                f"--from={PANDOC_FROM}",
                "--to=html5",
                "--mathml",
                "--toc",
                "--toc-depth=2",
                "--syntax-highlighting=pygments",
                f"--template={tpl}",
                "-o",
                str(out),
            ],
            check=True,
            cwd=str(BOOK_DIR),
        )
        raw = out.read_text(encoding="utf-8")
        toc = raw.split("<!--TOC_START-->", 1)[1].split("<!--TOC_END-->", 1)[0].strip()
        body = raw.split("<!--BODY_START-->", 1)[1].split("<!--BODY_END-->", 1)[0].strip()
        return toc, body


def build_interactive_html(toc_html: str, body_html: str, out_html: Path) -> None:
    css = (BOOK_DIR / "styles" / "book.css").read_text(encoding="utf-8")
    body_embedded = embed_svg_images_as_data_uri(body_html, BOOK_DIR)
    js = """
(function() {
  const root = document.documentElement;
  let fontSize = 16.5;
  window.setTheme = function(t) {
    root.setAttribute('data-theme', t);
    try { localStorage.setItem('book-theme', t); } catch (e) {}
  };
  window.bumpFont = function(delta) {
    fontSize = Math.max(13, Math.min(22, fontSize + delta));
    root.style.setProperty('--base-size', fontSize + 'px');
  };
  try {
    const saved = localStorage.getItem('book-theme');
    if (saved) root.setAttribute('data-theme', saved);
  } catch (e) {}

  const search = document.getElementById('toc-search');
  const links = Array.from(document.querySelectorAll('.book-sidebar nav a'));
  search.addEventListener('input', function() {
    const q = this.value.trim().toLowerCase();
    links.forEach(a => {
      const li = a.closest('li');
      const match = !q || a.textContent.toLowerCase().includes(q);
      if (li) li.style.display = match ? '' : 'none';
    });
  });

  const bar = document.getElementById('reading-progress');
  window.addEventListener('scroll', function() {
    const h = document.documentElement.scrollHeight - window.innerHeight;
    const pct = h > 0 ? (window.scrollY / h) * 100 : 0;
    bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
  }, { passive: true });
})();
"""
    page = f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html_lib.escape(BOOK_TITLE)}</title>
  <style>{css}</style>
</head>
<body>
  <div id="reading-progress"></div>
  <div class="book-shell">
    <aside class="book-sidebar">
      <div class="sidebar-brand">📘 从零到一：大模型底层与后训练</div>
      <div class="toolbar">
        <button onclick="setTheme('light')" title="浅色模式">☀️ 亮色</button>
        <button onclick="setTheme('sepia')" title="护眼纸张">📜 护眼</button>
        <button onclick="setTheme('dark')" title="夜间模式">🌙 暗黑</button>
        <button onclick="bumpFont(-1)" title="缩小字号">A-</button>
        <button onclick="bumpFont(1)" title="放大字号">A+</button>
      </div>
      <input id="toc-search" type="search" placeholder="搜索章节标题..." />
      <nav aria-label="目录">{toc_html}</nav>
    </aside>
    <main class="book-main">
      <header class="book-hero">
        <div class="badge">第一版（修订精校版）· HTML / PDF / EPUB 全格式同步</div>
        <h1 class="hero-title">{html_lib.escape(BOOK_TITLE)}</h1>
        <p class="hero-sub">{html_lib.escape(BOOK_SUBTITLE)}</p>
        <div class="hero-meta">作者：{html_lib.escape(BOOK_AUTHOR)} · 18 张原创高清原理图 · 精准核验 Qwen3-ASR-0.6B 全部 782,426,112 参数</div>
      </header>
      {body_embedded}
    </main>
  </div>
  <script>{js}</script>
</body>
</html>
"""
    out_html.write_text(page, encoding="utf-8")


def build_pdf(toc_html: str, body_html: str, out_pdf: Path) -> None:
    css = (BOOK_DIR / "styles" / "print.css").read_text(encoding="utf-8")
    cover_b64 = base64.b64encode((BOOK_DIR / "assets" / "cover.svg").read_bytes()).decode("ascii")
    body_embedded = embed_svg_images_as_data_uri(body_html, BOOK_DIR)
    print_html_str = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html_lib.escape(BOOK_TITLE)}</title>
  <style>{css}</style>
</head>
<body>
  <section class="print-cover">
    <img src="data:image/svg+xml;base64,{cover_b64}" alt="封面" />
  </section>
  <section class="print-toc">
    <h1>目 录</h1>
    <nav>{toc_html}</nav>
  </section>
  <main>
    {body_embedded}
  </main>
</body>
</html>
"""
    with tempfile.TemporaryDirectory() as tmp:
        print_html_path = Path(tmp) / "print.html"
        print_html_path.write_text(print_html_str, encoding="utf-8")
        subprocess.run(
            [
                str(CHROME_BIN),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--print-to-pdf-no-header",
                f"--print-to-pdf={out_pdf}",
                f"file://{print_html_path}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def build_epub(combined_md: Path, cover_png: Path, out_epub: Path) -> None:
    subprocess.run(
        [
            "pandoc",
            str(combined_md),
            f"--from={PANDOC_FROM}",
            "--to=epub3",
            "--mathml",
            "--toc",
            "--toc-depth=2",
            "--split-level=1",
            "--syntax-highlighting=pygments",
            f"--css={BOOK_DIR / 'styles' / 'epub.css'}",
            f"--epub-cover-image={cover_png}",
            f"--resource-path={BOOK_DIR}",
            "--metadata",
            f"title={BOOK_TITLE}",
            "--metadata",
            f"subtitle={BOOK_SUBTITLE}",
            "--metadata",
            f"author={BOOK_AUTHOR}",
            "--metadata",
            "lang=zh-Hans",
            "-o",
            str(out_epub),
        ],
        check=True,
        cwd=str(BOOK_DIR),
    )


def main() -> None:
    dist_dir = BOOK_DIR / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combine & preprocess all chapters
    parts = []
    for rel in CHAPTER_FILES:
        path = BOOK_DIR / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing chapter: {path}")
        parts.append(preprocess_markdown(path.read_text(encoding="utf-8")))
    combined_text = "\n\n".join(parts)

    with tempfile.TemporaryDirectory() as tmp:
        combined_md = Path(tmp) / "book_combined.md"
        combined_md.write_text(combined_text, encoding="utf-8")

        # 2. Render PNG cover for EPUB3
        cover_svg = BOOK_DIR / "assets" / "cover.svg"
        cover_png = BOOK_DIR / "assets" / "cover.png"
        render_cover_png(cover_svg, cover_png)
        print(f"[OK] Cover PNG rendered: {cover_png.name} ({cover_png.stat().st_size:,} bytes)")

        # 3. Generate shared HTML fragment + TOC
        toc_html, body_html = pandoc_to_fragment_and_toc(combined_md)

        # 4. Build HTML
        out_html = dist_dir / f"{OUTPUT_BASENAME}.html"
        build_interactive_html(toc_html, body_html, out_html)
        print(f"[OK] HTML built: {out_html.name} ({out_html.stat().st_size:,} bytes)")

        # 5. Build PDF
        out_pdf = dist_dir / f"{OUTPUT_BASENAME}.pdf"
        build_pdf(toc_html, body_html, out_pdf)
        print(f"[OK] PDF built:  {out_pdf.name} ({out_pdf.stat().st_size:,} bytes)")

        # 6. Build EPUB3
        out_epub = dist_dir / f"{OUTPUT_BASENAME}.epub"
        build_epub(combined_md, cover_png, out_epub)
        print(f"[OK] EPUB built: {out_epub.name} ({out_epub.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
