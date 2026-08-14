#!/usr/bin/env python3
"""Generic HTML and PDF export for multi-file standalone factor reports.

The canonical report remains a set of Markdown files.  This module discovers
numbered chapters (``01`` through ``10``) followed by Markdown files under
``appendix/`` and renders them as:

* one browser-openable HTML file with local images embedded as data URIs; and
* one paginated PDF produced with Matplotlib/PdfPages.

The implementation deliberately contains no factor-specific metrics or figure
names.  Callers provide the report title and may provide an explicit Markdown
file sequence when the standard chapter layout is not appropriate.
"""

from __future__ import annotations

import base64
import html
import io
import mimetypes
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union
from urllib.parse import unquote, urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mistune
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image


PathInput = Union[str, Path]
DEFAULT_TITLE = "Standalone Factor Research Report"
DEFAULT_MATHJAX_URL = (
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
)
_NUMBERED_MARKDOWN = re.compile(
    r"^(0[1-9]|10)(?:[^0-9].*)?\.md$",
    flags=re.IGNORECASE,
)
_CJK_FONT_CANDIDATES: Sequence[Path] = (
    Path("/usr/share/fonts/chinese/simhei.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


class ReportSection(NamedTuple):
    """Resolved source and presentation metadata for one Markdown section."""

    source_path: Path
    relative_path: Path
    anchor: str
    title: str


class ExportedReport(NamedTuple):
    """Paths created by :func:`export_standalone_report`."""

    html_path: Path
    pdf_path: Path


def discover_markdown_files(report_root: PathInput) -> List[Path]:
    """Return canonical chapters in ``01..10`` then ``appendix`` order.

    Numbered chapters must be top-level Markdown files whose names begin with
    ``01`` through ``10``.  Appendix files are discovered recursively and
    sorted by their report-relative POSIX paths.  ``README.md`` is intentionally
    not included because it is normally navigation for the source tree rather
    than a report chapter.
    """

    root = Path(report_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Report root does not exist: {}".format(root))

    numbered: List[Tuple[int, str, Path]] = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = _NUMBERED_MARKDOWN.fullmatch(path.name)
        if match:
            numbered.append((int(match.group(1)), path.name.casefold(), path.resolve()))

    appendix_root = root / "appendix"
    appendix: List[Path] = []
    if appendix_root.is_dir():
        appendix = sorted(
            (
                path.resolve()
                for path in appendix_root.rglob("*.md")
                if path.is_file()
            ),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )

    numbered_paths = [item[2] for item in sorted(numbered)]
    return numbered_paths + appendix


def _safe_text(value: str) -> str:
    """Remove common Markdown presentation syntax while preserving content."""

    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"__([^_]+)__", r"\1", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\\\((.*?)\\\)", r"\1", value)
    value = value.replace("\\[", "").replace("\\]", "")
    value = value.replace("−", "-").replace("•", "-")
    return html.unescape(value).strip()


def _first_heading(source: str, fallback: str) -> str:
    match = re.search(
        r"^\s*#{1,6}\s+(.+?)\s*#*\s*$",
        source,
        flags=re.MULTILINE,
    )
    if match:
        heading = _safe_text(match.group(1))
        if heading:
            return heading
    return fallback.replace("_", " ").replace("-", " ").strip()


def _anchor_for(relative_path: Path) -> str:
    raw = unicodedata.normalize("NFKC", relative_path.with_suffix("").as_posix())
    slug = re.sub(r"[^\w-]+", "-", raw, flags=re.UNICODE).strip("-_").lower()
    return "section-{}".format(slug or "report")


def _resolve_sections(
    report_root: PathInput,
    markdown_files: Optional[Sequence[PathInput]],
) -> Tuple[Path, List[ReportSection]]:
    root = Path(report_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Report root does not exist: {}".format(root))

    if markdown_files is None:
        selected = discover_markdown_files(root)
    else:
        selected = []
        for path_input in markdown_files:
            path = Path(path_input).expanduser()
            selected.append((path if path.is_absolute() else root / path).resolve())
    if not selected:
        raise ValueError(
            "No Markdown chapters found; expected 01..10 files and/or appendix/*.md"
        )

    sections: List[ReportSection] = []
    used_anchors: Dict[str, int] = {}
    for source_path in selected:
        if not source_path.is_file():
            raise FileNotFoundError("Markdown source does not exist: {}".format(source_path))
        try:
            relative_path = source_path.relative_to(root)
        except ValueError:
            relative_path = Path(source_path.name)
        source = source_path.read_text(encoding="utf-8")
        anchor_base = _anchor_for(relative_path)
        duplicate_index = used_anchors.get(anchor_base, 0) + 1
        used_anchors[anchor_base] = duplicate_index
        anchor = (
            anchor_base
            if duplicate_index == 1
            else "{}-{}".format(anchor_base, duplicate_index)
        )
        sections.append(
            ReportSection(
                source_path=source_path,
                relative_path=relative_path,
                anchor=anchor,
                title=_first_heading(source, relative_path.stem),
            )
        )
    return root, sections


def _protect_math(markdown_text: str) -> str:
    """Protect TeX delimiters from Mistune 0.8 backslash escaping."""

    def display(match: re.Match) -> str:
        return '\n<div class="math-block">\\[{}\\]</div>\n'.format(match.group(1))

    def inline(match: re.Match) -> str:
        return '<span class="math-inline">\\({}\\)</span>'.format(match.group(1))

    markdown_text = re.sub(
        r"\\\[(.*?)\\\]",
        display,
        markdown_text,
        flags=re.DOTALL,
    )
    return re.sub(r"\\\((.*?)\\\)", inline, markdown_text)


def _is_external_target(target: str) -> bool:
    parsed = urlsplit(target)
    return bool(parsed.scheme or parsed.netloc or target.startswith("//"))


def _local_target(source_path: Path, target: str) -> Path:
    parsed = urlsplit(target)
    decoded_path = unquote(parsed.path)
    return (source_path.parent / decoded_path).resolve()


class EmbeddedMarkdownRenderer(mistune.Renderer):
    """Mistune renderer that embeds local images and joins report links."""

    def __init__(
        self,
        source_path: Path,
        section_links: Dict[Path, str],
    ) -> None:
        super().__init__(escape=False)
        self.source_path = source_path
        self.section_links = section_links

    def image(self, src: str, title: str, text: str) -> str:
        if _is_external_target(src):
            return super().image(src, title, text)

        image_path = _local_target(self.source_path, src)
        if not image_path.is_file():
            return (
                '<span class="missing-image">Missing image: {}</span>'.format(
                    html.escape(src)
                )
            )

        mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        title_attr = (
            ' title="{}"'.format(html.escape(title, quote=True)) if title else ""
        )
        return (
            '<img src="data:{mime};base64,{payload}" alt="{alt}"{title} '
            'loading="eager">'.format(
                mime=html.escape(mime, quote=True),
                payload=payload,
                alt=html.escape(text or image_path.stem, quote=True),
                title=title_attr,
            )
        )

    def link(self, link: str, title: str, text: str) -> str:
        href = link
        if link.startswith("#") or _is_external_target(link):
            href = link
        else:
            target_path = _local_target(self.source_path, link)
            target_anchor = self.section_links.get(target_path)
            if target_anchor:
                href = "#{}".format(target_anchor)
        title_attr = (
            ' title="{}"'.format(html.escape(title, quote=True)) if title else ""
        )
        return '<a href="{}"{}>{}</a>'.format(
            html.escape(href, quote=True),
            title_attr,
            text,
        )


def _html_css() -> str:
    return """
:root {
  --ink: #172033;
  --muted: #5e6a7d;
  --line: #d9e0ea;
  --navy: #17365d;
  --blue: #2f6f9f;
  --soft: #f3f6fa;
  --accent: #b23a48;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: #eef2f6;
  font-family: "Noto Sans CJK SC", "SimHei", "Microsoft YaHei", Arial, sans-serif;
  line-height: 1.72;
}
main {
  width: min(1120px, calc(100% - 32px));
  margin: 24px auto 60px;
  background: white;
  padding: 58px 72px;
  box-shadow: 0 8px 30px rgba(20, 35, 55, .10);
}
.cover {
  border-top: 8px solid var(--navy);
  padding: 32px 0 28px;
}
.cover .eyebrow {
  color: var(--blue);
  font-size: 13px;
  letter-spacing: .12em;
  text-transform: uppercase;
  font-weight: 700;
}
.cover h1 {
  color: var(--navy);
  font-size: 42px;
  line-height: 1.18;
  margin: 18px 0 12px;
  border: 0;
}
.cover p { color: var(--muted); font-size: 16px; }
.toc {
  border: 1px solid var(--line);
  background: #fbfcfe;
  padding: 26px 32px;
  margin: 28px 0 60px;
}
.toc h2 { margin-top: 0; }
.toc ol { columns: 2; column-gap: 42px; }
.toc li { break-inside: avoid; margin: 6px 0; }
.document-section { padding-top: 18px; }
h1, h2, h3, h4, h5, h6 { color: var(--navy); line-height: 1.35; }
h1 {
  margin-top: 54px;
  padding-bottom: 10px;
  border-bottom: 3px solid var(--navy);
}
h2 {
  margin-top: 38px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--line);
}
h3 { margin-top: 28px; color: #24517c; }
p { margin: 10px 0 15px; }
a { color: #1f6698; text-decoration: none; }
a:hover { text-decoration: underline; }
blockquote {
  margin: 18px 0;
  padding: 10px 18px;
  background: #f6f8fb;
  border-left: 5px solid var(--blue);
  color: #34445a;
}
code {
  background: #f0f3f7;
  color: #8a2635;
  border-radius: 3px;
  padding: 1px 4px;
  font-family: "Noto Sans Mono CJK SC", monospace;
}
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #172033;
  color: #edf4ff;
  padding: 16px 18px;
  border-radius: 6px;
  line-height: 1.48;
  font-size: 12px;
}
pre code { color: inherit; background: none; padding: 0; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0 26px;
  font-size: 13px;
}
th {
  color: white;
  background: var(--navy);
  font-weight: 650;
}
th, td {
  border: 1px solid var(--line);
  padding: 8px 9px;
  vertical-align: top;
}
tbody tr:nth-child(even) { background: #f7f9fc; }
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 22px auto 28px;
  border: 1px solid #e2e7ef;
}
.math-block {
  overflow-x: auto;
  text-align: center;
  margin: 20px 0;
  padding: 8px;
}
.math-inline { white-space: nowrap; }
.missing-image { color: var(--accent); font-weight: 700; }
.generated {
  margin-top: 50px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 12px;
}
@page { size: A4; margin: 16mm 14mm 17mm; }
@media print {
  body { background: white; }
  main { width: auto; margin: 0; padding: 0; box-shadow: none; }
  .cover, .toc { page-break-after: always; }
  .document-section { page-break-before: always; }
  h1, h2, h3 { break-after: avoid; }
  table, img, blockquote, pre { break-inside: avoid; }
  a { color: inherit; }
}
"""


def _mathjax_markup(mathjax_url: str) -> str:
    if not mathjax_url:
        raise ValueError("mathjax_url must not be empty")
    return """
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
    displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']]
  }},
  svg: {{ fontCache: 'global' }},
  options: {{ enableMenu: false }}
}};
</script>
<script defer src="{url}"></script>
""".format(url=html.escape(mathjax_url, quote=True))


def build_html(
    report_root: PathInput,
    output_path: PathInput,
    *,
    title: str = DEFAULT_TITLE,
    markdown_files: Optional[Sequence[PathInput]] = None,
    mathjax_url: str = DEFAULT_MATHJAX_URL,
    language: str = "zh-CN",
) -> Path:
    """Build a single HTML file with all local Markdown images embedded."""

    _, sections = _resolve_sections(report_root, markdown_files)
    section_links = {
        section.source_path.resolve(): section.anchor for section in sections
    }
    rendered_sections: List[str] = []
    for section in sections:
        renderer = EmbeddedMarkdownRenderer(section.source_path, section_links)
        markdown = mistune.Markdown(renderer=renderer)
        source = section.source_path.read_text(encoding="utf-8")
        body = markdown(_protect_math(source))
        rendered_sections.append(
            '<section id="{anchor}" class="document-section" '
            'data-source="{source}">{body}</section>'.format(
                anchor=html.escape(section.anchor, quote=True),
                source=html.escape(section.relative_path.as_posix(), quote=True),
                body=body,
            )
        )

    toc_items = "\n".join(
        '<li><a href="#{anchor}">{label}</a></li>'.format(
            anchor=html.escape(section.anchor, quote=True),
            label=html.escape(section.title),
        )
        for section in sections
    )
    document = """<!doctype html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
{mathjax}
</head>
<body>
<main>
<header class="cover">
  <div class="eyebrow">Standalone Factor Research</div>
  <h1>{title}</h1>
  <p>Generated from the canonical multi-file Markdown report.</p>
</header>
<nav class="toc" aria-label="Table of contents">
  <h2>Table of Contents</h2>
  <ol>{toc}</ol>
</nav>
{sections}
<div class="generated">Generated from canonical Markdown sources.</div>
</main>
</body>
</html>
""".format(
        language=html.escape(language, quote=True),
        title=html.escape(title),
        css=_html_css(),
        mathjax=_mathjax_markup(mathjax_url),
        toc=toc_items,
        sections="".join(rendered_sections),
    )

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def _display_units(text: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in text
    )


def _wrap_units(text: str, max_units: int) -> List[str]:
    """Wrap mixed Chinese/Latin text using approximate display width."""

    if not text:
        return [""]
    output: List[str] = []
    for original_line in text.splitlines() or [""]:
        source_line = original_line.rstrip()
        while _display_units(source_line) > max_units:
            used = 0
            cut = 0
            preferred = 0
            for index, character in enumerate(source_line):
                used += (
                    2
                    if unicodedata.east_asian_width(character) in {"W", "F"}
                    else 1
                )
                if character.isspace() or character in "，。；：、,.!?;:)）]】":
                    preferred = index + 1
                if used > max_units:
                    cut = index
                    break
            if preferred >= int(cut * 0.65):
                cut = preferred
            cut = max(cut, 1)
            output.append(source_line[:cut].rstrip())
            source_line = source_line[cut:].lstrip()
        output.append(source_line)
    return output


def _parse_table_row(line: str) -> List[str]:
    return [_safe_text(cell.strip()) for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    cells = line.strip().strip("|").split("|")
    return bool(cells) and all(
        re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells
    )


def _parse_image_line(line: str) -> Optional[Tuple[str, str]]:
    match = re.fullmatch(r"\s*!\[([^\]]*)\]\((.+)\)\s*", line)
    if not match:
        return None
    description = match.group(1)
    target_with_title = match.group(2).strip()
    if target_with_title.startswith("<") and ">" in target_with_title:
        target = target_with_title[1 : target_with_title.index(">")]
    else:
        target_match = re.match(r"(\S+)", target_with_title)
        target = target_match.group(1) if target_match else target_with_title
    return target, _safe_text(description)


def parse_markdown_blocks(text: str) -> Iterator[Tuple[str, object]]:
    """Yield a small, dependency-light block model for the PDF renderer."""

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            yield "heading", (len(heading.group(1)), _safe_text(heading.group(2)))
            index += 1
            continue

        fence = re.match(r"^```(.*)$", stripped)
        if fence:
            code_lines: List[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            yield "code", "\n".join(code_lines)
            continue

        if stripped.startswith(r"\["):
            if stripped.endswith(r"\]") and len(stripped) > 4:
                yield "math", stripped[2:-2].strip()
                index += 1
                continue
            math_lines: List[str] = []
            opening_remainder = stripped[2:].strip()
            if opening_remainder:
                math_lines.append(opening_remainder)
            index += 1
            while index < len(lines) and lines[index].strip() != r"\]":
                math_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            yield "math", "\n".join(math_lines)
            continue

        image = _parse_image_line(line)
        if image:
            yield "image", image
            index += 1
            continue

        if (
            line.lstrip().startswith("|")
            and index + 1 < len(lines)
            and _is_table_separator(lines[index + 1])
        ):
            header = _parse_table_row(line)
            index += 2
            rows: List[List[str]] = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                rows.append(_parse_table_row(lines[index]))
                index += 1
            yield "table", (header, rows)
            continue

        if re.match(r"^\s*>\s?", line):
            quote_lines: List[str] = []
            while index < len(lines) and re.match(r"^\s*>\s?", lines[index]):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[index]))
                index += 1
            yield "quote", _safe_text(" ".join(quote_lines))
            continue

        list_item = re.match(r"^\s*([-*+]|\d+\.)\s+(.+)$", line)
        if list_item:
            prefix = (
                "-"
                if list_item.group(1) in {"-", "*", "+"}
                else list_item.group(1)
            )
            yield "list", "{} {}".format(prefix, _safe_text(list_item.group(2)))
            index += 1
            continue

        if re.fullmatch(r"\s*(?:---+|\*\*\*+)\s*", line):
            yield "rule", ""
            index += 1
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index]
            candidate_stripped = candidate.strip()
            if not candidate_stripped:
                break
            if re.match(r"^(#{1,6})\s+", candidate):
                break
            if candidate_stripped.startswith(("```", r"\[")):
                break
            if _parse_image_line(candidate):
                break
            if re.match(r"^\s*(?:[-*+]|\d+\.)\s+", candidate):
                break
            if re.match(r"^\s*>\s?", candidate):
                break
            if re.fullmatch(r"\s*(?:---+|\*\*\*+)\s*", candidate):
                break
            if candidate.lstrip().startswith("|") and index + 1 < len(lines):
                if _is_table_separator(lines[index + 1]):
                    break
            paragraph.append(candidate_stripped)
            index += 1
        yield "paragraph", _safe_text(" ".join(paragraph))


def _font_properties(font_path: Optional[PathInput]) -> Tuple[FontProperties, FontProperties]:
    if font_path is not None:
        selected = Path(font_path).expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError("PDF font does not exist: {}".format(selected))
    else:
        selected = next((path for path in _CJK_FONT_CANDIDATES if path.is_file()), None)

    if selected is not None:
        font = FontProperties(fname=str(selected))
        mono_font = FontProperties(fname=str(selected))
    else:
        font = FontProperties(family="DejaVu Sans")
        mono_font = FontProperties(family="DejaVu Sans Mono")
    return font, mono_font


class MatplotlibPdfReport:
    """Small paginated PDF layout engine for the supported Markdown blocks."""

    PAGE_WIDTH = 8.27
    PAGE_HEIGHT = 11.69
    LEFT = 0.07
    RIGHT = 0.95
    TOP = 0.94
    BOTTOM = 0.065

    def __init__(
        self,
        output_path: Path,
        title: str,
        font_path: Optional[PathInput] = None,
    ) -> None:
        self.output_path = output_path
        self.title = title
        self.pdf = PdfPages(str(output_path))
        self.figure = None
        self.y = self.TOP
        self.page_number = 0
        self.font, self.mono_font = _font_properties(font_path)

    def close(self) -> None:
        self._flush_page()
        self.pdf.close()

    def _flush_page(self) -> None:
        if self.figure is None:
            return
        footer = "{}  ·  {}".format(self.title, self.page_number)
        self.figure.text(
            0.5,
            0.027,
            footer,
            ha="center",
            color="#667085",
            fontsize=7.2,
            fontproperties=self.font,
        )
        self.pdf.savefig(self.figure)
        plt.close(self.figure)
        self.figure = None

    def new_page(self) -> None:
        self._flush_page()
        self.figure = plt.figure(
            figsize=(self.PAGE_WIDTH, self.PAGE_HEIGHT),
            dpi=140,
        )
        self.figure.patch.set_facecolor("white")
        self.page_number += 1
        self.y = self.TOP

    def ensure(self, required_height: float) -> None:
        if self.figure is None:
            self.new_page()
        if self.y - required_height < self.BOTTOM:
            self.new_page()

    def add_text(
        self,
        text: str,
        *,
        fontsize: float = 9.4,
        color: str = "#172033",
        weight: str = "normal",
        indent: float = 0,
        spacing_after: float = 0.010,
        background: str = "",
        max_units: int = 108,
        font: Optional[FontProperties] = None,
    ) -> None:
        remaining = _wrap_units(text, max_units)
        line_height = fontsize * 1.48 / 72 / self.PAGE_HEIGHT
        while remaining:
            self.ensure(line_height + spacing_after)
            available_lines = max(
                1,
                int((self.y - self.BOTTOM - spacing_after) / line_height),
            )
            chunk = remaining[:available_lines]
            remaining = remaining[available_lines:]
            final_spacing = spacing_after if not remaining else 0.005
            height = len(chunk) * line_height + final_spacing
            bbox = None
            if background:
                bbox = {
                    "boxstyle": "round,pad=0.45",
                    "facecolor": background,
                    "edgecolor": "#d9e0ea",
                    "linewidth": 0.6,
                }
            self.figure.text(
                self.LEFT + indent,
                self.y,
                "\n".join(chunk),
                ha="left",
                va="top",
                color=color,
                fontsize=fontsize,
                fontweight=weight,
                fontproperties=font or self.font,
                linespacing=1.48,
                bbox=bbox,
            )
            self.y -= height
            if remaining:
                self.new_page()

    def add_heading(self, level: int, text: str) -> None:
        normalized_level = min(max(level, 1), 4)
        sizes = {1: 18.0, 2: 14.0, 3: 11.5, 4: 10.2}
        colors = {1: "#17365d", 2: "#17365d", 3: "#24517c", 4: "#24517c"}
        if normalized_level == 1 and self.y < self.TOP - 0.04:
            self.new_page()
        self.y -= 0.010 if normalized_level > 1 else 0
        self.add_text(
            text,
            fontsize=sizes[normalized_level],
            color=colors[normalized_level],
            weight="bold",
            spacing_after=0.018 if normalized_level <= 2 else 0.011,
            max_units=max(48, int(1120 / sizes[normalized_level])),
        )
        if normalized_level <= 2:
            self.ensure(0.008)
            self.figure.add_artist(
                plt.Line2D(
                    [self.LEFT, self.RIGHT],
                    [self.y + 0.006, self.y + 0.006],
                    transform=self.figure.transFigure,
                    color="#d9e0ea",
                    linewidth=0.8 if normalized_level == 2 else 1.5,
                )
            )

    def add_code(self, code: str) -> None:
        wrapped: List[str] = []
        for line in code.replace("−", "-").replace("•", "-").splitlines() or [""]:
            wrapped.extend(_wrap_units(line, 112))
        line_height = 7.2 * 1.35 / 72 / self.PAGE_HEIGHT
        while wrapped:
            self.ensure(line_height + 0.020)
            available_lines = max(
                1,
                int((self.y - self.BOTTOM - 0.018) / line_height),
            )
            chunk, wrapped = wrapped[:available_lines], wrapped[available_lines:]
            height = len(chunk) * line_height + 0.020
            self.figure.text(
                self.LEFT + 0.012,
                self.y - 0.006,
                "\n".join(chunk),
                ha="left",
                va="top",
                color="#f4f7fb",
                fontsize=7.2,
                fontproperties=self.mono_font,
                linespacing=1.35,
                bbox={
                    "boxstyle": "round,pad=0.65",
                    "facecolor": "#172033",
                    "edgecolor": "#172033",
                },
            )
            self.y -= height + 0.008
            if wrapped:
                self.new_page()

    def add_math(self, expression: str) -> None:
        compact = " ".join(line.strip() for line in expression.splitlines()).strip()
        can_render = (
            len(compact) < 145
            and r"\begin" not in compact
            and r"\end" not in compact
            and r"\\" not in compact
        )
        if can_render:
            try:
                rendered = io.BytesIO()
                math_expression = compact.replace(r"\text{", r"\mathrm{")
                math_to_image(
                    "${}$".format(math_expression),
                    rendered,
                    dpi=180,
                    format="png",
                    color="#172033",
                )
                rendered.seek(0)
                with Image.open(rendered) as image:
                    image.load()
                    self._add_pil_image(
                        image.copy(),
                        "",
                        max_height=0.105,
                        border=False,
                    )
                return
            except Exception:
                pass
        self.add_text(
            expression,
            fontsize=8.5,
            color="#25364c",
            indent=0.025,
            background="#f6f8fb",
            max_units=96,
            font=self.mono_font,
        )

    def _add_pil_image(
        self,
        image: Image.Image,
        caption: str,
        *,
        max_height: float = 0.57,
        border: bool = True,
    ) -> None:
        width_px, height_px = image.size
        normalized_width = self.RIGHT - self.LEFT
        normalized_height = (
            normalized_width
            * height_px
            / max(width_px, 1)
            * self.PAGE_WIDTH
            / self.PAGE_HEIGHT
        )
        if normalized_height > max_height:
            scale = max_height / normalized_height
            normalized_height *= scale
            normalized_width *= scale
        caption_height = 0.025 if caption else 0.008
        self.ensure(normalized_height + caption_height)
        x_position = (1 - normalized_width) / 2
        image_bottom = self.y - normalized_height
        axis = self.figure.add_axes(
            [x_position, image_bottom, normalized_width, normalized_height]
        )
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        if border:
            for spine in axis.spines.values():
                spine.set_edgecolor("#d9e0ea")
                spine.set_linewidth(0.6)
        else:
            axis.set_frame_on(False)
        self.y = image_bottom - 0.007
        if caption:
            self.figure.text(
                0.5,
                self.y,
                caption,
                ha="center",
                va="top",
                color="#667085",
                fontsize=7.3,
                fontproperties=self.font,
            )
            self.y -= 0.024

    def add_image(
        self,
        source_path: Path,
        target: str,
        caption: str,
        *,
        max_height: float = 0.57,
    ) -> None:
        if _is_external_target(target):
            self.add_text(
                "External image not embedded in PDF: {}".format(target),
                color="#b23a48",
                weight="bold",
            )
            return
        image_path = _local_target(source_path, target)
        if not image_path.is_file():
            self.add_text(
                "Missing image: {}".format(target),
                color="#b23a48",
                weight="bold",
            )
            return
        try:
            with Image.open(str(image_path)) as image:
                image.load()
                self._add_pil_image(
                    image.copy(),
                    caption,
                    max_height=max_height,
                )
        except (OSError, ValueError) as error:
            self.add_text(
                "Unreadable image {}: {}".format(target, error),
                color="#b23a48",
                weight="bold",
            )

    def add_table(
        self,
        header: Sequence[str],
        rows: Sequence[Sequence[str]],
    ) -> None:
        if not header:
            return
        column_count = len(header)
        normalized_rows = [
            list(row)[:column_count] + [""] * max(0, column_count - len(row))
            for row in rows
        ]
        rows_per_page = 10 if column_count >= 8 else 15
        chunks = [
            normalized_rows[offset : offset + rows_per_page]
            for offset in range(0, len(normalized_rows), rows_per_page)
        ] or [[]]
        for chunk_index, chunk in enumerate(chunks):
            font_size = (
                5.0
                if column_count >= 8
                else (5.8 if column_count >= 6 else 6.6)
            )
            raw_widths: List[int] = []
            for column in range(column_count):
                values = [header[column]] + [row[column] for row in chunk]
                raw_widths.append(
                    max(7, min(28, max(_display_units(value) for value in values)))
                )
            width_total = sum(raw_widths)
            column_widths = [width / width_total for width in raw_widths]
            wrapped_data: List[List[str]] = []
            maximum_lines = 1
            for row in [list(header)] + list(chunk):
                wrapped_row: List[str] = []
                for column, value in enumerate(row):
                    cell_units = max(7, int(105 * column_widths[column]))
                    wrapped = _wrap_units(_safe_text(value), cell_units)
                    maximum_lines = max(maximum_lines, len(wrapped))
                    wrapped_row.append("\n".join(wrapped))
                wrapped_data.append(wrapped_row)
            row_height = 0.028 + max(0, maximum_lines - 1) * 0.010
            height = min(0.76, row_height * len(wrapped_data) + 0.010)
            self.ensure(height + 0.018)
            axis = self.figure.add_axes(
                [self.LEFT, self.y - height, self.RIGHT - self.LEFT, height]
            )
            axis.axis("off")
            table = axis.table(
                cellText=wrapped_data[1:],
                colLabels=wrapped_data[0],
                colWidths=column_widths,
                cellLoc="left",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)
            for (row_index, _), cell in table.get_celld().items():
                cell.get_text().set_fontproperties(self.font)
                cell.get_text().set_fontsize(font_size)
                cell.set_edgecolor("#d9e0ea")
                cell.set_linewidth(0.5)
                if row_index == 0:
                    cell.set_facecolor("#17365d")
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                elif row_index % 2 == 0:
                    cell.set_facecolor("#f7f9fc")
                else:
                    cell.set_facecolor("white")
            self.y -= height + 0.018
            if chunk_index + 1 < len(chunks):
                self.new_page()

    def add_rule(self) -> None:
        self.ensure(0.022)
        self.figure.add_artist(
            plt.Line2D(
                [self.LEFT, self.RIGHT],
                [self.y - 0.006, self.y - 0.006],
                transform=self.figure.transFigure,
                color="#d9e0ea",
                linewidth=0.8,
            )
        )
        self.y -= 0.022

    def add_block(
        self,
        source_path: Path,
        block_type: str,
        payload: object,
    ) -> None:
        if block_type == "heading":
            level, text = payload
            self.add_heading(level, text)
        elif block_type == "paragraph":
            self.add_text(payload)
        elif block_type == "list":
            self.add_text(
                payload,
                indent=0.018,
                max_units=102,
                spacing_after=0.004,
            )
        elif block_type == "quote":
            self.add_text(
                payload,
                color="#34445a",
                indent=0.018,
                background="#f6f8fb",
                max_units=100,
            )
        elif block_type == "code":
            self.add_code(payload)
        elif block_type == "math":
            self.add_math(payload)
        elif block_type == "image":
            target, caption = payload
            self.add_image(source_path, target, caption)
        elif block_type == "table":
            header, rows = payload
            self.add_table(header, rows)
        elif block_type == "rule":
            self.add_rule()

    def add_cover(
        self,
        sections: Sequence[ReportSection],
        report_root: Path,
    ) -> None:
        self.new_page()
        self.figure.add_artist(
            plt.Line2D(
                [0.07, 0.93],
                [0.90, 0.90],
                transform=self.figure.transFigure,
                color="#17365d",
                linewidth=5,
            )
        )
        self.y = 0.855
        self.add_text(
            self.title,
            fontsize=22,
            color="#17365d",
            weight="bold",
            max_units=74,
            spacing_after=0.016,
        )
        self.add_text(
            "Standalone multi-section factor research report",
            fontsize=10,
            color="#5e6a7d",
            weight="bold",
            spacing_after=0.014,
        )
        self.add_text(
            "Source: {}".format(report_root),
            fontsize=7.6,
            color="#667085",
            max_units=118,
            spacing_after=0.025,
        )
        self.add_heading(2, "Table of Contents")
        for index, section in enumerate(sections, 1):
            self.add_text(
                "{:02d}. {}".format(index, section.title),
                fontsize=9.5,
                color="#24517c",
                indent=0.015,
                spacing_after=0.007,
                max_units=96,
            )


def build_pdf(
    report_root: PathInput,
    output_path: PathInput,
    *,
    title: str = DEFAULT_TITLE,
    markdown_files: Optional[Sequence[PathInput]] = None,
    font_path: Optional[PathInput] = None,
) -> Path:
    """Build a paginated PDF from canonical Markdown sections."""

    root, sections = _resolve_sections(report_root, markdown_files)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = MatplotlibPdfReport(destination, title, font_path=font_path)
    try:
        report.add_cover(sections, root)
        for section in sections:
            source = section.source_path.read_text(encoding="utf-8")
            report.new_page()
            for block_type, payload in parse_markdown_blocks(source):
                report.add_block(section.source_path, block_type, payload)
    finally:
        report.close()
    return destination


def export_standalone_report(
    report_root: PathInput,
    output_dir: PathInput,
    *,
    title: str = DEFAULT_TITLE,
    output_stem: str = "standalone_factor_report",
    markdown_files: Optional[Sequence[PathInput]] = None,
    mathjax_url: str = DEFAULT_MATHJAX_URL,
    font_path: Optional[PathInput] = None,
) -> ExportedReport:
    """Create both single-file HTML and PDF exports with a shared section set."""

    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("output_stem must be a non-empty file stem")
    _, sections = _resolve_sections(report_root, markdown_files)
    selected_files = [section.source_path for section in sections]
    destination = Path(output_dir).expanduser().resolve()
    html_path = build_html(
        report_root,
        destination / "{}.html".format(output_stem),
        title=title,
        markdown_files=selected_files,
        mathjax_url=mathjax_url,
    )
    pdf_path = build_pdf(
        report_root,
        destination / "{}.pdf".format(output_stem),
        title=title,
        markdown_files=selected_files,
        font_path=font_path,
    )
    return ExportedReport(html_path=html_path, pdf_path=pdf_path)


__all__ = [
    "DEFAULT_MATHJAX_URL",
    "DEFAULT_TITLE",
    "EmbeddedMarkdownRenderer",
    "ExportedReport",
    "MatplotlibPdfReport",
    "ReportSection",
    "build_html",
    "build_pdf",
    "discover_markdown_files",
    "export_standalone_report",
    "parse_markdown_blocks",
]
