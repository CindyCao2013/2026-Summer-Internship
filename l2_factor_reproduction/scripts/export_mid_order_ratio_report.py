#!/usr/bin/env python3
"""Export the multi-file mid_order_ratio report to HTML and PDF.

The exporter intentionally uses packages already present in the research
environment:

* Mistune renders Markdown tables, lists, links, and code to HTML.
* Matplotlib creates a paginated, searchable PDF with Chinese font support.
* The local Jupyter MathJax bundle renders equations in the HTML export.

Usage:
    python l2_factor_reproduction/scripts/export_mid_order_ratio_report.py
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import mimetypes
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mistune
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = (
    WORKSPACE / "research" / "reports" / "factors" / "mid_order_ratio"
)
CJK_FONT_PATH = Path("/usr/share/fonts/chinese/simhei.ttf")

REPORT_FILES: Sequence[Tuple[str, str, str]] = (
    ("README.md", "readme", "Executive Overview"),
    ("01_executive_summary.md", "summary", "01 — Executive Summary"),
    (
        "02_data_and_factor_construction.md",
        "construction",
        "02 — Data and Factor Construction",
    ),
    (
        "03_standalone_signal_validation.md",
        "validation",
        "03 — Standalone Signal Validation",
    ),
    ("04_robustness_analysis.md", "robustness", "04 — Robustness Analysis"),
    (
        "05_factor_exposure_diagnostics.md",
        "exposure-diagnostics",
        "05 — Factor Exposure Diagnostics",
    ),
    (
        "06_economic_interpretation.md",
        "interpretation",
        "06 — Economic Interpretation",
    ),
    (
        "07_limitations_and_future_research.md",
        "limitations",
        "07 — Limitations and Future Research",
    ),
    (
        "08_research_decision_framework.md",
        "decision-framework",
        "08 — Research Decision Framework",
    ),
    ("appendix/code_reference.md", "code-reference", "Appendix A — Code Reference"),
    ("appendix/data_lineage.md", "data-lineage", "Appendix B — Data Lineage"),
    (
        "appendix/reproduction_commands.md",
        "reproduction",
        "Appendix C — Reproduction Commands",
    ),
)


def _safe_text(value: str) -> str:
    """Remove Markdown presentation syntax while preserving content."""
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"__([^_]+)__", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"\\\((.*?)\\\)", r"\1", value)
    value = value.replace("\\[", "").replace("\\]", "")
    value = value.replace("−", "-").replace("•", "-")
    return html.unescape(value).strip()


def _display_units(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)


def _wrap_units(text: str, max_units: int) -> List[str]:
    """Wrap mixed Chinese/Latin text using approximate display width."""
    if not text:
        return [""]
    output: List[str] = []
    for source_line in text.splitlines() or [""]:
        source_line = source_line.rstrip()
        while _display_units(source_line) > max_units:
            used = 0
            cut = 0
            preferred = 0
            for idx, ch in enumerate(source_line):
                used += 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
                if ch.isspace() or ch in "，。；：、,.!?;:)）]】":
                    preferred = idx + 1
                if used > max_units:
                    cut = idx
                    break
            if preferred >= int(cut * 0.65):
                cut = preferred
            cut = max(cut, 1)
            output.append(source_line[:cut].rstrip())
            source_line = source_line[cut:].lstrip()
        output.append(source_line)
    return output


def _protect_math(markdown_text: str) -> str:
    """Protect TeX delimiters from Mistune's backslash escaping."""

    def display(match: re.Match) -> str:
        return f'\n<div class="math-block">\\[{match.group(1)}\\]</div>\n'

    def inline(match: re.Match) -> str:
        return f'<span class="math-inline">\\({match.group(1)}\\)</span>'

    markdown_text = re.sub(
        r"\\\[(.*?)\\\]",
        display,
        markdown_text,
        flags=re.DOTALL,
    )
    return re.sub(r"\\\((.*?)\\\)", inline, markdown_text)


class EmbeddedReportRenderer(mistune.Renderer):
    """Mistune renderer that embeds local figures and fixes report links."""

    def __init__(self, report_root: Path, section_links: dict):
        super().__init__(escape=False)
        self.report_root = report_root
        self.section_links = section_links

    def image(self, src: str, title: str, text: str) -> str:
        if src.startswith(("http://", "https://", "data:")):
            return super().image(src, title, text)
        image_path = (self.report_root / src).resolve()
        if not image_path.exists():
            return (
                f'<span class="missing-image">Missing image: '
                f"{html.escape(src)}</span>"
            )
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return (
            f'<img src="data:{mime};base64,{payload}" '
            f'alt="{html.escape(text or image_path.stem)}" loading="eager">'
        )

    def link(self, link: str, title: str, text: str) -> str:
        if link in self.section_links:
            href = f"#{self.section_links[link]}"
        elif link.startswith(("http://", "https://", "mailto:", "#")):
            href = link
        elif link.startswith("export/"):
            href = link.split("/", 1)[1]
        else:
            href = f"../{link}"
        title_attr = f' title="{html.escape(title)}"' if title else ""
        return f'<a href="{html.escape(href)}"{title_attr}>{text}</a>'


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
  padding-top: 32px;
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
}
.cover .subtitle { color: var(--muted); font-size: 20px; }
.cover-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 36px;
}
.cover-card {
  background: var(--soft);
  border-left: 4px solid var(--blue);
  padding: 17px 20px;
}
.cover-card strong { display: block; color: var(--navy); font-size: 23px; }
.disclaimer {
  margin-top: 28px;
  color: var(--muted);
  font-size: 13px;
}
.brief-page {
  min-height: 760px;
  padding: 20px 4px 30px;
  border-bottom: 1px solid var(--line);
}
.brief-page > h1 {
  margin-top: 0;
  font-size: 29px;
}
.brief-page h2 {
  margin-top: 18px;
  font-size: 18px;
}
.brief-page table {
  margin: 10px 0 16px;
  font-size: 11px;
  line-height: 1.38;
}
.brief-page th, .brief-page td { padding: 5px 7px; }
.brief-page img {
  max-height: 520px;
  width: auto;
  margin: 12px auto 16px;
}
.brief-note {
  color: var(--muted);
  font-size: 12px;
  margin: 8px 0 12px;
}
.brief-definition {
  color: var(--navy);
  font-size: 15px;
  font-weight: 650;
  padding: 10px 14px;
  border-left: 4px solid var(--blue);
  background: var(--soft);
}
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
h1, h2, h3, h4 { color: var(--navy); line-height: 1.35; }
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
  .brief-page {
    min-height: 252mm;
    max-height: 252mm;
    overflow: hidden;
    break-after: page;
    page-break-after: always;
  }
  .brief-page img { max-height: 118mm; }
  .brief-page table { font-size: 8.5px; }
  .toc { page-break-after: always; }
  .document-section { page-break-before: always; }
  h1, h2, h3 { break-after: avoid; }
  table, img, blockquote, pre { break-inside: avoid; }
  a { color: inherit; }
}
"""


def _image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def build_html(
    report_root: Path,
    output_path: Path,
    *,
    mathjax_mode: str = "cdn",
) -> None:
    """Build a browser-openable HTML report.

    ``mathjax_mode``:
      - ``cdn``: single-file public HTML (images embedded; MathJax from CDN)
      - ``local``: keep a local ``assets/mathjax`` copy for offline intranet use
    """
    section_links = {name: anchor for name, anchor, _ in REPORT_FILES}
    renderer = EmbeddedReportRenderer(report_root, section_links)
    markdown = mistune.Markdown(renderer=renderer)
    sections = []
    for file_name, anchor, label in REPORT_FILES:
        text = (report_root / file_name).read_text(encoding="utf-8")
        if file_name == "README.md":
            text = re.sub(
                r"\n## (?:报告导航|Report navigation|Full report export)\n.*?(?=\n## )",
                "\n",
                text,
                flags=re.DOTALL,
            )
        body = markdown(_protect_math(text))
        sections.append(
            f'<section id="{anchor}" class="document-section" '
            f'data-source="{html.escape(file_name)}">{body}</section>'
        )

    toc_items = "\n".join(
        f'<li><a href="#{anchor}">{html.escape(label)}</a></li>'
        for _, anchor, label in REPORT_FILES
    )
    if mathjax_mode == "local":
        mathjax_source = Path(
            "/opt/conda/anaconda3/envs/base_93/lib/python3.8/site-packages/"
            "notebook/static/components/MathJax"
        )
        mathjax_target = output_path.parent / "assets" / "mathjax"
        if mathjax_source.exists() and not mathjax_target.exists():
            mathjax_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(mathjax_source, mathjax_target)
        mathjax_script = (
            '<script type="text/x-mathjax-config">\n'
            "MathJax.Hub.Config({\n"
            '  messageStyle: "none",\n'
            '  tex2jax: {inlineMath: [["\\\\(","\\\\)"]], '
            'displayMath: [["\\\\[","\\\\]"]]},\n'
            '  SVG: {font: "TeX"}\n'
            "});\n"
            "</script>\n"
            '<script src="assets/mathjax/MathJax.js?config=TeX-AMS-MML_SVG"></script>'
        )
    else:
        # Public single-file HTML: no local assets folder required.
        mathjax_script = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']]
  },
  svg: { fontCache: 'global' },
  options: { enableMenu: false }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""

    pipeline_uri = _image_data_uri(report_root / "figures" / "01_pipeline_architecture.png")
    universe_uri = _image_data_uri(report_root / "figures" / "03_universe_comparison_table.png")
    decile_uri = _image_data_uri(
        report_root / "figures" / "04_decile_cumulative_csi1000_index_excess.png"
    )
    stability_uri = _image_data_uri(
        report_root / "figures" / "07b_ic_stability_combined.png"
    )
    exposure_uri = _image_data_uri(
        report_root / "figures" / "08_neutralization_comparison.png"
    )

    brief_pages = f"""
<section class="brief-page cover">
  <div class="eyebrow">Standalone Factor Validation</div>
  <h1>mid_order_ratio — Single Factor Research Report</h1>
  <p><strong>Research status:</strong> Completed factor construction audit, standalone predictive validation, robustness testing, and exposure diagnostics.</p>
  <div class="brief-definition">mid_order_ratio measures the fraction of daily traded amount contributed by trades with transaction amount between RMB 40k and RMB 200k.</div>
  <h2>Research universe</h2>
  <table>
    <tr><th>Sample</th><td>2023-01-04 to 2024-06-28 · 358 days</td><th>Universe</th><td>CSI1000 PIT · avg. 990 valid names</td></tr>
    <tr><th>Signal lag</th><td>T-1 factor</td><th>Return horizon</th><td>Next-day close-to-close</td></tr>
    <tr><th>Factor data</th><td>ClickHouse strict L2 Tick</td><th>Reference data</th><td>DolphinDB Wind</td></tr>
  </table>
  <h2>Table 1 — Standalone Factor Performance Summary</h2>
  <table>
    <tr><th>Metric</th><th>Result</th><th>Interpretation</th></tr>
    <tr><td>RankIC</td><td>Raw -4.80% · effective +4.80%</td><td>Higher raw factor predicts lower next-day return rank</td></tr>
    <tr><td>ICIR</td><td>Raw -6.33 · effective +6.33</td><td>Sign follows factor orientation</td></tr>
    <tr><td>IC t-stat</td><td>-7.58</td><td>Sample mean differs from zero; no Newey-West adjustment</td></tr>
    <tr><td>IC negative days</td><td>67.9%</td><td>Not driven only by a few dates</td></tr>
    <tr><td>Effective H-L Sharpe</td><td>2.74 gross</td><td>Standalone sorting diagnostic; fee=0</td></tr>
    <tr><td>H-L MDD</td><td>-9.68%</td><td>Compounded H-L diagnostic path</td></tr>
    <tr><td>H-L turnover</td><td>1.69 / day</td><td>High turnover; costs not modeled</td></tr>
  </table>
  <h2>Figure 1 — Factor construction pipeline</h2>
  <img src="{pipeline_uri}" alt="Factor construction pipeline">
</section>
<section class="brief-page">
  <h1>Figure 2 — Universe comparison</h1>
  <p class="brief-note">The raw-factor RankIC direction is negative in SSE/SZSE A-share (excluding BSE), CSI300, CSI500, and CSI1000. H-L metrics are standalone sorting diagnostics.</p>
  <img src="{universe_uri}" alt="Universe comparison table">
  <table>
    <tr><th>Universe</th><th>RankIC</th><th>ICIR</th><th>H-L Sharpe</th><th>MDD</th><th>Turnover</th></tr>
    <tr><td>ALL</td><td>-5.53%</td><td>-7.29</td><td>3.05</td><td>-20.51%</td><td>1.73</td></tr>
    <tr><td>CSI300</td><td>-2.12%</td><td>-3.34</td><td>1.26</td><td>-9.90%</td><td>1.68</td></tr>
    <tr><td>CSI500</td><td>-3.45%</td><td>-4.43</td><td>1.49</td><td>-9.57%</td><td>1.72</td></tr>
    <tr><td>CSI1000</td><td>-4.80%</td><td>-6.33</td><td>2.74</td><td>-9.68%</td><td>1.69</td></tr>
  </table>
  <p><strong>Interpretation:</strong> signal direction is consistent across universes, while magnitude varies by market-cap segment.</p>
</section>
<section class="brief-page">
  <h1>Figure 3 — CSI1000 index-excess decile diagnostic</h1>
  <p class="brief-note">Effective signal = -mid_order_ratio; T-1 signal; cumulative sum of next-day CSI1000 index-excess returns. This is a standalone factor sorting diagnostic, not a recommendation.</p>
  <img src="{decile_uri}" alt="CSI1000 index-excess decile cumulative return">
  <table>
    <tr><th>Decile</th><th>G1</th><th>G2</th><th>G3</th><th>G4</th><th>G5</th><th>G6</th><th>G7</th><th>G8</th><th>G9</th><th>G10</th></tr>
    <tr><th>Annualized excess</th><td>-30.07%</td><td>-15.51%</td><td>3.89%</td><td>8.21%</td><td>5.95%</td><td>11.48%</td><td>10.48%</td><td>10.22%</td><td>12.05%</td><td>11.05%</td></tr>
  </table>
  <p><strong>Interpretation:</strong> tails are clearly separated and the overall ordering is consistent, but middle deciles are not perfectly monotonic.</p>
</section>
<section class="brief-page">
  <h1>Figure 4 — IC stability</h1>
  <p class="brief-note">Daily raw RankIC, 63-trading-day rolling mean, and monthly averages address whether the relationship is concentrated in a small number of dates.</p>
  <img src="{stability_uri}" alt="Daily rolling and monthly IC stability">
  <table>
    <tr><th>Mean raw RankIC</th><th>Negative-IC months</th><th>Excluding 2024-01</th><th>Largest positive month</th></tr>
    <tr><td>-4.80%</td><td>16 / 18</td><td>-4.79%</td><td>2024-02 · +3.10%</td></tr>
  </table>
</section>
<section class="brief-page">
  <h1>Figure 5 — Factor exposure diagnostics</h1>
  <p class="brief-note">Industry and size controls test whether common exposures fully explain the standalone predictive relationship. They are not additional alpha inputs.</p>
  <img src="{exposure_uri}" alt="Industry and size exposure diagnostic">
  <table>
    <tr><th>Method</th><th>Raw-direction RankIC</th><th>ICIR</th><th>Effective H-L Sharpe</th><th>MDD</th></tr>
    <tr><td>Raw</td><td>-4.80%</td><td>-6.33</td><td>2.74</td><td>-9.68%</td></tr>
    <tr><td>Industry</td><td>-4.62%</td><td>-9.53</td><td>4.17</td><td>-6.62%</td></tr>
    <tr><td>Market cap</td><td>-4.52%</td><td>-5.82</td><td>2.42</td><td>-10.05%</td></tr>
    <tr><td>Industry + market cap</td><td>-4.41%</td><td>-9.09</td><td>4.42</td><td>-7.22%</td></tr>
  </table>
  <p><strong>Interpretation:</strong> industry and size explain part of the factor level, but do not fully explain the negative next-day RankIC.</p>
</section>
"""

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mid_order_ratio — Single Factor Research Report</title>
<style>{_html_css()}</style>
{mathjax_script}
</head>
<body>
<main>
{brief_pages}
<nav class="toc">
  <h2>Table of Contents</h2>
  <ol>{toc_items}</ol>
</nav>
{''.join(sections)}
<div class="generated">Generated from the canonical multi-file Markdown report.</div>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def _parse_table_row(line: str) -> List[str]:
    return [_safe_text(cell.strip()) for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    cells = line.strip().strip("|").split("|")
    return bool(cells) and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)


def parse_markdown_blocks(text: str) -> Iterator[Tuple[str, object]]:
    """Yield a small block model sufficient for this report's Markdown."""
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            yield "heading", (len(heading.group(1)), _safe_text(heading.group(2)))
            idx += 1
            continue

        fence = re.match(r"^```(.*)$", line.strip())
        if fence:
            code_lines = []
            idx += 1
            while idx < len(lines) and not lines[idx].strip().startswith("```"):
                code_lines.append(lines[idx])
                idx += 1
            idx += 1
            yield "code", "\n".join(code_lines)
            continue

        if line.strip() == r"\[":
            math_lines = []
            idx += 1
            while idx < len(lines) and lines[idx].strip() != r"\]":
                math_lines.append(lines[idx])
                idx += 1
            idx += 1
            yield "math", "\n".join(math_lines)
            continue

        image = re.fullmatch(r"\s*!\[([^\]]*)\]\(([^)]+)\)\s*", line)
        if image:
            yield "image", (image.group(2), image.group(1))
            idx += 1
            continue

        if (
            line.lstrip().startswith("|")
            and idx + 1 < len(lines)
            and _is_table_separator(lines[idx + 1])
        ):
            header = _parse_table_row(line)
            idx += 2
            rows = []
            while idx < len(lines) and lines[idx].lstrip().startswith("|"):
                rows.append(_parse_table_row(lines[idx]))
                idx += 1
            yield "table", (header, rows)
            continue

        if re.match(r"^\s*>\s?", line):
            quote_lines = []
            while idx < len(lines) and re.match(r"^\s*>\s?", lines[idx]):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[idx]))
                idx += 1
            yield "quote", _safe_text(" ".join(quote_lines))
            continue

        list_item = re.match(r"^\s*([-*]|\d+\.)\s+(.+)$", line)
        if list_item:
            prefix = "-" if list_item.group(1) in {"-", "*"} else list_item.group(1)
            yield "list", f"{prefix} {_safe_text(list_item.group(2))}"
            idx += 1
            continue

        if re.fullmatch(r"\s*---+\s*", line):
            yield "rule", ""
            idx += 1
            continue

        paragraph = [line.strip()]
        idx += 1
        while idx < len(lines):
            candidate = lines[idx]
            if not candidate.strip():
                break
            if re.match(r"^(#{1,4})\s+", candidate):
                break
            if candidate.strip().startswith(("```", r"\[")):
                break
            if re.fullmatch(r"\s*!\[([^\]]*)\]\(([^)]+)\)\s*", candidate):
                break
            if re.match(r"^\s*([-*]|\d+\.)\s+", candidate):
                break
            if candidate.lstrip().startswith("|") and idx + 1 < len(lines):
                if _is_table_separator(lines[idx + 1]):
                    break
            paragraph.append(candidate.strip())
            idx += 1
        yield "paragraph", _safe_text(" ".join(paragraph))


class MatplotlibPdfReport:
    PAGE_WIDTH = 8.27
    PAGE_HEIGHT = 11.69
    LEFT = 0.07
    RIGHT = 0.95
    TOP = 0.94
    BOTTOM = 0.065

    def __init__(self, output_path: Path, report_root: Path):
        self.output_path = output_path
        self.report_root = report_root
        self.pdf = PdfPages(output_path)
        self.figure = None
        self.y = self.TOP
        self.page_number = 0
        if CJK_FONT_PATH.exists():
            self.font = FontProperties(fname=str(CJK_FONT_PATH))
            self.mono_font = FontProperties(fname=str(CJK_FONT_PATH))
        else:
            self.font = FontProperties(family="DejaVu Sans")
            self.mono_font = FontProperties(family="DejaVu Sans Mono")

    def close(self) -> None:
        self._flush_page()
        self.pdf.close()

    def _flush_page(self) -> None:
        if self.figure is None:
            return
        self.figure.text(
            0.5,
            0.027,
            f"mid_order_ratio research report  ·  {self.page_number}",
            ha="center",
            color="#667085",
            fontsize=7.5,
            fontproperties=self.font,
        )
        self.pdf.savefig(self.figure)
        plt.close(self.figure)
        self.figure = None

    def new_page(self) -> None:
        self._flush_page()
        self.figure = plt.figure(figsize=(self.PAGE_WIDTH, self.PAGE_HEIGHT), dpi=140)
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
        font: FontProperties = None,
    ) -> None:
        lines = _wrap_units(text, max_units)
        line_height = fontsize * 1.48 / 72 / self.PAGE_HEIGHT
        height = max(len(lines), 1) * line_height + spacing_after
        self.ensure(height)
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
            "\n".join(lines),
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

    def add_heading(self, level: int, text: str) -> None:
        sizes = {1: 18, 2: 14, 3: 11.5, 4: 10.2}
        colors = {1: "#17365d", 2: "#17365d", 3: "#24517c", 4: "#24517c"}
        if level == 1 and self.y < self.TOP - 0.04:
            self.new_page()
        self.y -= 0.010 if level > 1 else 0
        self.add_text(
            text,
            fontsize=sizes[level],
            color=colors[level],
            weight="bold",
            spacing_after=0.018 if level <= 2 else 0.011,
            max_units=max(48, int(1120 / sizes[level])),
        )
        if level <= 2:
            self.ensure(0.008)
            self.figure.add_artist(
                plt.Line2D(
                    [self.LEFT, self.RIGHT],
                    [self.y + 0.006, self.y + 0.006],
                    transform=self.figure.transFigure,
                    color="#d9e0ea",
                    linewidth=0.8 if level == 2 else 1.5,
                )
            )

    def add_code(self, code: str) -> None:
        code = code.replace("−", "-").replace("•", "-")
        wrapped = []
        for line in code.splitlines() or [""]:
            wrapped.extend(_wrap_units(line, 112))
        line_height = 7.2 * 1.35 / 72 / self.PAGE_HEIGHT
        while wrapped:
            available_lines = max(
                1,
                int((self.y - self.BOTTOM - 0.018) / line_height),
            )
            chunk, wrapped = wrapped[:available_lines], wrapped[available_lines:]
            height = len(chunk) * line_height + 0.020
            self.ensure(height)
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
                    f"${math_expression}$",
                    rendered,
                    dpi=180,
                    format="png",
                    color="#172033",
                )
                rendered.seek(0)
                with Image.open(rendered) as image:
                    image.load()
                    self._add_pil_image(image.copy(), "", max_height=0.105, border=False)
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
        x = (1 - normalized_width) / 2
        image_bottom = self.y - normalized_height
        axis = self.figure.add_axes([x, image_bottom, normalized_width, normalized_height])
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
        relative_path: str,
        caption: str,
        *,
        max_height: float = 0.57,
    ) -> None:
        image_path = (self.report_root / relative_path).resolve()
        if not image_path.exists():
            self.add_text(
                f"Missing image: {relative_path}",
                color="#b23a48",
                weight="bold",
            )
            return
        with Image.open(image_path) as image:
            image.load()
            self._add_pil_image(image.copy(), caption, max_height=max_height)

    def add_table(self, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        if not rows:
            return
        ncols = len(header)
        normalized_rows = [
            list(row)[:ncols] + [""] * max(0, ncols - len(row)) for row in rows
        ]
        max_rows = 10 if ncols >= 8 else 15
        for offset in range(0, len(normalized_rows), max_rows):
            chunk = normalized_rows[offset : offset + max_rows]
            chunk_header = list(header)
            font_size = 5.0 if ncols >= 8 else (5.8 if ncols >= 6 else 6.6)
            raw_widths = []
            for col in range(ncols):
                values = [chunk_header[col]] + [row[col] for row in chunk]
                raw_widths.append(max(7, min(28, max(_display_units(v) for v in values))))
            width_total = sum(raw_widths)
            col_widths = [width / width_total for width in raw_widths]
            wrapped_data = []
            max_lines = 1
            for row in [chunk_header] + chunk:
                wrapped_row = []
                for col, value in enumerate(row):
                    cell_units = max(7, int(105 * col_widths[col]))
                    wrapped = _wrap_units(_safe_text(value), cell_units)
                    max_lines = max(max_lines, len(wrapped))
                    wrapped_row.append("\n".join(wrapped))
                wrapped_data.append(wrapped_row)
            row_height = 0.028 + max(0, max_lines - 1) * 0.010
            height = min(0.76, row_height * len(wrapped_data) + 0.010)
            self.ensure(height + 0.018)
            axis = self.figure.add_axes(
                [self.LEFT, self.y - height, self.RIGHT - self.LEFT, height]
            )
            axis.axis("off")
            table = axis.table(
                cellText=wrapped_data[1:],
                colLabels=wrapped_data[0],
                colWidths=col_widths,
                cellLoc="left",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)
            for (row_idx, _), cell in table.get_celld().items():
                cell.get_text().set_fontproperties(self.font)
                cell.get_text().set_fontsize(font_size)
                cell.set_edgecolor("#d9e0ea")
                cell.set_linewidth(0.5)
                if row_idx == 0:
                    cell.set_facecolor("#17365d")
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                elif row_idx % 2 == 0:
                    cell.set_facecolor("#f7f9fc")
                else:
                    cell.set_facecolor("white")
            self.y -= height + 0.018
            if offset + max_rows < len(normalized_rows):
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

    def add_block(self, block_type: str, payload: object) -> None:
        if block_type == "heading":
            level, text = payload
            self.add_heading(level, text)
        elif block_type == "paragraph":
            self.add_text(payload)
        elif block_type == "list":
            self.add_text(payload, indent=0.018, max_units=102, spacing_after=0.004)
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
            self.add_image(*payload)
        elif block_type == "table":
            self.add_table(*payload)
        elif block_type == "rule":
            self.add_rule()


def build_pdf(report_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = MatplotlibPdfReport(output_path, report_root)
    try:
        report.new_page()
        report.figure.add_artist(
            plt.Line2D(
                [0.07, 0.93],
                [0.90, 0.90],
                transform=report.figure.transFigure,
                color="#17365d",
                linewidth=5,
            )
        )
        report.y = 0.855
        report.add_text(
            "mid_order_ratio — Single Factor Research Report",
            fontsize=22,
            color="#17365d",
            weight="bold",
            max_units=74,
            spacing_after=0.010,
        )
        report.add_text(
            "Research status: Completed factor construction audit, standalone predictive "
            "validation, robustness testing, and exposure diagnostics.",
            fontsize=8.6,
            color="#5e6a7d",
            weight="bold",
            max_units=112,
            spacing_after=0.010,
        )
        report.add_text(
            "mid_order_ratio measures the fraction of daily traded amount contributed by "
            "trades with transaction amount between RMB 40k and RMB 200k.",
            fontsize=8.8,
            color="#17365d",
            weight="bold",
            background="#f6f8fb",
            max_units=106,
            spacing_after=0.008,
        )
        report.add_text(
            "Research universe",
            fontsize=9.5,
            color="#24517c",
            weight="bold",
            spacing_after=0.004,
        )
        report.add_text(
            "Sample: 2023-01-04 to 2024-06-28 · 358 days    |    "
            "Universe: CSI1000 PIT · avg. 990 valid names\n"
            "Signal lag: T-1 factor    |    Return horizon: next-day close-to-close\n"
            "Factor data: ClickHouse strict L2 Tick    |    Reference data: DolphinDB Wind",
            fontsize=7.5,
            color="#25364c",
            max_units=118,
            spacing_after=0.006,
        )
        report.add_text(
            "Table 1 — Standalone Factor Performance Summary",
            fontsize=9.5,
            color="#24517c",
            weight="bold",
            spacing_after=0.004,
        )
        report.add_table(
            ["Metric", "Result", "Interpretation"],
            [
                ["RankIC", "Raw -4.80% · effective +4.80%", "Higher raw factor predicts lower return rank"],
                ["ICIR", "Raw -6.33 · effective +6.33", "Sign follows factor orientation"],
                ["IC t-stat", "-7.58", "Sample mean differs from zero"],
                ["IC negative days", "67.9%", "Not driven only by a few dates"],
                ["Effective H-L Sharpe", "2.74 gross", "Standalone sorting diagnostic; fee=0"],
                ["H-L MDD", "-9.68%", "Compounded diagnostic path"],
                ["H-L turnover", "1.69 / day", "High turnover; costs not modeled"],
            ],
        )
        report.add_text(
            "Figure 1 — Factor construction pipeline",
            fontsize=9.5,
            color="#24517c",
            weight="bold",
            spacing_after=0.004,
        )
        report.add_image(
            "figures/01_pipeline_architecture.png",
            "Tick data -> strict transaction filtering -> trade-size bucket -> daily factor -> next-day return",
            max_height=0.175,
        )

        report.new_page()
        report.add_text(
            "Figure 2 — Universe comparison",
            fontsize=18,
            color="#17365d",
            weight="bold",
            spacing_after=0.012,
        )
        report.add_text(
            "Raw-factor RankIC direction is negative in SSE/SZSE A-share (excluding BSE), CSI300, CSI500, "
            "and CSI1000. H-L metrics are standalone sorting diagnostics.",
            fontsize=9,
            color="#5e6a7d",
            spacing_after=0.008,
        )
        report.add_image(
            "figures/03_universe_comparison_table.png",
            "Figure 2. Same strict-trade definition, T-1 signal, and PIT membership rules.",
            max_height=0.48,
        )
        report.add_table(
            ["Universe", "RankIC", "ICIR", "H-L Sharpe", "MDD", "Turnover"],
            [
                ["ALL", "-5.53%", "-7.29", "3.05", "-20.51%", "1.73"],
                ["CSI300", "-2.12%", "-3.34", "1.26", "-9.90%", "1.68"],
                ["CSI500", "-3.45%", "-4.43", "1.49", "-9.57%", "1.72"],
                ["CSI1000", "-4.80%", "-6.33", "2.74", "-9.68%", "1.69"],
            ],
        )
        report.add_text(
            "Interpretation: signal direction is consistent across universes, while "
            "magnitude varies by market-cap segment.",
            fontsize=9,
            color="#17365d",
            weight="bold",
            max_units=104,
        )

        report.new_page()
        report.add_text(
            "Figure 3 — CSI1000 index-excess decile diagnostic",
            fontsize=18,
            color="#17365d",
            weight="bold",
            spacing_after=0.010,
        )
        report.add_text(
            "Effective signal = -mid_order_ratio; T-1 signal; cumulative sum of next-day "
            "CSI1000 index-excess returns. This is a standalone sorting diagnostic.",
            fontsize=8.8,
            color="#5e6a7d",
            spacing_after=0.008,
        )
        report.add_image(
            "figures/04_decile_cumulative_csi1000_index_excess.png",
            "Figure 3. Tails separate clearly; middle deciles are not perfectly monotonic.",
            max_height=0.46,
        )
        report.add_table(
            ["Decile", "Annualized excess", "Raw-factor meaning"],
            [
                ["G1", "-30.07%", "highest mid_order_ratio"],
                ["G2", "-15.51%", "higher"],
                ["G3", "3.89%", ""],
                ["G4", "8.21%", ""],
                ["G5", "5.95%", ""],
                ["G6", "11.48%", ""],
                ["G7", "10.48%", ""],
                ["G8", "10.22%", ""],
                ["G9", "12.05%", "lower"],
                ["G10", "11.05%", "lowest mid_order_ratio"],
            ],
        )

        report.new_page()
        report.add_text(
            "Figure 4 — IC stability",
            fontsize=18,
            color="#17365d",
            weight="bold",
            spacing_after=0.010,
        )
        report.add_text(
            "Daily raw RankIC, 63-trading-day rolling mean, and monthly averages test "
            "whether the relationship is concentrated in a small number of dates.",
            fontsize=9,
            color="#5e6a7d",
            spacing_after=0.008,
        )
        report.add_image(
            "figures/07b_ic_stability_combined.png",
            "Figure 4. Mean raw RankIC -4.80%; 16 of 18 monthly means are negative.",
            max_height=0.70,
        )
        report.add_table(
            ["Mean RankIC", "Negative months", "Excluding 2024-01", "Largest positive month"],
            [["-4.80%", "16 / 18", "-4.79%", "2024-02 · +3.10%"]],
        )

        report.new_page()
        report.add_text(
            "Figure 5 — Factor exposure diagnostics",
            fontsize=18,
            color="#17365d",
            weight="bold",
            spacing_after=0.010,
        )
        report.add_text(
            "Industry and size controls test whether common exposures fully explain the "
            "standalone predictive relationship. They are not additional alpha inputs.",
            fontsize=9,
            color="#5e6a7d",
            spacing_after=0.008,
        )
        report.add_image(
            "figures/08_neutralization_comparison.png",
            "Figure 5. Daily CSI1000 PIT cross-sectional residual diagnostics.",
            max_height=0.48,
        )
        report.add_table(
            ["Method", "RankIC", "ICIR", "H-L Sharpe", "MDD"],
            [
                ["Raw", "-4.80%", "-6.33", "2.74", "-9.68%"],
                ["Industry", "-4.62%", "-9.53", "4.17", "-6.62%"],
                ["Market cap", "-4.52%", "-5.82", "2.42", "-10.05%"],
                ["Industry + market cap", "-4.41%", "-9.09", "4.42", "-7.22%"],
            ],
        )
        report.add_text(
            "Interpretation: industry and size explain part of the factor level, but do "
            "not fully explain the negative next-day RankIC.",
            fontsize=9,
            color="#17365d",
            weight="bold",
            max_units=104,
        )

        report.new_page()
        report.add_heading(1, "Table of Contents")
        for index, (_, _, label) in enumerate(REPORT_FILES, 1):
            report.add_text(
                f"{index:02d}. {label}",
                fontsize=10.2,
                color="#24517c",
                indent=0.015,
                spacing_after=0.009,
                max_units=92,
            )

        for file_name, _, _ in REPORT_FILES:
            source = (report_root / file_name).read_text(encoding="utf-8")
            if file_name == "README.md":
                source = re.sub(
                    r"\n## (?:报告导航|Report navigation)\n.*?(?=\n## )",
                    "\n",
                    source,
                    flags=re.DOTALL,
                )
            report.new_page()
            for block_type, payload in parse_markdown_blocks(source):
                report.add_block(block_type, payload)
    finally:
        report.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
        help="Canonical multi-file report directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Export directory; defaults to <report-root>/export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    output_dir = (args.output_dir or report_root / "export").resolve()
    missing = [name for name, _, _ in REPORT_FILES if not (report_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing report files: {missing}")

    html_path = output_dir / "mid_order_ratio_report.html"
    public_html_path = output_dir / "public" / "index.html"
    pdf_path = output_dir / "mid_order_ratio_report.pdf"
    build_html(report_root, html_path, mathjax_mode="local")
    build_html(report_root, public_html_path, mathjax_mode="cdn")
    build_pdf(report_root, pdf_path)
    print(f"HTML (local assets): {html_path}")
    print(f"HTML (public single-file): {public_html_path}")
    print(f"PDF:  {pdf_path}")
    print(f"file://{public_html_path.resolve()}")


if __name__ == "__main__":
    main()
