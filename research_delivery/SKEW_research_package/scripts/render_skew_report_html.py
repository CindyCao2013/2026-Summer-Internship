#!/usr/bin/env python3
"""Render the SKEW Markdown report as a self-contained HTML memo."""

import base64
import html
import io
import mimetypes
import re
from pathlib import Path

import mistune
from matplotlib.mathtext import math_to_image


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "research_delivery/SKEW_research_package").is_dir()
)
REPORT_DIR = ROOT / "research_delivery/SKEW_research_package/report"
SOURCE = REPORT_DIR / "SKEW_买方因子研究报告.md"
TARGET = REPORT_DIR / "SKEW_买方因子研究报告.html"

CSS = """
:root { color-scheme: light; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC",
       "Microsoft YaHei", sans-serif; color:#17202a; line-height:1.65; margin:0; }
main { max-width:1120px; margin:36px auto 80px; padding:0 34px; }
h1 { color:#102a43; border-bottom:2px solid #2f80ed; padding-bottom:10px; margin-top:42px; }
h2 { color:#1f4e79; margin-top:32px; }
h3 { color:#365f91; }
blockquote { border-left:4px solid #2f80ed; background:#f4f8fc; margin:18px 0;
             padding:10px 18px; color:#334e68; }
table { border-collapse:collapse; width:100%; margin:16px 0 24px; font-size:14px; }
th { background:#eaf2f8; color:#163a5f; }
th, td { border:1px solid #cbd5e1; padding:7px 9px; text-align:left; }
tr:nth-child(even) td { background:#fafcff; }
img { display:block; max-width:100%; margin:22px auto 30px; border:1px solid #d8e1ea;
      box-shadow:0 3px 14px rgba(30,55,80,.10); }
img.math-display { border:0; box-shadow:none; margin:18px auto; width:auto; }
img.math-inline { display:inline-block; border:0; box-shadow:none; margin:0 .12em;
                  height:1.2em; width:auto; vertical-align:-.18em; }
code { background:#eef2f6; border-radius:3px; padding:2px 5px; }
pre { background:#132238; color:#eef5ff; border-radius:6px; padding:15px; overflow:auto; }
pre code { background:transparent; padding:0; }
a { color:#1769aa; }
hr { border:0; border-top:1px solid #d8e1ea; margin:34px 0; }
"""


def inline_cell(text: str) -> str:
    value = html.escape(text.strip())
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    return value


def expand_tables(text: str) -> str:
    lines = text.splitlines()
    output = []
    i = 0
    separator = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
    while i < len(lines):
        if (
            i + 1 < len(lines)
            and "|" in lines[i]
            and separator.match(lines[i + 1])
        ):
            rows = []
            j = i
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                if j != i + 1:
                    rows.append(
                        [cell.strip() for cell in lines[j].strip().strip("|").split("|")]
                    )
                j += 1
            header, body = rows[0], rows[1:]
            output.append("<table><thead><tr>")
            output.extend(f"<th>{inline_cell(cell)}</th>" for cell in header)
            output.append("</tr></thead><tbody>")
            for row in body:
                output.append("<tr>")
                output.extend(f"<td>{inline_cell(cell)}</td>" for cell in row)
                output.append("</tr>")
            output.append("</tbody></table>")
            i = j
            continue
        output.append(lines[i])
        i += 1
    return "\n".join(output)


def extract_math(text: str):
    replacements = {}
    pattern = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.DOTALL)

    def replace(match):
        token = f"SKEWMATHPLACEHOLDER_{len(replacements)}_END"
        replacements[token] = (
            " ".join((match.group(1) or match.group(2)).split()),
            match.group(1) is not None,
        )
        return token

    return pattern.sub(replace, text), replacements


def render_math(expression: str, display: bool) -> str:
    output = io.BytesIO()
    math_to_image(f"${expression}$", output, format="svg", dpi=180)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    css_class = "math-display" if display else "math-inline"
    alt = html.escape(expression, quote=True)
    return (
        f'<img class="{css_class}" '
        f'src="data:image/svg+xml;base64,{encoded}" alt="{alt}">'
    )


def inline_report_images(body: str) -> str:
    pattern = re.compile(r'<img src="([^"]+)" alt="([^"]*)">')

    def replace(match):
        source = match.group(1)
        if source.startswith("data:"):
            return match.group(0)
        path = (REPORT_DIR / source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Report image not found: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        alt = html.escape(match.group(2), quote=True)
        return f'<img src="data:{mime};base64,{encoded}" alt="{alt}">'

    return pattern.sub(replace, body)


def main() -> None:
    markdown = mistune.Markdown(renderer=mistune.Renderer(escape=False))
    source, math_replacements = extract_math(SOURCE.read_text(encoding="utf-8"))
    body = markdown(expand_tables(source))
    body = inline_report_images(body)
    for token, (expression, display) in math_replacements.items():
        body = body.replace(token, render_math(expression, display))
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SKEW 买方单因子研究报告</title>
<style>{CSS}</style>
</head>
<body><main>{body}</main></body>
</html>"""
    TARGET.write_text(document, encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
