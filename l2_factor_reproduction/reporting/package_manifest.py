"""Integrity manifest for a completed standalone research-report package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


DEFAULT_REQUIRED_CHAPTERS: Sequence[str] = tuple(
    f"{number:02d}_" for number in range(1, 11)
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_numbered_chapters(root: Path, prefixes: Iterable[str]) -> None:
    markdown = [path.name for path in root.glob("*.md") if path.is_file()]
    missing = [
        prefix
        for prefix in prefixes
        if not any(name.startswith(prefix) for name in markdown)
    ]
    if missing:
        raise ValueError(f"missing numbered report chapters: {missing}")


def build_final_package_manifest(
    report_root: Path,
    *,
    output_path: Path = None,
    required_chapter_prefixes: Sequence[str] = DEFAULT_REQUIRED_CHAPTERS,
) -> Dict[str, Any]:
    """Hash every report file after Markdown, HTML, and PDF export."""
    root = Path(report_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    output = (
        Path(output_path).resolve()
        if output_path is not None
        else root / "artifacts/final_package_manifest.json"
    )
    _require_numbered_chapters(root, required_chapter_prefixes)

    html_files = sorted((root / "export").glob("*.html"))
    pdf_files = sorted((root / "export").glob("*.pdf"))
    figures = sorted((root / "figures").glob("*.png"))
    if not html_files or not pdf_files:
        raise ValueError("completed package requires HTML and PDF exports")
    if len(figures) < 10:
        raise ValueError("completed package requires at least ten PNG figures")
    for html_path in html_files:
        html = html_path.read_text(encoding="utf-8")
        if "data:image/png;base64," not in html or "Missing image:" in html:
            raise ValueError(f"HTML image embedding check failed: {html_path}")
    for pdf_path in pdf_files:
        payload = pdf_path.read_bytes()
        if not payload.startswith(b"%PDF") or b"/Type /Page" not in payload:
            raise ValueError(f"PDF page check failed: {pdf_path}")

    records: List[Dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() == output:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "version": "standalone_research_package_v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "report_root": str(root),
        "file_count": len(records),
        "figure_count": len(figures),
        "html_count": len(html_files),
        "pdf_count": len(pdf_files),
        "files": records,
        "hash_policy": (
            "SHA256 covers every regular file under report_root except this "
            "self-referential manifest."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


__all__ = ["build_final_package_manifest", "sha256_file"]

