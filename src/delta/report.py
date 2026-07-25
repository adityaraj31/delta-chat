"""Render a delta report in Markdown and JSON."""

from __future__ import annotations

import json
from pathlib import Path

from src.canonical.model import CanonicalDocument, ChangeKind
from src.delta.engine import DeltaEntry


def _format_entry_md(entry: DeltaEntry, old_doc: CanonicalDocument, new_doc: CanonicalDocument) -> str:
    lines: list[str] = []
    kind = entry.kind
    marker = {"added": "**+**", "removed": "**-**", "modified": "**~**"}[kind.value]

    lines.append(f"### {marker} {kind.value.upper()} — page {entry.page}")

    if entry.old:
        lines.append(f"**Old** `{entry.old.id}`: {entry.old.raw_text}")
    if entry.new:
        lines.append(f"**New** `{entry.new.id}`: {entry.new.raw_text}")

    if entry.similarity < 1.0:
        lines.append(f"Similarity: {entry.similarity:.0%}")

    if entry.reasons:
        lines.append(f"Reasons: {', '.join(entry.reasons)}")

    return "\n".join(lines)


def render_markdown(
    entries: list[DeltaEntry],
    old_doc: CanonicalDocument,
    new_doc: CanonicalDocument,
) -> str:
    """Render full delta report as Markdown."""
    parts: list[str] = []

    parts.append("# PID Delta Report\n")
    parts.append(f"**Old revision:** `{old_doc.source_path}` — {old_doc.page_count} pages, {len(old_doc.elements)} elements")
    parts.append(f"**New revision:** `{new_doc.source_path}` — {new_doc.page_count} pages, {len(new_doc.elements)} elements")
    parts.append("")

    # Summary
    added = sum(1 for e in entries if e.kind == ChangeKind.ADDED)
    removed = sum(1 for e in entries if e.kind == ChangeKind.REMOVED)
    modified = sum(1 for e in entries if e.kind == ChangeKind.MODIFIED)
    parts.append("## Summary")
    parts.append(f"- **Added:** {added}")
    parts.append(f"- **Removed:** {removed}")
    parts.append(f"- **Modified:** {modified}")
    parts.append(f"- **Total changes:** {len(entries)}")
    parts.append("")

    # Detailed entries
    parts.append("## Detailed Changes\n")
    for entry in entries:
        parts.append(_format_entry_md(entry, old_doc, new_doc))
        parts.append("")

    return "\n".join(parts)


def render_json(
    entries: list[DeltaEntry],
    old_doc: CanonicalDocument,
    new_doc: CanonicalDocument,
) -> str:
    """Render delta report as JSON."""
    data = {
        "old_source": old_doc.source_path,
        "new_source": new_doc.source_path,
        "summary": {
            "added": sum(1 for e in entries if e.kind == ChangeKind.ADDED),
            "removed": sum(1 for e in entries if e.kind == ChangeKind.REMOVED),
            "modified": sum(1 for e in entries if e.kind == ChangeKind.MODIFIED),
            "total": len(entries),
        },
        "entries": [
            {
                "kind": e.kind.value,
                "page": e.page,
                "old_id": e.element_id_old,
                "new_id": e.element_id_new,
                "old_text": e.old.raw_text if e.old else None,
                "new_text": e.new.raw_text if e.new else None,
                "similarity": e.similarity,
                "reasons": e.reasons,
            }
            for e in entries
        ],
    }
    return json.dumps(data, indent=2)


def write_report(
    entries: list[DeltaEntry],
    old_doc: CanonicalDocument,
    new_doc: CanonicalDocument,
    output_dir: Path,
) -> dict[str, Path]:
    """Write both Markdown and JSON reports, return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "delta_report.md"
    json_path = output_dir / "delta_report.json"

    md_path.write_text(render_markdown(entries, old_doc, new_doc))
    json_path.write_text(render_json(entries, old_doc, new_doc))

    return {"markdown": md_path, "json": json_path}
