"""Render a delta report in Markdown, HTML, and JSON."""

from __future__ import annotations

import json
from pathlib import Path

from src.canonical.model import CanonicalDocument, ChangeKind
from src.delta.engine import DeltaEntry


def _is_noop_modified(entry: DeltaEntry) -> bool:
    return (
        entry.kind == ChangeKind.MODIFIED
        and entry.old is not None
        and entry.new is not None
        and entry.old.raw_text.strip() == entry.new.raw_text.strip()
    )


def _filter_report_entries(entries: list[DeltaEntry]) -> list[DeltaEntry]:
    return [entry for entry in entries if not _is_noop_modified(entry)]


def format_delta_entry(change: DeltaEntry) -> str | None:
    """Format one delta entry for compact human-readable report text."""
    if _is_noop_modified(change):
        return None

    if change.kind == ChangeKind.MODIFIED and change.old is not None and change.new is not None:
        element = change.new or change.old
        label = element.human_readable_label()
        return (
            f"- [{change.kind.value.upper()}] {label}: "
            f"Changed from '{change.old.raw_text}' to '{change.new.raw_text}'"
        )

    if change.kind == ChangeKind.ADDED and change.new is not None:
        return f"- [ADDED] {change.new.human_readable_label()}: '{change.new.raw_text}'"

    if change.kind == ChangeKind.REMOVED and change.old is not None:
        return f"- [REMOVED] {change.old.human_readable_label()}: '{change.old.raw_text}'"

    return None


def _group_entries(entries: list[DeltaEntry]) -> dict[ChangeKind, list[DeltaEntry]]:
    """Group entries by change kind."""
    groups: dict[ChangeKind, list[DeltaEntry]] = {
        ChangeKind.ADDED: [],
        ChangeKind.REMOVED: [],
        ChangeKind.MODIFIED: [],
    }
    for e in entries:
        groups[e.kind].append(e)
    return groups


def _format_entry_md(entry: DeltaEntry) -> str:
    lines: list[str] = []
    kind = entry.kind
    marker = {"added": "**+**", "removed": "**-**", "modified": "**~**"}[kind.value]

    lines.append(f"### {marker} page {entry.page}")

    if entry.old:
        lines.append(f"**Old** `{entry.old.id}`: {entry.old.raw_text}")
    if entry.new:
        lines.append(f"**New** `{entry.new.id}`: {entry.new.raw_text}")

    if entry.description:
        lines.append(f"*{entry.description}*")

    if entry.similarity < 1.0:
        lines.append(f"Similarity: {entry.similarity:.0%} | Confidence: {entry.confidence:.0%}")

    if entry.reasons:
        lines.append(f"Reasons: {', '.join(entry.reasons)}")

    return "\n".join(lines)


def render_markdown(
    entries: list[DeltaEntry],
    old_doc: CanonicalDocument,
    new_doc: CanonicalDocument,
) -> str:
    """Render full delta report as Markdown, grouped by change type."""
    entries = _filter_report_entries(entries)
    parts: list[str] = []

    parts.append("# PID Delta Report\n")
    parts.append(f"**Old revision:** `{old_doc.source_path}` — {old_doc.page_count} pages, {len(old_doc.elements)} elements")
    parts.append(f"**New revision:** `{new_doc.source_path}` — {new_doc.page_count} pages, {len(new_doc.elements)} elements")
    parts.append("")

    # Summary
    groups = _group_entries(entries)
    parts.append("## Summary")
    parts.append(f"- **Added:** {len(groups[ChangeKind.ADDED])}")
    parts.append(f"- **Removed:** {len(groups[ChangeKind.REMOVED])}")
    parts.append(f"- **Modified:** {len(groups[ChangeKind.MODIFIED])}")
    parts.append(f"- **Total changes:** {len(entries)}")
    parts.append("")

    # Grouped sections
    for kind, label in [
        (ChangeKind.REMOVED, "Removed"),
        (ChangeKind.ADDED, "Added"),
        (ChangeKind.MODIFIED, "Modified"),
    ]:
        group = groups[kind]
        if not group:
            continue
        parts.append(f"## {label} ({len(group)})\n")
        for entry in group:
            parts.append(_format_entry_md(entry))
            parts.append("")

    return "\n".join(parts)


def render_html(
    entries: list[DeltaEntry],
    old_doc: CanonicalDocument,
    new_doc: CanonicalDocument,
) -> str:
    """Render delta report as HTML with color-coded sections."""
    entries = _filter_report_entries(entries)
    groups = _group_entries(entries)

    css = """
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        .summary { background: #f5f5f5; padding: 16px; border-radius: 8px; margin-bottom: 24px; }
        .section { margin-bottom: 24px; }
        .section h2 { border-bottom: 2px solid #ddd; padding-bottom: 8px; }
        .entry { border-left: 4px solid #ddd; padding: 8px 16px; margin-bottom: 12px; background: #fafafa; }
        .entry.added { border-color: #2d8f2d; }
        .entry.removed { border-color: #c00; }
        .entry.modified { border-color: #d90; }
        .old { color: #c00; }
        .new { color: #2d8f2d; }
        .desc { font-style: italic; color: #666; }
        .meta { font-size: 0.85em; color: #888; }
    </style>
    """

    parts = [
        "<!DOCTYPE html><html><head>",
        "<title>PID Delta Report</title>",
        css,
        "</head><body>",
        "<h1>PID Delta Report</h1>",
        f"<p><strong>Old:</strong> {old_doc.source_path} ({old_doc.page_count} pages, {len(old_doc.elements)} elements)</p>",
        f"<p><strong>New:</strong> {new_doc.source_path} ({new_doc.page_count} pages, {len(new_doc.elements)} elements)</p>",
        '<div class="summary">',
        f"<strong>Added:</strong> {len(groups[ChangeKind.ADDED])} | ",
        f"<strong>Removed:</strong> {len(groups[ChangeKind.REMOVED])} | ",
        f"<strong>Modified:</strong> {len(groups[ChangeKind.MODIFIED])} | ",
        f"<strong>Total:</strong> {len(entries)}",
        "</div>",
    ]

    for kind, label in [
        (ChangeKind.REMOVED, "Removed"),
        (ChangeKind.ADDED, "Added"),
        (ChangeKind.MODIFIED, "Modified"),
    ]:
        group = groups[kind]
        if not group:
            continue
        parts.append('<div class="section">')
        parts.append(f"<h2>{label} ({len(group)})</h2>")
        for entry in group:
            cls = kind.value
            parts.append(f'<div class="entry {cls}">')
            if entry.old:
                parts.append(f'<div class="old">Old: <code>{entry.old.id}</code> — {_esc(entry.old.raw_text)}</div>')
            if entry.new:
                parts.append(f'<div class="new">New: <code>{entry.new.id}</code> — {_esc(entry.new.raw_text)}</div>')
            if entry.description:
                parts.append(f'<div class="desc">{_esc(entry.description)}</div>')
            meta_parts = [f"page {entry.page}"]
            if entry.similarity < 1.0:
                meta_parts.append(f"similarity {entry.similarity:.0%}")
            meta_parts.append(f"confidence {entry.confidence:.0%}")
            parts.append(f'<div class="meta">{" | ".join(meta_parts)}</div>')
            parts.append("</div>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _esc(text: str) -> str:
    """Escape HTML entities."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_json(
    entries: list[DeltaEntry],
    old_doc: CanonicalDocument,
    new_doc: CanonicalDocument,
) -> str:
    """Render delta report as JSON."""
    entries = _filter_report_entries(entries)
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
                "confidence": e.confidence,
                "description": e.description,
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
    """Write Markdown, HTML, and JSON reports, return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "delta_report.md"
    html_path = output_dir / "delta_report.html"
    json_path = output_dir / "delta_report.json"

    md_path.write_text(render_markdown(entries, old_doc, new_doc))
    html_path.write_text(render_html(entries, old_doc, new_doc))
    json_path.write_text(render_json(entries, old_doc, new_doc))

    return {"markdown": md_path, "html": html_path, "json": json_path}
