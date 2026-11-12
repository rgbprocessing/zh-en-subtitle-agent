"""Deterministic QA and file writers."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import MasterCue
from .sources import fmt_ts


def qa_timing(master: list[MasterCue], max_cps: float = 17.0, min_dur: float = 0.8,
              max_extend: float = 1.5, gap: float = 0.05) -> list[int]:
    """Lengthen too-fast or too-short English cues where there is room before the
    next line (overlays may overlap freely). Returns ids still too fast."""
    lines = sorted((m for m in master if m.kind != "overlay"), key=lambda m: m.start)
    flagged = []
    for i, m in enumerate(lines):
        if not m.en:
            continue
        nxt = lines[i + 1].start - gap if i + 1 < len(lines) else m.end + max_extend
        need = max(len(m.en) / max_cps, min_dur)
        if m.end - m.start < need:
            m.end = round(max(m.end, min(m.start + need, nxt, m.end + max_extend)), 3)
        if len(m.en) / max(m.end - m.start, 1e-6) > max_cps:
            flagged.append(m.id)
    return flagged


def write_srt(master: list[MasterCue], path: Path, field: str) -> None:
    rows = [m for m in sorted(master, key=lambda m: m.start) if getattr(m, field)]
    with open(path, "w", encoding="utf-8") as f:
        for n, m in enumerate(rows, 1):
            f.write(f"{n}\n{fmt_ts(m.start)} --> {fmt_ts(m.end)}\n{getattr(m, field)}\n\n")


def write_outputs(master: list[MasterCue], out_dir: Path, episode: str,
                  report: dict, unresolved: list[int] | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / f"{episode}_master.json"
    p_zh = out_dir / f"{episode}_zh.srt"
    p_en = out_dir / f"{episode}_en.srt"
    p_notes = out_dir / f"{episode}_notes.md"
    p_report = out_dir / f"{episode}_report.json"

    p_json.write_text(json.dumps([asdict(m) for m in master], ensure_ascii=False, indent=1), encoding="utf-8")
    write_srt(master, p_zh, "zh")
    written = [p_json, p_zh]
    if any(m.en for m in master):
        write_srt(master, p_en, "en")
        written.append(p_en)

    unresolved = set(unresolved or [])
    lines = [f"# {episode} notes\n"]
    for m in master:
        if m.note or m.id in unresolved:
            flag = " ⚠ still too fast to read" if m.id in unresolved else ""
            lines.append(f"- **{fmt_ts(m.start)}** ({m.decided_by}) {m.zh} → {m.en or '—'}"
                         f"{': ' + m.note if m.note else ''}{flag}")
    p_notes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    p_report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return written + [p_notes, p_report]
