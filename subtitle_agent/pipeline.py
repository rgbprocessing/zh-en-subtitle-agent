"""Loads an episode once; shared by the dry run and the agent graph."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .glossary import Glossary
from .schema import Cue, MasterCue
from .sources import Manifest
from .timeline import EpisodeIndex, Window, attach_overlays, auto_accept, build_windows


@dataclass
class Episode:
    manifest: Manifest
    cues: list[Cue]
    windows: list[Window]
    index: EpisodeIndex
    glossary: Glossary
    rules: str
    master: list[MasterCue]
    pending: list[int]
    report: dict

    @classmethod
    def load(cls, manifest_path: str | Path) -> "Episode":
        m = Manifest.load(manifest_path)
        cues = m.load_all()
        windows = build_windows(cues, m.priority, m.trusted_alone, m.agree_threshold)
        overlays, ov_report = attach_overlays(windows, cues)
        master, pending = auto_accept(windows, overlays, m.priority)
        rules = m.rules.read_text(encoding="utf-8") if m.rules and m.rules.exists() else ""
        report = {
            "episode": m.episode,
            "cues_per_source": dict(Counter(c.source for c in cues)),
            "windows": len(windows),
            "window_status": dict(Counter(w.status for w in windows)),
            "auto_accepted_lines": sum(1 for x in master if x.kind != "overlay"),
            "pending_for_agent": len(pending),
            **ov_report,
        }
        return cls(m, cues, windows, EpisodeIndex(cues, windows), Glossary.load(m.glossary),
                   rules, master, pending, report)
