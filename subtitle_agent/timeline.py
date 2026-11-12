"""Deterministic alignment: no LLM here.

Groups cues from all line sources (subtitle + speech) into time windows,
decides which windows the sources agree on (auto-accepted) and which need
the agent. Full-frame overlays are de-duplicated against subtitle text and
recurring scrolls are dropped, per the standing rules.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .schema import LINE_KINDS, Cue, MasterCue
from .sources import fmt_ts


def normalize(text: str) -> str:
    """Letters and digits only (keeps CJK), NFKC so full/half-width match."""
    t = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in "LN").lower()


def similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass
class Window:
    id: int
    start: float
    end: float
    by_source: dict[str, list[Cue]]
    overlays: list[Cue] = field(default_factory=list)
    status: str = ""                 # agree | conflict | single | single_trusted
    sims: dict[str, float] = field(default_factory=dict)  # vs. reference source

    def text_of(self, source: str) -> str:
        return " ".join(c.text for c in self.by_source.get(source, []))

    def describe(self) -> str:
        out = [f"Window {self.id}  {fmt_ts(self.start)} → {fmt_ts(self.end)}  status={self.status}"]
        for src, cues in self.by_source.items():
            for c in cues:
                conf = f" conf={c.conf:.2f}" if c.conf is not None else ""
                out.append(f"  [{c.start:.2f}–{c.end:.2f}] {src} ({c.kind}{conf}): {c.text}")
        for c in self.overlays:
            out.append(f"  [{c.start:.2f}–{c.end:.2f}] {c.source} (overlay): {c.text}")
        if self.sims:
            out.append("  similarity to reference: " +
                       ", ".join(f"{k}={v:.2f}" for k, v in self.sims.items()))
        return "\n".join(out)


def _group(lines: list[Cue], min_overlap: float) -> list[list[Cue]]:
    groups, cur, cur_end = [], [], -1.0
    for c in sorted(lines, key=lambda c: (c.start, c.end)):
        if cur and c.start < cur_end - min_overlap:
            cur.append(c)
            cur_end = max(cur_end, c.end)
        else:
            if cur:
                groups.append(cur)
            cur, cur_end = [c], c.end
    if cur:
        groups.append(cur)
    return groups


def build_windows(cues: list[Cue], priority: list[str], trusted_alone: list[str],
                  agree_threshold: float = 0.8, low_threshold: float = 0.5,
                  min_overlap: float = 0.05) -> list[Window]:
    lines = [c for c in cues if c.kind in LINE_KINDS]
    windows = []
    for i, grp in enumerate(_group(lines, min_overlap)):
        by_source: dict[str, list[Cue]] = {}
        for c in grp:
            by_source.setdefault(c.source, []).append(c)
        order = [s for s in priority if s in by_source]
        by_source = {s: by_source[s] for s in order}
        w = Window(i, min(c.start for c in grp), max(c.end for c in grp), by_source)
        if len(order) == 1:
            w.status = "single_trusted" if order[0] in trusted_alone else "single"
        else:
            ref = order[0]
            w.sims = {s: similarity(w.text_of(ref), w.text_of(s)) for s in order[1:]}
            ok = max(w.sims.values()) >= agree_threshold and min(w.sims.values()) >= low_threshold
            # Outvoted: two other sources match each other exactly but not the reference
            # (e.g. OCR misread 备 as 各 while both ASR models heard it right).
            others = [normalize(w.text_of(s)) for s in order[1:]]
            if ok and len(others) >= 2 and len(set(others)) == 1 and others[0] != normalize(w.text_of(ref)):
                ok = False
            w.status = "agree" if ok else "conflict"
        windows.append(w)
    return windows


def attach_overlays(windows: list[Window], cues: list[Cue], dup_threshold: float = 0.7,
                    recurring_min: int = 5) -> tuple[list[Cue], dict]:
    """Returns overlays to keep plus a report of what was dropped and why."""
    overlays = sorted((c for c in cues if c.kind not in LINE_KINDS), key=lambda c: c.start)
    counts = Counter(normalize(c.text) for c in overlays)
    kept, dup, recurring = [], 0, 0
    for c in overlays:
        if counts[normalize(c.text)] >= recurring_min:
            recurring += 1           # constant sponsor scroll / channel bug
            continue
        near = [w for w in windows if w.start < c.end and c.start < w.end]
        if any(similarity(c.text, w.text_of(s)) >= dup_threshold or normalize(c.text) in normalize(w.text_of(s))
               for w in near for s in w.by_source):
            dup += 1                 # full-frame OCR re-reading the subtitle line
            continue
        for w in near:
            w.overlays.append(c)
        kept.append(c)
    return kept, {"overlays_kept": len(kept), "overlays_duplicate_of_subtitle": dup,
                  "overlays_recurring_dropped": recurring}


def auto_accept(windows: list[Window], overlays: list[Cue], priority: list[str]
                ) -> tuple[list[MasterCue], list[int]]:
    """Accept agreed windows using the top-priority source's text and timing.
    Returns (master cues so far, ids of windows the agent must decide)."""
    master, pending = [], []
    for w in windows:
        if w.status in ("agree", "single_trusted"):
            ref = next(s for s in priority if s in w.by_source)
            for c in w.by_source[ref]:
                master.append(MasterCue(0, c.start, c.end, c.text, c.kind,
                                        origin=list(w.by_source), window_id=w.id))
        else:
            pending.append(w.id)
    for c in overlays:
        master.append(MasterCue(0, c.start, c.end, c.text, "overlay", origin=[c.source]))
    renumber(master)
    return master, pending


def renumber(master: list[MasterCue]) -> None:
    master.sort(key=lambda m: (m.start, m.kind == "overlay"))
    for i, m in enumerate(master):
        m.id = i


class EpisodeIndex:
    """What the agent's tools query: any time range, across every source."""

    def __init__(self, cues: list[Cue], windows: list[Window]):
        self.cues = sorted(cues, key=lambda c: c.start)
        self.windows = {w.id: w for w in windows}

    def format_range(self, start: float, end: float, limit: int = 60) -> str:
        hits = [c for c in self.cues if c.start < end and start < c.end][:limit]
        if not hits:
            return f"No cues between {start:.2f}s and {end:.2f}s."
        return "\n".join(f"[{c.start:.2f}–{c.end:.2f}] {c.source} ({c.kind}): {c.text}" for c in hits)
