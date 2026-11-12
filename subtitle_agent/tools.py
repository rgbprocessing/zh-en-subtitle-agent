"""Tools the reconcile agent can call. They read the precomputed index; none run models."""
from __future__ import annotations

from langchain_core.tools import tool

from .pipeline import Episode
from .glossary import Glossary


def make_tools(ep: Episode):
    @tool
    def get_window(start: float, end: float) -> str:
        """Show every source's cues (subtitle OCR, ASR, overlays) between start and end
        seconds. Use it to check neighbouring lines or see whether a line continues."""
        return ep.index.format_range(start, end)

    @tool
    def lookup_glossary(term: str) -> str:
        """Look up a Chinese name or term (or an English rendering) in the show glossary."""
        return Glossary.fmt(ep.glossary.search(term))

    @tool
    def get_rules() -> str:
        """Return the standing editing/translation rules for this show."""
        return ep.rules or "(no extra rules)"

    @tool
    def compare_sources(window_id: int) -> str:
        """Character-level similarity between each source's text in a window and the
        most trusted source, plus each source's full text."""
        w = ep.index.windows.get(window_id)
        return w.describe() if w else f"No window {window_id}."

    return [get_window, lookup_glossary, get_rules, compare_sources]
