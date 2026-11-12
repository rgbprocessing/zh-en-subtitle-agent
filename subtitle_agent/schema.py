"""Common data shapes. Every source file, whatever its format, becomes a list of Cues."""
from __future__ import annotations

from dataclasses import dataclass, field

# kind:
#   "subtitle" – burned-in subtitle text (cropped OCR)
#   "speech"   – what was said (ASR: FunASR, Whisper)
#   "overlay"  – other on-screen text (full-frame OCR: signs, captions, graphics)
LINE_KINDS = {"subtitle", "speech"}


@dataclass
class Cue:
    start: float
    end: float
    text: str
    source: str
    kind: str = "speech"
    conf: float | None = None

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


@dataclass
class MasterCue:
    """One line of the final Chinese master (and later its English translation)."""
    id: int
    start: float
    end: float
    zh: str
    kind: str
    origin: list[str] = field(default_factory=list)   # sources that supported it
    window_id: int | None = None
    decided_by: str = "auto"                            # auto | agent | fallback
    en: str = ""
    note: str = ""                                      # reconcile/translator note
