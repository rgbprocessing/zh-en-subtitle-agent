"""Loaders: turn precomputed OCR/ASR output files into Cues.

The models never run inside the agent. You run them however you do now
(Colab, your pipeline script), and point a manifest.toml at the files.
"""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .schema import Cue

_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def parse_ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    h, mi, se, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000


def fmt_ts(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def load_srt(path: Path, source: str, kind: str) -> list[Cue]:
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if "-->" in ln:
                a, b = ln.split("-->")
                body = " ".join(lines[i + 1:]).strip()
                if body:
                    cues.append(Cue(parse_ts(a), parse_ts(b), body, source, kind))
                break
    return cues


def _get(d, dotted: str):
    for k in dotted.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def load_json(path: Path, source: str, kind: str, fields: dict | None = None,
              list_key: str | None = None, time_scale: float = 1.0) -> list[Cue]:
    """Generic JSON loader. `fields` maps start/end/text/conf to keys in each item
    (dotted paths allowed, e.g. "box.score"). `time_scale` converts units
    (0.001 if times are in milliseconds)."""
    f = {"start": "start", "end": "end", "text": "text", "conf": None, **(fields or {})}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = _get(data, list_key) if list_key else data
    cues = []
    for it in items:
        text = str(_get(it, f["text"])).strip()
        if not text:
            continue
        conf = float(_get(it, f["conf"])) if f.get("conf") else None
        cues.append(Cue(float(_get(it, f["start"])) * time_scale,
                        float(_get(it, f["end"])) * time_scale,
                        text, source, kind, conf))
    return cues


@dataclass
class SourceSpec:
    name: str
    kind: str
    path: Path
    format: str = "srt"
    options: dict = field(default_factory=dict)

    def load(self) -> list[Cue]:
        if self.format == "srt":
            return load_srt(self.path, self.name, self.kind)
        if self.format == "json":
            return load_json(self.path, self.name, self.kind, **self.options)
        raise ValueError(f"unknown format {self.format!r} for source {self.name}")


@dataclass
class Manifest:
    episode: str
    sources: list[SourceSpec]
    priority: list[str]                 # most-trusted first
    trusted_alone: list[str]            # sources accepted even with no second opinion
    agree_threshold: float = 0.8
    glossary: Path | None = None
    rules: Path | None = None
    settings: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        base = path.parent
        specs = []
        for s in raw["sources"]:
            opts = {k: v for k, v in s.items() if k not in {"name", "kind", "path", "format"}}
            specs.append(SourceSpec(s["name"], s["kind"], base / s["path"], s.get("format", "srt"), opts))
        names = [s.name for s in specs]
        priority = raw.get("priority", names)
        def opt_path(key):
            return base / raw[key] if raw.get(key) else None
        return cls(
            episode=raw["episode"],
            sources=specs,
            priority=priority + [n for n in names if n not in priority],
            trusted_alone=raw.get("trusted_alone", priority[:1]),
            agree_threshold=raw.get("agree_threshold", 0.8),
            glossary=opt_path("glossary"),
            rules=opt_path("rules"),
            settings=raw.get("settings", {}),
        )

    def load_all(self) -> list[Cue]:
        cues = []
        for s in self.sources:
            cues.extend(s.load())
        return cues
