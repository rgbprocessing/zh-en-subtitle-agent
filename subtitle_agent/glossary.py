"""Glossary retrieval (the RAG piece).

glossary.json is a list of {"zh": ..., "en": ..., "note": ...} entries:
cast names, show terms, recurring phrases. Lookup is exact-substring first,
then fuzzy. Swap in embeddings later if the glossary gets large; the tool
interface stays the same.
"""
from __future__ import annotations

import json
from pathlib import Path

from .timeline import normalize, similarity


class Glossary:
    def __init__(self, entries: list[dict]):
        self.entries = entries

    @classmethod
    def load(cls, path: Path | None) -> "Glossary":
        if not path or not Path(path).exists():
            return cls([])
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def hits_in(self, text: str) -> list[dict]:
        """Entries whose Chinese term appears in `text` (used to prime translation)."""
        t = normalize(text)
        return [e for e in self.entries if normalize(e["zh"]) and normalize(e["zh"]) in t]

    def search(self, term: str, k: int = 5) -> list[dict]:
        exact = [e for e in self.entries
                 if normalize(term) in normalize(e["zh"]) or term.lower() in e.get("en", "").lower()]
        if exact:
            return exact[:k]
        scored = sorted(self.entries, key=lambda e: similarity(term, e["zh"]), reverse=True)
        return [e for e in scored[:k] if similarity(term, e["zh"]) >= 0.4]

    @staticmethod
    def fmt(entries: list[dict]) -> str:
        if not entries:
            return "(no glossary entries)"
        return "\n".join(f"{e['zh']} → {e['en']}" + (f"  ({e['note']})" if e.get("note") else "")
                         for e in entries)
