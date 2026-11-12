"""Counts LLM calls and tokens across a graph run (parallel-safe)."""
from __future__ import annotations

import threading

from langchain_core.callbacks import BaseCallbackHandler


class UsageCounter(BaseCallbackHandler):
    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs):
        with self._lock:
            self.calls += 1
            for gens in response.generations:
                for g in gens:
                    um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                    self.input_tokens += um.get("input_tokens", 0)
                    self.output_tokens += um.get("output_tokens", 0)

    def summary(self) -> dict:
        return {"llm_calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}
