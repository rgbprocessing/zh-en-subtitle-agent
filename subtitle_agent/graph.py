"""The LangGraph agent, in two architectures for comparison.

single (baseline):
  prepare ─► reconcile (one window at a time) ─► merge ─► translate ─► qa ─► write
                                                              ▲          │ too fast
                                                              └──────────┘ (1 retry)
multi:
  prepare ─┬─► reconciler ─┐                      ┌───── revise (1 round) ─────┐
           ├─► reconciler ─┼─► merge ─► translator ─► reviewer ─► qa ─► write
           └─► reconciler ─┘   (parallel via Send; each one a tool-using agent)

Both use the same tools, prompts, schemas and QA, so the comparison isolates
the architecture: parallel reconcilers + a translator/reviewer team vs. one pass.
"""
from __future__ import annotations

import json
import operator
import re
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

try:
    from langgraph.types import Send
except ImportError:  # older langgraph
    from langgraph.constants import Send

from . import prompts
from .output import qa_timing, write_outputs
from .pipeline import Episode
from .schema import MasterCue
from .timeline import renumber
from .tools import make_tools


# ---------- structured outputs (schema-constrained, so small models can't break the format) ----------

class CueOut(BaseModel):
    start: float
    end: float
    zh: str = Field(description="final Chinese text")
    kind: Literal["subtitle", "speech"] = "subtitle"


class Decision(BaseModel):
    keep: bool = Field(description="false to drop the window entirely")
    cues: list[CueOut] = Field(default_factory=list)
    note: str = ""


class Translation(BaseModel):
    id: int
    en: str
    note: str = ""


class TranslationBatch(BaseModel):
    items: list[Translation]


class Review(BaseModel):
    id: int
    ok: bool
    issue: str = ""
    suggestion: str = ""


class ReviewBatch(BaseModel):
    items: list[Review]


# ---------- state ----------

class State(TypedDict, total=False):
    master: list[dict]
    pending: list[int]
    decisions: Annotated[list[dict], operator.add]   # parallel reconcilers append here
    feedback: dict                                   # id -> reviewer feedback
    review_rounds: int
    qa_flags: list[int]
    retries: int
    report: dict
    written: list[str]


# ---------- helpers ----------

def _text(msg) -> str:
    c = msg.content
    if isinstance(c, list):
        c = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text)
    for open_, close in (("{", "}"), ("[", "]")):
        a, b = text.find(open_), text.rfind(close)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
    return None


def ask_structured(llm, schema: type[BaseModel], messages: list):
    """Schema-constrained call; falls back to free text + JSON parse.
    Returns (parsed or None, error string)."""
    errors = []
    try:
        out = llm.with_structured_output(schema).invoke(messages)
        if isinstance(out, schema):
            return out, ""
        if isinstance(out, dict):
            return schema.model_validate(out), ""
        errors.append(f"structured: got {type(out).__name__}")
    except Exception as e:  # provider without structured output, or validation error
        errors.append(f"structured: {e}")
    raw = ""
    try:
        raw = _text(llm.invoke(messages + [HumanMessage(
            "Reply with ONLY JSON matching this schema:\n" + json.dumps(schema.model_json_schema()))]))
        data = extract_json(raw)
        if isinstance(data, list) and "items" in schema.model_fields:
            data = {"items": data}
        return schema.model_validate(data), ""
    except Exception as e:
        errors.append(f"text fallback: {e}; raw={raw[:300]!r}")
    return None, " | ".join(errors)


def investigate(llm, tools, messages, max_steps: int = 3) -> list:
    """Let the agent call tools until it stops (or max_steps). Returns the transcript."""
    by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools)
    for _ in range(max_steps):
        ai = bound.invoke(messages)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            break
        for tc in ai.tool_calls:
            tool = by_name.get(tc["name"])
            out = tool.invoke(tc["args"]) if tool else f"Unknown tool {tc['name']}"
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return messages


def _rows(master) -> list[dict]:
    return [asdict(m) for m in master]


def _to_master(rows: list[dict]) -> list[MasterCue]:
    return [MasterCue(**r) for r in rows]


# ---------- graph ----------

def build_graph(ep: Episode, llm, out_dir: Path, mode: str = "single",
                batch_size: int = 25, max_cps: float = 17.0):
    if mode not in ("single", "multi"):
        raise ValueError("mode must be 'single' or 'multi'")
    tools = make_tools(ep)
    rules = ep.rules or "(none)"

    # --- reconcile one window (the transcription agent) ---
    def reconcile_window(wid: int) -> dict:
        w = ep.index.windows[wid]
        before = [m for m in ep.master if m.end <= w.start and m.kind != "overlay"][-3:]
        nxt = ep.index.windows.get(wid + 1)
        msgs = [SystemMessage(prompts.RECONCILE_SYSTEM.format(rules=rules)),
                HumanMessage(prompts.RECONCILE_USER.format(
                    before="\n".join(f"[{m.start:.2f}] {m.zh}" for m in before) or "(start)",
                    window=w.describe(),
                    after=nxt.describe() if nxt else "(end)"))]
        msgs = investigate(llm, tools, msgs)
        decision, err = ask_structured(llm, Decision, msgs + [HumanMessage(prompts.DECIDE)])
        return {"wid": wid, "decision": decision.model_dump() if decision else None, "error": err}

    def prepare(state: State) -> State:
        return {"master": _rows(ep.master), "pending": list(ep.pending), "decisions": [],
                "feedback": {}, "review_rounds": 0, "retries": 0,
                "report": {**ep.report, "mode": mode}}

    def reconcile_all(state: State) -> State:          # single: sequential
        return {"decisions": [reconcile_window(w) for w in state["pending"]]}

    def reconcile_one(task: dict) -> State:             # multi: one per Send
        return {"decisions": [reconcile_window(task["wid"])]}

    def fan_out(state: State):
        if not state["pending"]:
            return "merge"
        return [Send("reconciler", {"wid": w}) for w in state["pending"]]

    def merge(state: State) -> State:
        master = _to_master(state["master"])
        counts = {"kept": 0, "dropped": 0, "fallback": 0}
        for d in sorted(state.get("decisions", []), key=lambda d: d["wid"]):
            w = ep.index.windows[d["wid"]]
            dec = d["decision"]
            if dec is None:
                ref = next(iter(w.by_source))
                for c in w.by_source[ref]:
                    master.append(MasterCue(0, c.start, c.end, c.text, c.kind, list(w.by_source), w.id,
                                            "fallback", note=f"agent failed, used {ref}: {d['error'][:200]}"))
                counts["fallback"] += 1
            elif not dec["keep"] or not dec["cues"]:
                counts["dropped"] += 1
            else:
                for c in dec["cues"]:
                    master.append(MasterCue(0, c["start"], c["end"], c["zh"], c["kind"], list(w.by_source),
                                            w.id, "agent", note=dec.get("note", "")))
                counts["kept"] += 1
        renumber(master)
        return {"master": _rows(master), "report": {**state["report"], "agent_decisions": counts}}

    # --- translator agent ---
    def translate(state: State) -> State:
        master = _to_master(state["master"])
        flags = set(state.get("qa_flags") or [])
        feedback = state.get("feedback") or {}
        fb_ids = {int(k) for k in feedback}
        todo = [m for m in master if not m.en or m.id in flags or m.id in fb_ids]
        done_ctx = [m for m in master if m.en and m.id not in flags and m.id not in fb_ids]
        by_id = {m.id: m for m in master}
        failed = 0
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            extra = ""
            fl = [m for m in batch if m.id in flags]
            if fl:
                extra += prompts.SHORTEN_EXTRA.format(flagged="\n".join(
                    f"{m.id}: {m.en} ({m.end - m.start:.1f}s)" for m in fl))
            fb = [m for m in batch if m.id in fb_ids]
            if fb:
                extra += prompts.REVISE_EXTRA.format(flagged="\n".join(
                    f"{m.id}: {m.en}  — {feedback[str(m.id)]}" for m in fb))
            ctx = [m for m in done_ctx if m.start < batch[0].start][-5:]
            msgs = [SystemMessage(prompts.TRANSLATE_SYSTEM.format(rules=rules)),
                    HumanMessage(prompts.TRANSLATE_USER.format(
                        glossary=ep.glossary.fmt(ep.glossary.hits_in(" ".join(m.zh for m in batch))),
                        context="\n".join(f"{m.zh} → {m.en}" for m in ctx) or "(none)",
                        lines="\n".join(f"{m.id} [{m.kind}] {m.zh}" for m in batch),
                        extra=extra))]
            result, _ = ask_structured(llm, TranslationBatch, msgs)
            if result is None:
                failed += len(batch)
                continue
            for t in result.items:
                m = by_id.get(t.id)
                if m and t.en.strip():
                    m.en = t.en.strip()
                    if t.note:
                        m.note = (m.note + " | " if m.note else "") + t.note
            done_ctx.extend(m for m in batch if m.en)
        report = {**state["report"]}
        report["translate_failed_lines"] = report.get("translate_failed_lines", 0) + failed
        return {"master": _rows(master), "feedback": {}, "qa_flags": [], "report": report}

    # --- reviewer agent (multi only) ---
    def review(state: State) -> State:
        master = _to_master(state["master"])
        lines = [m for m in master if m.en]
        feedback = {}
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            msgs = [SystemMessage(prompts.REVIEW_SYSTEM.format(rules=rules)),
                    HumanMessage(prompts.REVIEW_USER.format(
                        glossary=ep.glossary.fmt(ep.glossary.hits_in(" ".join(m.zh for m in batch))),
                        lines="\n".join(f"{m.id} [{m.kind}] {m.zh} → {m.en}" for m in batch)))]
            result, _ = ask_structured(llm, ReviewBatch, msgs)
            for r in (result.items if result else []):
                if not r.ok and r.id in {m.id for m in batch}:
                    feedback[str(r.id)] = f"{r.issue} Suggestion: {r.suggestion}".strip()
        report = {**state["report"], "reviewer_flagged": len(feedback)}
        return {"feedback": feedback, "review_rounds": state.get("review_rounds", 0) + 1, "report": report}

    def after_translate(state: State) -> str:
        if mode == "multi" and state.get("review_rounds", 0) == 0:
            return "reviewer"
        return "qa"

    def after_review(state: State) -> str:
        return "translator" if state.get("feedback") else "qa"

    # --- deterministic QA + write ---
    def qa(state: State) -> State:
        master = _to_master(state["master"])
        flags = qa_timing(master, max_cps=max_cps)
        return {"master": _rows(master), "qa_flags": flags,
                "retries": state.get("retries", 0) + (1 if flags else 0)}

    def after_qa(state: State) -> str:
        return "translator" if state.get("qa_flags") and state["retries"] <= 1 else "write"

    def write(state: State) -> State:
        master = _to_master(state["master"])
        report = {**state["report"], "untranslated": sum(1 for m in master if not m.en),
                  "still_too_fast": len(state.get("qa_flags") or [])}
        paths = write_outputs(master, out_dir, ep.manifest.episode, report, state.get("qa_flags"))
        return {"report": report, "written": [str(p) for p in paths]}

    g = StateGraph(State)
    g.add_node("prepare", prepare)
    g.add_node("merge", merge)
    g.add_node("translator", translate)
    g.add_node("qa", qa)
    g.add_node("write", write)
    g.add_edge(START, "prepare")
    if mode == "single":
        g.add_node("reconcile", reconcile_all)
        g.add_edge("prepare", "reconcile")
        g.add_edge("reconcile", "merge")
    else:
        g.add_node("reconciler", reconcile_one)
        g.add_node("reviewer", review)
        g.add_conditional_edges("prepare", fan_out, ["reconciler", "merge"])
        g.add_edge("reconciler", "merge")
        g.add_conditional_edges("reviewer", after_review, {"translator": "translator", "qa": "qa"})
    g.add_edge("merge", "translator")
    g.add_conditional_edges("translator", after_translate,
                            {"reviewer": "reviewer", "qa": "qa"} if mode == "multi" else {"qa": "qa"})
    g.add_conditional_edges("qa", after_qa, {"translator": "translator", "write": "write"})
    g.add_edge("write", END)
    return g.compile()
