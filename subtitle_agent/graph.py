"""The LangGraph agent.

prepare ─► reconcile ─► translate ─► qa ─┬─► write
                            ▲             │
                            └─(too fast)──┘   (one retry: shorten flagged lines)

`prepare` is deterministic (alignment already done in Episode.load).
`reconcile` is a tool-using agent, run once per conflicting window.
`translate` batches lines with glossary hits and prior context.
`qa` is deterministic timing repair; lines still unreadable go back once to be shortened.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from . import prompts
from .output import qa_timing, write_outputs
from .pipeline import Episode
from .schema import MasterCue
from .timeline import renumber
from .tools import make_tools


class State(TypedDict, total=False):
    master: list[dict]
    pending: list[int]
    qa_flags: list[int]
    retries: int
    report: dict
    written: list[str]


# ---------- helpers ----------

def _text(msg) -> str:
    c = msg.content
    if isinstance(c, list):  # some providers return content blocks
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


def run_tool_loop(llm, tools, messages, max_steps: int = 4) -> str:
    by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools)
    for _ in range(max_steps):
        ai = bound.invoke(messages)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            return _text(ai)
        for tc in ai.tool_calls:
            tool = by_name.get(tc["name"])
            out = tool.invoke(tc["args"]) if tool else f"Unknown tool {tc['name']}"
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    messages.append(HumanMessage("Stop using tools. Give your final JSON answer now."))
    return _text(llm.invoke(messages))


def _to_master(rows: list[dict]) -> list[MasterCue]:
    return [MasterCue(**r) for r in rows]


# ---------- graph ----------

def build_graph(ep: Episode, llm, out_dir: Path, batch_size: int = 25, max_cps: float = 17.0):
    tools = make_tools(ep)
    rules = ep.rules or "(none)"

    def prepare(state: State) -> State:
        return {"master": [asdict(m) for m in ep.master], "pending": list(ep.pending),
                "retries": 0, "report": dict(ep.report)}

    def reconcile(state: State) -> State:
        master = _to_master(state["master"])
        decided = {"kept": 0, "dropped": 0, "fallback": 0}
        for wid in state["pending"]:
            w = ep.index.windows[wid]
            before = [m for m in master if m.end <= w.start and m.kind != "overlay"][-3:]
            nxt = ep.index.windows.get(wid + 1)
            msgs = [SystemMessage(prompts.RECONCILE_SYSTEM.format(rules=rules)),
                    HumanMessage(prompts.RECONCILE_USER.format(
                        before="\n".join(f"[{m.start:.2f}] {m.zh}" for m in before) or "(start)",
                        window=w.describe(),
                        after=nxt.describe() if nxt else "(end)"))]
            result = extract_json(run_tool_loop(llm, tools, msgs)) or {}
            if not isinstance(result, dict) or "keep" not in result:
                # fallback: most trusted source, flagged for review
                ref = next(iter(w.by_source))
                for c in w.by_source[ref]:
                    master.append(MasterCue(0, c.start, c.end, c.text, c.kind, list(w.by_source),
                                            wid, "fallback", note="agent answer unreadable; check"))
                decided["fallback"] += 1
                continue
            if not result["keep"]:
                decided["dropped"] += 1
                continue
            for c in result.get("cues", []):
                master.append(MasterCue(0, float(c.get("start", w.start)), float(c.get("end", w.end)),
                                        c["zh"], c.get("kind", "subtitle"), list(w.by_source), wid,
                                        "agent", note=result.get("note", "")))
            decided["kept"] += 1
        renumber(master)
        report = {**state["report"], "agent_decisions": decided}
        return {"master": [asdict(m) for m in master], "pending": [], "report": report}

    def translate(state: State) -> State:
        master = _to_master(state["master"])
        flags = set(state.get("qa_flags") or [])
        todo = [m for m in master if (m.id in flags) or not m.en]
        done_ctx = [m for m in master if m.en and m.id not in flags]
        by_id = {m.id: m for m in master}
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            gl = ep.glossary.hits_in(" ".join(m.zh for m in batch))
            ctx = [m for m in done_ctx if m.start < batch[0].start][-5:]
            extra = ""
            flagged = [m for m in batch if m.id in flags]
            if flagged:
                extra = prompts.SHORTEN_EXTRA.format(flagged="\n".join(
                    f"{m.id}: {m.en} ({m.end - m.start:.1f}s)" for m in flagged))
            msgs = [SystemMessage(prompts.TRANSLATE_SYSTEM.format(rules=rules)),
                    HumanMessage(prompts.TRANSLATE_USER.format(
                        glossary=ep.glossary.fmt(gl),
                        context="\n".join(f"{m.zh} → {m.en}" for m in ctx) or "(none)",
                        lines="\n".join(f"{m.id} [{m.kind}] {m.zh}" for m in batch),
                        extra=extra))]
            rows = extract_json(_text(llm.invoke(msgs))) or []
            for r in rows if isinstance(rows, list) else []:
                m = by_id.get(int(r.get("id", -1)))
                if m and r.get("en"):
                    m.en = r["en"].strip()
                    if r.get("note"):
                        m.note = (m.note + " | " if m.note else "") + r["note"]
            done_ctx.extend(m for m in batch if m.en)
        return {"master": [asdict(m) for m in master]}

    def qa(state: State) -> State:
        master = _to_master(state["master"])
        flags = qa_timing(master, max_cps=max_cps)
        return {"master": [asdict(m) for m in master], "qa_flags": flags,
                "retries": state["retries"] + (1 if flags else 0)}

    def route_after_qa(state: State) -> str:
        return "translate" if state.get("qa_flags") and state["retries"] <= 1 else "write"

    def write(state: State) -> State:
        master = _to_master(state["master"])
        report = {**state["report"], "untranslated": sum(1 for m in master if not m.en),
                  "still_too_fast": len(state.get("qa_flags") or [])}
        paths = write_outputs(master, out_dir, ep.manifest.episode, report, state.get("qa_flags"))
        return {"report": report, "written": [str(p) for p in paths]}

    g = StateGraph(State)
    for name, fn in [("prepare", prepare), ("reconcile", reconcile), ("translate", translate),
                     ("qa", qa), ("write", write)]:
        g.add_node(name, fn)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "reconcile")
    g.add_edge("reconcile", "translate")
    g.add_edge("translate", "qa")
    g.add_conditional_edges("qa", route_after_qa, {"translate": "translate", "write": "write"})
    g.add_edge("write", END)
    return g.compile()
