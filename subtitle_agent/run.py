"""CLI.

  python -m subtitle_agent.run sample/manifest.toml --dry-run
  python -m subtitle_agent.run ep05/manifest.toml --model ollama:qwen2.5:7b --mode multi --out out/

--dry-run needs no LLM and no LangGraph: it aligns sources, auto-accepts agreed
lines, and writes the partial master plus a report of what the agent would get.
Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to trace full runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .output import write_outputs
from .pipeline import Episode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default="ollama:qwen2.5:7b",
                    help="any init_chat_model id, e.g. ollama:qwen2.5:7b, openai:gpt-4o-mini")
    ap.add_argument("--mode", choices=["single", "multi"], default="single")
    ap.add_argument("--show-pending", type=int, default=3, help="print N pending windows in dry run")
    args = ap.parse_args(argv)

    ep = Episode.load(args.manifest)
    out = Path(args.out)

    if args.dry_run:
        paths = write_outputs(ep.master, out, ep.manifest.episode + "_dryrun", ep.report)
        print(json.dumps(ep.report, ensure_ascii=False, indent=1))
        for wid in ep.pending[:args.show_pending]:
            print("\n" + ep.index.windows[wid].describe())
        print("\nwrote:", *paths, sep="\n  ")
        return

    from langchain.chat_models import init_chat_model
    from .graph import build_graph
    from .usage import UsageCounter

    llm = init_chat_model(args.model, temperature=0)
    app = build_graph(ep, llm, out, mode=args.mode)
    counter = UsageCounter()
    final = app.invoke({}, config={"run_name": f"subtitle-agent:{ep.manifest.episode}:{args.mode}",
                                   "recursion_limit": 50, "callbacks": [counter], "tags": [args.mode]})
    print(json.dumps({**final["report"], **counter.summary()}, ensure_ascii=False, indent=1))
    print("wrote:", *final["written"], sep="\n  ")


if __name__ == "__main__":
    main()
