"""Run both architectures on the same episode and score them side by side.

  python -m subtitle_agent.compare sample/manifest.toml --model ollama:qwen2.5:7b \\
      --ref-zh sample/reference_zh.srt --ref-en sample/reference_en.srt --out out/compare

Writes out/compare/<mode>/... for each mode, plus compare.json and compare.md.
Run it a few times (or on several episodes) before drawing conclusions: LLM
outputs vary, and one 5-line sample is a smoke test, not an eval.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .eval_cer import cer
from .eval_mt import chrf_srt
from .pipeline import Episode
from .sources import load_srt


def run_mode(manifest: str, llm, mode: str, out: Path, ref_zh=None, ref_en=None) -> dict:
    from .graph import build_graph
    from .usage import UsageCounter

    ep = Episode.load(manifest)
    counter = UsageCounter()
    t0 = time.time()
    final = build_graph(ep, llm, out / mode, mode=mode).invoke(
        {}, config={"callbacks": [counter], "recursion_limit": 50,
                    "run_name": f"subtitle-agent:{ep.manifest.episode}:{mode}", "tags": [mode]})
    row = {"mode": mode, "seconds": round(time.time() - t0, 1), **counter.summary()}
    rep = final["report"]
    row.update({k: rep.get(k) for k in ("agent_decisions", "reviewer_flagged", "still_too_fast")})
    ep_name = ep.manifest.episode
    if ref_zh:
        row["cer_zh"] = cer(load_srt(Path(ref_zh), "ref", "subtitle"),
                            load_srt(out / mode / f"{ep_name}_zh.srt", "hyp", "subtitle"))["cer"]
    en_path = out / mode / f"{ep_name}_en.srt"
    if ref_en and en_path.exists():
        row["chrf_en"] = chrf_srt(load_srt(Path(ref_en), "ref", "subtitle"),
                                  load_srt(en_path, "hyp", "subtitle"))["chrf"]
    return row


def to_markdown(rows: list[dict]) -> str:
    cols = ["mode", "cer_zh", "chrf_en", "llm_calls", "input_tokens", "output_tokens", "seconds",
            "agent_decisions", "reviewer_flagged"]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--model", default="ollama:qwen2.5:7b")
    ap.add_argument("--modes", default="single,multi")
    ap.add_argument("--ref-zh")
    ap.add_argument("--ref-en")
    ap.add_argument("--out", default="out/compare")
    a = ap.parse_args(argv)

    from langchain.chat_models import init_chat_model
    llm = init_chat_model(a.model, temperature=0)
    out = Path(a.out)
    rows = [run_mode(a.manifest, llm, m.strip(), out, a.ref_zh, a.ref_en) for m in a.modes.split(",")]
    for r in rows:
        r["model"] = a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_markdown(rows)
    (out / "compare.md").write_text(f"Model: `{a.model}`\n\n{md}\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
