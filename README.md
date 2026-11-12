# Subtitle Agent

An agent that turns several noisy machine transcripts of a Chinese show (subtitle OCR, full-frame OCR, two ASR models) into a clean Chinese master and English subtitles with translator's notes.

## Key design decision: the agent doesn't run the models

OCR and ASR run beforehand, in Colab or wherever is cheapest. Their outputs are listed in a `manifest.toml`. The agent only works on the cases where the sources disagree.

- **Why:** the models are slow, need a GPU, and give the same output every time. Re-running them inside an agent loop costs time and money for nothing. The judgment calls are the hard part: which source to trust, whether a line is a hallucination, how to keep names consistent.
- **Tradeoff:** the agent can't ask for a fresh model pass (for example, "re-OCR this 3 s at higher resolution"). It can only look at what was precomputed. A future `request_rerun` tool could queue that work instead of running it inline.
- **Measured effect:** most windows are auto-accepted with no LLM call at all. `report.json` gives the split for each episode.

## Two architectures (for comparison)

| | `--mode single` (baseline) | `--mode multi` |
|---|---|---|
| Reconcile | one agent, windows one at a time | one **reconciler agent per window, in parallel** (LangGraph `Send`) |
| Translate | translator, one pass | **translator ↔ reviewer**: reviewer flags problems, translator revises (1 round) |
| QA | deterministic timing repair | same |

Both share tools, prompts, schemas and QA, so the comparison isolates the architecture:

```bash
python -m subtitle_agent.compare sample/manifest.toml --model ollama:qwen2.5:7b \
    --ref-zh sample/reference_zh.srt --ref-en sample/reference_en.srt
```
→ a table of Chinese CER, English chrF, LLM calls, tokens and time per mode (`out/compare/compare.md`).

## Lessons from evals

- **Run 1 (qwen2.5:7b, single):** CER 0.025. The agent dropped the Whisper hallucination, but its free-text JSON for the OCR-typo window was unparseable, so it fell back to the OCR text and kept the typo. **Fix:** schema-constrained structured output (Pydantic + `with_structured_output`), with a text-parse fallback and the raw error logged in the notes.

## How it works

```
precomputed files ─► loaders ─► align into time windows ─► agree? ──yes──► auto-accept
  (manifest.toml)    (SRT/JSON)  (no LLM)                    │
                                                             no
                                                             ▼
                          LangGraph:  reconcile agent (tools) ─► translate ─► QA ─► write
                                                                    ▲           │
                                                                    └─ shorten ─┘ (1 retry)
```

| Piece | File | LLM? |
|---|---|---|
| Load any SRT/JSON into one `Cue` format | `sources.py` | no |
| Group cues into windows, score agreement, find OCR typos outvoted by ASR, drop duplicate or recurring overlays | `timeline.py` | no |
| Glossary retrieval (names, show terms) | `glossary.py` | no |
| Agent tools: `get_window`, `compare_sources`, `lookup_glossary`, `get_rules` | `tools.py` | — |
| Graph (single + multi modes), structured-output schemas | `graph.py` | yes |
| Run both modes and score them | `compare.py` | yes |
| chrF translation score | `eval_mt.py` | no |
| Reading-speed QA: lengthen cues where there's room | `output.py` | no |
| CER evaluation + per-source ablation | `eval_cer.py` | no |

## Run it

```bash
pip install -r requirements.txt

# 1. No LLM needed: see what the agent would receive
python -m subtitle_agent.run sample/manifest.toml --dry-run

# 2. Full agent with a free local model (install Ollama, then: ollama pull qwen2.5:7b)
python -m subtitle_agent.run sample/manifest.toml --model ollama:qwen2.5:7b --out out/

# 3. Score against a hand-finished reference, and score each raw source alone
python -m subtitle_agent.eval_cer --ref sample/reference_zh.srt --hyp out/demo_zh.srt --manifest sample/manifest.toml

python -m unittest discover -s tests
```

Tracing: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` before a full run, and every tool call and decision appears in LangSmith.

## Using a real episode

Copy `sample/manifest.toml` next to your episode's files and edit the `[[sources]]` blocks. SRT works as-is. For JSON, map the field names:

```toml
[[sources]]
name = "paddle_fullframe"
kind = "overlay"            # subtitle | speech | overlay
path = "ocr_raw.json"
format = "json"
list_key = "frames"         # where the list of items lives
fields = { start = "t0", end = "t1", text = "rec_text", conf = "rec_score" }
# time_scale = 0.001        # if times are in ms
```

## Roadmap

- [ ] Eval set from finished episodes (ep4 Chinese master as reference); track CER per source vs. agent
- [ ] Translation eval: LLM-judge rubric + human spot checks against the finished English SRTs
- [ ] Prompt-injection test: on-screen text containing instructions, check the agent transcribes rather than obeys
- [ ] Dockerfile + cloud deploy of a demo running on cached outputs from a short clip
- [x] Multi-agent mode: parallel reconcilers (`Send`) + translator/reviewer loop
- [ ] Supervisor that routes each conflict to a specialist (OCR-vs-ASR, hallucination check, overlay)
- [ ] Test on a video with official zh + en subtitles (held-out test set; don't tune on it)

*No show footage or full transcripts are included in this repo. Sample data is invented.*
