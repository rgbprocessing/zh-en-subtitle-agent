# zh-en-subtitle-agent

A LangGraph agent that turns several noisy machine transcripts of a Chinese TV show into a clean Chinese transcript and English subtitles with translator's notes.

**Who it's for:** fans and translators subtitling shows that have no official English subs. I built it to replace my own manual workflow: running several OCR and speech-recognition models over each episode, then hand-merging their disagreements.

## The problem

No single model gets a variety show right:

| Source | Good at | Fails at |
|---|---|---|
| Subtitle OCR (cropped) | Exact wording of burned-in subs | Look-alike characters (备 → 各) |
| Full-frame OCR | Signs, captions, graphics | Re-reads the subtitles; sponsor scrolls |
| ASR (FunASR, Whisper) | Lines with no subtitle | Names; hallucinated "字幕由…提供" credits over silence |

The agent's job is to make the judgment calls I used to make by hand.

## How it works

```
precomputed OCR/ASR files ─► align into time windows ─► sources agree? ─yes─► auto-accept
       (manifest.toml)          (plain code, no LLM)          │
                                                             no
                                                              ▼
              reconciler agent (tools) ─► translator ─► [reviewer] ─► timing QA ─► .srt + notes
```

- **Alignment (deterministic):** groups cues from every source by time and scores agreement. It sends a window to the agent only when sources disagree, and also when one OCR reading is outvoted by two ASR models that agree with each other.
- **Reconciler (the agentic step):** a tool-using LLM decides what was really said. It can look at neighbouring time (`get_window`), compare sources (`compare_sources`), look up names (`lookup_glossary`), and check the rules (`get_rules`). It answers in a fixed schema (`Decision`), so even small models can't break the format.
- **Translator → reviewer:** batches lines with glossary hits and prior context. In multi-agent mode, a reviewer flags problems and the translator revises them once.
- **Timing QA (deterministic):** lengthens cues that are too fast to read, where there's room.

Built with LangGraph, LangChain, and LangSmith (tracing). It runs on any chat model; I use **Qwen 2.5 7B via Ollama** on a Colab GPU.

### Key design decisions:

OCR and ASR run beforehand, and the agent reads their saved outputs.
- **Why:** the models are slow, need a GPU, and give the same output every time. Running them inside an agent loop would cost time and money for nothing. The hard part is judgment, not extraction.
- **Tradeoff:** the agent can't ask for a fresh pass (for example, "re-OCR these 3 seconds at higher resolution"). It can only use what was precomputed.
- **Benefit:** most windows are auto-accepted with **no LLM call at all**. The LLM only sees real disputes.

Honest framing: this is mostly a **workflow with one agentic step**. LLM autonomy is used only where judgment is needed, and the rest is plain, testable code.

## Single vs multi-agent

| | `--mode single` | `--mode multi` |
|---|---|---|
| Reconcile | one agent, one window at a time | one reconciler per window, **in parallel** (LangGraph `Send`) |
| Translate | one pass | translator ↔ **reviewer** loop |

Both modes share tools, prompts, and QA, so a comparison isolates the architecture.

## Evaluation

- **Chinese:** character error rate (CER) against a hand-corrected transcript, plus each raw source scored alone as a baseline.
- **English:** chrF (character n-gram F-score) against a reference translation.
- **Cost:** LLM calls, tokens, and time, per mode.

### Results so far (sample clip, qwen2.5:7b)

| Run | What happened | Change |
|---|---|---|
| 1 | CER 0.025 (best single source: 0.175). Dropped the Whisper hallucination, but the model's free-text JSON for the OCR-typo window was unparseable, so the typo survived. | Switched to schema-constrained structured output |
| 2 | 0 parse failures; typo fixed. CER 0.05. The only difference from the reference was a filler sound (嗯嗯) the agent kept. | Decided that confident filler sounds belong in the subtitles: updated rules + reference |

Single vs multi on the sample: identical quality, with multi using 1 extra LLM call. The reviewer found nothing to fix. The 5-line sample is too easy to separate the architectures; full-episode comparisons are next.

## Run it

```bash
pip install -r requirements.txt
python -m unittest discover -s tests

# no LLM needed: see what the agent would receive
python -m subtitle_agent.run sample/manifest.toml --dry-run

# full run (Ollama: `ollama pull qwen2.5:7b`)
python -m subtitle_agent.run sample/manifest.toml --model ollama:qwen2.5:7b --mode multi

# compare architectures
python -m subtitle_agent.compare sample/manifest.toml --model ollama:qwen2.5:7b \
    --ref-zh sample/reference_zh.srt --ref-en sample/reference_en.srt
```

On a GPU: open `notebooks/colab_run.ipynb` in Colab. Tracing: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`.

### Using your own episode

Point a `manifest.toml` at your files. SRT works as-is; for JSON, map the field names:

```toml
[[sources]]
name = "paddle_fullframe"
kind = "overlay"            # subtitle | speech | overlay
path = "ocr_raw.json"
format = "json"
list_key = "frames"
fields = { start = "t0", end = "t1", text = "rec_text", conf = "rec_score" }
```

## Code map

| File | What | LLM? |
|---|---|---|
| `sources.py` | Load SRT/JSON into one cue format; manifest | no |
| `timeline.py` | Alignment, agreement scoring, overlay de-dup | no |
| `glossary.py` | Name/term retrieval | no |
| `tools.py` | The reconciler's tools | — |
| `graph.py` | LangGraph graph (both modes), output schemas | yes |
| `output.py` | Timing QA, SRT + notes writers | no |
| `eval_cer.py`, `eval_mt.py`, `compare.py` | Evaluation | — |

Roadmap: see [Issues](../../issues).

*No show footage or full transcripts are included. The sample data is invented.*
