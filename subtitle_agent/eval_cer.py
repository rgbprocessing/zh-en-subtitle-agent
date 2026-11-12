"""Character error rate against a hand-finished reference transcript.

  # score the agent's master
  python -m subtitle_agent.eval_cer --ref ep04_final_zh.srt --hyp out/ep04_zh.srt
  # ablation: score every raw source alone, from the manifest
  python -m subtitle_agent.eval_cer --ref ep04_final_zh.srt --manifest ep04/manifest.toml

Text is compared in 60-second blocks (by cue midpoint) so small timing
differences don't matter and long episodes stay fast.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .schema import LINE_KINDS, Cue
from .sources import Manifest, load_srt
from .timeline import normalize


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _blocks(cues: list[Cue], size: float) -> dict[int, str]:
    out = defaultdict(list)
    for c in sorted(cues, key=lambda c: c.start):
        out[int(c.mid // size)].append(normalize(c.text))
    return {k: "".join(v) for k, v in out.items()}


def cer(ref: list[Cue], hyp: list[Cue], block: float = 60.0) -> dict:
    r, h = _blocks(ref, block), _blocks(hyp, block)
    edits = sum(levenshtein(r.get(k, ""), h.get(k, "")) for k in set(r) | set(h))
    n = sum(len(v) for v in r.values())
    return {"cer": round(edits / max(n, 1), 4), "edits": edits, "ref_chars": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp")
    ap.add_argument("--manifest")
    ap.add_argument("--block", type=float, default=60.0)
    a = ap.parse_args(argv)
    ref = load_srt(Path(a.ref), "ref", "subtitle")
    results = {}
    if a.hyp:
        results["hypothesis"] = cer(ref, load_srt(Path(a.hyp), "hyp", "subtitle"), a.block)
    if a.manifest:
        m = Manifest.load(a.manifest)
        for s in m.sources:
            if s.kind in LINE_KINDS:
                results[s.name] = cer(ref, s.load(), a.block)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
