"""chrF translation score (0–100) against a reference English SRT.

chrF compares character n-grams, so it gives partial credit for close wording
("Are you ready, Captain Kim?" vs "Captain Kim, are you ready?") and needs no
extra packages. Text is grouped into 60-second blocks by cue midpoint, like CER.

  python -m subtitle_agent.eval_mt --ref sample/reference_en.srt --hyp out/demo_en.srt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .schema import Cue
from .sources import load_srt


def _ngrams(text: str, n: int) -> Counter:
    t = "".join(text.lower().split())
    return Counter(t[i:i + n] for i in range(len(t) - n + 1))


def chrf(refs: list[str], hyps: list[str], max_n: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF: n-gram matches summed over all segments, then averaged over n."""
    p_sum = r_sum = 0.0
    used = 0
    for n in range(1, max_n + 1):
        match = hyp_total = ref_total = 0
        for r, h in zip(refs, hyps):
            rg, hg = _ngrams(r, n), _ngrams(h, n)
            match += sum((rg & hg).values())
            hyp_total += sum(hg.values())
            ref_total += sum(rg.values())
        if hyp_total and ref_total:
            p_sum += match / hyp_total
            r_sum += match / ref_total
            used += 1
    if not used:
        return 0.0
    p, r = p_sum / used, r_sum / used
    if p + r == 0:
        return 0.0
    return round(100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r), 2)


def chrf_srt(ref: list[Cue], hyp: list[Cue], block: float = 60.0) -> dict:
    rb, hb = defaultdict(list), defaultdict(list)
    for c in ref:
        rb[int(c.mid // block)].append(c.text)
    for c in hyp:
        hb[int(c.mid // block)].append(c.text)
    keys = sorted(set(rb) | set(hb))
    return {"chrf": chrf([" ".join(rb[k]) for k in keys], [" ".join(hb[k]) for k in keys])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(chrf_srt(load_srt(Path(a.ref), "ref", "subtitle"),
                              load_srt(Path(a.hyp), "hyp", "subtitle"))))


if __name__ == "__main__":
    main()
