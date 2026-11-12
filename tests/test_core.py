import unittest
from pathlib import Path

from subtitle_agent.eval_cer import levenshtein
from subtitle_agent.output import qa_timing
from subtitle_agent.pipeline import Episode
from subtitle_agent.schema import MasterCue
from subtitle_agent.sources import fmt_ts, parse_ts
from subtitle_agent.timeline import normalize, similarity

SAMPLE = Path(__file__).resolve().parent.parent / "sample" / "manifest.toml"


class Core(unittest.TestCase):
    def test_timestamps_roundtrip(self):
        self.assertEqual(fmt_ts(parse_ts("01:02:03,045")), "01:02:03,045")

    def test_normalize_ignores_punct_and_width(self):
        self.assertEqual(normalize("大家好，欢迎！ ABC"), normalize("大家好欢迎abc"))

    def test_similarity(self):
        self.assertGreater(similarity("这个泡菜太辣了", "这个泡菜太辣了吧"), 0.9)
        self.assertEqual(similarity("嗯嗯", "字幕由某某提供"), 0.0)

    def test_levenshtein(self):
        self.assertEqual(levenshtein("准备好", "准各好"), 1)

    def test_sample_episode(self):
        ep = Episode.load(SAMPLE)
        self.assertEqual(ep.report["window_status"], {"agree": 3, "conflict": 2})
        self.assertEqual(ep.report["overlays_recurring_dropped"], 6)
        self.assertEqual(ep.report["overlays_duplicate_of_subtitle"], 1)
        # OCR typo outvoted by both ASR models goes to the agent
        texts = [ep.index.windows[w].text_of("cropped_ocr") for w in ep.pending]
        self.assertIn("金队长你准各好了吗", texts)

    def test_qa_lengthens_where_room(self):
        a = MasterCue(0, 0.0, 1.0, "x", "subtitle", en="A" * 30)
        b = MasterCue(1, 5.0, 6.0, "y", "subtitle", en="ok")
        flags = qa_timing([a, b])
        self.assertGreater(a.end, 1.0)
        self.assertLessEqual(a.end, 4.95)
        self.assertEqual(flags, [])


if __name__ == "__main__":
    unittest.main()
