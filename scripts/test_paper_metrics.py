import unittest
from paper_metrics import StreamMetrics, empirical_cdf


def content(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


class MetricsTests(unittest.TestCase):
    def test_empty_role_and_delayed_done_do_not_change_token_timing(self):
        metric = StreamMetrics(100)
        metric.observe({"choices": [{"delta": {"role": "assistant", "content": ""}}]}, 101)
        metric.observe(content("a"), 110)
        metric.observe(content("bc"), 120)  # two tokens in one content chunk
        metric.observe(content("d"), 140)
        metric.observe({"choices": [], "usage": {"completion_tokens": 4}}, 180)
        metric.observe("[DONE]", 200)
        self.assertEqual(metric.result(), {"ttft_ms": 10, "tpot_ms": 10,
            "output_tokens": 4, "e2e_ms": 100, "count_source": "usage"})

    def test_missing_usage_does_not_treat_chunks_as_tokens(self):
        metric = StreamMetrics(0)
        metric.observe(content("many tokens in one chunk"), 10)
        metric.observe(content("more"), 40)
        metric.observe("[DONE]", 50)
        self.assertIsNone(metric.result()["tpot_ms"])
        self.assertEqual(metric.result(4)["tpot_ms"], 10)

    def test_one_output_token_has_no_decode_interval(self):
        metric = StreamMetrics(0)
        metric.observe(content("one"), 10)
        metric.observe("[DONE]", 20)
        self.assertIsNone(metric.result(1)["tpot_ms"])

    def test_incomplete_or_failed_streams_are_rejected(self):
        metric = StreamMetrics(0)
        with self.assertRaises(ValueError):
            metric.result()
        with self.assertRaises(ValueError):
            metric.observe({"error": "engine failed"}, 10)

    def test_timestamp_reversals_are_rejected(self):
        metric = StreamMetrics(10)
        with self.assertRaises(ValueError):
            metric.observe(content("x"), 9)
        metric.observe(content("x"), 15)
        with self.assertRaises(ValueError):
            metric.observe("[DONE]", 14)

    def test_cdf_is_over_requests(self):
        self.assertEqual(empirical_cdf([30, 10, 20]), [(10, 1/3), (20, 2/3), (30, 1)])
        self.assertEqual(empirical_cdf([]), [])


if __name__ == "__main__":
    unittest.main()
