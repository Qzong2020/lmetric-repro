import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import aiohttp

from cpu_backend import CpuEngine, RemoteCpuEngine, Request
from cpu_benchmark import make_tokenizer, make_trace, policy_metrics


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/cpu_benchmark.json").read_text())["backend"]


class CacheTests(unittest.IsolatedAsyncioTestCase):
    def request(self, rid, tokens):
        return Request(rid, tokens, 2, asyncio.get_running_loop().create_future())

    async def test_only_committed_prefix_is_reusable(self):
        engine = CpuEngine(0, CONFIG)
        a, b, c = [self.request(i, list(range(32))) for i in range(3)]
        engine.prepare(a, [])
        engine.prepare(b, [])
        self.assertEqual(b.hits, 0, "in-flight, uncomputed KV must not count as a hit")
        engine.commit(a)
        engine.prepare(c, [])
        self.assertEqual(c.hits, 16, "last token must be recomputed even for a full cached prompt")
        self.assertEqual(c.bids[0], a.bids[0])
        for req in [a, b, c]:
            engine.release(req)
        engine.assert_drained()

    async def test_eviction_preserves_pinned_blocks_and_reuses_ids(self):
        engine = CpuEngine(0, {**CONFIG, "cache_blocks": 6})
        evicted = []
        pinned = self.request(0, list(range(32)))
        engine.prepare(pinned, evicted)
        engine.commit(pinned)
        pinned_ids = set(pinned.bids)
        for i in range(1, 12):
            req = self.request(i, list(range(i * 100, i * 100 + 32)))
            engine.prepare(req, evicted)
            engine.commit(req)
            engine.release(req)
        self.assertTrue(evicted)
        self.assertTrue(pinned_ids.isdisjoint(evicted))
        self.assertLessEqual(max(engine.blocks), 6)
        self.assertEqual(len(engine.blocks), 6)
        engine.release(pinned)
        engine.assert_drained()

    async def test_http_and_sse_agree_on_cache_and_completion(self):
        engine = CpuEngine(0, CONFIG)
        url = await engine.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url + "/v1/metrics") as events:
                    for rid in [1, 2]:
                        post = asyncio.create_task(session.post(
                            url + "/v1/completions", headers={"X-Request-Id": str(rid)},
                            json={"prompt": list(range(32)), "max_tokens": 2}))
                        statuses = []
                        async for line in events.content:
                            if not line.startswith(b"data:"):
                                continue
                            step = json.loads(line[5:])
                            statuses += step["outputs"]
                            if any(s["is_finished"] for s in step["outputs"]):
                                break
                        response = await post
                        self.assertEqual(response.status, 200)
                        await response.read()
                        self.assertEqual([s["state"] for s in statuses], ["PREFILL", "DECODE"])
                        self.assertEqual(statuses[0]["hit_token_cnt"], 0 if rid == 1 else 16)
                        self.assertEqual(sum(len(s["new_token_ids"]) for s in statuses), 2)
            engine.assert_drained()
        finally:
            await engine.close()

    async def test_spawned_backend_has_separate_process_and_drain_state(self):
        import os
        engine = RemoteCpuEngine(0, CONFIG)
        url = await engine.start()
        try:
            self.assertNotEqual(engine.process.pid, os.getpid())
            async with aiohttp.ClientSession() as session:
                async with session.get(url + "/v1/metrics") as events:
                    response_task = asyncio.create_task(session.post(
                        url + "/v1/completions", headers={"X-Request-Id": "7"},
                        json={"prompt": list(range(32)), "max_tokens": 2}))
                    async for line in events.content:
                        if line.startswith(b"data:") and any(s["is_finished"] for s in json.loads(line[5:])["outputs"]):
                            break
                    response = await response_task
                    self.assertEqual(response.status, 200)
                    await response.read()
                await engine.refresh(session)
                engine.assert_drained()
                self.assertEqual(len(engine.records), 1)
                await engine.reset_remote(session)
                await engine.refresh(session)
                self.assertEqual(engine.records, [])
        finally:
            await engine.close()


class TraceTests(unittest.TestCase):
    def test_policy_histogram_excludes_warmup(self):
        prefix = "blitz_policy_schedule_duration_seconds"
        before = f'{prefix}_count 2\n{prefix}_sum 0.000002\n{prefix}_bucket{{le="0.000001"}} 2\n{prefix}_bucket{{le="0.00001"}} 2\n'
        after = f'{prefix}_count 4\n{prefix}_sum 0.000022\n{prefix}_bucket{{le="0.000001"}} 2\n{prefix}_bucket{{le="0.00001"}} 4\n'
        measured = policy_metrics(after, before)
        self.assertEqual(measured["schedule_count"], 2)
        self.assertAlmostEqual(measured["schedule_mean_us"], 10)
        self.assertEqual(measured["schedule_p95_upper_us"], 10)

    def test_lengths_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as path:
            tokenizer = make_tokenizer(Path(path))
            for scenario in ["no_share", "shared", "mixed", "hotspot_burst"]:
                trace = make_trace(scenario, 17, 10, 120)
                self.assertEqual(trace, make_trace(scenario, 17, 10, 120))
                for request in trace:
                    self.assertEqual(len(tokenizer.encode(request["prompt"]).ids), request["input_length"])


if __name__ == "__main__":
    unittest.main()
