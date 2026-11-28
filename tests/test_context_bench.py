"""Regression tests for qualification, device scope and interrupted streams."""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from context_bench import summarize_burst
from device_journal import DeviceJournal
from llm_stream import Endpoint, burst
from standard_bench import PublicLoadPolicy
from llm_evidence import SweepBudget, decode_peak, public_outcomes, read_metrics

SLO = {"llm_ttft_p95_s": 3, "llm_tpot_p95_ms": 100, "llm_min_joint_pass_fraction": 0.95}


class EvidenceTests(unittest.TestCase):
    def test_public_load_policy_accepts_partial_config_and_preserves_defaults(self):
        default = PublicLoadPolicy.from_config({})
        self.assertEqual([default.prompt_count(n) for n in (2, 8, 64)], [100, 200, 1200])
        policy = PublicLoadPolicy.from_config(
            {"min_prompts": 5, "prompts_per_concurrency": 2, "datasets": {}}
        )
        self.assertEqual([policy.prompt_count(n) for n in (1, 4)], [5, 8])
        with self.assertRaises(ValueError):
            PublicLoadPolicy.from_config({"max_prompts": 10})

    def test_budget_uses_previous_level_and_remaining_wall_time(self):
        budget = SweepBudget(20, start=0)
        with patch("llm_evidence.time.monotonic", return_value=11):
            self.assertTrue(budget.allows(4, "first level"))
            budget.completed(work=2, elapsed=6)
            self.assertTrue(budget.allows(3, "fits"))
            self.assertFalse(budget.allows(4, "exceeds remaining time"))

    def test_missing_request_timing_is_unknown(self):
        raw = dict(completed=1, ttfts=[None], latencies=[1.2], output_lens=[2], start_times=[0])
        result = public_outcomes(raw, 1, 1, SLO)
        self.assertEqual(result["capacity_status"], "insufficient evidence")
        self.assertEqual(result["latency_status"], "insufficient evidence")

    def test_touching_intervals_do_not_overlap(self):
        self.assertEqual(decode_peak([(0, 1), (1, 2), (1, 1)]), 1)

    def test_queued_requests_can_complete_without_capacity_or_latency(self):
        raw = dict(
            completed=2, start_times=[0, 0], ttfts=[0.1, 4.0], latencies=[0.15, 4.05], output_lens=[2, 2]
        )
        result = public_outcomes(raw, 2, 2, SLO)
        self.assertEqual(result["completion_status"], "passed")
        self.assertEqual(result["verified_decode_concurrency"], 1)
        self.assertEqual(result["capacity_status"], "failed")
        self.assertEqual(result["joint_fraction"], 0.5)

    def test_missing_details_are_unknown(self):
        result = public_outcomes({"completed": 16}, 16, 16, SLO)
        self.assertEqual(result["latency_status"], "insufficient evidence")
        self.assertEqual(result["capacity_status"], "insufficient evidence")

    def test_metrics_exclude_other_models(self):
        text = (
            'vllm:num_preemptions_total{model_name="other"} 7\n'
            'vllm:num_preemptions_total{model_name="gemma"} 0'
        )
        self.assertEqual(read_metrics(text, "gemma"), {"num_preemptions_total": 0})

    def test_kernel_scope_and_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel"
            path.write_text("xe 0000:64:00.0 Engine reset\n")
            journal = DeviceJournal(["0000:64:00.0"], path)

            def append(text):
                with path.open("a") as f:
                    f.write(text + "\n")

            append("xe 0000:90:00.0 Engine reset")
            self.assertIs(journal.finish(Path(tmp) / "slice"), True)
            append("xe 0000:64:00.0 Timedout job")
            self.assertIs(journal.finish(Path(tmp) / "slice"), False)
            path.write_text("")
            self.assertIsNone(journal.finish(Path(tmp) / "slice"))


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.release = asyncio.Event()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    async def exercise(self, handler, stall, timeout):
        async def metrics(_):
            return web.Response(text='vllm:num_preemptions_total{model_name="m"} 0')

        app = web.Application()
        app.router.add_get("/metrics", metrics)
        app.router.add_post("/v1/completions", handler)
        async with TestServer(app) as server, ClientSession() as session:
            endpoint = Endpoint(session, str(server.make_url("/")), "m")
            try:
                return await burst(endpoint, [[1, 2]], 2, self.folder, stall, timeout)
            finally:
                self.release.set()

    async def test_truncated_stream(self):
        async def truncated(_):
            return web.Response(body=b'data: {"choices":[{"text":"x","token_ids":[1]}]}\n\n')

        requests, _, seconds, reason = await self.exercise(truncated, None, 3)
        level = summarize_burst(requests, 2, 2, seconds, reason, {}, {}, (True, None), SLO)
        self.assertEqual(level["failed"], 1)
        self.assertEqual(level["capacity_status"], "failed")
        events = [json.loads(l) for l in (self.folder / "events.jsonl").read_text().splitlines()]
        self.assertIn("output", [e.get("state") for e in events])
        self.assertIn("failed", [e.get("state") for e in events])

    async def test_stall_retains_interrupted_request(self):
        async def hanging(_):
            await self.release.wait()
            return web.Response()

        requests, _, _, reason = await self.exercise(hanging, 0.1, 5)
        self.assertEqual(reason, "stalled")
        self.assertEqual(requests[0].state, "interrupted")

    async def test_wall_deadline_without_stall(self):
        async def hanging(_):
            await self.release.wait()
            return web.Response()

        requests, _, _, reason = await self.exercise(hanging, None, 0.2)
        self.assertEqual(reason, "wall_timeout")
        self.assertEqual(requests[0].state, "interrupted")


if __name__ == "__main__":
    unittest.main()
