# SPDX-License-Identifier: Apache-2.0
"""Opt-in request/execution protocol. No KV mutations or device timing.

This module intentionally has no vLLM/torch imports so its contract can be
exercised on CPU without initializing an engine. Callers supply real outputs.
"""
import asyncio
import copy
import json
import os
import time
import uuid

PROTOCOL = "request-v1"


def enabled():
    return os.getenv("VLLM_LMETRIC_EVENTS", "0") == "1"


def validate_config(config):
    """Reject unsupported modes instead of exporting plausible empty metrics."""
    p = config.parallel_config
    if p.pipeline_parallel_size != 1 or p.data_parallel_size != 1:
        raise ValueError("LMetric request-v1 requires PP=DP=1")
    if config.speculative_config or config.kv_transfer_config or config.ec_transfer_config:
        raise ValueError("LMetric request-v1 excludes speculation and external KV/EC")
    if config.model_config.runner_type != "generate":
        raise ValueError("LMetric request-v1 requires generation")
    if config.lora_config or config.model_config.is_multimodal_model:
        raise ValueError("LMetric request-v1 requires plain text without LoRA")


class RequestEvents:
    """Freeze schedule data, then emit only when that schedule completes.

    The timer includes scheduling, host/device waits, and (for a batch queue)
    queue residence. It is not device time, TTFT, or a non-overlapping tick.
    """
    def __init__(self, scheduler, block_size, hash_block_size, async_scheduling,
                 batch_queue_size):
        groups = scheduler.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or block_size != hash_block_size:
            raise ValueError("LMetric request-v1 requires one KV group and equal block/hash sizes")
        self.epoch = uuid.uuid4().hex
        self.seq = 0
        self.first_outputs = set()
        self.hits = {}
        cls = type(scheduler)
        # Wrap EngineCore's dispatch, including Ascend overrides, rather than
        # monkey-patching only the upstream Scheduler.schedule implementation.
        self.runtime = dict(scheduler=f"{cls.__module__}.{cls.__qualname__}",
                            scheduler_block_size=block_size,
                            hash_block_size=hash_block_size, kv_groups=len(groups),
                            group_spec=type(groups[0].kv_cache_spec).__name__,
                            async_scheduling=async_scheduling,
                            batch_queue_size=batch_queue_size,
                            kv_events=False, protocol=PROTOCOL)

    def capture(self, scheduler, output, started):
        if output.preempted_req_ids or output.scheduled_cached_reqs.resumed_req_ids:
            raise RuntimeError("LMetric request-v1 does not support preempt/resume")
        prev = {r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs}
        prev.update(zip(output.scheduled_cached_reqs.req_ids,
                        output.scheduled_cached_reqs.num_computed_tokens))
        requests = {}
        for rid, count in output.num_scheduled_tokens.items():
            req = scheduler.requests[rid]
            if req.client_index != 0 or req.mm_features or req.prompt_token_ids is None:
                raise RuntimeError("LMetric request-v1 requires one frontend, token IDs, text only")
            if req.prefill_stats is not None:
                self.hits[rid] = req.prefill_stats.num_local_cached_tokens
            if rid not in self.hits:
                raise RuntimeError("Missing first-admission hit snapshot")
            start = prev[rid]
            requests[rid] = dict(prev=start, prompt=req.num_prompt_tokens,
                                 hits=self.hits[rid],
                                 prefill=max(0, min(count, req.num_prompt_tokens-start)))
        output.lmetric_snapshot = dict(started=started, requests=requests)

    def complete(self, output, core_outputs, make_output):
        snap = output.lmetric_snapshot
        if snap is None:
            raise RuntimeError("Missing per-schedule LMetric snapshot")
        if any(client != 0 for client in core_outputs):
            raise RuntimeError("LMetric request-v1 requires one frontend")
        eco = core_outputs.setdefault(0, make_output())
        statuses = []
        for result in eco.outputs:
            rid = result.request_id
            if rid not in snap["requests"]:
                raise RuntimeError("Unscheduled result/control output unsupported by request-v1")
            r = snap["requests"][rid]
            first = rid not in self.first_outputs
            # An empty intermediate chunk is represented by outputs=[], never
            # by a fake generated token or a repeated PREFILL status.
            if not result.new_token_ids and not result.finished:
                continue
            statuses.append(dict(request_id=rid, new_token_ids=list(result.new_token_ids),
                                 state="PREFILL" if first else "DECODE",
                                 is_finished=result.finished, hit_token_cnt=r["hits"],
                                 prev_computed_tokens=r["prev"]))
            self.first_outputs.add(rid)
            if result.finished:
                self.hits.pop(rid, None)
                self.first_outputs.discard(rid)
        self.seq += 1
        eco.lmetric_event = dict(lmetric_protocol=PROTOCOL, engine_epoch=self.epoch,
                                step_id=self.seq, latency=int((time.monotonic()-snap["started"])*1000),
                                latency_kind="schedule_to_completion_host_ms",
                                execution_mode="async" if self.runtime["async_scheduling"] else "sync",
                                kv_events=False, block_size=self.runtime["scheduler_block_size"],
                                kv_groups=self.runtime["kv_groups"],
                                prefill_tokens=sum(r["prefill"] for r in snap["requests"].values()),
                                outputs=statuses)
        return core_outputs


class EventHub:
    """Single subscriber, no replay. Loss/disconnect permanently invalidates it.

    Submission requires a subscriber. A fresh engine/frontend/router session is
    needed after failure; reconnecting with an old incremental ledger is unsafe.
    """
    def __init__(self, capacity=1024):
        self.queue = asyncio.Queue(maxsize=capacity)
        self.active = False
        self.failed = None
        self.identities = {}
        self.external_ids = set()

    def require_ready(self):
        if self.failed or not self.active:
            raise ValueError(self.failed or "Connect /v1/metrics before submitting requests")

    def subscribe(self):
        if self.active or self.failed:
            raise ValueError(self.failed or "Only one LMetric subscriber is supported")
        self.active = True

    def disconnect(self):
        self.active = False
        self.failed = self.failed or "LMetric subscriber disconnected; restart session"

    def register(self, internal_id, external_id):
        self.require_ready()
        if not isinstance(external_id, str) or not external_id.isascii() or not external_id.isdecimal():
            raise ValueError("X-Request-Id must be an unsigned decimal u64")
        n = int(external_id)
        if n > 2**64-1 or n in self.external_ids or internal_id in self.identities:
            raise ValueError("Duplicate or invalid LMetric request ID")
        self.identities[internal_id] = n
        self.external_ids.add(n)

    def publish(self, event):
        self.require_ready()
        event = copy.deepcopy(event)
        for status in event["outputs"]:
            rid = status["request_id"]
            if rid not in self.identities:
                self.failed = "Unknown internal request ID"
                raise RuntimeError(self.failed)
            status["request_id"] = self.identities[rid]
            if status["is_finished"]:
                self.external_ids.remove(self.identities.pop(rid))
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.failed = "LMetric queue overflow; restart session"
            raise RuntimeError(self.failed) from None

    async def frames(self):
        try:
            yield ": ready\n\n"
            while not self.failed:
                try:
                    event = await asyncio.wait_for(self.queue.get(), timeout=1)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"
            yield "data: " + json.dumps({"lmetric_protocol": PROTOCOL, "error": self.failed}) + "\n\n"
        finally:
            self.disconnect()
