#!/usr/bin/env python3
"""CPU-only synthetic yaullm HTTP/SSE backend. Does not run an LLM.

FCFS, continuous decode batching, unchunked prefill, finite prompt-only KV cache.
Step time is an independent service model, never a router policy score.
"""
import asyncio
import hashlib
import heapq
import json
import multiprocessing
import struct
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from aiohttp import web


@dataclass
class Block:
    key: bytes
    refs: int = 1
    ready: bool = False


@dataclass
class Request:
    rid: int
    tokens: list
    output_length: int
    done: asyncio.Future
    arrival: float = field(default_factory=time.perf_counter)
    generated: int = 0
    hits: int = 0
    bids: list = field(default_factory=list)
    new_bids: list = field(default_factory=list)
    first: float = 0


class CpuEngine:
    def __init__(self, index, config):
        self.index, self.config = index, config
        self.waiting, self.running = deque(), []
        self.free = list(range(1, config["cache_blocks"] + 1))
        self.blocks, self.lookup, self.lru = {}, {}, OrderedDict()
        self.subscribers = set()
        self.wake = asyncio.Event()
        self.task = None
        self.step_id = 0
        self.records = []
        self.evictions = 0
        self.high_water_active = 0
        self.high_water_waiting = 0
        self.failure = None

    def keys(self, tokens):
        previous = b"cpu-cache"
        size = self.config["block_size"]
        for pos in range(0, len(tokens) - size + 1, size):
            previous = hashlib.blake2b(
                previous + struct.pack("<" + "I" * size, *tokens[pos:pos + size]),
                digest_size=16,
            ).digest()
            yield previous

    def touch(self, bid):
        self.lru.pop(bid, None)
        self.lru[bid] = None

    def allocate(self, key, evicted):
        if not self.free:
            victim = next((b for b in self.lru if self.blocks[b].refs == 0), None)
            if victim is None:
                raise RuntimeError("KV capacity cannot hold the active prompts")
            old = self.blocks.pop(victim)
            self.lru.pop(victim)
            if self.lookup.get(old.key) == victim:
                del self.lookup[old.key]
            heapq.heappush(self.free, victim)
            evicted.append(victim)
            self.evictions += 1
        bid = heapq.heappop(self.free)
        self.blocks[bid] = Block(key)
        self.touch(bid)
        return bid

    def prepare(self, request, evicted):
        keys = list(self.keys(request.tokens))
        # Recompute at least one token to produce the first generated token.
        max_hit = (len(request.tokens) - 1) // self.config["block_size"]
        for key in keys[:max_hit]:
            bid = self.lookup.get(key)
            if bid is None:
                break
            assert self.blocks[bid].ready
            self.blocks[bid].refs += 1
            self.touch(bid)
            request.bids.append(bid)
        request.hits = len(request.bids) * self.config["block_size"]
        for key in keys[len(request.bids):]:
            bid = self.allocate(key, evicted)
            request.bids.append(bid)
            request.new_bids.append(bid)

    def commit(self, request):
        for bid in request.new_bids:
            block = self.blocks[bid]
            block.ready = True
            self.lookup[block.key] = bid

    def release(self, request):
        for bid in request.bids:
            self.blocks[bid].refs -= 1
            assert self.blocks[bid].refs >= 0

    def assert_drained(self):
        assert not self.waiting and not self.running
        assert all(block.refs == 0 for block in self.blocks.values())
        assert len(self.blocks) + len(self.free) == self.config["cache_blocks"]

    async def run(self):
        try:
            while True:
                if not self.waiting and not self.running:
                    self.wake.clear()
                    await self.wake.wait()
                decoders = list(self.running)
                prefills, evicted = [], []
                prefill_tokens = 0
                # Estimate budget before allocating/pinning blocks.
                while self.waiting and len(self.running) < self.config["max_running"]:
                    req = self.waiting[0]
                    hit_blocks = 0
                    for key in list(self.keys(req.tokens))[:(len(req.tokens) - 1) // self.config["block_size"]]:
                        if key not in self.lookup:
                            break
                        hit_blocks += 1
                    work = len(req.tokens) - hit_blocks * self.config["block_size"]
                    if prefills and prefill_tokens + work > self.config["prefill_budget"]:
                        break
                    req = self.waiting.popleft()
                    self.prepare(req, evicted)
                    prefill_tokens += len(req.tokens) - req.hits
                    prefills.append(req)
                    self.running.append(req)
                self.high_water_active = max(self.high_water_active, len(self.running))
                context = sum(len(r.tokens) + r.generated for r in decoders)
                delay_ms = (
                    self.config["base_step_ms"]
                    + 1000 * prefill_tokens / self.config["prefill_tokens_per_second"]
                    + len(decoders) * self.config["decode_request_ms"]
                    + context / 1024 * self.config["context_1024_ms"]
                )
                await asyncio.sleep(delay_ms / 1000)
                outputs, used, new = [], {}, {}
                now = time.perf_counter()
                for req in prefills:
                    self.commit(req)
                    req.first = now
                    used[str(req.rid)] = req.bids
                    new[str(req.rid)] = req.new_bids
                for req in decoders + prefills:
                    req.generated += 1
                    finished = req.generated == req.output_length
                    outputs.append({
                        "request_id": req.rid, "new_token_ids": [2],
                        "state": "PREFILL" if req in prefills else "DECODE",
                        "is_finished": finished, "hit_token_cnt": req.hits,
                        "prev_computed_tokens": req.hits if req in prefills else len(req.tokens) + req.generated - 2,
                    })
                    if finished:
                        self.running.remove(req)
                        self.release(req)
                        self.records.append({
                            "rid": req.rid, "engine": self.index,
                            "input_tokens": len(req.tokens), "output_tokens": req.generated,
                            "cached_tokens": req.hits,
                            "backend_ttft_ms": (req.first - req.arrival) * 1000,
                            "backend_e2e_ms": (now - req.arrival) * 1000,
                        })
                        req.done.set_result(True)
                self.step_id += 1
                payload = {
                    "step_id": self.step_id, "latency": round(delay_ms),
                    "prefill_tokens": prefill_tokens,
                    "prefill_token_budget": max(0, self.config["prefill_budget"] - prefill_tokens),
                    "outputs": outputs, "cur_used_block_ids": used,
                    "new_block_hashes_ids": new, "evicted_block_ids": evicted,
                    "op_exec_log": None,
                }
                for queue in self.subscribers:
                    queue.put_nowait(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = repr(exc)
            for req in list(self.waiting) + self.running:
                if not req.done.done():
                    req.done.set_exception(exc)
            raise

    async def completions(self, http_request):
        body = await http_request.json()
        tokens = body["prompt"]
        if not isinstance(tokens, list) or not tokens:
            raise web.HTTPBadRequest(text="CPU backend requires nonempty token IDs")
        req = Request(int(http_request.headers["X-Request-Id"]), tokens,
                      int(body["max_tokens"]), asyncio.get_running_loop().create_future())
        self.waiting.append(req)
        self.high_water_waiting = max(self.high_water_waiting, len(self.waiting))
        self.wake.set()
        await asyncio.shield(req.done)
        return web.json_response({"choices": [{"text": "out", "finish_reason": "length"}]})

    async def metrics(self, http_request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                               "Cache-Control": "no-cache"})
        queue = asyncio.Queue()
        self.subscribers.add(queue)
        await response.prepare(http_request)
        try:
            await response.write(b": cpu simulator connected\n\n")
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), 1)
                    await response.write(("data: " + json.dumps(payload, separators=(",", ":")) + "\n\n").encode())
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            self.subscribers.discard(queue)
        return response

    async def start(self, port=0):
        app = web.Application()
        app.router.add_post("/v1/completions", self.completions)
        app.router.add_route("*", "/v1/metrics", self.metrics)
        app.router.add_get("/cpu/status", self.status)
        app.router.add_post("/cpu/reset", self.reset)
        self.runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", port)
        await site.start()
        self.task = asyncio.create_task(self.run())
        self.port = site._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{self.port}"

    async def status(self, _):
        return web.json_response({
            "records": self.records, "step_id": self.step_id,
            "subscribers": len(self.subscribers), "evictions": self.evictions,
            "high_water_active": self.high_water_active,
            "high_water_waiting": self.high_water_waiting,
            "failure": self.failure, "waiting": len(self.waiting),
            "running": len(self.running),
            "references": sum(b.refs for b in self.blocks.values()),
            "block_accounting_ok": len(self.blocks) + len(self.free) == self.config["cache_blocks"],
        })

    async def reset(self, _):
        self.assert_drained()
        self.records.clear()
        self.evictions = self.high_water_active = self.high_water_waiting = 0
        return web.json_response({"ok": True})

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.runner.cleanup()


def _worker_main(index, config, pipe):
    async def main():
        engine = CpuEngine(index, config)
        url = await engine.start()
        pipe.send(url)
        pipe.close()
        try:
            await asyncio.Event().wait()
        finally:
            await engine.close()
    asyncio.run(main())


class RemoteCpuEngine:
    """One spawned CPU backend per process; load generation runs independently."""
    def __init__(self, index, config):
        self.index, self.config = index, config
        self.subscribers = self.step_id = 0
        self.records = []
        self.failure = None

    async def start(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_worker_main, args=(self.index, self.config, child), daemon=True)
        self.process.start()
        child.close()
        if not parent.poll(15):
            raise TimeoutError("CPU backend process startup timed out")
        self.url = parent.recv()
        parent.close()
        return self.url

    async def refresh(self, session):
        async with session.get(self.url + "/cpu/status") as response:
            response.raise_for_status()
            state = await response.json()
        self.__dict__.update(state)

    async def reset_remote(self, session):
        async with session.post(self.url + "/cpu/reset") as response:
            response.raise_for_status()
            await response.read()

    def assert_drained(self):
        assert self.waiting == self.running == self.references == 0
        assert self.block_accounting_ok

    async def close(self):
        if hasattr(self, "process"):
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
