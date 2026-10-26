# SPDX-License-Identifier: Apache-2.0
"""Bounded scheduler-metadata journal, independent of native KV publishers.

This observes successful cache-map changes, not NPU content completion. The
owner and all mutation/drain calls live on the EngineCore scheduler thread.
"""
import json
import os


def kv_enabled():
    return os.getenv("VLLM_LMETRIC_KV_EVENTS", "0") == "1"


class KVJournal:
    def __init__(self, epoch, block_size, num_blocks, max_events=8192,
                 max_bytes=16 * 1024 * 1024):
        self.epoch = epoch
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.seq = 0
        self.pending = []
        self.pending_bytes = 0
        self.failed = None
        self.requests = {}
        self.mappings = {}
        self.reset("initial_empty")

    def _fail(self, reason):
        self.failed = self.failed or reason
        raise RuntimeError("LMetric KV journal invalid: " + self.failed)

    def _check(self):
        if self.failed:
            self._fail(self.failed)

    def _record(self, kind, **fields):
        self._check()
        event = dict(mutation_seq=self.seq + 1, type=kind, group_id=0,
                     block_size=self.block_size, **fields)
        size = len(json.dumps(event, separators=(",", ":")))
        if len(self.pending) >= self.max_events or self.pending_bytes + size > self.max_bytes:
            self._fail("bounded mutation buffer exceeded")
        self.seq += 1
        self.pending.append(event)
        self.pending_bytes += size

    def _bid(self, block_id):
        if not 0 < block_id < self.num_blocks:
            self._fail("invalid/null physical block ID")

    def insert(self, block_id, cache_hash, group_id, block_size, logical_index,
               prefix_token_ids):
        self._check()
        self._bid(block_id)
        if group_id != 0 or block_size != self.block_size:
            self._fail("unsupported KV group or block size")
        tokens = list(prefix_token_ids)
        if (len(tokens) != (logical_index + 1) * block_size
                or len(tokens) > 8192 or any(t < 0 for t in tokens)):
            self._fail("invalid or oversized finalized token prefix")
        if block_id in self.mappings:
            self._fail("physical ID inserted without prior removal")
        cache_hash = bytes(cache_hash).hex()
        self._record("cache_insert", block_id=block_id, cache_hash=cache_hash,
                     logical_index=logical_index, prefix_token_ids=tokens)
        self.mappings[block_id] = cache_hash

    def remove(self, block_id, cache_hash, group_id):
        self._check()
        self._bid(block_id)
        cache_hash = bytes(cache_hash).hex()
        if group_id != 0 or self.mappings.get(block_id) != cache_hash:
            self._fail("removed physical mapping identity mismatch")
        self._record("cache_remove", block_id=block_id, cache_hash=cache_hash)
        del self.mappings[block_id]

    def request_blocks(self, request_id, groups):
        self._check()
        if len(groups) != 1:
            self._fail("multiple KV groups are unsupported")
        if request_id not in self.requests and len(self.requests) >= 1024:
            self._fail("active occupancy request bound exceeded")
        blocks = list(groups[0])
        for bid in blocks:
            self._bid(bid)
        if self.requests.get(request_id) != blocks:
            self._record("request_blocks", request_id=request_id, block_ids=blocks)
            self.requests[request_id] = blocks

    def release(self, request_id):
        self._check()
        if request_id in self.requests:
            self._record("request_release", request_id=request_id, block_ids=[])
            del self.requests[request_id]

    def reset(self, reason="explicit_reset"):
        self._record("cache_reset", reason=reason)
        self.mappings.clear()

    def drain(self):
        self._check()
        result = dict(version="kv-journal-v1",
                      observation_kind="scheduler_cache_metadata",
                      num_blocks=self.num_blocks, mutation_seq=self.seq,
                      mutations=self.pending)
        self.pending = []
        self.pending_bytes = 0
        return result
