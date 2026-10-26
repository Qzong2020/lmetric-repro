#!/usr/bin/env python3
"""Validate pinned public traces and record CPU-only workload characteristics."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def scan(path, sample_count=2000):
    pointer = path.with_name(path.name + ".lfs-pointer").read_text().splitlines()
    expected_sha = next(s.split(":", 1)[1] for s in pointer if s.startswith("oid sha256:"))
    expected_size = int(next(s.split()[1] for s in pointer if s.startswith("size ")))
    assert path.stat().st_size == expected_size
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == expected_sha
    sample_path = ROOT / "data/paper_cpu/samples" / path.name
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    inputs, outputs = [], []
    types = collections.Counter()
    buckets = collections.Counter()
    errors = collections.Counter()
    prefix_nodes = {}
    next_node = 1
    cached_tokens = 0
    first_time = previous = None
    with path.open() as source, sample_path.open("w") as sample:
        for i, line in enumerate(source):
            row = json.loads(line)
            length, output = row["input_length"], row["output_length"]
            timestamp, hashes = row["timestamp"], row["hash_ids"]
            errors["invalid_lengths"] += int(length <= 0 or output <= 0)
            errors["invalid_block_count"] += int(len(hashes) != (length + 15) // 16)
            errors["non_integer_hash"] += int(any(not isinstance(h, int) or h < 0 for h in hashes))
            errors["timestamp_reversals"] += int(previous is not None and timestamp < previous)
            if first_time is None:
                first_time = timestamp
            previous = timestamp
            inputs.append(length)
            outputs.append(output)
            types[row["type"]] += 1
            buckets[int((timestamp - first_time) // 60)] += 1
            parent, hits, matching = 0, 0, True
            # An infinite prefix cache AFTER every preceding request's prefill
            # completes. This is a structural upper bound, not an engine hit rate.
            for h in hashes[:length // 16]:
                key = parent, h
                if key in prefix_nodes:
                    if matching:
                        hits += 1
                else:
                    matching = False
                    prefix_nodes[key] = next_node
                    next_node += 1
                parent = prefix_nodes[key]
            cached_tokens += hits * 16
            if i < sample_count:
                sample.write(line)
    result = {"file": path.name, "sha256": sha, "bytes": path.stat().st_size,
              "requests": len(inputs), "duration_seconds": previous - first_time,
              "mean_arrival_rps": len(inputs) / (previous - first_time),
              "schema_errors": dict(errors), "types": dict(types),
              "requests_per_minute": dict(sorted(buckets.items())),
              "potential_prefix_cached_tokens": cached_tokens,
              "potential_prefix_token_ratio": cached_tokens / sum(inputs),
              "cache_assumption": "infinite, previous requests completed, full 16-token prefix blocks only",
              "sample": str(sample_path.relative_to(ROOT)), "sample_requests": min(sample_count, len(inputs)),
              "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest()}
    for key, values in [("input_tokens", inputs), ("output_tokens", outputs)]:
        result[key] = {"min": min(values), "mean": sum(values) / len(values),
                       "p50": quantile(values, .5), "p95": quantile(values, .95),
                       "p99": quantile(values, .99), "max": max(values)}
    assert not any(errors.values()), result["schema_errors"]
    return result


if __name__ == "__main__":
    rows = [scan(ROOT / "data/paper_cpu" / f"qwen_trace{letter}_blksz_16.jsonl") for letter in "AB"]
    output = ROOT / "results/paper_cpu/trace_inventory.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row["file"], row["requests"], "requests", "schema OK", "sample", row["sample_requests"])
