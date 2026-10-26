#!/usr/bin/env python3
"""Replay fixed open-loop workloads through the real router and CPU backends.

Requires aiohttp, psutil and tokenizers; no model weights or accelerators.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import time

import aiohttp
import psutil
from tokenizers import Tokenizer, models, pre_tokenizers

from cpu_backend import RemoteCpuEngine

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def make_tokenizer(path):
    path.mkdir(parents=True, exist_ok=True)
    vocab = {"[UNK]": 0, "[PAD]": 1, "out": 2}
    vocab.update({f"w{i}": i + 3 for i in range(4096)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    save_json(path / "config.json", {"model_type": "cpu-synthetic", "max_position_embeddings": 8192})
    # Check controlled prompt lengths and the token returned by the backend.
    assert tokenizer.encode("w0 w1 out").ids == [3, 4, 2]
    assert tokenizer.decode([2]) == "out"
    return tokenizer


def make_trace(scenario, seed, count, rate):
    rng = random.Random(seed)
    groups = []
    for group in range(8):
        prng = random.Random(1000 + group)
        groups.append([f"w{prng.randrange(4096)}" for _ in range(2048)])
    now, trace = 0.0, []
    for i in range(count):
        length, output = 512, 32
        group, shared = None, 0
        if scenario == "shared":
            group, shared = rng.randrange(8), 384
        elif scenario == "mixed":
            length = rng.choices([64, 256, 1024, 2048], [4, 3, 2, 1])[0]
            output = rng.choice([16, 32, 64])
            group, shared = rng.randrange(8), max(0, length - 64)
        elif scenario == "hotspot_burst":
            length, shared = 1024, 896
            group = (i // 100) % 3 if rng.random() < 0.9 else rng.randrange(8)
        elif scenario != "no_share":
            raise ValueError(scenario)
        interval = rng.expovariate(rate)
        if scenario == "hotspot_burst":
            interval *= 0.25 if i % 60 < 15 else 1.25
        now += interval
        tokens = groups[group][:shared] if shared else []
        tokens += [f"w{rng.randrange(4096)}" for _ in range(length - shared)]
        trace.append({"index": i, "at": now, "prompt": " ".join(tokens),
                      "input_length": length, "output_length": output})
    return trace


def percentile(values, pct):
    if not values:
        return None
    vals = sorted(values)
    index = (len(vals) - 1) * pct / 100
    lo, hi = math.floor(index), math.ceil(index)
    return vals[lo] + (vals[hi] - vals[lo]) * (index - lo)


def metrics_values(raw, name):
    vals = []
    for line in raw.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            vals.append(float(line.split()[-1]))
    return vals


def policy_metrics(raw, before_raw=""):
    prefix = "blitz_policy_schedule_duration_seconds"
    counts, sums = metrics_values(raw, prefix + "_count"), metrics_values(raw, prefix + "_sum")
    total = sum(counts) - sum(metrics_values(before_raw, prefix + "_count"))
    previous_buckets = {}
    for line in before_raw.splitlines():
        if line.startswith(prefix + "_bucket{"):
            previous_buckets[float(line.split('le="')[1].split('"')[0])] = float(line.split()[-1])
    buckets = []
    for line in raw.splitlines():
        if line.startswith(prefix + "_bucket{"):
            bound = float(line.split('le="')[1].split('"')[0])
            buckets.append((bound, float(line.split()[-1]) - previous_buckets.get(bound, 0)))
    result = {"schedule_count": total,
              "schedule_mean_us": (sum(sums) - sum(metrics_values(before_raw, prefix + "_sum"))) / total * 1e6 if total else None}
    for q in [50, 95, 99]:
        # Upper bucket bound, not an exact quantile.
        bound = next((b for b, n in sorted(buckets) if n >= total * q / 100), None)
        result[f"schedule_p{q}_upper_us"] = bound * 1e6 if bound is not None else None
    return result


async def replay(session, endpoint, trace):
    started = time.perf_counter()

    async def one(item):
        await asyncio.sleep(max(0, started + item["at"] - time.perf_counter()))
        sent = time.perf_counter()
        rec = {"index": item["index"], "scheduled_at": item["at"],
               "dispatch_lag_ms": (sent - started - item["at"]) * 1000,
               "input_length": item["input_length"], "expected_tokens": item["output_length"],
               "tokens": 0, "rid": None, "ttft_ms": None, "success": False}
        body = {"model": "cpu-synthetic", "messages": [{"role": "user", "content": item["prompt"]}],
                "max_tokens": item["output_length"], "stream": True}
        first, finish, done = None, False, False
        try:
            async with session.post(endpoint, json=body) as response:
                response.raise_for_status()
                async for line in response.content:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        continue
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError(chunk)
                    rec["rid"] = int(chunk["id"].split("-")[-1])
                    choice = chunk["choices"][0]
                    if choice["delta"].get("content"):
                        first = first if first is not None else time.perf_counter()
                        rec["tokens"] += 1
                    if choice.get("finish_reason"):
                        finish = True
            rec["success"] = first is not None and finish and done and rec["tokens"] == item["output_length"]
            if not rec["success"]:
                rec["error"] = f"incomplete stream: finish={finish}, done={done}, tokens={rec['tokens']}"
        except Exception as exc:
            rec["error"] = repr(exc)
        end = time.perf_counter()
        rec["e2e_ms"] = (end - sent) * 1000
        if first is not None:
            rec["ttft_ms"] = (first - sent) * 1000
            rec["scheduled_ttft_ms"] = (first - started - item["at"]) * 1000
        return rec

    rows = await asyncio.gather(*(one(item) for item in trace))
    return rows, time.perf_counter() - started


async def run_case(config, policy, scenario, load, seed, out, count=None, replicas=None):
    out.mkdir(parents=True, exist_ok=True)
    replica_count = replicas or config["replicas"]
    rate = config["loads_rps"][load]
    n = count or config["requests"]
    trace = make_trace(scenario, seed, n, rate)
    encoded_trace = json.dumps(trace, sort_keys=True).encode()
    trace_hash = hashlib.sha256(encoded_trace).hexdigest()
    # Trace files shared across policies; exact hash also written per case.
    trace_path = out.parent.parent / "traces" / f"{scenario}-{load}-{seed}-{n}.json"
    if not trace_path.exists():
        save_json(trace_path, trace)
    tokenizer_path = ROOT / "models" / "cpu-synthetic-tokenizer"
    engines = [RemoteCpuEngine(i, config["backend"]) for i in range(replica_count)]
    proc, monitor_task = None, None
    log = (out / "router.log").open("w")
    client_config = out / "clients.json"
    resources = []
    try:
        urls = [await engine.start() for engine in engines]
        save_json(client_config, urls)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = os.environ.copy()
        env.update({"TOKIO_WORKER_THREADS": str(config["router_workers"]),
                    "RUST_LOG": "warn", "LOG_LEVEL": "warn",
                    "CUDA_VISIBLE_DEVICES": "", "ASCEND_RT_VISIBLE_DEVICES": "",
                    "HIP_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
        binary = ROOT / "results" / "cpu" / "bins" / policy
        command = [str(binary), "--hostname", "127.0.0.1", "--port", str(port),
                   "--client-config", str(client_config), "--model-name", "cpu-synthetic",
                   "--tokenizer-name", str(tokenizer_path), "--use-tokenizer",
                   "--max-concurrent-requests", "4096", "--validation-workers", "2"]
        save_json(out / "run.json", {"command": command, "policy": policy, "scenario": scenario,
                                    "load": load, "seed": seed, "replicas": replica_count,
                                    "requests": n, "trace_sha256": trace_hash})
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        process = psutil.Process(proc.pid)
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=90)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=timeout) as session:
            for _ in range(300):
                if proc.poll() is not None:
                    raise RuntimeError(f"router exited {proc.returncode}; see {out / 'router.log'}")
                try:
                    await asyncio.gather(*(engine.refresh(session) for engine in engines))
                    async with session.get(base + "/health") as response:
                        ready = response.status == 200
                    if ready and all(engine.subscribers for engine in engines):
                        break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.05)
            else:
                raise TimeoutError("router or SSE not ready")
            idle_before = sum(process.cpu_times()[:2])
            idle_start = time.perf_counter()
            await asyncio.sleep(0.2)
            idle_cpu = (sum(process.cpu_times()[:2]) - idle_before) / (time.perf_counter() - idle_start) * 100
            warmup_trace = make_trace(scenario, seed + 10000, config["warmup_requests"], 200)
            warmup, _ = await replay(session, base + "/v1/chat/completions", warmup_trace)
            if not all(r["success"] for r in warmup):
                save_json(out / "warmup-errors.json", [r for r in warmup if not r["success"]])
                raise RuntimeError("warmup failed")
            await asyncio.gather(*(engine.refresh(session) for engine in engines))
            for engine in engines:
                engine.assert_drained()
                await engine.reset_remote(session)
            async with session.get(base + "/metrics") as response:
                before_metrics = await response.text()
            before_cpu = sum(process.cpu_times()[:2])
            before_runner_cpu = time.process_time()
            monitor_start = time.perf_counter()

            async def monitor():
                while True:
                    resources.append({"at": time.perf_counter() - monitor_start,
                                      "router_rss_bytes": process.memory_info().rss,
                                      "runner_rss_bytes": psutil.Process().memory_info().rss,
                                      "backends_rss_bytes": sum(psutil.Process(e.process.pid).memory_info().rss for e in engines)})
                    await asyncio.sleep(0.1)

            monitor_task = asyncio.create_task(monitor())
            rows, duration = await replay(session, base + "/v1/chat/completions", trace)
            after_cpu = sum(process.cpu_times()[:2])
            runner_cpu = time.process_time() - before_runner_cpu
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
            async with session.get(base + "/metrics") as response:
                after_metrics = await response.text()
            (out / "metrics-before.prom").write_text(before_metrics)
            (out / "metrics-after.prom").write_text(after_metrics)
            await asyncio.gather(*(engine.refresh(session) for engine in engines))
            for engine in engines:
                if engine.failure:
                    raise RuntimeError(engine.failure)
                engine.assert_drained()
            backend = [r for engine in engines for r in engine.records]
            by_rid = {r["rid"]: r for r in backend}
            for row in rows:
                row.update({k: v for k, v in by_rid.get(row["rid"], {}).items() if k != "rid"})
            for name in ["active_requests", "waiting_requests", "pending_prefill_tokens", "context_tokens"]:
                values = metrics_values(after_metrics, "blitz_replica_" + name)
                # A never-used replica has no completion-step gauge yet.
                observed_replicas = sum(engine.step_id > 0 for engine in engines)
                if len(values) != observed_replicas or any(v != 0 for v in values):
                    raise AssertionError(f"router did not drain: {name}={values}")
            passed = [r for r in rows if r["success"]]
            dispatch = [sum(r["engine"] == i for r in backend) for i in range(replica_count)]
            summary = {"policy": policy, "scenario": scenario, "load": load, "seed": seed,
                       "replicas": replica_count, "requests": n, "successes": len(passed),
                       "backend_completions": len(backend), "duration_s": duration,
                       "total_input_tokens": sum(r["input_tokens"] for r in backend),
                       "total_cached_tokens": sum(r["cached_tokens"] for r in backend),
                       "total_output_tokens": sum(r["tokens"] for r in passed),
                       "offered_rps": rate, "completed_rps": len(passed) / duration,
                       "output_tok_s": sum(r["tokens"] for r in passed) / duration,
                       "cache_hit_ratio": sum(r["cached_tokens"] for r in backend) / sum(r["input_tokens"] for r in backend),
                       "evicted_blocks": sum(e.evictions for e in engines),
                       "dispatch": dispatch,
                       "dispatch_cv": statistics.pstdev(dispatch) / statistics.mean(dispatch),
                       "router_cpu_seconds": after_cpu - before_cpu,
                       "router_idle_cpu_percent_one_core": idle_cpu,
                       "router_cpu_percent_one_core": (after_cpu - before_cpu) / duration * 100,
                       "runner_cpu_seconds": runner_cpu,
                       "router_peak_rss_mib": max(r["router_rss_bytes"] for r in resources) / 2**20,
                       "runner_peak_rss_mib": max(r["runner_rss_bytes"] for r in resources) / 2**20,
                       "backends_peak_rss_mib": max(r["backends_rss_bytes"] for r in resources) / 2**20,
                       "backend_processes": replica_count,
                       "trace_sha256": trace_hash, "drained": True,
                       "max_backend_running": max(e.high_water_active for e in engines),
                       "max_backend_waiting": max(e.high_water_waiting for e in engines)}
            for name in ["ttft_ms", "scheduled_ttft_ms", "e2e_ms", "dispatch_lag_ms"]:
                values = [r[name] for r in passed]
                summary[name + "_mean"] = statistics.mean(values)
                for q in [50, 95, 99]:
                    summary[name + f"_p{q}"] = percentile(values, q)
            # Remove warmup observations from cumulative policy histograms.
            measured = policy_metrics(after_metrics, before_metrics)
            measured_count = measured["schedule_count"]
            summary.update(measured)
            save_json(out / "summary.json", summary)
            (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            save_json(out / "resources.json", resources)
            if len(passed) != n or len(backend) != n or measured_count != n:
                raise AssertionError(f"request accounting mismatch: {len(passed)}/{len(backend)}/{measured_count}/{n}")
            return summary
    finally:
        if monitor_task:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        await asyncio.gather(*(engine.close() for engine in engines))
        log.close()


async def main_async(args):
    config = json.loads(args.config.read_text())
    make_tokenizer(ROOT / "models" / "cpu-synthetic-tokenizer")
    save_json(args.output / "config.json", config)
    if args.mode == "smoke":
        cases = [("lmetric-q", "shared", "medium", 17, 100, 2)]
    elif args.mode == "scaling":
        cases = [("lmetric-q", "mixed", "high", 17, 300, n) for n in [2, 4, 8, 16]]
    elif args.mode == "pilot":
        cases = [(p, s, "high", 17, 100, 4) for s in config["scenarios"] for p in config["policies"]]
    else:
        cases = []
        for scenario in config["scenarios"]:
            for load in config["loads_rps"]:
                for seed in config["seeds"]:
                    policies = config["policies"].copy()
                    random.Random(seed + len(scenario) + len(load)).shuffle(policies)
                    cases += [(p, scenario, load, seed, None, None) for p in policies]
    for i, (policy, scenario, load, seed, count, replicas) in enumerate(cases):
        key = f"{scenario}-{load}-{seed}-{replicas or config['replicas']}r-{policy}"
        case_dir = args.output / "cases" / key
        if (case_dir / "summary.json").exists() and args.resume:
            print(f"[{i+1}/{len(cases)}] resume {key}", flush=True)
            continue
        summary = await run_case(config, policy, scenario, load, seed, case_dir, count, replicas)
        print(f"[{i+1}/{len(cases)}] {key} success={summary['successes']}/{summary['requests']} "
              f"TTFT.p95={summary['ttft_ms_p95']:.1f}ms hit={summary['cache_hit_ratio']:.1%}", flush=True)
    summaries = [json.loads(path.read_text()) for path in sorted((args.output / "cases").glob("*/summary.json"))]
    save_json(args.output / "summaries.json", summaries)
    flat = [{k: v for k, v in s.items() if not isinstance(v, (dict, list))} for s in summaries]
    with (args.output / "summaries.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "pilot", "full", "scaling"], default="smoke")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cpu_benchmark.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(main_async(parser.parse_args()))
