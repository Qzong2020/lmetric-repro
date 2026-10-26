#!/usr/bin/env python3
"""Record only experiment sources, binary hashes, and non-secret resource data."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cpu"


def command(*args):
    return subprocess.check_output(args, cwd=ROOT / "blitz-router", text=True).strip()


def main():
    env = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "rustc": command("rustc", "+1.88.0", "--version"),
        "cargo": command("cargo", "+1.88.0", "--version"),
        "source_commit": command("git", "rev-parse", "HEAD"),
        "python_dependencies": {name: importlib.metadata.version(name) for name in ["aiohttp", "psutil", "tokenizers"]},
        "backend": "synthetic CPU only; no model weights or accelerator API",
        "features": "--no-default-features vllm-backend,radixtree-blockhash,default-hash-algo,determinent-schedule,<one-policy>",
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    paths = list((ROOT / "scripts").glob("*.py")) + [
        ROOT / "scripts/run_cpu_tests.sh", ROOT / "configs/cpu_benchmark.json",
        ROOT / "configs/cpu_requirements.txt", ROOT / "blitz-router/Cargo.lock",
        ROOT / "blitz-router/router/src/engine/colocation.rs",
        ROOT / "blitz-router/router/src/gateway/server.rs",
        ROOT / "blitz-router/router/src/scheduler/policies/lmetric.rs",
        ROOT / "blitz-router/router/src/scheduler/policies/policy_runner.rs",
    ] + list((OUT / "bins").glob("*-q"))
    paths += [ROOT / name for name in [
        "blitz-router/.cargo/config.toml", "blitz-router/router/Cargo.toml",
        "blitz-router/docs/cpu-testing.md", "models/cpu-synthetic-tokenizer/tokenizer.json",
        "results/cpu/environment.json", "results/cpu/full/summaries.json",
        "results/cpu/full/summaries.csv", "results/cpu/full/aggregates.csv",
        "results/cpu/scaling/summaries.json", "notes/cpu_test_report.md",
    ] if (ROOT / name).exists()]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    changes = command("git", "diff", "--", "router/src/engine/colocation.rs", "router/src/gateway/server.rs",
                      "router/src/scheduler/policies/lmetric.rs", "router/src/scheduler/policies/policy_runner.rs")
    (OUT / "validation/router-changes.patch").write_text(changes + "\n")


if __name__ == "__main__":
    main()
