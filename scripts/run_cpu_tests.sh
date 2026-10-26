#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
mkdir -p results/cpu/validation results/cpu/bins
python3 -m unittest discover -s scripts -p test_cpu_backend.py -v > results/cpu/validation/python-tests.log 2>&1
cd blitz-router
# Two existing request-sim tests require private external dataset/tokenizer paths.
# Preserve the unfiltered run log from the initial audit; the CPU harness uses
# its own controlled tokenizer and trace instead of these external fixtures.
cargo +1.88.0 test --workspace --lib --features lmetric-q --locked --jobs 6 \
    -- --skip test_gen_string_latency_scaling --skip test_inflate_latency \
    > ../results/cpu/validation/workspace-filtered-tests.log 2>&1
for policy in lmetric-q round-robin-q join-shortest-weight-q most-hit-q; do
    cargo +1.88.0 build -p router --release --no-default-features \
        --features "vllm-backend,radixtree-blockhash,default-hash-algo,determinent-schedule,$policy" \
        --locked --jobs 6 > "../results/cpu/validation/build-$policy.log" 2>&1
    cp target/release/router "../results/cpu/bins/$policy"
done
cd "$task_root"
python3 scripts/cpu_manifest.py
python3 scripts/cpu_benchmark.py --mode smoke --output results/cpu/smoke
python3 scripts/cpu_benchmark.py --mode full --output results/cpu/full
python3 scripts/cpu_benchmark.py --mode scaling --output results/cpu/scaling
python3 scripts/report_cpu_benchmark.py
python3 scripts/cpu_manifest.py
