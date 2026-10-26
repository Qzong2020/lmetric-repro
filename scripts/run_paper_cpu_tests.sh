#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
mkdir -p results/paper_cpu
python3 scripts/fetch_paper_cpu_inputs.py
python3 scripts/paper_trace_inventory.py > results/paper_cpu/trace_inventory.log 2>&1
python3 -m unittest discover -s scripts -p test_paper_metrics.py -v > results/paper_cpu/metrics_tests.log 2>&1
cd blitz-router
cargo +1.88.0 build -p request-sim --example paper_trace_check --locked --jobs 4 > ../results/paper_cpu/trace_build.log 2>&1
for letter in A B; do
    RAYON_NUM_THREADS=2 target/debug/examples/paper_trace_check \
        "$task_root/data/paper_cpu/samples/qwen_trace${letter}_blksz_16.jsonl" \
        "$task_root/models/paper-qwen2-7b-tokenizer" \
        "$task_root/results/paper_cpu/replay-$letter" 2000 \
        > "../results/paper_cpu/replay-$letter.log" 2>&1
done
PAPER_CPU_RESULTS="$task_root/results/paper_cpu/after" \
    cargo +1.88.0 test --workspace --lib --features lmetric-q --locked --jobs 4 \
        -- --skip test_gen_string_latency_scaling --skip test_inflate_latency \
        > ../results/paper_cpu/workspace_tests.log 2>&1
PAPER_CPU_RESULTS="$task_root/results/paper_cpu" \
    cargo +1.88.0 test -p router --lib --features lmetric-q --locked \
        real_trace_tokens_match_paper_oracles -- --ignored --nocapture \
        > ../results/paper_cpu/real_trace_policy_tests.log 2>&1
cd "$task_root"
python3 scripts/report_paper_cpu.py
