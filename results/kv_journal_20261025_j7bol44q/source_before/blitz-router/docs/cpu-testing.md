# CPU scheduling experiments

The surrounding reproduction workspace provides `../scripts/cpu_benchmark.py`
(relative to the repository root) and `../configs/cpu_benchmark.json`.
They replay controlled synthetic prompts through the real router into a
CPU-only HTTP/SSE backend implementing finite prompt-prefix caching and
an independent synthetic prefill/decode service-time model. Each backend runs
in a separate spawned process from the request generator. No model weights,
GPU, NPU, or actual inference kernels are used. See `../notes/cpu_test_report.md`
in that workspace for results and limitations.

This experiment adds the following metrics to the router's `/metrics` endpoint:

- `blitz_policy_schedule_duration_seconds`: wall time inside each policy call,
  including observation-lock acquisition and admission bookkeeping, excluding
  command-queue waiting and the histogram update itself. Histogram buckets
  begin at 1 microsecond. Both Append and NextRequest paths are measured.
- `blitz_replica_active_requests`, `blitz_replica_waiting_requests`,
  `blitz_replica_pending_prefill_tokens`, `blitz_replica_context_tokens`:
  gauges labeled by `engine`, published after each engine-step metric update.
  They can lag admissions until the next step and are absent for replicas that
  have never produced a step. The benchmark checks all observed replicas
  return to zero after requests finish.

The `lmetric-q` scoring formula is unchanged. Added policy tests check empty
replica sets, cache affinity versus load, admission bookkeeping, and burst
distribution. The benchmark uses Rust 1.88.0 because the current local lockfile
contains proptest 1.12.0, which cannot compile with pinned Rust 1.86.0. It does
not update the project's toolchain or lockfile.

The existing idle work loop repeatedly polls `next_request` and yields. CPU
utilization therefore includes idle polling as well as actual policy work.
Interpret the measured policy-call latency separately from process CPU usage.
