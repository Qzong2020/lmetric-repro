# Experimental LMetric request events

This local 0.23.0 patch is opt-in via `VLLM_LMETRIC_EVENTS=1`.
A single subscriber must connect to `GET /v1/metrics` before submitting a
single-prompt `n=1` completion with a decimal u64 `X-Request-Id`.
Internal randomized IDs are mapped explicitly, not parsed by suffix.

The `request-v1` envelope contains engine epoch, contiguous completion sequence,
block size, KV group count, host schedule-to-completion milliseconds, per-step
prompt compute work and incremental request outputs. The first real output is
PREFILL; later outputs are DECODE. Empty-token chunks retain their envelopes.
The EngineCore dispatch wrapper covers both step and step_with_batch_queue;
snapshots belong to the returned SchedulerOutput, including when a plugin
rewrites schedule. Asynchronous latency includes queue residence and may overlap
other events; it is not device time or a throughput tick.

Only one frontend, PP=DP=1, one KV group, text token IDs, no speculation/external
KV/EC/LoRA are supported. Stop strings and echo are rejected. Preempt/resume and
abort explicitly fail this prototype session. KV mutation export is absent
(`kv_events=false`). Queue loss or disconnect requires a fresh session; there
is no replay or automatic router replica removal.

CPU tests live in `/root/lmetric-repro/scripts/ascend_minimal`; run
`python /root/lmetric-repro/scripts/run_ascend_minimal_tests.py`.
They use selected original structs/methods and real loopback ZMQ/HTTP/SSE with
synthetic inputs, without full engine initialization or NPU execution.
The full report is `/root/lmetric-repro/docs/ascend_minimal_events.md`.
