# CPU checks against arXiv:2603.15202v3

The vLLM baseline now follows Figure 6(a): `4 * queued + running`.
`LMetric.bs` includes queued and running requests, so the DSL expression is
`3 * waiting + bs`. The old `4 * waiting + bs` over-weighted queued requests.
LMetric's Figure 17 expression remains unchanged; both prefill work and batch
size include the candidate hypothetically before comparing replicas.

`scheduler/policies/paper_tests.rs` calls the actual generated Rust policy
implementations and compares their choices with independently written paper
expressions. It also checks admission bookkeeping, prefix-cache matching, and
zero correction when actual cache hits agree with the decision snapshot.

The ignored real-data test requires `PAPER_CPU_RESULTS` to point at prepared
`replay-A/tokenized.jsonl` and `replay-B/tokenized.jsonl` files. It is invoked
explicitly by the surrounding workspace's `scripts/run_paper_cpu_tests.sh`.
Those inputs come from the actual request-sim inflater and a pinned real Qwen2
tokenizer; the snapshots used to exercise the policy are controlled fixtures,
not engine telemetry. No inference latency or throughput is reproduced here.

Full results are in the surrounding workspace's
`notes/paper_cpu_test_report.md`. The earlier synthetic CPU performance report
used the old baseline and remains an engineering preflight artifact.
