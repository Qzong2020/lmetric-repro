# Ascend request/execution protocol prototype

`request-v1` is an experimental, versioned extension for vLLM-Ascend 0.23.0.
It reports completed-schedule prompt work and per-request outputs, not KV cache mutations.
The engine patch lives in the sibling `engines/vllm` checkout; installed packages are unchanged.

The first target uses one frontend and one KV group, with Router flag
`--kvcache-block-size 128`. The default Router block size remains 16.
Compute-only Prefill work plus the first-output actual-minus-predicted hit
correction closes the new protocol ledger, including first-output finish.
Legacy yaullm ledger semantics are preserved.

Run `python ../scripts/run_ascend_minimal_tests.py` from this repository.
It starts a loopback synthetic SSE server and runs the real Rust receiver and
completion loop. No model or NPU is initialized. The network test is ignored in
ordinary cargo test and enabled explicitly by this driver.

The receiver stops on invalid sequences or block-size mismatch. The original
`request-v1` path above is compute-only. The opt-in `request-kv-v1` extension
now maintains independent physical mappings, prefix trees and request ledgers
for each registered backend. A failed journal session is latched invalid,
excluded during scoring and rechecked under the admission lock. Its idle
sender exits when the completion receiver closes. Stream recovery and safe
in-flight cancellation remain unsupported; restart affected sessions.

The journal capability gate permits `lmetric-q` and the existing
`join-shortest-weight-q` comparison baseline with `radixtree-blockhash`, with
`simulator` disabled. Other policy builds remain rejected. Both consume the
same journal and retain sequence/epoch validation. The baseline minimizes
`3 * waiting + bs`; LMetric minimizes `(max(0, prefill) + uncached_tokens) *
(bs + 1)`. The engine still requires `max_num_seqs=1` per instance.

Dual-instance validation and any bounded preliminary timing are documented in
[the dual-NPU report](../../docs/dual_npu_validation.md). Production debug
records `CANDIDATE` and `KV_STATE` support score/state reconciliation. Histogram
`blitz_kv_prepare_duration_seconds` covers journal staging/tree rebuilding;
`blitz_step_apply_duration_seconds` covers completion application excluding
receive wait and the final debug snapshot. These instrumented host durations
do not establish isolated incremental event-collection overhead.

See the [full implementation/validation report](../../docs/ascend_minimal_events.md)
and [machine-readable results](../../results/ascend_minimal_events/tests.json).
