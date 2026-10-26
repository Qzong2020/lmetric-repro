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

The receiver stops on invalid sequences or block-size mismatch. Automatic
replica removal, stream recovery, abort ACKs and KV events are not implemented;
stop and restart the experimental session on failure. This is not a completed
cache-aware routing integration or a performance result.

See the [full implementation/validation report](../../docs/ascend_minimal_events.md)
and [machine-readable results](../../results/ascend_minimal_events/tests.json).
