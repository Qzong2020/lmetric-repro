# yaullm

`yaullm` stands for **yet-another-micro-vllm** modification. This repository is a
vLLM fork used as the backend engine in the **x-metric** artifact evaluation.

In the AE workflow, the cluster-level pipeline is:

```text
request-sim -> blitz-router -> yaullm
```

`yaullm` extends vLLM with richer metric reporting over SSE. The stream gives
the router visibility into each engine replica's internal state, so routing
decisions can account for what the engine actually scheduled and how its prefix
cache changed over time.

## Metric SSE

`yaullm` adds a metric Server-Sent Events endpoint to the OpenAI-compatible
server:

```text
GET /v1/metrics
Content-Type: text/event-stream
```

The metric SSE stream is designed for two purposes:

1. Expose the inner batch scheduler behavior of the engine, including step
   boundaries, per-step latency, request progress, and request state changes.
2. Expose the inner prefix-cache management behavior of the engine, including
   cache hits, newly materialized blocks, currently used blocks, evictions, and
   request abort/preemption events.

This stream is separate from normal token streaming. The router consumes it to
observe what each engine replica actually did after a routing decision.

## Configuration

Metric reporting is enabled by default:

```bash
VLLM_REPORT_METRICS=1
```

Set `VLLM_REPORT_METRICS=0` to disable metric reporting and skip the extra
bookkeeping used only for this stream.

`VLLM_SSE_MODE` controls the amount of serialized data:

- `full` (default): scheduler progress plus prefix-cache block updates.
- `minimal`: scheduler progress only.

Example:

```bash
VLLM_USE_V1=1 VLLM_REPORT_METRICS=1 VLLM_SSE_MODE=full \
  vllm serve /path/to/model --port 50180
```

Read the stream:

```bash
curl -N http://localhost:50180/v1/metrics
```

For the router-side SSE parser and consumer, see
[`router/src/engine/colocation.rs`](https://github.com/blitz-serving/blitz-router/blob/osdi26-ae-workflow/router/src/engine/colocation.rs)
in `blitz-router`.
