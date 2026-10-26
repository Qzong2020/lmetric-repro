# SPDX-License-Identifier: Apache-2.0
"""Opt-in request/execution SSE. Does not expose KV cache events."""
from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

router = APIRouter()


@router.get("/v1/metrics")
async def metrics(request: Request):
    hub = getattr(request.app.state.engine_client, "lmetric_hub", None)
    if hub is None:
        raise HTTPException(404, "Set VLLM_LMETRIC_EVENTS=1 to enable request-v1")
    try:
        hub.subscribe()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return StreamingResponse(hub.frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
