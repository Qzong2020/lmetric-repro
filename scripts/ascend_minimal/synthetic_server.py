"""Loopback-only test server. Events are explicitly SYNTHETIC, no engine/model."""
import argparse
import asyncio
import json
from types import SimpleNamespace as NS
import uvicorn
from fastapi import FastAPI,HTTPException
from support import events,module
from test_events import ipc_roundtrip,make_fixtures

app=FastAPI()
hub=events.EventHub()
app.state.engine_client=NS(lmetric_hub=hub)
app.include_router(module('lmetric_test_http','vllm/entrypoints/openai/serving_metrics.py').router)

@app.get('/test/health')
async def health():return {'synthetic':True,'active':hub.active}

@app.post('/test/run')
async def run():
    for _ in range(500):
        if hub.active:break
        await asyncio.sleep(.01)
    else:raise HTTPException(409,'No SSE subscriber')
    hub.register('engine-internal-uuid','101')
    packets=await asyncio.to_thread(ipc_roundtrip,make_fixtures(async_mode=True))
    for packet in packets:hub.publish(packet)
    return {'synthetic':True,'packets':len(packets),'real_zmq_subprocess':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args();uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='warning')
