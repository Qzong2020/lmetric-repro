"""CPU-only loading of exact selected definitions, without engine startup.

Unused schema dependency types are Any. This tests the real selected structs
and methods, not importability/initialization of the complete vLLM package.
"""
import ast
import asyncio
import contextlib
import enum
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Mapping
import msgspec

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT/'engines/vllm'


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ENGINE/relative)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


events = module('lmetric_cpu_events', 'vllm/v1/engine/lmetric_events.py')
output_types = module('lmetric_cpu_sched_output', 'vllm/v1/core/sched/output.py')


def selected(relative, names, scope):
    tree = ast.parse((ENGINE/relative).read_text())
    nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ENGINE/relative), 'exec'), scope)


schema = dict(msgspec=msgspec, enum=enum, time=time, Any=Any, Mapping=Mapping,
              torch=NS(Tensor=Any), np=NS(ndarray=Any),
              FINISH_REASON_STRINGS=('stop','length','abort'))
for name in ['LogprobsLists','LogprobsTensors','EngineCoreEvent','PrefillStats',
             'SchedulerStats','UtilityOutput']:
    schema[name] = Any
selected('vllm/v1/engine/__init__.py', ['FinishReason','EngineCoreOutput','EngineCoreOutputs'], schema)
CoreOutput = schema['EngineCoreOutput']
CoreOutputs = schema['EngineCoreOutputs']

# Execute actual EngineCore dispatch methods against a synthetic scheduler and
# executor. No claim that a real EngineCore/Scheduler/NPU has been initialized.
tree = ast.parse((ENGINE/'vllm/v1/engine/core.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EngineCore')
names = ['_schedule_with_lmetric','_complete_with_lmetric','step','step_with_batch_queue']
methods = [n for n in cls.body if getattr(n,'name',None) in names]
new_cls = ast.ClassDef(name='CoreHarness',bases=[],keywords=[],body=methods,decorator_list=[])
scope = dict(time=time, SchedulerOutput=output_types.SchedulerOutput,
             EngineCoreOutputs=CoreOutputs)
exec(compile(ast.fix_missing_locations(ast.Module(body=[new_cls],type_ignores=[])),
             str(ENGINE/'vllm/v1/engine/core.py'),'exec'),scope)
CoreHarness = scope['CoreHarness']


def scheduler(async_mode=False):
    req = NS(client_index=0, mm_features=[], prompt_token_ids=list(range(64)),
             num_prompt_tokens=64, prefill_stats=NS(num_local_cached_tokens=32))
    return NS(kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=object())]),
              requests={'engine-internal-uuid':req})


def schedule_output(prev, count, new=False):
    rid = 'engine-internal-uuid'
    cached = output_types.CachedRequestData.make_empty()
    new_reqs = []
    if new:
        new_reqs = [output_types.NewRequestData(rid,list(range(64)),[],None,None,
                                               ([1],),prev,None)]
    else:
        cached.req_ids=[rid]; cached.num_computed_tokens=[prev]
    return output_types.SchedulerOutput(new_reqs,cached,{rid:count},count,{}, {},
                                        [0],set(),[],preempted_req_ids=set())


def make_fixtures(async_mode=False):
    s = scheduler()
    s.requests['engine-internal-uuid'].num_prompt_tokens=384
    s.requests['engine-internal-uuid'].prompt_token_ids=list(range(384))
    s.requests['engine-internal-uuid'].prefill_stats.num_local_cached_tokens=128
    tracker = events.RequestEvents(s,128,128,async_mode,2 if async_mode else 1)
    result=[]
    for i,(prev,count,tokens,finished) in enumerate([
        (128,128,[],False),(256,128,[9],False),(384,1,[10],False),(385,1,[11],True)]):
        out=schedule_output(prev,count,new=i==0)
        if out.scheduled_new_reqs:
            out.scheduled_new_reqs[0].prompt_token_ids=list(range(384))
        tracker.capture(s,out,time.monotonic())
        eco={}
        if tokens:
            eco[0]=CoreOutputs(outputs=[CoreOutput(request_id='engine-internal-uuid',
                new_token_ids=tokens,finish_reason=schema['FinishReason'].LENGTH if finished else None)])
        result.append(tracker.complete(out,eco,CoreOutputs)[0].lmetric_event)
    return result
