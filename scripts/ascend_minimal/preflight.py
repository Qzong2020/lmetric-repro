"""Focused CPU preflight; configuration intent is not a running-engine fact."""
import ast
import glob
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from support import ROOT,ENGINE

out=ROOT/'results/ascend_minimal_events'
source=Path('/vllm-workspace/vllm-ascend/vllm_ascend/utils.py')
tree=ast.parse(source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_block_size')
class Log:
    def __getattr__(self,_):return lambda *a,**kw:None
scope={'logger':Log(),'get_ascend_config':lambda:NS(xlite_graph_config=NS(enabled=False)),
       'vllm_version_is':lambda _:True}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
block_probes=[]
for requested in [16,32,128]:
    for prefix,chunk in [(True,False),(False,True),(True,True)]:
        config=NS(cache_config=NS(block_size=requested,enable_prefix_caching=prefix),
                  scheduler_config=NS(enable_chunked_prefill=chunk),
                  model_config=NS(hf_config=NS(model_type='qwen2'),is_hybrid=False))
        scope['refresh_block_size'](config)
        assert config.cache_config.block_size==128
        block_probes.append(dict(requested=requested,prefix_cache=prefix,chunked_prefill=chunk,
                                 effective_in_cpu_method=config.cache_config.block_size))
report={'npu_executed':False,'full_engine_initialized':False,
        'devices':glob.glob('/dev/davinci*'),'npu_smi':'15 second timeout during initial read-only probe; no retry',
        'block_size_probes':block_probes,'block_size_source':str(source),
        'block_size_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'actual_scheduler':None,'actual_kv_groups':None,'actual_engine_core_path':None,
        'reason':'No initialized target model engine; runtime report remains pending',
        'modules':{m:importlib.util.find_spec(m) is not None for m in ['msgspec','zmq','fastapi','httpx','uvicorn','torch_npu']}}
processes=subprocess.run(['ps','-eo','pid,comm,args'],capture_output=True,text=True,check=True).stdout.splitlines()
report['serving_processes']=[s for s in processes if ('vllm serve' in s or 'vllm.entrypoints.openai.api_server' in s)
                             and 'bash -' not in s and 'preflight.py' not in s]
(out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
