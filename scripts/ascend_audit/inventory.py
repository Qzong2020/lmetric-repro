#!/usr/bin/env python3
"""Read-only fixed-source inventory for the Ascend metric audit."""
import ast, datetime, hashlib, importlib.metadata as metadata, json, platform, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ascend_metric_audit'
REPOS={'router':ROOT/'blitz-router','vllm':Path('/vllm-workspace/vllm'),'ascend':Path('/vllm-workspace/vllm-ascend')}
FILES={
'router':['router/src/engine/client.rs','router/src/engine/colocation.rs','router/src/engine/vllm_http.rs','router/src/scheduler/kvcache.rs','router/src/scheduler/state.rs','router/src/scheduler/policies/dsl_runtime.rs','router/src/scheduler/policies/mod.rs','router/src/scheduler/policies/lmetric.rs','router/src/scheduler/simulator/mirror.rs','router/src/scheduler/simulator/mod.rs','radixtree/src/block_hash.rs'],
'vllm':['vllm/v1/core/sched/scheduler.py','vllm/v1/core/sched/async_scheduler.py','vllm/v1/core/sched/output.py','vllm/v1/core/kv_cache_manager.py','vllm/v1/core/kv_cache_coordinator.py','vllm/v1/core/single_type_kv_cache_manager.py','vllm/v1/core/block_pool.py','vllm/v1/core/kv_cache_utils.py','vllm/v1/engine/core.py','vllm/v1/engine/core_client.py','vllm/v1/engine/__init__.py','vllm/v1/engine/async_llm.py','vllm/v1/engine/input_processor.py','vllm/v1/request.py','vllm/v1/metrics/stats.py','vllm/distributed/kv_events.py','vllm/config/cache.py','vllm/v1/engine/output_processor.py','vllm/entrypoints/openai/api_server.py','vllm/config/scheduler.py','vllm/config/vllm.py','vllm/entrypoints/openai/completion/serving.py','vllm/entrypoints/serve/instrumentator/metrics.py'],
'ascend':['vllm_ascend/attention/attention_v1.py','vllm_ascend/platform.py','vllm_ascend/utils.py','vllm_ascend/ascend_config.py','vllm_ascend/patch/platform/__init__.py','vllm_ascend/patch/platform/patch_balance_schedule.py','vllm_ascend/patch/platform/patch_kv_cache_coordinator.py','vllm_ascend/patch/platform/patch_async_swa_kv_lifetime.py','vllm_ascend/patch/platform/patch_kv_cache_utils.py','vllm_ascend/patch/platform/patch_use_v2_model_runner.py','vllm_ascend/core/recompute_scheduler.py','vllm_ascend/core/scheduler_dynamic_batch.py','vllm_ascend/core/scheduler_profiling_chunk.py','vllm_ascend/core/single_type_kv_cache_manager.py','vllm_ascend/worker/worker.py','vllm_ascend/worker/model_runner_v1.py']}
def git(path,*args):
    r=subprocess.run(['git','-C',str(path),*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return r.stdout.strip() if not r.returncode else {'error':r.stderr.strip()}
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    env={'date_client_timezone':'2026-10-24 Asia/Shanghai','captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'python':platform.python_version(),'machine':platform.machine(),'repos':{},'packages':{},
         'device_check':{'npu_smi_info_exit':187,'output':'DrvMngGetConsoleLogLevel failed. (ret=4)\ndcmi module initialize failed. ret is -8005','dev_davinci_glob':list(map(str,Path('/dev').glob('davinci*')))},
         'no_runtime_engine_or_model_loaded':True}
    for name,repo in REPOS.items():
        env['repos'][name]={'path':str(repo),'head':git(repo,'rev-parse','HEAD'),'branch':git(repo,'branch','--show-current'),
            'describe':git(repo,'describe','--always','--tags'),'status':git(repo,'status','--short','--branch'),
            'tracked_diff_sha256':hashlib.sha256(git(repo,'diff','HEAD').encode()).hexdigest()}
    for name in ['vllm','vllm-ascend','torch','torch-npu','triton-ascend','msgspec']:
        try:
            d=metadata.distribution(name)
            env['packages'][name]={'version':d.version,'metadata_path':str(d._path),'direct_url':d.read_text('direct_url.json')}
        except metadata.PackageNotFoundError: env['packages'][name]={'installed':False}
    sources={}
    for repo,files in FILES.items():
        for f in files:
            p=REPOS[repo]/f; s=p.read_text()
            item={'repo':repo,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'line_count':len(s.splitlines())}
            if p.suffix=='.py':
                item['symbols']={}
                for n in ast.parse(s).body:
                    if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                        item['symbols'][n.name]=n.lineno
                        if isinstance(n,ast.ClassDef):
                            item['symbols'].update({n.name+'.'+m.name:m.lineno for m in n.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef))})
            sources[repo+'/'+f]=item
    (OUT/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2)+'\n')
    (OUT/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    # Verify existing archive files against the existing manifest; do not regenerate it.
    manifest=json.loads((ROOT/'results/yaullm_audit/manifest.json').read_text())
    archive={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in manifest.items() if p.startswith('results/yaullm_audit/')}
    (OUT/'archive_integrity.json').write_text(json.dumps(archive,indent=2)+'\n')
    assert all(archive.values()),'Existing yaullm archive changed; investigate before using it.'
    print('Inventoried',len(sources),'fixed source files;',len(archive),'archive hashes matched.')
if __name__=='__main__':main()
