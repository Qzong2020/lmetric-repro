#!/usr/bin/env python3
"""Execute selected ORIGINAL vLLM v0.23.0 methods with SIMULATED storage.
AST extraction avoids importing vLLM/NPU. This is NOT a real engine test.
"""
import ast, hashlib, json, importlib.util
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ascend_metric_audit'
V=Path('/vllm-workspace/vllm/vllm')
def original_nodes(path, names):
    return [n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
def execute(nodes,ns):
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[])),'<original-vllm-source>','exec'),ns)
class Event(SimpleNamespace): pass
class Block(SimpleNamespace):
    def reset_hash(self): self.block_hash=None
class Queue:
    def __init__(self): self.items=[]
    @property
    def num_free_blocks(self): return len(self.items)
    def popleft_n(self,n): r=self.items[:n]; del self.items[:n]; return r
    def append_n(self,items): self.items.extend(items)
    def prepend_n(self,items): self.items[:0]=items

def main():
    pool_path=V/'v1/core/block_pool.py'; utils_path=V/'v1/core/kv_cache_utils.py'
    ns={'KVCacheBlock':Block,'BlockHashWithGroupId':lambda x:x,'BlockHash':lambda x:x,
        'envs':SimpleNamespace(VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=False),
        'BlockStored':Event,'BlockRemoved':Event,'AllBlocksCleared':Event,'MEDIUM_GPU':'GPU',
        'logger':SimpleNamespace(info=lambda *a:None,warning=lambda *a:None)}
    execute(original_nodes(utils_path,{'make_block_hash_with_group_id','get_block_hash','get_group_id','maybe_convert_block_hash'}),ns)
    nodes=original_nodes(pool_path,{'BlockHashToBlockMap','BlockPool'})
    keep={'get_new_blocks','_maybe_evict_cached_block','free_blocks','cache_full_blocks','take_events','get_num_free_blocks','reset_prefix_cache'}
    nodes[1].body=[n for n in nodes[1].body if isinstance(n,ast.FunctionDef) and n.name in keep]
    execute(nodes,ns)
    p=ns['BlockPool'].__new__(ns['BlockPool'])
    p.cached_block_hash_to_block=ns['BlockHashToBlockMap'](); p.metrics_collector=None
    p.enable_caching=True; p.enable_kv_cache_events=False; p.hash_block_size=16
    p.free_block_queue=Queue(); p.kv_event_queue=[]
    b=Block(block_id=7,ref_cnt=0,block_hash=None,is_null=False)
    p.free_block_queue.append_n([b])
    assert p.get_new_blocks(1)==[b] and b.block_hash is None
    results=[{'case':'allocation_is_not_cache_commit','passed':True}]
    req=SimpleNamespace(request_id='SIMULATED',block_hashes=[b'a'*32],all_token_ids=list(range(16)))
    p.cache_full_blocks(req,[b],0,1,16,0)
    key=b.block_hash
    assert key is not None and p.cached_block_hash_to_block.get_one_block(key) is b
    p.free_blocks([b]); assert b.ref_cnt==0 and b.block_hash==key
    results.append({'case':'free_zero_ref_keeps_cache_mapping','passed':True})
    p.enable_kv_cache_events=True
    assert p.get_new_blocks(1)==[b] and b.block_hash is None
    assert len(p.kv_event_queue)==1 and p.kv_event_queue[0].group_idx==0
    assert not hasattr(p.kv_event_queue[0],'block_id')
    results.append({'case':'reuse_evicts_mapping_but_native_event_has_no_physical_id','passed':True})
    events=p.take_events(); assert len(events)==1 and p.take_events()==[]
    results.append({'case':'take_events_is_destructive','passed':True})
    # Duplicate hash -> two physical blocks. Removing one retains the other.
    b1=Block(block_id=8,ref_cnt=0,block_hash=key,is_null=False)
    b2=Block(block_id=9,ref_cnt=0,block_hash=key,is_null=False)
    p.cached_block_hash_to_block.insert(key,b1); p.cached_block_hash_to_block.insert(key,b2)
    assert p._maybe_evict_cached_block(b1)
    assert p.cached_block_hash_to_block.get_one_block(key) is b2
    assert not p._maybe_evict_cached_block(b1)
    results.append({'case':'duplicate_content_removal_preserves_other_physical_copy','passed':True})
    p.num_gpu_blocks=3; p.blocks=[b,b1,b2]
    p.free_block_queue.items=[b1,b2]
    assert p.reset_prefix_cache()
    assert all(x.block_hash is None for x in p.blocks)
    assert len(p.cached_block_hash_to_block)==0
    results.append({'case':'reset_is_global_clear','passed':True})
    ascend_utils=Path('/vllm-workspace/vllm-ascend/vllm_ascend/utils.py')
    execute(original_nodes(ascend_utils,{'refresh_block_size'}),ns)
    config=SimpleNamespace(cache_config=SimpleNamespace(block_size=32,enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='qwen2'),is_hybrid=False))
    ns['refresh_block_size'](config)
    assert config.cache_config.block_size==128
    results.append({'case':'ascend_refresh_overrides_32_to_128_with_prefix_cache','passed':True})
    # Existing author serializer tests, unchanged, output redirected into this audit.
    spec=importlib.util.spec_from_file_location('prior_audit',ROOT/'scripts/audit_yaullm_field_contract.py')
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    old.OUT=OUT/'author_serializer'; old.main()
    report={'scope':'ORIGINAL selected source methods with SIMULATED blocks, queue, request and event containers; NOT Scheduler or NPU execution',
            'npu_used':False,'vllm_imported':False,'cases':results,'author_serializer_cases':3,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pool_path,utils_path,ascend_utils]}}
    (OUT/'target_probes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Target original-method CPU probes:',len(results),'passed.')
if __name__=='__main__':main()
