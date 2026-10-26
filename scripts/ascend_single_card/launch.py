"""Observation-only process wrapper; no synthetic scheduler or model outputs."""
import functools, inspect, json, os, runpy, sys
from pathlib import Path
import vllm, vllm_ascend
import vllm.v1.engine.core as core

def emit(kind, **data):
 print('VALIDATION_'+kind+' '+json.dumps(dict(pid=os.getpid(), **data),default=repr),flush=True)
emit('IMPORT',vllm=vllm.__file__,ascend=vllm_ascend.__file__,core=core.__file__,events=os.getenv('VLLM_LMETRIC_EVENTS'))
original_init=core.EngineCore.__init__
@functools.wraps(original_init)
def observed_init(self,*args,**kwargs):
 original_init(self,*args,**kwargs)
 s=self.scheduler;c=self.vllm_config;kv=s.kv_cache_config
 from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
 sizes=resolve_kv_cache_block_sizes(kv,c)
 emit('RUNTIME',scheduler=type(s).__module__+'.'+type(s).__qualname__,mro=[x.__module__+'.'+x.__qualname__ for x in type(s).__mro__],scheduler_source=inspect.getsourcefile(type(s)),engine_step_fn=self.step_fn.__name__,async_scheduling=self.async_scheduling,enforce_eager=c.model_config.enforce_eager,scheduler_block_size=sizes[0],hash_block_size=sizes[1],kv_groups=len(kv.kv_cache_groups),kv_specs=[repr(g.kv_cache_spec) for g in kv.kv_cache_groups],num_blocks=kv.num_blocks,cache_num_gpu_blocks=c.cache_config.num_gpu_blocks,available_memory=self.available_gpu_memory_for_kv_cache,batch_queue_size=self.batch_queue_size,batch_queue_length=None if self.batch_queue is None else len(self.batch_queue),parallel=repr(c.parallel_config),speculative=repr(c.speculative_config),kv_transfer=repr(c.kv_transfer_config),ec_transfer=repr(c.ec_transfer_config),lora=repr(c.lora_config))
core.EngineCore.__init__=observed_init
for name in ['step','step_with_batch_queue']:
 original=getattr(core.EngineCore,name)
 def observed(self,*args,_original=original,_name=name,**kwargs):
  result=_original(self,*args,**kwargs)
  emit('STEP',path=_name,queue_length=None if self.batch_queue is None else len(self.batch_queue),running=len(self.scheduler.running),waiting=len(self.scheduler.waiting),requests=list(self.scheduler.requests),seq=getattr(getattr(self,'lmetric_tracker',None),'seq',None))
  return result
 observed.__name__=name
 setattr(core.EngineCore,name,observed)
if os.getenv('VLLM_LMETRIC_EVENTS') == '1':
 from vllm.v1.engine.lmetric_events import EventHub
 register_original=EventHub.register
 def observed_register(self,internal_id,external_id):
  result=register_original(self,internal_id,external_id)
  emit('IDENTITY',internal_id=internal_id,external_id=external_id)
  return result
 EventHub.register=observed_register
 frames_original=EventHub.frames
 async def observed_frames(self):
  async for frame in frames_original(self):
   yield frame
   emit('HUB',queue_size=self.queue.qsize(),active=self.active,failed=self.failed,identities=len(self.identities))
 EventHub.frames=observed_frames
if __name__=='__main__':
 runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
