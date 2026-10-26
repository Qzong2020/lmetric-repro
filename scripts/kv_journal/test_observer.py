"""CPU-only validation of the independent observation script."""
import ast,contextlib,io,json,os,unittest
from pathlib import Path
from types import SimpleNamespace as NS
class ObserverTest(unittest.TestCase):
 def test_real_snapshot_function_does_not_collide_with_emitter(self):
  source=Path(__file__).with_name('launch.py')
  nodes=[x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in ['emit','pool_snapshot']]
  scope={'json':json,'os':os};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
  grouped=b'hash'+(0).to_bytes(4,'big')
  pool=NS(cached_block_hash_to_block=NS(_cache={grouped:NS(block_id=1)}),lmetric_journal=NS(seq=3,pending=[]),num_gpu_blocks=8,get_num_free_blocks=lambda:6)
  obj=NS(scheduler=NS(requests={'r':object()},kv_cache_manager=NS(block_pool=pool,get_block_ids=lambda rid:([1,2],))),lmetric_tracker=NS(seq=1))
  with contextlib.redirect_stdout(io.StringIO()) as f:scope['pool_snapshot'](obj,'step')
  d=json.loads(f.getvalue().split('VALIDATION_KV_POOL ',1)[1]);self.assertEqual(d['actual'],{'1':b'hash'.hex()});self.assertEqual(d['phase'],'step');self.assertEqual(d['occupancy'],{'r':[[1,2]]})
if __name__=='__main__':unittest.main(verbosity=2)
