"""CPU-check the independent verifier, including cross-backend contamination."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from experiment import Session, prompt, tagged

class VerifierTests(unittest.TestCase):
    def fixture(self,out,idx):
        cache_hash=f'{idx+7:064x}'
        mutations=[dict(mutation_seq=1,type='cache_reset',group_id=0,block_size=128,reason='initial_empty'),dict(mutation_seq=2,type='cache_insert',group_id=0,block_size=128,block_id=1,cache_hash=cache_hash,logical_index=0,prefix_token_ids=[idx+7]*128)]
        event=dict(step_id=1,engine_epoch=f'engine-{idx}',kv_journal=dict(mutation_seq=2,mutations=mutations))
        pool=dict(completed_step=1,mutation_seq=2,pending_mutations=0,actual={'1':cache_hash},occupancy={})
        (out/f'engine_{idx}.log').write_text('VALIDATION_GATEWAY_SSE '+json.dumps({'event':event})+'\nVALIDATION_KV_POOL '+json.dumps(pool)+'\n')
        return dict(engine=idx,step_id=1,engine_epoch=f'engine-{idx}',mutation_seq=2,physical={'1':cache_hash},requests={},valid=True,bs=0,waiting=0,prefill=0,all_tokens=0,active_entries=0,exception_entries=0)
    def test_verify_two_distinct_namespaces_and_reject_contamination(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);s=Session.__new__(Session);s.out=out
            states=[self.fixture(out,i) for i in range(2)]
            def log(): (out/'gateway.log').write_text(''.join('DEBUG cache_tracking: KV_STATE state='+json.dumps(x)+'\n' for x in states))
            log();result=s.verify();self.assertNotEqual(result[0]['epoch'],result[1]['epoch'])
            self.assertNotEqual(result[0]['physical'],result[1]['physical'])
            states[1]['physical']=copy.deepcopy(states[0]['physical']);log()
            with self.assertRaises(AssertionError):s.verify()
            self.assertEqual(s.verify([0])[0]['physical'],result[0]['physical'])
    def test_shared_stdout_records_and_partial_live_write(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'shared.log'
            path.write_text('OBS {"a":1}(EngineCore) OTHER {"b":2}\nOBS {"a":2}OBS {"a":3}\nOBS {"a":')
            self.assertEqual(tagged(path,'OBS'),[{'a':1},{'a':2},{'a':3}])
            path.write_text('OBS {broken}\n')
            with self.assertRaises(json.JSONDecodeError):tagged(path,'OBS')

    def test_all_prompts_round_trip_including_long_warmup_and_benchmark(self):
        for seed in list(range(5))+list(range(10,18)):
            self.assertEqual(len(prompt(seed)['tokens']),313)
        self.assertEqual(len(prompt(0,700)['tokens']),700)
        self.assertEqual(prompt(0,700)['tokens'][:313],prompt(0)['tokens'])

if __name__=='__main__':unittest.main(verbosity=2)
