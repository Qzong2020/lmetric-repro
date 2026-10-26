"""Synthetic CPU inputs against real BlockPool/manager hooks; no NPU claim."""
import asyncio
import hashlib
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.engine.lmetric_kv_journal import KVJournal
from vllm.v1.engine.lmetric_events import EventHub


def request(rid='real-internal', length=384, seed=1):
    tokens=[seed+i%17 for i in range(length)]
    hashes=[hashlib.sha256(bytes([seed,i])).digest() for i in range(length//128)]
    return NS(request_id=rid,all_token_ids=tokens,block_hashes=hashes)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.pool=BlockPool(8,True,128)
        self.j=KVJournal('epoch',128,8)
        self.pool.lmetric_journal=self.j
        self.assertEqual(self.j.drain()['mutations'][0]['type'],'cache_reset')

    def cache(self,req,n=2):
        blocks=self.pool.get_new_blocks(n)
        self.pool.cache_full_blocks(req,blocks,0,n,128,0)
        return blocks

    def test_insert_only_after_success_and_full_prefix_proof(self):
        req=request();blocks=self.cache(req)
        es=self.j.drain()['mutations']
        self.assertEqual([e['type'] for e in es],['cache_insert']*2)
        self.assertEqual(es[1]['prefix_token_ids'],req.all_token_ids[:256])
        self.assertEqual(es[1]['logical_index'],1)
        self.assertIs(self.pool.get_cached_block(req.block_hashes[0],[0])[0],blocks[0])
        with patch.object(self.pool.cached_block_hash_to_block,'insert',side_effect=RuntimeError('insert fails')):
            with self.assertRaises(RuntimeError):self.cache(request('second'))
        self.assertEqual(self.j.drain()['mutations'],[])

    def test_free_is_not_evict_and_exact_copy_removal(self):
        req=request();a=self.cache(req,1);b=self.cache(req,1);self.j.drain()
        self.pool.free_blocks(a+b)
        self.assertEqual(self.j.drain()['mutations'],[])
        self.pool.evict_blocks({a[0].block_id})
        es=self.j.drain()['mutations'];self.assertEqual(len(es),1)
        self.assertEqual(es[0]['block_id'],a[0].block_id)
        self.assertEqual(es[0]['type'],'cache_remove')
        self.assertIs(self.pool.get_cached_block(req.block_hashes[0],[0])[0],b[0])
        self.pool.evict_blocks({a[0].block_id})
        self.assertEqual(self.j.drain()['mutations'],[])

    def test_physical_reuse_order(self):
        self.pool=BlockPool(3,True,128);self.pool.lmetric_journal=self.j
        old=self.cache(request(),2);self.j.drain();self.pool.free_blocks(old)
        new=self.cache(request('new',seed=2),2)
        es=self.j.drain()['mutations']
        self.assertEqual([e['type'] for e in es],['cache_remove']*2+['cache_insert']*2)
        self.assertEqual({e['block_id'] for e in es[:2]},{e['block_id'] for e in es[2:]})
        self.assertNotEqual(es[0]['cache_hash'],es[2]['cache_hash'])

    def test_reset_failure_then_success(self):
        bs=self.cache(request(),1);self.j.drain()
        self.assertFalse(self.pool.reset_prefix_cache());self.assertEqual(self.j.drain()['mutations'],[])
        self.pool.free_blocks(bs);self.assertTrue(self.pool.reset_prefix_cache())
        self.assertEqual([e['type'] for e in self.j.drain()['mutations']],['cache_reset'])

    def test_request_full_lists_release_and_deduplicate(self):
        self.j.request_blocks('r',([1,2],));self.j.request_blocks('r',([1,2],))
        self.j.request_blocks('r',([1,2,3],));self.j.release('r')
        es=self.j.drain()['mutations']
        self.assertEqual([e['type'] for e in es],['request_blocks','request_blocks','request_release'])
        self.assertEqual([e['block_ids'] for e in es],[[1,2],[1,2,3],[]])
        self.assertEqual(self.j.requests,{})
        with self.assertRaises(RuntimeError):self.j.request_blocks('multi',([1],[2]))

    def test_mask_and_null_not_reported(self):
        req=request();bs=self.pool.get_new_blocks(3)
        self.pool.cache_full_blocks(req,bs,0,3,128,0,[True,False,True])
        es=self.j.drain()['mutations'];self.assertEqual([e['logical_index'] for e in es],[0,2])
        self.assertEqual(es[-1]['prefix_token_ids'],req.all_token_ids)

    def test_bounded_failure_latches_without_sequence_reset(self):
        j=KVJournal('e',128,8,max_events=2)
        j.request_blocks('r',([1],))
        with self.assertRaises(RuntimeError):j.release('r')
        with self.assertRaises(RuntimeError):j.drain()
        j=KVJournal('e',128,8);self.assertEqual(j.drain()['mutations'][0]['mutation_seq'],1)
        j.request_blocks('r',([1],));self.assertEqual(j.drain()['mutations'][0]['mutation_seq'],2)

    def test_manager_actual_allocate_and_free_hooks(self):
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.kv_cache_interface import KVCacheConfig,KVCacheGroupSpec,FullAttentionSpec
        from vllm.v1.request import Request
        from vllm.sampling_params import SamplingParams
        import torch
        cfg=KVCacheConfig(num_blocks=8,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(['layer'],FullAttentionSpec(block_size=128,num_kv_heads=1,head_size=16,dtype=torch.float16))])
        manager=KVCacheManager(cfg,1024,128,128)
        manager.block_pool.lmetric_journal=self.j
        req=Request('managed',list(range(313)),SamplingParams(max_tokens=8),None)
        req.block_hashes=request().block_hashes
        manager.allocate_slots(req,128)
        es=self.j.drain()['mutations']
        used=[e for e in es if e['type']=='request_blocks']
        self.assertEqual(used[0]['block_ids'],manager.get_block_ids('managed')[0])
        manager.free(req)
        self.assertEqual(self.j.drain()['mutations'][-1]['type'],'request_release')

    def test_idle_reset_uses_real_core_control_output(self):
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.engine.lmetric_events import RequestEvents
        from queue import Queue
        tracker=RequestEvents.__new__(RequestEvents)
        tracker.kv_journal=self.j;tracker.epoch='epoch';tracker.seq=0
        tracker.protocol='request-kv-v1'
        tracker.runtime={'async_scheduling':False,'scheduler_block_size':128}
        obj=EngineCore.__new__(EngineCore)
        obj.lmetric_tracker=tracker;obj.batch_queue=None;obj.output_queue=Queue()
        obj.scheduler=NS(has_unfinished_requests=lambda:False,
                         reset_prefix_cache=lambda *a:self.pool.reset_prefix_cache())
        self.assertTrue(obj.reset_prefix_cache())
        client,out=obj.output_queue.get_nowait()
        self.assertEqual(client,0)
        self.assertEqual(out.lmetric_event['event_kind'],'control')
        self.assertEqual(out.lmetric_event['outputs'],[])
        self.assertEqual(out.lmetric_event['kv_journal']['mutations'][-1]['type'],'cache_reset')
        obj.batch_queue=[object()]
        with self.assertRaises(ValueError):obj.reset_prefix_cache()
        self.assertEqual(self.j.drain()['mutations'],[])

    def test_kv_runtime_rejects_unverified_config(self):
        from vllm.v1.engine.lmetric_events import validate_config
        cfg=NS(parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1),model_config=NS(enforce_eager=True,max_model_len=1024,runner_type='generate',is_multimodal_model=False),cache_config=NS(enable_prefix_caching=True),scheduler_config=NS(max_num_seqs=1),speculative_config=None,kv_transfer_config=None,ec_transfer_config=None,lora_config=None)
        with patch.dict(os.environ,{'VLLM_LMETRIC_KV_EVENTS':'1'}):
            validate_config(cfg)
            cfg.scheduler_config.max_num_seqs=2
            with self.assertRaises(ValueError):validate_config(cfg)
            cfg.scheduler_config.max_num_seqs=1;cfg.model_config.enforce_eager=False
            with self.assertRaises(ValueError):validate_config(cfg)

    def test_disabled_pool_keeps_native_behavior(self):
        pool=BlockPool(4,True,128)
        bs=pool.get_new_blocks(1);pool.cache_full_blocks(request(),bs,0,1,128,0)
        pool.free_blocks(bs)
        self.assertIsNotNone(pool.get_cached_block(request().block_hashes[0],[0]))


class HubTests(unittest.IsolatedAsyncioTestCase):
    async def test_journal_identity_mapped_before_final_output_unregistration(self):
        hub=EventHub();hub.subscribe();hub.register('internal','101')
        event={'lmetric_protocol':'request-kv-v1','outputs':[{'request_id':'internal','is_finished':True}],
               'kv_journal':{'mutations':[{'type':'request_blocks','request_id':'internal','block_ids':[1]},{'type':'request_release','request_id':'internal','block_ids':[]}]}}
        hub.publish(event);out=await hub.queue.get()
        self.assertEqual([x['request_id'] for x in out['kv_journal']['mutations']],[101,101])
        self.assertEqual(hub.identities,{})

    async def test_sse_byte_bound_and_drain(self):
        hub=EventHub();hub.subscribe();hub.max_queued_bytes=10
        with self.assertRaises(RuntimeError):hub.publish({'outputs':[],'padding':'x'*20})
        self.assertIsNotNone(hub.failed)
        hub=EventHub();hub.subscribe();frames=hub.frames()
        self.assertIn('ready',await anext(frames))
        hub.publish({'outputs':[]})
        self.assertGreater(hub.queued_bytes,0)
        self.assertIn('data:',await anext(frames))
        self.assertEqual(hub.queued_bytes,0)
        await frames.aclose()

    async def test_kv_only_envelope_survives_real_msgpack(self):
        from vllm.v1.engine import EngineCoreOutputs
        from vllm.v1.serial_utils import MsgpackEncoder,MsgpackDecoder
        event={'lmetric_protocol':'request-kv-v1','outputs':[],'kv_journal':{'mutations':[{'type':'cache_reset','mutation_seq':1}]}}
        obj=EngineCoreOutputs(lmetric_event=event)
        result=MsgpackDecoder(EngineCoreOutputs).decode(MsgpackEncoder().encode(obj))
        self.assertEqual(result.lmetric_event,event)

if __name__=='__main__':unittest.main(verbosity=2)
