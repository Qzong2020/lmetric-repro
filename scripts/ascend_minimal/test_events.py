"""SIMULATED inputs only; selected real production methods and wire schemas."""
import asyncio
import ast
from collections import deque
from concurrent.futures import Future
import contextlib
import copy
import json
import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
import msgspec
import zmq
from support import (events,output_types,CoreOutput,CoreOutputs,CoreHarness,ENGINE,
                     ROOT,schema,scheduler,schedule_output,make_fixtures,module)


def ipc_sender(endpoint, packets):
    ctx=zmq.Context(); sock=ctx.socket(zmq.PUSH); sock.connect(endpoint)
    for packet in packets:
        sock.send(msgspec.msgpack.encode(CoreOutputs(lmetric_event=packet)))
    sock.close();ctx.term()


def ipc_roundtrip(packets):
    ctx=zmq.Context(); sock=ctx.socket(zmq.PULL);sock.setsockopt(zmq.RCVTIMEO,10000)
    port=sock.bind_to_random_port('tcp://127.0.0.1')
    child=multiprocessing.get_context('spawn').Process(target=ipc_sender,
        args=(f'tcp://127.0.0.1:{port}',packets));child.start()
    try:
        decoder=msgspec.msgpack.Decoder(CoreOutputs)
        result=[decoder.decode(sock.recv()).lmetric_event for _ in packets]
    finally:
        child.join(10)
        if child.is_alive():child.terminate();child.join()
        sock.close();ctx.term()
    assert child.exitcode==0
    return result


class ContractTests(unittest.TestCase):
    def test_real_struct_and_zmq_subprocess(self):
        packets=make_fixtures()
        received=ipc_roundtrip(packets)
        self.assertEqual(packets,received)
        self.assertEqual(received[0]['outputs'],[])
        self.assertEqual([p['prefill_tokens'] for p in received],[128,128,0,0])
        self.assertEqual([p['step_id'] for p in received],[1,2,3,4])
        self.assertEqual(received[1]['outputs'][0]['state'],'PREFILL')
        self.assertEqual(received[2]['outputs'][0]['state'],'DECODE')
        out=ROOT/'results/ascend_minimal_events';out.mkdir(exist_ok=True)
        (out/'synthetic_ipc_events.json').write_text(json.dumps(received,indent=2)+'\n')

    def test_async_snapshot_not_future_state(self):
        s=scheduler(); t=events.RequestEvents(s,128,128,True,2)
        first=schedule_output(32,16,True); second=schedule_output(48,16)
        t.capture(s,first,time.monotonic());t.capture(s,second,time.monotonic())
        s.requests['engine-internal-uuid'].num_prompt_tokens=999
        a=t.complete(first,{},CoreOutputs)[0].lmetric_event
        b=t.complete(second,{0:CoreOutputs(outputs=[CoreOutput('engine-internal-uuid',[9])])},CoreOutputs)[0].lmetric_event
        self.assertEqual(a['prefill_tokens'],16);self.assertEqual(b['prefill_tokens'],16)
        self.assertEqual(b['outputs'][0]['prev_computed_tokens'],48)
        self.assertEqual(b['execution_mode'],'async')

    def test_rewritten_scheduler_dispatch_and_both_core_paths(self):
        # A rewritten schedule(), not an upstream Scheduler.schedule stub.
        for async_mode in [False,True]:
            class RewrittenScheduler:
                def __init__(self):
                    self.__dict__.update(scheduler().__dict__)
                    self.pending=deque([schedule_output(32,16,True),schedule_output(48,16)])
                def has_requests(self): return bool(self.pending)
                def schedule(self): return self.pending.popleft()
                def get_grammar_bitmask(self,out): return None
                def update_from_output(self,out,result):
                    return {} if out.scheduled_new_reqs else {0:CoreOutputs(outputs=[CoreOutput('engine-internal-uuid',[9])])}
            core=CoreHarness();core.scheduler=RewrittenScheduler()
            core.lmetric_tracker=events.RequestEvents(core.scheduler,128,128,async_mode,2 if async_mode else 1)
            def future(value):
                f=Future();f.set_result(value);return f
            core.model_executor=NS(execute_model=lambda *a,**kw:future(NS()),
                                   sample_tokens=lambda *a,**kw:future(NS()))
            core.log_error_detail=lambda *a:contextlib.nullcontext()
            core.log_iteration_details=lambda *a:contextlib.nullcontext()
            core._process_aborts_queue=lambda:None
            core.batch_queue=deque(maxlen=2) if async_mode else None
            core.batch_queue_size=2 if async_mode else 1
            core.use_spec_decode=False
            core.is_ec_consumer=True
            core.is_pooling_model=False
            if async_mode:
                a,executed=core.step_with_batch_queue();self.assertIsNone(a)
                a,_=core.step_with_batch_queue(); b,_=core.step_with_batch_queue()
            else:
                a,_=core.step();b,_=core.step()
            self.assertEqual(a[0].lmetric_event['outputs'],[])
            self.assertEqual(a[0].lmetric_event['step_id'],1)
            self.assertEqual(b[0].lmetric_event['step_id'],2)
            self.assertEqual(b[0].lmetric_event['outputs'][0]['state'],'PREFILL')

    def test_core_client_filter_keeps_metric_only_packets(self):
        tree=ast.parse((ENGINE/'vllm/v1/engine/core_client.py').read_text())
        matches=[n.test for n in ast.walk(tree) if isinstance(n,ast.If)
                 and 'outputs.lmetric_event' in ast.unparse(n.test)]
        self.assertEqual(len(matches),1)
        test=compile(ast.Expression(matches[0]),'<production-filter>','eval')
        self.assertTrue(eval(test,{'outputs':CoreOutputs(lmetric_event={'step_id':1})}))
        self.assertFalse(eval(test,{'outputs':CoreOutputs()}))

    def test_config_validation_and_first_finish_after_free(self):
        c=NS(parallel_config=NS(pipeline_parallel_size=1,data_parallel_size=1),
             speculative_config=None,kv_transfer_config=None,ec_transfer_config=None,
             lora_config=None,model_config=NS(runner_type='generate',is_multimodal_model=False))
        events.validate_config(c)
        for field in ['speculative_config','kv_transfer_config','ec_transfer_config','lora_config']:
            changed=copy.deepcopy(c);setattr(changed,field,object())
            with self.assertRaises(ValueError):events.validate_config(changed)
        s=scheduler();t=events.RequestEvents(s,128,128,False,1)
        o=schedule_output(32,32,True);t.capture(s,o,time.monotonic())
        s.requests.clear() # the real scheduler frees a finished Request before returning
        packet=t.complete(o,{0:CoreOutputs(outputs=[CoreOutput('engine-internal-uuid',[9],
            finish_reason=schema['FinishReason'].LENGTH)])},CoreOutputs)[0].lmetric_event
        self.assertEqual(packet['outputs'][0]['hit_token_cnt'],32)
        self.assertTrue(packet['outputs'][0]['is_finished'])
        self.assertFalse(t.hits)

    def test_config_and_preemption_rejected(self):
        s=scheduler();s.kv_cache_config.kv_cache_groups*=2
        with self.assertRaises(ValueError):events.RequestEvents(s,128,128,False,1)
        s=scheduler();t=events.RequestEvents(s,128,128,False,1)
        o=schedule_output(0,16,True);o.preempted_req_ids={'x'}
        with self.assertRaises(RuntimeError):t.capture(s,o,time.monotonic())


class HubTests(unittest.IsolatedAsyncioTestCase):
    async def test_id_mapping_frames_and_disconnect(self):
        h=events.EventHub();h.subscribe();h.register('engine-internal-uuid','101')
        packets=make_fixtures()
        for p in packets:h.publish(p)
        frames=h.frames();self.assertEqual(await anext(frames),': ready\n\n')
        got=[json.loads((await anext(frames)).removeprefix('data: ')) for _ in packets]
        self.assertEqual(got[1]['outputs'][0]['request_id'],101)
        self.assertEqual(packets[1]['outputs'][0]['request_id'],'engine-internal-uuid')
        self.assertFalse(h.identities);await frames.aclose()
        with self.assertRaises(ValueError):h.subscribe()

    async def test_overflow_and_invalid_identity_fail_closed(self):
        h=events.EventHub(1)
        with self.assertRaises(ValueError):h.register('a','101')
        h.subscribe()
        for bad in ['-1','cmpl-101','18446744073709551616','１２']:
            with self.assertRaises(ValueError):h.register('a',bad)
        h.register('a','101')
        with self.assertRaises(ValueError):h.register('b','101')
        p=make_fixtures()[0];h.publish(p)
        with self.assertRaises(RuntimeError):h.publish(p)
        self.assertTrue(h.failed)

    async def test_real_async_llm_output_handler_publishes_empty_event(self):
        # Exact production method, synthetic core and OutputProcessor.
        tree=ast.parse((ENGINE/'vllm/v1/engine/async_llm.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AsyncLLM')
        method=next(n for n in cls.body if getattr(n,'name',None)=='_run_output_handler')
        scope={'asyncio':asyncio,'envs':NS(VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=1)}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<production-AsyncLLM>','exec'),scope)
        h=events.EventHub();h.subscribe();q=asyncio.Queue()
        await q.put(CoreOutputs(lmetric_event=make_fixtures()[0]))
        obj=NS(output_handler=None,engine_core=NS(get_output_async=q.get),
               output_processor=NS(update_scheduler_stats=lambda _:None),
               log_stats=False,logger_manager=None,renderer=None,lmetric_hub=h)
        scope['_run_output_handler'](obj)
        packet=await asyncio.wait_for(h.queue.get(),1)
        self.assertEqual(packet['outputs'],[])
        obj.output_handler.cancel()
        with self.assertRaises(asyncio.CancelledError):await obj.output_handler


if __name__=='__main__':unittest.main(verbosity=2)
