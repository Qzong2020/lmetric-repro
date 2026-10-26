#!/usr/bin/env python3
"""CPU-only serializer checks and source-bound arithmetic audit, not engine execution."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/yaullm_audit'
OUT = SOURCE / 'field_contract'


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = SOURCE / 'engine_types.py'
    tree = ast.parse(path.read_text())
    names = {'split_hash_to_64bit_list', 'simplified_engine_core_outputs_to_dict'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {'Any': Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    serialize = namespace['simplified_engine_core_outputs_to_dict']
    def output(**changes):
        fields = dict(outputs=[SimpleNamespace(request_id='101', new_token_ids=[7], state='PREFILL',
                      finished=False, num_cached_tokens=32, num_computed_tokens=32, ttft=0.125)],
                      step_id=1, latency=0.012345, prefill_token_budget=256, prefill_tokens=64,
                      new_block_hashes=['0x123'], evicted_block_hashes=['0x456'],
                      evicted_block_ids=[9], used_block_ids={'101':[2,3,4,5]},
                      new_block_hashes_ids={'101':[4,5]}, preempted_ids=[], aborted_requests=set())
        fields.update(changes)
        return SimpleNamespace(**fields)
    full = serialize(output(), 'full')
    minimal = serialize(output(), 'minimal')
    empty = serialize(output(outputs=[], prefill_tokens=16), 'full')
    assert full['latency'] == 12
    assert full['outputs'][0]['request_id'] == 101
    assert full['outputs'][0]['hit_token_cnt'] == 32
    assert full['outputs'][0]['prev_computed_tokens'] == 32
    assert full['new_block_hashes'] == [[0x123]]
    assert full['cur_used_block_ids'] == {'101':[2,3,4,5]}
    assert minimal['prefill_tokens'] == 0 and minimal['new_block_hashes_ids'] == {}
    assert empty['outputs'] == [] and empty['prefill_tokens'] == 16
    dump('full.json', full)
    dump('minimal.json', minimal)
    dump('empty_prefill.json', empty)
    # Check the precise expressions used in the static derivation. These
    # guards bind the example to the inspected sources; they are not tests of
    # the complete Rust event loop or the Scheduler.
    scheduler = (SOURCE/'scheduler.py').read_text()
    rust_admission = (ROOT/'blitz-router/router/src/scheduler/policies/dsl_runtime.rs').read_text()
    rust_consumer = (ROOT/'blitz-router/router/src/engine/colocation.rs').read_text()
    rust_hits = (ROOT/'blitz-router/router/src/scheduler/kvcache.rs').read_text()
    assert 'prefill_tokens += num_new_tokens + num_computed_tokens' in scheduler
    assert 'prefill_tokens_inc: new_ntkns' in rust_admission
    assert 'metric_delta.prefill_tokens_dec = m.prefill_tokens as isize' in rust_consumer
    assert 'metric_delta.prefill_tokens_dec += inc_hit_nblks' in rust_consumer
    assert 'self.real_hit_nblks.unwrap() as isize - hit_nblks as isize' in rust_hits
    cases = []
    for predicted, actual in [(0,0),(32,32),(16,32),(32,16)]:
        length = 64
        initial = length-predicted
        emitted = (length-actual)+actual
        correction = actual-predicted
        cases.append(dict(prompt_length=length, predicted_hit_tokens=predicted,
                          actual_hit_tokens=actual, admitted_prefill=initial,
                          producer_prefill=emitted, consumer_extra_decrement=correction,
                          derived_prefill_after_first_nonfinal_output=initial-emitted-correction,
                          independently_expected_remaining_prefill=0))
    dump('prefill_arithmetic.json', {'scope':'static arithmetic under isolated one-step prefill, first output nonfinal, no preempt/cancel; no Rust event loop or real inference executed',
        'cases':cases,'finding':'selected producer/consumer pair over-decrements by actual cached tokens under stated assumptions; verify compatible artifact versions before changing either side'})
    dump('checks.json', {'actual_author_serializer_executed':True,'serializer_cases':3,
        'serializer_assertions_passed':True,'npu_used':False,'rust_event_loop_executed':False,
        'engine_schedule_executed':False,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [path,SOURCE/'scheduler.py',ROOT/'blitz-router/router/src/scheduler/policies/dsl_runtime.rs',ROOT/'blitz-router/router/src/engine/colocation.rs',ROOT/'blitz-router/router/src/scheduler/kvcache.rs']}})
    print('Author serializer: 3 CPU-only cases passed.')
    print('Static prefill derivation:', [(c['predicted_hit_tokens'],c['actual_hit_tokens'],c['derived_prefill_after_first_nonfinal_output']) for c in cases])
    print('No model, NPU, full scheduler or Rust event loop was executed.')


if __name__ == '__main__':
    main()
