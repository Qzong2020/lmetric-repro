"""Single-NPU smoke for a larger model: load, prefill/decode, journal, router stream.

Reuses Session's engine/router command, SSE observation, journal verifier and
request/response plumbing. One engine only; the router has one real backend.
Bounded: one model load, two tracked requests, no foreign processes touched.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from experiment import (Session, fields, post, prompt, tagged, wait,
                        MODEL, TOKENIZER, GPU_UTIL, NUM_GPU_BLOCKS, MAX_MODEL_LEN)


class Single(Session):
    def health(self, port, p):
        def check():
            if p.poll() is not None:
                raise RuntimeError(f'owned process {p.pid} exited during startup')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as r:
                    return r.status == 200
            except Exception:
                return False
        wait(check, 900, 'bounded single-model load timeout')

    def start_single(self):
        before = self.smi('npu_before.txt')
        assert before.count('No running processes found') == 2, 'NPU occupied: never stop foreign processes'
        self.engines = []
        cmd = self.engine_cmd(0)
        p = self.spawn(cmd, 'engine_0.log', dict(self.env, ASCEND_RT_VISIBLE_DEVICES='0'))
        self.engines.append(p)
        self.status['engines'] = [{'device': 0, 'runtime_id': 0, 'visible_devices': '0',
                                   'pid': p.pid, 'command': cmd}]
        self.save()
        t0 = time.monotonic()
        self.health(18181, p)
        self.status['load_seconds'] = time.monotonic() - t0
        self.smi('npu_loaded_0.txt')
        (self.out / 'clients.json').write_text(json.dumps(['http://127.0.0.1:18181']))
        cmd = self.router_cmd()
        self.gateway = self.spawn(cmd, 'gateway.log', self.env)
        self.status['gateway'] = {'pid': self.gateway.pid, 'command': cmd}
        self.save()
        self.health(18184, self.gateway)
        wait(lambda: 'GET /v1/metrics HTTP/1.1" 200' in (self.out / 'engine_0.log').read_text(),
             30, 'SSE before any tracked request')
        self.status['subscribed_before_requests'] = True
        self.save()
        self.reset([0])
        self.smi('npu_after_reset.txt')

    def reconcile_single(self, row):
        from experiment import strip_ansi
        log = strip_ansi((self.out / 'gateway.log').read_text())
        rid = row['request_id']
        cand = dec = adm = None
        for line in log.splitlines():
            if not any(t in line for t in [' CANDIDATE ', ' DECISION ', ' REQUEST_ADMIT ']):
                continue
            f = fields(line)
            if f.get('request_id') != str(rid):
                continue
            if ' CANDIDATE ' in line:
                cand = f
            elif ' DECISION ' in line:
                dec = f
            else:
                adm = f
        assert cand and dec and adm, ('missing routing observation', rid)
        assert int(dec['engine']) == int(adm['engine']) == 0
        key = 'lmetric_score' if self.policy == 'lmetric-q' else 'shortest_weight_score'
        assert int(dec['engine']) == int(cand['engine']), ('misroute', cand, dec)
        events = self.events(0)
        outs = [o for e in events for o in e.get('outputs', []) if o['request_id'] == rid]
        assert outs, 'no engine SSE output for request'
        ids = [t for o in outs for t in o['new_token_ids']]
        assert len(ids) == row['request']['max_tokens']
        assert sum(o['is_finished'] for o in outs) == 1
        hits = outs[0]['hit_token_cnt']
        work = 0
        active = None
        for e in events:
            owners = {m['request_id'] for m in e['kv_journal']['mutations'] if m['type'] == 'request_blocks'}
            output_owners = {o['request_id'] for o in e['outputs']}
            if owners:
                assert len(owners) == 1
                active = next(iter(owners))
            if output_owners:
                active = next(iter(output_owners))
            if active == rid:
                work += e['prefill_tokens']
            if any(o['is_finished'] for o in e['outputs']):
                active = None
        assert work == len(row['input_tokens']) - hits, (work, hits)
        if 'response' in row:
            assert row['response']['choices'][0]['message']['content'] == TOKENIZER.decode(ids)
        else:
            chunks = [c for c in row['chunks']
                      if c['chunk']['choices'][0]['delta'].get('role') is None
                      and c['chunk']['choices'][0]['delta'].get('content') is not None
                      and c['chunk']['choices'][0].get('finish_reason') is None]
            assert len(chunks) == len(ids), 'one gateway token delta per generated token'
            row['ttft_ms'] = (chunks[0]['received_ns'] - row['sent_ns']) / 1e6
            row['tpot_ms'] = (chunks[-1]['received_ns'] - chunks[0]['received_ns']) / 1e6 / (len(ids) - 1)
            row['token_events'] = len(chunks)
        row.update(engine=0, candidates=[cand], decision=dec, actual_hits=hits,
                   prefill_work=work, generated_ids=ids,
                   latency_ms=(row['end_ns'] - row['sent_ns']) / 1e6)
        return row


def main():
    base = Path(sys.argv[1])
    name = sys.argv[2]
    s = Single(base / name, policy='lmetric-q', mode='sync')
    try:
        s.start_single()
        fixture = prompt(0)
        assert len(fixture['tokens']) == 313
        (s.out / 'fixtures.json').write_text(json.dumps({'X': fixture}, indent=2))
        # Non-stream: full response, real usage accounting.
        a = s.request(fixture, 'smoke_nonstream', max_tokens=8, stream=False)
        s.drain([0]); s.verify([0]); s.reconcile_single(a)
        assert a['response']['usage'] == dict(prompt_tokens=313, completion_tokens=8, total_tokens=321)
        # Stream: TTFT/TPOT semantics on the real gateway path.
        b = s.request(fixture, 'smoke_stream', max_tokens=8, stream=True)
        s.drain([0]); s.verify([0]); s.reconcile_single(b)
        assert b['token_events'] == 8
        s.metrics('metrics_final.txt')
        s.status.update(passed=True, requests=2, journal_verified=True,
                        stream_verified=True, usage_verified=True,
                        model=MODEL, gpu_memory_utilization=GPU_UTIL,
                        num_gpu_blocks_override=NUM_GPU_BLOCKS, max_model_len=MAX_MODEL_LEN)
        s.save()
        (s.out / 'smoke.json').write_text(json.dumps({
            'passed': True, 'model': MODEL, 'gpu_memory_utilization': GPU_UTIL,
            'num_gpu_blocks_override': NUM_GPU_BLOCKS, 'max_model_len': MAX_MODEL_LEN,
            'load_seconds': s.status.get('load_seconds'),
            'nonstream': {k: a.get(k) for k in ['engine', 'actual_hits', 'prefill_work', 'latency_ms']},
            'stream': {k: b.get(k) for k in ['engine', 'actual_hits', 'prefill_work', 'ttft_ms', 'tpot_ms', 'token_events']},
            'usage': a['response']['usage'],
            'generated_ids': {'nonstream': a['generated_ids'], 'stream': b['generated_ids']},
        }, indent=2) + '\n')
        print(s.out.name, ': single-card smoke passed', flush=True)
    except BaseException as e:
        s.status['failure'] = repr(e); s.save(); raise
    finally:
        s.close()


if __name__ == '__main__':
    main()
