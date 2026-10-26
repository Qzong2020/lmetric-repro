"""Bounded dual-NPU production-gateway validation; never touches foreign PIDs."""
import concurrent.futures as futures
import collections
import datetime
import importlib.metadata
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path('/root/lmetric-repro')
# The container exports HTTP(S)_PROXY; urllib would send loopback health and
# inference traffic through it. Bypass the proxy for loopback only.
for _k in ('NO_PROXY','no_proxy'):
    os.environ[_k]=','.join(filter(None,[os.environ.get(_k),'127.0.0.1','localhost']))
# Defaults preserve the Qwen3-0.6B dual-NPU session exactly. A new model/run
# overrides these without editing this file, so prior evidence stays reproducible.
MODEL=os.environ.get('LMETRIC_MODEL','/models/Qwen3-0.6B')
TOKENIZER=Tokenizer.from_file(os.environ.get('LMETRIC_TOKENIZER',MODEL+'/tokenizer.json'))
GPU_UTIL=os.environ.get('LMETRIC_GPU_UTIL','0.20')
NUM_GPU_BLOCKS=os.environ.get('LMETRIC_NUM_GPU_BLOCKS','8')
MAX_MODEL_LEN=os.environ.get('LMETRIC_MAX_MODEL_LEN','1024')
# Physical NPU IDs as seen by npu-smi, in the same order as logical runtime IDs
# 0,1. The container's exposed pair changes across restarts (0/6, then 0/7).
PHYS_DEVICES=[int(x) for x in os.environ.get('LMETRIC_PHYSICAL_DEVICES','0,6').split(',')]

ANSI=re.compile(r'\x1b\[[0-9;]*m')
def strip_ansi(text):return ANSI.sub('',text)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def tagged(path,tag):
    # EngineCore and frontend share stdout. A complete JSON object can be
    # followed immediately by another process's log without a newline.
    # tracing's fmt layer may emit ANSI escapes around the structured line.
    source=strip_ansi(path.read_text(errors='replace'));decoder=json.JSONDecoder();rows=[]
    for match in re.finditer(re.escape(tag+' '),source):
        tail=source[match.end():]
        try:row,_=decoder.raw_decode(tail)
        except json.JSONDecodeError:
            if '\n' not in tail:break  # live final write is not complete yet
            raise
        rows.append(row)
    return rows
def fields(line):return {k:v for k,v in re.findall(r'(\w+)=("[^"]*"|\S+)',line)}
def wait(check,seconds,label):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if check():return
        time.sleep(.02)
    raise TimeoutError(label)
def post(url,body=None,timeout=30):
    req=urllib.request.Request(url,data=b'' if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,r.read().decode()
def prompt(seed,n=313):
    texts=['Please explain how a request passes through a language model inference server. ',
           'Ocean currents distribute heat through the deep sea and affect coastal weather patterns. ',
           'Astronomy studies distant galaxies, stellar evolution, and the history of the universe. ',
           'Botanical gardens protect diverse plants and teach visitors about local ecosystems. ',
           'Railway networks connect cities through carefully scheduled passenger and freight services. ']
    text=texts[seed%5] if seed<5 else f'Unique record {seed}: measurements in a carefully bounded inference experiment. '
    ids=TOKENIZER.encode(text*100).ids[:n];text=TOKENIZER.decode(ids)
    assert TOKENIZER.encode(text).ids==ids and len(ids)==n
    return {'text':text,'tokens':ids}

def stop_owned(p):
    if p is None:return
    # The entire group belongs to our start_new_session Popen, including workers.
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)

class Session:
    def __init__(self,out,policy='lmetric-q',mode='sync'):
        self.out=out;out.mkdir();self.policy=policy;self.mode=mode;self.processes=[];self.logs=[];self.lock=threading.Lock();self.responses=[]
        self.status={'started':stamp(),'mode':mode,'policy':policy,'devices':PHYS_DEVICES,'physical_devices':PHYS_DEVICES,'scope':'bounded real dual-NPU production gateway','packages':{n:importlib.metadata.version(n) for n in ['vllm','vllm-ascend','torch','torch-npu','tokenizers']},'model':MODEL,'tokenizer':os.environ.get('LMETRIC_TOKENIZER',MODEL+'/tokenizer.json'),'gpu_memory_utilization':GPU_UTIL,'num_gpu_blocks_override':NUM_GPU_BLOCKS,'max_model_len':MAX_MODEL_LEN}
        self.env=dict(os.environ,VLLM_LMETRIC_EVENTS='1',VLLM_LMETRIC_KV_EVENTS='1',VLLM_SERVER_DEV_MODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',RAYON_NUM_THREADS='2',NO_COLOR='1',PYTHONPATH=str(ROOT/'engines/vllm')+':'+os.getenv('PYTHONPATH',''),LMETRIC_REQUIRE_KV_JOURNAL='1',LOG_LEVEL='info,cache_tracking=debug,correction=info,scheduling=debug')
        self.save()
    def save(self): (self.out/'run.json').write_text(json.dumps(self.status,indent=2)+'\n')
    def smi(self,name):
        r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15);(self.out/name).write_text(r.stdout+r.stderr);return r.stdout
    def spawn(self,cmd,name,env):
        log=(self.out/name).open('w');self.logs.append(log)
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);self.processes.append(p);return p
    def health(self,port,p):
        def check():
            if p.poll() is not None:raise RuntimeError(f'owned process {p.pid} exited during startup')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as r:return r.status==200
            except Exception:return False
        wait(check,240 if port!=18184 else 30,'bounded startup timeout')
    def engine_cmd(self,idx):
        return [str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/kv_journal/launch_gateway.py'),'--model',MODEL,'--served-model-name','ascend-small','--host','127.0.0.1','--port',str(18181+idx),'--tensor-parallel-size','1','--pipeline-parallel-size','1','--max-model-len',MAX_MODEL_LEN,'--max-num-seqs','1','--max-num-batched-tokens','128','--gpu-memory-utilization',GPU_UTIL,'--num-gpu-blocks-override',NUM_GPU_BLOCKS,'--enforce-eager','--async-scheduling' if self.mode=='async' else '--no-async-scheduling','--enable-prefix-caching','--enable-chunked-prefill','--block-size','128']
    def router_cmd(self):
        return [str(self.out.parent/'bins'/self.policy),'--client-config',str(self.out/'clients.json'),'--hostname','127.0.0.1','--port','18184','--tokenizer-name',MODEL,'--use-tokenizer','--model-name','ascend-small','--kvcache-block-size','128','--max-concurrent-requests','4','--max-best-of','1','--validation-workers','1','--max-input-length','800','--max-total-tokens',MAX_MODEL_LEN,'--chat-template-mode','none']
    def start(self):
        before=self.smi('npu_before.txt')
        assert before.count('No running processes found')==2,'NPU occupied: never stop foreign processes'
        for port in [18181,18182,18184]:
            # SO_REUSEADDR tolerates TIME_WAIT from a just-closed owned session
            # but still fails on a live listener, so foreign services are never
            # disturbed and never silently shared.
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                s.bind(('127.0.0.1',port))
        self.engines=[];commands=[]
        for idx,(device,runtime_id) in enumerate(zip(PHYS_DEVICES,[0,1])):
            cmd=self.engine_cmd(idx)
            p=self.spawn(cmd,f'engine_{idx}.log',dict(self.env,ASCEND_RT_VISIBLE_DEVICES=str(runtime_id)));self.engines.append(p);commands.append({'device':device,'runtime_id':runtime_id,'visible_devices':str(runtime_id),'pid':p.pid,'command':cmd})
            self.status['engines']=commands;self.save()
            # Sequential initialization avoids conflating startup resource contention.
            self.health(18181+idx,p)
            self.smi(f'npu_loaded_{idx}.txt')
        (self.out/'clients.json').write_text(json.dumps(['http://127.0.0.1:18181','http://127.0.0.1:18182']))
        cmd=self.router_cmd()
        self.gateway=self.spawn(cmd,'gateway.log',self.env);self.status['gateway']={'pid':self.gateway.pid,'command':cmd};self.save();self.health(18184,self.gateway)
        for idx in range(2):wait(lambda idx=idx:'GET /v1/metrics HTTP/1.1" 200' in (self.out/f'engine_{idx}.log').read_text(),15,'SSE before any tracked request')
        self.status['subscribed_before_requests']=True;self.save()
        self.reset([0,1])
        print(f'{self.out.name}: both engines ready, subscribed and reset',flush=True)
    def events(self,idx):return [x['event'] for x in tagged(self.out/f'engine_{idx}.log','VALIDATION_GATEWAY_SSE')]
    def states(self):
        rows=[]
        for l in strip_ansi((self.out/'gateway.log').read_text()).splitlines():
            if 'KV_STATE state=' in l:rows.append(json.loads(l.split('KV_STATE state=',1)[1]))
        return rows
    def reset(self,idxs):
        self.drain(idxs)
        for idx in idxs:
            prior=len(self.events(idx));assert post(f'http://127.0.0.1:{18181+idx}/reset_prefix_cache')[0]==200
            wait(lambda idx=idx,prior=prior:len(self.events(idx))>prior,5,'reset SSE arrival')
            last=self.events(idx)[-1]['step_id']
            wait(lambda idx=idx,last=last:any(s['engine']==idx and s['step_id']==last and not s['physical'] for s in self.states()),5,'Rust reset application')
        self.verify(idxs)
    def drain(self,idxs=(0,1)):
        def check():
            states=self.states();latest={s['engine']:s for s in states}
            for idx in idxs:
                events=self.events(idx)
                assert not any('error' in e for e in events),('engine session failed',idx)
                if not events:continue
                s=latest.get(idx)
                if s is None or s['step_id']!=events[-1]['step_id']:return False
                assert s['valid'],'invalid session: stop affected experiment'
                if any(s[k] for k in ['bs','waiting','prefill','all_tokens','active_entries','exception_entries']) or s['requests']:return False
            return True
        wait(check,30,'per-replica request table/ledger drain')
    def request(self,fixture,label,max_tokens=8,stream=False):
        body={'model':'ascend-small','messages':[{'role':'user','content':fixture['text']}],'max_tokens':max_tokens,'stream':stream,'temperature':0.6}
        row={'label':label,'request':body,'input_tokens':fixture['tokens'],'sent_utc':stamp(),'sent_ns':time.monotonic_ns(),'chunks':[]}
        try:
            req=urllib.request.Request('http://127.0.0.1:18184/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as r:
                row['status']=r.status
                if not stream:row['response']=json.loads(r.read())
                else:
                    done=False
                    for line in r:
                        if not line.startswith(b'data:'):continue
                        value=line[5:].strip();now=time.monotonic_ns()
                        if value==b'[DONE]':done=True;break
                        chunk=json.loads(value);row['chunks'].append({'received_ns':now,'chunk':chunk})
                    assert done,'stream incomplete'
            row['end_ns']=time.monotonic_ns();row['completed_utc']=stamp()
            rid=(row['chunks'][0]['chunk']['id'] if stream else row['response']['id']).split('-')[-1]
            row['request_id']=int(rid)
            if not stream:assert row['response']['usage']==dict(prompt_tokens=len(fixture['tokens']),completion_tokens=max_tokens,total_tokens=len(fixture['tokens'])+max_tokens)
        except BaseException as e:
            row['error']=repr(e);raise
        finally:
            with self.lock:
                self.responses.append(row)
                with (self.out/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return row
    def controlled_pair(self,first,second,label,first_tokens=8):
        old=len(self.responses)
        log=self.out/'gateway.log';before=log.read_text().count('REQUEST_ADMIT')
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(self.request,first,label+'_first',first_tokens)
            wait(lambda:log.read_text().count('REQUEST_ADMIT')>before,3,'first admission before controlled second request')
            b=pool.submit(self.request,second,label+'_second')
            rows=[a.result(timeout=180),b.result(timeout=180)]
        self.drain();assert len(self.responses)==old+2;self.verify();self.reconcile(rows)
        return rows
    def reconcile(self,rows,output_name="reconciled.jsonl"):
        log=strip_ansi((self.out/'gateway.log').read_text());lines=log.splitlines();result=[]
        for row in rows:
            rid=row['request_id'];candidates=[];decisions=[];admits=[]
            for l in lines:
                if any(tag in l for tag in [' CANDIDATE ',' DECISION ',' REQUEST_ADMIT ']):
                    f=fields(l)
                    if f.get('request_id')!=str(rid):continue
                    if ' CANDIDATE ' in l:candidates.append(f)
                    elif ' DECISION ' in l:decisions.append(f)
                    else:admits.append(f)
            assert len(decisions)==len(admits)==1,(rid,decisions,admits)
            candidates=candidates[-2:];assert len(candidates)==2
            d=decisions[0];idx=int(d['engine']);assert idx==int(admits[0]['engine'])
            key='lmetric_score' if self.policy=='lmetric-q' else 'shortest_weight_score'
            eligible=[c for c in candidates if c['routing_enabled']=='true'];expected=int(min(eligible,key=lambda c:(int(c[key]),int(c['engine'])))['engine'])
            assert idx==expected,('misroute',rid,candidates,d)
            observed=[]
            for backend in range(2):
                es=self.events(backend);os_=[o for e in es for o in e.get('outputs',[]) if o['request_id']==rid]
                if os_:observed.append((backend,es,os_))
            assert len(observed)==1 and observed[0][0]==idx
            _,es,os_=observed[0];es=[e for e in es if 'error' not in e];ids=[t for o in os_ for t in o['new_token_ids']]
            assert len(ids)==row['request']['max_tokens'] and os_[-1]['is_finished'] and sum(o['is_finished'] for o in os_)==1
            hits=os_[0]['hit_token_cnt'];predicted=int(d['predicted_hits'])*128
            # Journal is metadata; a queued request can see a newer cache at execution.
            work=sum(e['prefill_tokens'] for e in es if any(o['request_id']==rid for o in e['outputs']))
            # Empty-output chunked prefill must also be attributed. max_num_seqs=1
            # leaves each frame's request_blocks identity as an unambiguous owner.
            active=None;work=0
            for e in es:
                owners={m['request_id'] for m in e['kv_journal']['mutations'] if m['type']=='request_blocks'}
                output_owners={o['request_id'] for o in e['outputs']}
                if owners:assert len(owners)==1;active=next(iter(owners))
                if output_owners:assert len(output_owners)==1;active=next(iter(output_owners))
                if active==rid:work+=e['prefill_tokens']
                if any(o['is_finished'] for o in e['outputs']):active=None
            assert work==len(row['input_tokens'])-hits,(rid,work,hits)
            if 'response' in row:assert row['response']['choices'][0]['message']['content']==TOKENIZER.decode(ids)
            else:
                token_chunks=[c for c in row['chunks'] if c['chunk']['choices'][0]['delta'].get('role') is None and c['chunk']['choices'][0]['delta'].get('content') is not None and c['chunk']['choices'][0].get('finish_reason') is None]
                assert len(token_chunks)==len(ids),'TTFT/TPOT require one gateway event per generated token'
                row['ttft_ms']=(token_chunks[0]['received_ns']-row['sent_ns'])/1e6
                row['tpot_ms']=(token_chunks[-1]['received_ns']-token_chunks[0]['received_ns'])/1e6/(len(ids)-1)
                row['token_events']=len(token_chunks)
            row.update(engine=idx,expected_engine=expected,candidates=candidates,decision=d,predicted_hits=predicted,actual_hits=hits,prefill_work=work,generated_ids=ids,latency_ms=(row['end_ns']-row['sent_ns'])/1e6)
            result.append(row)
        with (self.out/output_name).open('a') as f:
            for row in result:f.write(json.dumps(row)+'\n')
        return result
    def verify(self,idxs=(0,1),filename="verified_state.json",allow_terminal_error=()):
        states=self.states();summaries=[]
        for idx in idxs:
            es=self.events(idx)
            errors=[e for e in es if 'error' in e]
            assert not errors or idx in allow_terminal_error,('unexpected engine error',idx,errors)
            es=[e for e in es if 'error' not in e]
            applied=[s for s in states if s['engine']==idx]
            assert len(es)==len(applied),(idx,len(es),len(applied))
            assert [e['step_id'] for e in es]==list(range(1,len(es)+1))
            assert len({e['engine_epoch'] for e in es})<=1
            pools=tagged(self.out/f'engine_{idx}.log','VALIDATION_KV_POOL')
            identity={r['internal_id']:str(r['external_id']) for r in tagged(self.out/f'engine_{idx}.log','VALIDATION_IDENTITY')}
            physical={};requests={};seq=0
            for e,s in zip(es,applied):
                for m in e['kv_journal']['mutations']:
                    assert m['mutation_seq']==seq+1;seq+=1;kind=m['type'];bid=str(m.get('block_id'))
                    if kind=='cache_reset':assert not requests;physical.clear()
                    elif kind=='cache_insert':assert bid not in physical;physical[bid]=m['cache_hash']
                    elif kind=='cache_remove':assert physical.pop(bid)==m['cache_hash']
                    elif kind=='request_blocks':requests[str(m['request_id'])]=m['block_ids']
                    elif kind=='request_release':assert requests.pop(str(m['request_id'])) is not None
                assert e['kv_journal']['mutation_seq']==seq==s['mutation_seq']
                assert s['step_id']==e['step_id'] and s['engine_epoch']==e['engine_epoch'] and s['valid']
                assert s['physical']==physical and s['requests']==requests
                ps=[p for p in pools if p['completed_step']==e['step_id'] and p['mutation_seq']==seq and p['pending_mutations']==0]
                assert ps,('missing independent pool snapshot',idx,e['step_id'])
                assert ps[0]['actual']==physical
                live={identity[rid]:groups[0] for rid,groups in ps[0]['occupancy'].items() if groups and groups[0]}
                assert live==requests,(idx,e['step_id'],live,requests)
            summaries.append({'engine':idx,'events':len(es),'mutations':seq,'epoch':es[-1]['engine_epoch'] if es else None,'physical':physical,'requests':requests,'ledger':applied[-1] if applied else None})
        if len(summaries)==2 and all(s['epoch'] for s in summaries):assert summaries[0]['epoch']!=summaries[1]['epoch']
        (self.out/filename).write_text(json.dumps(summaries,indent=2)+'\n')
        return summaries
    def metrics(self,name):
        with urllib.request.urlopen('http://127.0.0.1:18184/metrics',timeout=5) as r:text=r.read().decode()
        (self.out/name).write_text(text);return text
    def close(self):
        for p in reversed(self.processes):stop_owned(p)
        for f in self.logs:f.close()
        self.smi('npu_after.txt');self.status['finished']=stamp();self.save()


def correctness(s):
    s.start();x,y,z=prompt(0),prompt(1),prompt(2,700)
    (s.out/'fixtures.json').write_text(json.dumps({'X':x,'Y':y,'Z':z},indent=2))
    # Both engines first prove inference through the tracked uniform gateway.
    pre=s.controlled_pair(x,y,'preflight')
    assert [r['engine'] for r in pre]==[0,1]
    a=s.request(x,'A_only_target');s.drain();s.verify();s.reconcile([a]);assert a['engine']==0 and a['actual_hits']==256
    assert [int(c['hit_tokens']) for c in a['candidates']]==[256,0]
    s.reset([0])
    warm_b=s.controlled_pair(z,x,'controlled_B_warm',first_tokens=16)
    assert [r['engine'] for r in warm_b]==[0,1]
    b=s.request(x,'B_only_same_target');s.drain();s.verify();s.reconcile([b]);assert b['engine']==1 and b['actual_hits']==256
    assert [int(c['hit_tokens']) for c in b['candidates']]==[0,256]
    long_x=prompt(0,700)
    warm_a=s.controlled_pair(long_x,x,'controlled_A_warm',first_tokens=16)
    assert [r['engine'] for r in warm_a]==[1,0]
    both=s.controlled_pair(x,x,'both_cached_load')
    assert [r['engine'] for r in both]==[0,1]
    for r in both:assert [int(c['hit_tokens']) for c in r['candidates']]==[256,256] and r['actual_hits']==256
    s.reset([0,1]);cold=s.controlled_pair(prompt(3),prompt(4),'both_cold')
    assert [r['engine'] for r in cold]==[0,1] and all(r['actual_hits']==0 for r in cold)
    s.drain();s.verify();s.metrics('metrics_before_failure.txt')
    s.status['correctness_before_failure']=True;s.save()
    # Only our idle A is terminated. This closes its affected experimental session.
    stop_owned(s.engines[0])
    wait(lambda:'WORK_LOOP_STOPPED engine=0' in (s.out/'gateway.log').read_text(),15,'invalid backend exclusion propagated to idle worker')
    survivor=s.request(prompt(3),'invalid_A_healthy_B');s.drain([1]);s.verify([1]);s.reconcile([survivor])
    assert survivor['engine']==1 and survivor['candidates'][0]['routing_enabled']=='false'
    assert 'panicked at' not in (s.out/'gateway.log').read_text()
    s.status.update(passed=True,requests=len(s.responses),failure_isolation_passed=True,cache_location_swap_passed=True,independent_namespaces_verified=True)
    s.metrics('metrics_final.txt');s.save()
    print(s.out.name,': correctness passed',len(s.responses),'requests',flush=True)

if __name__=='__main__':
    base=Path(sys.argv[1]);mode=sys.argv[2];name=sys.argv[3]
    s=Session(base/name,mode=mode)
    try:correctness(s)
    except BaseException as e:s.status['failure']=repr(e);s.save();raise
    finally:s.close()
