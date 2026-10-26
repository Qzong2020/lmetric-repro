"""Small paired trace replay. Correctness evidence is a mandatory prerequisite."""
import concurrent.futures as futures
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from experiment import Session, prompt

def metric_totals(text):
    totals={}
    for line in text.splitlines():
        if line.startswith('#'):continue
        parts=line.split()
        if len(parts)!=2:continue
        name=parts[0].split('{')[0]
        if name.endswith(('_sum','_count')):
            try:totals[name]=totals.get(name,0)+float(parts[1])
            except ValueError:pass
    return totals

def run_trace(s,name,fixtures,repeat):
    s.reset([0,1]);before=metric_totals(s.metrics(f'{name}_{repeat}_metrics_before.txt'))
    # Identical open-loop arrival offsets and prompt IDs for both policies.
    offsets=[0,.04,2,2.04,4,4.04,6,6.04][:len(fixtures)]
    start=time.monotonic();rows=[]
    def request_at(i):
        remaining=start+offsets[i]-time.monotonic()
        if remaining>0:time.sleep(remaining)
        return s.request(fixtures[i],f'{name}_r{repeat}_{i}',stream=True)
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(request_at,i) for i in range(len(fixtures))]
        rows=[f.result(timeout=180) for f in fs]
    s.drain();s.verify();s.reconcile(rows)
    after=metric_totals(s.metrics(f'{name}_{repeat}_metrics_after.txt'))
    hist={}
    for stem in ['blitz_policy_schedule_duration_seconds','blitz_kv_prepare_duration_seconds','blitz_step_apply_duration_seconds']:
        count=after.get(stem+'_count',0)-before.get(stem+'_count',0)
        total=after.get(stem+'_sum',0)-before.get(stem+'_sum',0)
        hist[stem]={'count':count,'total_seconds':total,'mean_us':total/count*1e6 if count else None}
    elapsed=(max(r['end_ns'] for r in rows)-min(r['sent_ns'] for r in rows))/1e9
    result={'workload':name,'repeat':repeat,'policy':s.policy,'requests':len(rows),'planned_arrival_offsets_seconds':offsets,'elapsed_seconds':elapsed,'request_throughput_rps':len(rows)/elapsed,'output_token_throughput_tps':sum(len(r['generated_ids']) for r in rows)/elapsed,'total_token_throughput_tps':sum(len(r['input_tokens'])+len(r['generated_ids']) for r in rows)/elapsed,'ttft_mean_ms':statistics.mean(r['ttft_ms'] for r in rows),'tpot_mean_ms':statistics.mean(r['tpot_ms'] for r in rows),'actual_hits':sum(r['actual_hits'] for r in rows),'predicted_hits':sum(r['predicted_hits'] for r in rows),'prefill_work':sum(r['prefill_work'] for r in rows),'distribution':{str(i):sum(r['engine']==i for r in rows) for i in range(2)},'instrumented_histograms':hist,'failures':0,'request_ids':[r['request_id'] for r in rows],'arrival_lateness_ms':[(r['sent_ns']/1e9-start-offsets[i])*1000 for i,r in enumerate(rows)]}
    with (s.out/'benchmark.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    return result

def main():
    base=Path(sys.argv[1]);policy=sys.argv[2];name=sys.argv[3]
    gates=json.loads((base/'correctness.json').read_text())
    for mode in ['sync','async']:
        run=json.loads((base/gates[mode]/'verified.json').read_text())
        assert run.get('passed') and run.get('failure_isolation_passed'),'correctness gates must pass before benchmarking'
    s=Session(base/name,policy=policy,mode='sync')
    try:
        s.start()
        # Streaming pilot validates timing semantics and the baseline end-to-end path.
        with futures.ThreadPoolExecutor(max_workers=2) as p:
            a=p.submit(s.request,prompt(3),'streaming_pilot_A',stream=True)
            time.sleep(.04)
            b=p.submit(s.request,prompt(4),'streaming_pilot_B',stream=True)
            pilot=[a.result(timeout=180),b.result(timeout=180)]
        s.drain();s.verify();s.reconcile(pilot)
        assert {r['engine'] for r in pilot}=={0,1}
        s.status['streaming_pilot_passed']=True;s.save()
        traces={'cache_friendly':[0,1,0,0,0,0,0,0],'mixed':[0,1,1,0,2,3,0,1],'low_hit':list(range(10,18))}
        (s.out/'trace_definition.json').write_text(json.dumps({name:[prompt(i) for i in ids] for name,ids in traces.items()},indent=2))
        summaries=[]
        for repeat in range(2):
            for trace,ids in traces.items():summaries.append(run_trace(s,trace,[prompt(i) for i in ids],repeat))
        s.status.update(passed=True,requests=len(s.responses),bounded_repeats=2,measured_requests=48)
        s.save()
    except BaseException as e:s.status['failure']=repr(e);s.save();raise
    finally:s.close()
if __name__=='__main__':main()
