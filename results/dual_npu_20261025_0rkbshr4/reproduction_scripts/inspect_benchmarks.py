"""Compare frozen benchmark traces, resets, request outcomes and raw clocks."""
import json,statistics,sys
from pathlib import Path
from experiment import Session, tagged, TOKENIZER
base=Path(sys.argv[1]);names=sys.argv[2:];summaries=[];traces=[]
for name in names:
    out=base/name;run=json.loads((out/'run.json').read_text());assert run.get('passed') and run.get('streaming_pilot_passed')
    s=Session.__new__(Session);s.out=out;s.policy=run['policy']
    states=s.verify(filename='final_independent_state.json')
    rows=[json.loads(l) for l in (out/'reconciled.jsonl').read_text().splitlines()]
    raw=[json.loads(l) for l in (out/'requests.jsonl').read_text().splitlines()]
    assert len(rows)==len(raw)==50 and all(r.get('status')==200 and 'error' not in r for r in raw)
    assert len({r['request_id'] for r in rows})==50
    assert all(r['token_events']==len(r['generated_ids'])==8 for r in rows)
    assert all(''.join(c['chunk']['choices'][0]['delta'].get('content') or '' for c in r['chunks'])==TOKENIZER.decode(r['generated_ids']) for r in rows)
    assert all(not x['requests'] and all(x['ledger'][k]==0 for k in ['bs','waiting','prefill','all_tokens','active_entries','exception_entries']) for x in states)
    data=[json.loads(l) for l in (out/'benchmark.jsonl').read_text().splitlines()];assert len(data)==6
    # A distinct real reset empties both backends before every measured trace.
    for idx in range(2):assert sum(e['event_kind']=='control' for e in s.events(idx))==7
    trace=json.loads((out/'trace_definition.json').read_text());traces.append(trace)
    for row in data:
        assert row['requests']==8 and row['failures']==0
        reqs=[r for r in rows if r['request_id'] in row['request_ids']]
        assert len(reqs)==8 and row['actual_hits']==sum(r['actual_hits'] for r in reqs)
        assert all(abs(a-b)<1e-9 for a,b in zip(row['planned_arrival_offsets_seconds'],[0,.04,2,2.04,4,4.04,6,6.04]))
        assert all(x['count']>0 and x['mean_us'] is not None for x in row['instrumented_histograms'].values())
    lateness=[x for r in data for x in r['arrival_lateness_ms']]
    shutdown_log=(out/'gateway.log').read_text().split('signal received, starting graceful shutdown',1)
    assert 'panicked at' not in shutdown_log[0]
    summaries.append({'arrival_lateness_max_ms':max(lateness),'shutdown_panic':len(shutdown_log)>1 and 'panicked at' in shutdown_log[1],'stream_text_matches_full_token_decode':True,'name':name,'policy':run['policy'],'requests':50,'measured_requests':48,'raw_token_timestamps_verified':True,'ledger_zero':True,'state_events':[x['events'] for x in states],'rows':data})
assert len(traces)==2 and traces[0]==traces[1]
comparison=[]
for trace in traces[0]:
    for s in summaries:
        rs=[r for r in s['rows'] if r['workload']==trace]
        comparison.append({'workload':trace,'policy':s['policy'],'repeats':len(rs),**{k:statistics.mean(r[k] for r in rs) for k in ['request_throughput_rps','output_token_throughput_tps','total_token_throughput_tps','ttft_mean_ms','tpot_mean_ms','actual_hits','prefill_work']},'request_distribution':{str(i):sum(r['distribution'][str(i)] for r in rs) for i in range(2)}})
result={'passed':True,'identical_trace_and_arrival_definition':True,'sessions':summaries,'comparison':comparison,'limitations':['debug instrumented eager, tiny model and KV pool','two repeats only, no confidence intervals or significance','max_num_seqs=1 per instance','event-collection incremental overhead not isolated','TTFT is first token delta receipt, TPOT is (last-first)/(N-1), role/finish events excluded','token throughput is generated tokens over first-send to last-DONE; separate total-token accounting provided']}
(base/'benchmark_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(comparison,indent=2))
