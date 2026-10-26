"""Offline independent audit; keep original driver failures intact."""
import collections,json,re,sys,tempfile
from pathlib import Path
from experiment import Session, tagged
base=Path(sys.argv[1]);out=base/sys.argv[2]
s=Session.__new__(Session);s.out=out;s.policy='lmetric-q'
run=json.loads((out/'run.json').read_text())
rows=[json.loads(l) for l in (out/'requests.jsonl').read_text().splitlines()]
assert len(rows)==13 and all('error' not in r for r in rows)
audit=Path(tempfile.mkdtemp(prefix='offline_audit_',dir=out))
states=s.verify(filename=str(audit.relative_to(out)/'verified_state.json'),allow_terminal_error=(0,))
s.reconcile(rows,output_name=str(audit.relative_to(out)/'reconciled.jsonl'))
labels={r['label']:r for r in rows}
assert [labels[x]['engine'] for x in ['preflight_first','preflight_second']]==[0,1]
for label,hits,engine in [('A_only_target',[256,0],0),('B_only_same_target',[0,256],1)]:
    r=labels[label];assert [int(c['hit_tokens']) for c in r['candidates']]==hits and r['engine']==engine and r['actual_hits']==256
assert labels['A_only_target']['input_tokens']==labels['B_only_same_target']['input_tokens']
for prefix in ['controlled_B_warm','controlled_A_warm','both_cached_load','both_cold']:
    pair=[labels[prefix+'_first'],labels[prefix+'_second']]
    assert [r['engine'] for r in pair]==([1,0] if prefix=='controlled_A_warm' else [0,1])
    if prefix=='both_cached_load':
        assert all([int(c['hit_tokens']) for c in r['candidates']]==[256,256] and r['actual_hits']==256 for r in pair)
        assert int(pair[1]['candidates'][0]['bs'])>0 and int(pair[1]['candidates'][1]['bs'])==0
    if prefix=='both_cold':assert all(r['actual_hits']==0 for r in pair)
last=labels['invalid_A_healthy_B'];assert last['engine']==1 and last['candidates'][0]['routing_enabled']=='false'
log=(out/'gateway.log').read_text();inference_log=log.split('signal received, starting graceful shutdown',1)[0]
assert 'WORK_LOOP_STOPPED engine=0' in inference_log and 'RECV_STEP_ERROR engine=0' in inference_log and 'panicked at' not in inference_log
assert all(not s['requests'] and all(s['ledger'][k]==0 for k in ['bs','waiting','prefill','all_tokens','active_entries','exception_entries']) for s in states)
loaded=(out/'npu_loaded_1.txt').read_text();runtimes=[]
for idx,physical in enumerate([0,6]):
    runtime=tagged(out/f'engine_{idx}.log','VALIDATION_RUNTIME')[-1];runtimes.append(runtime)
    assert re.search(r'\|\s+'+str(physical)+r'\s+0\s+\|\s+'+str(runtime['pid'])+r'\s+',loaded),('wrong physical binding',idx,runtime['pid'])
    assert runtime['async_scheduling']==(run['mode']=='async') and runtime['enforce_eager'] and runtime['kv_groups']==1 and runtime['num_blocks']==8
# Real physical IDs overlap at different contents, and both sequence spaces begin at 1.
views=[[r for r in s.states() if r['engine']==i] for i in range(2)]
assert views[0][0]['step_id']==views[1][0]['step_id']==1
assert any(any(k in b['physical'] and v!=b['physical'][k] for k,v in a['physical'].items()) for a in views[0] for b in views[1])
summary={'shutdown_panic_after_all_requests':'panicked at' in log[len(inference_log):],'offline_audit':str(audit.relative_to(out)),'passed':True,'mode':run['mode'],'requests':len(rows),'per_backend':dict(collections.Counter(r['engine'] for r in rows)),'subscribed_before_requests':run['subscribed_before_requests'],'cache_location_swap_passed':True,'independent_namespaces_verified':True,'failure_isolation_passed':True,'original_driver_failure':run.get('failure'),'offline_verifier_fix':'terminal error frames have no outputs; preserve and classify separately','events':[x['events'] for x in states],'mutations':[x['mutations'] for x in states],'engine_epochs':[x['epoch'] for x in states],'real_device_binding_verified':True,'runtime':runtimes,'decisions':[{'label':r['label'],'request_id':r['request_id'],'engine':r['engine'],'predicted_hits':r['predicted_hits'],'actual_hits':r['actual_hits'],'prefill_work':r['prefill_work'],'candidates':r['candidates']} for r in rows]}
(out/'verified.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['runtime','decisions']}))
