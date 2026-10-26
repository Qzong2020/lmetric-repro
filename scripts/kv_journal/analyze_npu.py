"""Verify real SSE/Rust journals against independently read live BlockPool maps."""
import collections,json,sys
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path('/root/lmetric-repro');OUT=ROOT/'results/kv_journal_20261025_j7bol44q'/sys.argv[1]
def rows(path,tag):
 return [json.loads(line.split(tag+' ',1)[1]) for line in path.read_text().splitlines() if tag+' ' in line]
run=json.loads((OUT/'run.json').read_text());assert run['passed_driver']
events=[json.loads(line) for line in (OUT/'sse.jsonl').read_text().splitlines()]
applied=rows(OUT/'rust.log','KV_APPLIED');requests=rows(OUT/'rust.log','KV_REAL_REQUEST');result=rows(OUT/'rust.log','KV_REAL_RESULT')[-1]
pools=rows(OUT/'server.log','VALIDATION_KV_POOL');runtime=rows(OUT/'server.log','VALIDATION_RUNTIME')[-1]
identities={r['internal_id']:str(r['external_id']) for r in rows(OUT/'server.log','VALIDATION_IDENTITY')}
assert len(requests)==8 and result['requests']==8
assert [e['step_id'] for e in events]==list(range(1,len(events)+1))
assert [a['step_id'] for a in applied]==list(range(1,len(events)+1))
assert max(p['completed_step'] for p in pools)==len(events)
assert len({e['engine_epoch'] for e in events})==1
assert runtime['num_blocks']==8 and runtime['kv_groups']==1 and runtime['scheduler_block_size']==runtime['hash_block_size']==128
assert runtime['async_scheduling']==(run['mode']=='async') and runtime['enforce_eager']
physical={};occupancy={};seq=0;counts=collections.Counter();verified=[];reused=[];last_removed={};snapshots=[]
T=Tokenizer.from_file('/models/Qwen3-0.6B/tokenizer.json')
for e,a in zip(events,applied):
 assert e['lmetric_protocol']=='request-kv-v1' and e['kv_events'] and e['kv_groups']==1
 assert a['valid'] and a['ledger']['prefill']>=0
 for m in e['kv_journal']['mutations']:
  assert m['mutation_seq']==seq+1 and m['group_id']==0 and m['block_size']==128
  seq+=1;kind=m['type'];counts[kind]+=1
  if kind=='cache_reset':
   assert not occupancy
   physical.clear()
  elif kind=='cache_insert':
   bid=str(m['block_id']);assert bid not in physical
   assert len(m['prefix_token_ids'])==(m['logical_index']+1)*128
   if bid in last_removed and last_removed[bid]!=m['cache_hash']:reused.append({'block_id':int(bid),'mutation_seq':seq,'old_hash':last_removed[bid],'new_hash':m['cache_hash']})
   physical[bid]=m['cache_hash']
  elif kind=='cache_remove':
   bid=str(m['block_id']);assert physical.pop(bid)==m['cache_hash'];last_removed[bid]=m['cache_hash']
  elif kind=='request_blocks':occupancy[str(m['request_id'])]=m['block_ids']
  elif kind=='request_release':assert occupancy.pop(str(m['request_id'])) is not None
  else:raise AssertionError(kind)
 assert e['kv_journal']['mutation_seq']==seq==a['mutation_seq']
 assert a['physical']==physical and a['requests']==occupancy
 matches=[p for p in pools if p['completed_step']==e['step_id'] and p['mutation_seq']==seq and p['pending_mutations']==0]
 assert matches,('No independent post-update pool snapshot',e['step_id'],seq)
 p=matches[0];assert p['actual']==physical,('physical divergence',e['step_id'])
 live={identities[rid]:groups[0] for rid,groups in p['occupancy'].items() if groups and groups[0]}
 assert live==occupancy,('occupancy divergence',e['step_id'],live,occupancy)
 verified.append(e['step_id'])
 snapshots.append({'step_id':e['step_id'],'mutation_seq':seq,'physical_maps':len(physical),'active_occupancies':len(occupancy),'free_blocks':p['free_blocks'],'ledger':a['ledger']})
for r in requests:
 rid=r['request_id'];statuses=[o for e in events for o in e['outputs'] if o['request_id']==rid]
 assert statuses[0]['state']=='PREFILL' and all(o['state']=='DECODE' for o in statuses[1:])
 assert sum(o['is_finished'] for o in statuses)==1 and statuses[-1]['is_finished']
 tokens=[t for o in statuses for t in o['new_token_ids']]
 assert len(tokens)==8 and T.decode(tokens)==r['response']['choices'][0]['text']
 assert r['actual_hits']==r['predicted_hits'] and r['prefill_work']==313-r['actual_hits']
 assert r['ledger']==dict(bs=0,waiting=0,prefill=0,all_tokens=0)
assert requests[1]['actual_hits']==256 and requests[6]['actual_hits']==requests[7]['actual_hits']==0
controls=[e for e in events if e['event_kind']=='control'];assert len(controls)==1 and not controls[0]['outputs']
assert [m['type'] for m in controls[0]['kv_journal']['mutations']]==['cache_reset']
assert reused and counts['cache_remove']>0 and counts['cache_reset']==2
assert not occupancy and result['ledger']==dict(bs=0,waiting=0,prefill=0,all_tokens=0)
assert pools[-1]['free_blocks']==7 and pools[-1]['actual']==physical
summary={'real_npu':True,'mode':run['mode'],'requests':8,'events':len(events),'mutations':seq,'mutation_types':dict(counts),'physical_snapshot_matches':len(verified),'physical_id_reuses':reused,'hits':[r['actual_hits'] for r in requests],'prefill_work':[r['prefill_work'] for r in requests],'runtime':runtime,'final':result,'snapshots':snapshots,'note':'scheduler cache metadata; no device content completion or performance claim'}
(OUT/'verified.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['mode','requests','events','mutations','mutation_types','physical_snapshot_matches','hits']}))
