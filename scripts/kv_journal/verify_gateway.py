"""Cross-check the production gateway, real SSE, and independent NPU pool."""
import collections,json,re,sys
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path('/root/lmetric-repro');OUT=ROOT/'results/kv_journal_20261025_j7bol44q'/(sys.argv[1] if len(sys.argv)>1 else 'gateway_release')
def rows(tag):return [json.loads(l.split(tag+' ',1)[1]) for l in (OUT/'server.log').read_text().splitlines() if tag+' ' in l]
events=[r['event'] for r in rows('VALIDATION_GATEWAY_SSE')];pools=rows('VALIDATION_KV_POOL')
responses=json.loads((OUT/'responses.json').read_text());run=json.loads((OUT/'run.json').read_text())
assert run['passed_requests']==3 and run['ledger_metrics_zero']
assert [e['step_id'] for e in events]==list(range(1,len(events)+1))
assert len({e['engine_epoch'] for e in events})==1
log=(OUT/'gateway.log').read_text();steps=[int(re.search(r'step_id=(\d+)',l).group(1)) for l in log.splitlines() if 'SSE_EVENT' in l]
assert steps==list(range(1,len(events)+1)),(steps,len(events))
physical={};seq=0;counts=collections.Counter()
for e in events:
 for m in e['kv_journal']['mutations']:
  assert m['mutation_seq']==seq+1;seq+=1;kind=m['type'];counts[kind]+=1
  if kind=='cache_reset':physical.clear()
  elif kind=='cache_insert':
   bid=str(m['block_id']);assert bid not in physical;physical[bid]=m['cache_hash']
  elif kind=='cache_remove':assert physical.pop(str(m['block_id']))==m['cache_hash']
 matches=[p for p in pools if p['completed_step']==e['step_id'] and p['mutation_seq']==seq and p['pending_mutations']==0]
 assert matches and matches[0]['actual']==physical
 assert e['kv_journal']['mutation_seq']==seq
T=Tokenizer.from_file('/models/Qwen3-0.6B/tokenizer.json');hits=[]
for i,r in enumerate(responses):
 statuses=[o for e in events for o in e['outputs'] if o['request_id']==i]
 hits.append(statuses[0]['hit_token_cnt']);ids=[t for o in statuses for t in o['new_token_ids']]
 assert len(ids)==8 and r['response']['choices'][0]['message']['content']==T.decode(ids)
 assert r['response']['usage']==dict(prompt_tokens=313,completion_tokens=8,total_tokens=321)
 assert statuses[-1]['is_finished'] and sum(o['is_finished'] for o in statuses)==1
assert hits==[0,256,0]
assert len([e for e in events if e['event_kind']=='control'])==1
assert log.count('REQUEST_FINISH')==3 and 'predicted=256 actual=256' in log
assert not [l for l in log.splitlines() if ' ERROR ' in l]
summary={'real_npu':True,'full_production_gateway':True,'requests':3,'sse_events':len(events),'consumer_step_log_matches':len(steps),'mutations':seq,'mutation_types':dict(counts),'actual_hits':hits,'usage_verified':True,'text_equals_true_sse_token_decode':True,'metrics_zero':True,'final_physical_map':physical}
(OUT/'verified.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
