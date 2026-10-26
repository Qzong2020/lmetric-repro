"""Check and export evidence from real server/Rust logs, never invent defaults."""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path('/root/lmetric-repro');OUT=ROOT/'results/ascend_single_card_validation'
def records(path,tag):
 out=[]
 for line in path.read_text().splitlines():
  if tag+' ' in line:
   out.append(json.loads(line.split(tag+' ',1)[1]))
 return out
mode=sys.argv[1];server=OUT/(mode+'_server.log');rust=OUT/(mode+'_rust.log')
events=records(rust,'REAL_RAW_SSE');applied=records(rust,'REAL_APPLIED');steps=records(server,'VALIDATION_STEP');runtime=records(server,'VALIDATION_RUNTIME');result=records(rust,'REAL_RESULT')
assert runtime and result and events
assert [e['step_id'] for e in events]==list(range(1,len(events)+1))
assert len({e['engine_epoch'] for e in events})==1
assert len(events)==max(s['seq'] for s in steps if s['seq'] is not None)
assert [e['step_id'] for e in applied]==[e['step_id'] for e in events]
assert result[-1]['ledger']==dict(bs=0,waiting=0,prefill=0,all_tokens=0)
assert sum(e['prefill_tokens'] for e in events)==313
assert sum(len(o['new_token_ids']) for e in events for o in e['outputs'])==8
statuses=[o for e in events for o in e['outputs']]
assert statuses[0]['state']=='PREFILL' and all(o['state']=='DECODE' for o in statuses[1:])
assert sum(o['is_finished'] for o in statuses)==1 and statuses[-1]['is_finished']
assert all(o['request_id']==2026102501 for o in statuses)
assert all(isinstance(e['latency'],int) and e['latency']>=0 and e['latency_kind']=='schedule_to_completion_host_ms' for e in events)
assert all(e['kv_groups']==1 and e['block_size']==128 and e['kv_events'] is False for e in events)
assert steps[-1]['running']==0 and steps[-1]['waiting']==0 and steps[-1]['requests']==[]
expected=list(range(0,313,128))+list(range(313,320))
assert [o['prev_computed_tokens'] for o in statuses]==list(range(256,257))+list(range(313,320))
http=records(rust,'REAL_HTTP')[-1]
row={'mode':mode,'runtime':runtime[-1],'imports':records(server,'VALIDATION_IMPORT'),'identity_map':records(server,'VALIDATION_IDENTITY'),'hub_states':records(server,'VALIDATION_HUB'),'step_calls':steps,'events':events,'ledger_after_each_event':applied,'http_response':http,'result':result[-1],'complete_sequence_and_ledger_verified':True,'latency_semantics':'host schedule to completion; no device or performance claim'}
(OUT/(mode+'_evidence.json')).write_text(json.dumps(row,indent=2,ensure_ascii=False)+'\n')
(OUT/(mode+'_events.jsonl')).write_text(''.join(json.dumps(e,separators=(',',':'))+'\n' for e in events))
print(json.dumps({'mode':mode,'events':len(events),'runtime':runtime[-1]['scheduler'],'final_ledger':result[-1]['ledger']}))
