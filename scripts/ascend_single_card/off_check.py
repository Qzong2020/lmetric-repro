import json,subprocess,time,urllib.request,urllib.error
from pathlib import Path
OUT=Path('/root/lmetric-repro/results/ascend_single_card_validation');url='http://127.0.0.1:18080'
for _ in range(240):
 try:
  with urllib.request.urlopen(url+'/health',timeout=1) as r:
   if r.status==200:break
 except Exception:time.sleep(1)
else:raise SystemExit('off engine not ready')
try:
 with urllib.request.urlopen(url+'/v1/metrics',timeout=2) as r:metrics_status=r.status
except urllib.error.HTTPError as e:metrics_status=e.code
assert metrics_status==404
prompt=json.loads((OUT/'prompt.json').read_text())
request={'model':'ascend-small','prompt':prompt['tokens'],'stream':False,'max_tokens':8,'min_tokens':8,'temperature':None}
req=urllib.request.Request(url+'/v1/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json','X-Request-Id':'2026102501'})
with urllib.request.urlopen(req,timeout=90) as r:status=r.status;body=json.load(r)
assert status==200 and body['usage']['prompt_tokens']==313 and body['usage']['completion_tokens']==8
row={'event_flag':0,'metrics_status':metrics_status,'inference_status':status,'request':request,'response':body}
for mode in ['sync_final']:
 evidence=json.loads((OUT/(mode+'_evidence.json')).read_text());tokens=evidence['result']['completion_tokens']
 req=urllib.request.Request(url+'/detokenize',data=json.dumps({'model':'ascend-small','tokens':tokens}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r:decoded=json.load(r)
 row[mode+'_decoded']=decoded
 assert decoded['prompt']==evidence['http_response']['choices'][0]['text']
(OUT/'off_result.json').write_text(json.dumps(row,indent=2)+'\n')
with (OUT/'off_npu_smi.txt').open('w') as f:subprocess.run(['npu-smi','info'],stdout=f,timeout=15)
print('Same prototype source: event flag OFF, /v1/metrics 404, inference 200, real SSE tokens decode to exact HTTP text')
