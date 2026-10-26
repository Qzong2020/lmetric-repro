"""Three actual gateway requests, one idle reset, fresh NPU session."""
import json,os,re,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/root/lmetric-repro');BASE=ROOT/'results/kv_journal_20261025_j7bol44q';OUT=BASE/(sys.argv[1] if len(sys.argv)>1 else 'gateway_final');OUT.mkdir()
status={'scope':'3 real HTTP gateway requests, one reset, one NPU; no performance test'}
def save(): (OUT/'run.json').write_text(json.dumps(status,indent=2)+'\n')
def smi(name):
 p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15);(OUT/name).write_text(p.stdout+p.stderr);return p.stdout
if 'No running processes found' not in smi('npu_before.txt'):raise SystemExit('NPU owned by another process; stop')
env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='0',VLLM_LMETRIC_EVENTS='1',VLLM_LMETRIC_KV_EVENTS='1',VLLM_SERVER_DEV_MODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(ROOT/'engines/vllm')+':'+os.getenv('PYTHONPATH',''),LMETRIC_REQUIRE_KV_JOURNAL='1',LOG_LEVEL='info,cache_tracking=debug,correction=info,scheduling=info',RUST_LOG='info,cache_tracking=debug,correction=info,scheduling=info')
cmd=json.loads((BASE/'sync_final/run.json').read_text())['server_command']
cmd=[str(ROOT/'scripts/kv_journal/launch_gateway.py') if x.endswith('/scripts/kv_journal/launch.py') else '18083' if x=='18081' else x for x in cmd]
(OUT/'clients.json').write_text(json.dumps(['http://127.0.0.1:18083']))
gcmd=[str(ROOT/'blitz-router/target/debug/router'),'--client-config',str(OUT/'clients.json'),'--hostname','127.0.0.1','--port','18084','--tokenizer-name','/models/Qwen3-0.6B','--use-tokenizer','--model-name','ascend-small','--kvcache-block-size','128','--max-concurrent-requests','1','--max-best-of','1','--validation-workers','1','--max-input-length','512','--max-total-tokens','1024','--chat-template-mode','none']
status.update(server_command=cmd,gateway_command=gcmd);save();server=gateway=None

def health(url,deadline,proc):
 while time.monotonic()<deadline:
  if proc.poll() is not None:raise RuntimeError('owned process exited during startup')
  try:
   with urllib.request.urlopen(url+'/health',timeout=1) as r:
    if r.status==200:return
  except Exception:time.sleep(.2)
 raise TimeoutError('bounded readiness timeout')
def post(url,body=None):
 req=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else b'',headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r:return r.status,r.read().decode()
logs=[]
try:
 sf=(OUT/'server.log').open('w');logs.append(sf);server=subprocess.Popen(cmd,env=env,stdout=sf,stderr=subprocess.STDOUT,start_new_session=True);status['server_pid']=server.pid;save()
 health('http://127.0.0.1:18083',time.monotonic()+240,server);smi('npu_loaded.txt')
 gf=(OUT/'gateway.log').open('w');logs.append(gf);gateway=subprocess.Popen(gcmd,env=env,stdout=gf,stderr=subprocess.STDOUT,start_new_session=True);status['gateway_pid']=gateway.pid;save()
 health('http://127.0.0.1:18084',time.monotonic()+30,gateway)
 deadline=time.monotonic()+15
 while time.monotonic()<deadline:
  if 'GET /v1/metrics HTTP/1.1" 200' in (OUT/'server.log').read_text():break
  if gateway.poll() is not None:raise RuntimeError('gateway exited before SSE subscription')
  time.sleep(.1)
 else:raise TimeoutError('real gateway must subscribe before first request')
 fixture=json.loads((BASE/'sync_final/fixture.json').read_text());text=fixture['prompts'][0]['text']
 results=[]
 for i in range(3):
  if i==2:
   code,body=post('http://127.0.0.1:18083/reset_prefix_cache');assert code==200
   deadline=time.monotonic()+5
   while time.monotonic()<deadline:
    frames=[json.loads(line.split('VALIDATION_GATEWAY_SSE ',1)[1])['event'] for line in (OUT/'server.log').read_text().splitlines() if 'VALIDATION_GATEWAY_SSE ' in line]
    if any(e.get('event_kind')=='control' for e in frames):break
    time.sleep(.05)
   else:raise TimeoutError('idle reset did not reach gateway SSE')
   time.sleep(.1)
  body={'model':'ascend-small','messages':[{'role':'user','content':text}],'max_tokens':8,'stream':False,'temperature':0.6}
  code,raw=post('http://127.0.0.1:18084/v1/chat/completions',body);response=json.loads(raw)
  results.append({'request':body,'status':code,'response':response})
  (OUT/'responses.json').write_text(json.dumps(results,indent=2)+'\n')
  assert code==200 and response['usage']['completion_tokens']==8 and response['usage']['prompt_tokens']==313
  time.sleep(.1)
 with urllib.request.urlopen('http://127.0.0.1:18084/metrics',timeout=5) as r:metrics=r.read().decode()
 (OUT/'metrics.txt').write_text(metrics)
 for name in ['active_requests','waiting_requests','pending_prefill_tokens','context_tokens']:
  values=re.findall(r'^blitz_replica_'+name+r'\{[^\n]*\}\s+([^\n]+)',metrics,re.M)
  assert values and all(float(v)==0 for v in values),(name,values)
 status['passed_requests']=3;status['ledger_metrics_zero']=True
except Exception as e:status['failure']=repr(e)
finally:
 for p in [gateway,server]:
  if p is not None and p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
 for f in logs:f.close()
 smi('npu_after.txt');save()
print(json.dumps(status),flush=True)
raise SystemExit(0 if status.get('passed_requests')==3 else 1)
