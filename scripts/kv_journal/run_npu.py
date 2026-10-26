"""Bounded real KV lifecycle; never kills processes started outside this run."""
import json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/root/lmetric-repro');BASE=ROOT/'results/kv_journal_20261025_j7bol44q'
mode=sys.argv[1];assert mode in ['sync','async']
name=sys.argv[2] if len(sys.argv)>2 else mode+'_attempt1'
OUT=BASE/name;OUT.mkdir()
status={'mode':mode,'scope':'8 real inference requests plus one idle reset; one NPU; no performance experiment'}
def save(): (OUT/'run.json').write_text(json.dumps(status,indent=2)+'\n')
def smi(name):
 p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15)
 (OUT/name).write_text(p.stdout+p.stderr);return p.stdout
before=smi('npu_before.txt')
if 'No running processes found' not in before:raise SystemExit('NPU has an existing process; do not start or terminate it')
from tokenizers import Tokenizer
T=Tokenizer.from_file('/models/Qwen3-0.6B/tokenizer.json')
texts=['Please explain how a request passes through a language model inference server. ',
       'Ocean currents distribute heat through the deep sea and affect coastal weather patterns. ',
       'Astronomy studies distant galaxies, stellar evolution, and the history of the universe. ',
       'Botanical gardens protect diverse plants and teach visitors about local ecosystems. ',
       'Railway networks connect cities through carefully scheduled passenger and freight services. ']
fixture={'scope':'real text tokenized by existing model tokenizer, no synthetic KV mutations','prompts':[]}
for t in texts:
 ids=T.encode(t*40).ids[:313];assert len(ids)==313
 fixture['prompts'].append({'text':T.decode(ids),'tokens':ids})
(OUT/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='0',VLLM_LMETRIC_EVENTS='1',VLLM_LMETRIC_KV_EVENTS='1',VLLM_SERVER_DEV_MODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(ROOT/'engines/vllm')+':'+os.getenv('PYTHONPATH',''))
cmd=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/kv_journal/launch.py'),'--model','/models/Qwen3-0.6B','--served-model-name','ascend-small','--host','127.0.0.1','--port','18081','--tensor-parallel-size','1','--pipeline-parallel-size','1','--max-model-len','1024','--max-num-seqs','1','--max-num-batched-tokens','128','--gpu-memory-utilization','0.20','--num-gpu-blocks-override','8','--enforce-eager','--async-scheduling' if mode=='async' else '--no-async-scheduling','--enable-prefix-caching','--enable-chunked-prefill','--block-size','128']
status['server_command']=cmd;save()
with (OUT/'server.log').open('w') as log:
 server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 status['server_pid']=server.pid;save()
 try:
  deadline=time.monotonic()+240
  while time.monotonic()<deadline:
   if server.poll() is not None:raise RuntimeError('engine exited before readiness')
   try:
    with urllib.request.urlopen('http://127.0.0.1:18081/health',timeout=1) as r:
     if r.status==200:break
   except Exception:time.sleep(1)
  else:raise TimeoutError('240 second engine startup bound')
  status['ready']=True;smi('npu_loaded.txt');save()
  driver=[str(ROOT/'blitz-router/target/debug/deps/router-edaf117dc4e07598'),'kv_journal_real_npu_cache_lifecycle','--ignored','--nocapture','--test-threads=1']
  denv=dict(env,LMETRIC_KV_URL='http://127.0.0.1:18081',LMETRIC_KV_FIXTURE=str(OUT/'fixture.json'),LMETRIC_KV_SSE_FILE=str(OUT/'sse.jsonl'),LMETRIC_REQUIRE_KV_JOURNAL='1')
  status['driver_command']=driver;save()
  with (OUT/'rust.log').open('w') as f:r=subprocess.run(driver,env=denv,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  status['driver_exit_code']=r.returncode;save()
  if r.returncode!=0:raise RuntimeError('real KV consumer failed; stop this session')
  status['passed_driver']=True
 except Exception as exc:
  status['failure']=repr(exc)
 finally:
  if server.poll() is None:
   os.killpg(server.pid,signal.SIGTERM)
   try:server.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
  status['server_exit_code']=server.returncode;smi('npu_after.txt');save()
print(json.dumps(status),flush=True)
raise SystemExit(0 if status.get('passed_driver') else 1)
