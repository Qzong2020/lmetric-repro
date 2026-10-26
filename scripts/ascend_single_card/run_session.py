"""Run one real request, requiring the native Rust SSE handshake first."""
import json,os,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/root/lmetric-repro');OUT=ROOT/'results/ascend_single_card_validation'
mode=sys.argv[1];url='http://127.0.0.1:18080';start=time.monotonic()
while time.monotonic()-start<240:
 try:
  with urllib.request.urlopen(url+'/health',timeout=1) as r:
   if r.status==200:break
 except Exception:time.sleep(1)
else:raise SystemExit('engine not ready in 240 seconds')
handshake=OUT/(mode+'_handshake.txt');assert not handshake.exists()
env=dict(os.environ,LMETRIC_REAL_URL=url,LMETRIC_REAL_PROMPT=str(OUT/'prompt.json'),LMETRIC_HANDSHAKE_FILE=str(handshake))
cmd=[str(ROOT/'blitz-router/target/debug/deps/router-edaf117dc4e07598'),'real_npu_sse_completion','--ignored','--nocapture','--test-threads=1']
with (OUT/(mode+'_npu_smi.txt')).open('w') as f:subprocess.run(['npu-smi','info'],stdout=f,stderr=subprocess.STDOUT,timeout=15)
with (OUT/(mode+'_rust.log')).open('w') as f:
 try:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=150);code=r.returncode
 except subprocess.TimeoutExpired:code='timeout'
(OUT/(mode+'_driver_result.json')).write_text(json.dumps({'command':cmd,'exit_code':code,'handshake':handshake.exists(),'health_200':True},indent=2)+'\n')
print(mode,'real driver exit',code,flush=True)
raise SystemExit(0 if code==0 else 1)
