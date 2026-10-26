#!/usr/bin/env python3
"""CPU-only checks, real loopback ZMQ/HTTP/SSE; no model/NPU execution."""
import hashlib,json,os,socket,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ascend_minimal_events';OUT.mkdir(exist_ok=True)
FEATURES='vllm-backend,radixtree-blockhash,default-hash-algo,determinent-schedule,lmetric-q'

def run(command,name,**kwargs):
    with (OUT/(name+'.log')).open('w') as log:
        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300,**kwargs)
    print(name,p.returncode,flush=True)
    return {'command':command,'exit_code':p.returncode,'log':str(OUT/(name+'.log'))}

def main():
    report={'npu_executed':False,'full_engine_initialized':False,'events':'SYNTHETIC ONLY'}
    report['python']=run([sys.executable,str(ROOT/'scripts/ascend_minimal/test_events.py')],'python_tests',cwd=ROOT)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    with tempfile.TemporaryDirectory(prefix='lmetric-libs-') as tmp, (OUT/'synthetic_server.log').open('w') as log:
        server=subprocess.Popen([sys.executable,str(ROOT/'scripts/ascend_minimal/synthetic_server.py'),'--port',str(port)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        try:
            for _ in range(100):
                try:
                    with urllib.request.urlopen(url+'/test/health',timeout=.2) as response:
                        assert json.load(response)['synthetic'];break
                except Exception:time.sleep(.05)
            else:raise RuntimeError('Synthetic server failed to start')
            for lib in ['ssl','crypto']:
                runtime=Path('/usr/lib64')/f'lib{lib}.so.3'
                if runtime.exists():(Path(tmp)/f'lib{lib}.so').symlink_to(runtime)
            env=dict(os.environ,LIBRARY_PATH=tmp+':'+os.environ.get('LIBRARY_PATH',''),LMETRIC_SYNTHETIC_URL=url)
            cmd=['cargo','+1.88.0','test','-p','router','--lib','--no-default-features','--features',FEATURES,'--offline','--locked','--jobs','4','ascend_request','--','--nocapture','--include-ignored','--test-threads=1']
            report['rust']=run(cmd,'rust_tests',cwd=ROOT/'blitz-router',env=env)
            broad=cmd[:cmd.index('ascend_request')]+['--','--test-threads=1']
            report['router_regression']=run(broad,'router_regression',cwd=ROOT/'blitz-router',env=env)
            simulator=cmd[:cmd.index('ascend_request')]+['--no-run']
            simulator[simulator.index('--features')+1]=FEATURES+',simulator'
            report['simulator_compile']=run(simulator,'simulator_compile',cwd=ROOT/'blitz-router',env=env)
        finally:
            server.terminate()
            try:server.wait(timeout=5)
            except subprocess.TimeoutExpired:server.kill();server.wait()
    lines=(OUT/'rust_tests.log').read_text().splitlines()
    report['rust']['cases']=[json.loads(s.split('AUDIT_RESULT ',1)[1]) for s in lines if 'AUDIT_RESULT ' in s]
    report['rust']['summary']=next((s for s in lines if s.startswith('test result:')),None)
    report['passed']=all(report[name]['exit_code']==0 for name in ['python','rust','router_regression','simulator_compile'])
    (OUT/'tests.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'passed':report['passed'],'rust_summary':report['rust']['summary']}))
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
