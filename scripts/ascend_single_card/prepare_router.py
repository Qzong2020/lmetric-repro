import json, os, shutil, subprocess
from pathlib import Path
ROOT=Path('/root/lmetric-repro');OUT=ROOT/'results/ascend_single_card_validation'
target=Path('/tmp/lmetric-real-npu-router')
assert not target.exists(), 'Do not overwrite an existing driver'
shutil.copytree(ROOT/'blitz-router',target,ignore=shutil.ignore_patterns('.git','target','__pycache__'))
p=target/'router/src/engine/colocation.rs'
s=p.read_text();marker='    async fn completion_event_loop('
assert marker in s
s=s.replace(marker,(ROOT/'scripts/ascend_single_card/real_npu.rs').read_text()+'\n'+marker,1);p.write_text(s)
p=target/'router/src/engine/client.rs';s=p.read_text()
marker='            self.sse_state = Some(SseState {'
assert s.count(marker)==1
s=s.replace(marker,'            if let Ok(p) = std::env::var("LMETRIC_HANDSHAKE_FILE") { std::fs::write(p, "real SSE HTTP 200 before request").unwrap(); }\n'+marker)
marker='                    Some(Ok(es::SSE::Event(e))) => {'
assert s.count(marker)==1
s=s.replace(marker,marker+'\n                        println!("REAL_RAW_SSE {}", e.data);')
p.write_text(s)
libs=target/'validation-libs';libs.mkdir()
for lib in ['ssl','crypto']:(libs/f'lib{lib}.so').symlink_to(f'/usr/lib64/lib{lib}.so.3')
cmd=['cargo','+1.88.0','test','-p','router','--lib','--no-default-features','--features','vllm-backend,radixtree-blockhash,default-hash-algo,determinent-schedule,lmetric-q','--offline','--locked','--jobs','4','real_npu_sse_completion','--no-run']
env=dict(os.environ,CARGO_TARGET_DIR=str(ROOT/'blitz-router/target'),LIBRARY_PATH=str(libs)+':'+os.getenv('LIBRARY_PATH',''))
(OUT/'rust_driver.json').write_text(json.dumps({'cwd':str(target),'command':cmd,'instrumentation':'copy only; add handshake observation and raw SSE log; production receiver and completion loop unchanged'},indent=2))
with (OUT/'rust_build.log').open('w') as f:p=subprocess.run(cmd,cwd=target,env=env,stdout=f,stderr=subprocess.STDOUT)
print('Rust driver build exit',p.returncode);raise SystemExit(p.returncode)
