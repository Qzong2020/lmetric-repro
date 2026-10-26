"""Offline builds, isolated logs, identical feature sets except policy."""
import os, shutil, subprocess, sys
from pathlib import Path
R=Path('/root/lmetric-repro');O=Path(sys.argv[1]);env=dict(os.environ,LIBRARY_PATH=str(O/'build_lib'))
base=['cargo','+1.88.0'];features='vllm-backend,radixtree-blockhash,default-hash-algo,determinent-schedule,'
for policy in ['lmetric-q','join-shortest-weight-q']:
    common=['-p','router','--no-default-features','--features',features+policy,'--locked','--offline','--jobs','6']
    for label,cmd in [('tests',base+['test',*common,'--lib','--','--test-threads=1']),('build',base+['build',*common])]:
        with (O/(policy+'_'+label+'.log')).open('w') as f:
            p=subprocess.run(cmd,cwd=R/'blitz-router',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=360)
        if p.returncode:raise SystemExit(f'{policy} {label} failed: {p.returncode}')
        print(policy,label,'passed',flush=True)
    (O/'bins').mkdir(exist_ok=True);shutil.copy2(R/'blitz-router/target/debug/router',O/'bins'/policy)
