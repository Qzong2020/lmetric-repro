"""Preserve the incoming work and prior evidence before dual-NPU experiments."""
import hashlib, json, shutil, subprocess, tempfile
from pathlib import Path
ROOT = Path('/root/lmetric-repro')
OUT = Path(tempfile.mkdtemp(prefix='dual_npu_20261025_', dir=ROOT/'results'))
rows = {}
for name in ['engines/vllm', 'blitz-router', 'blitz-router/request-sim', '/vllm-workspace/vllm', '/vllm-workspace/vllm-ascend']:
    repo = ROOT/name
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True)
    label = name.strip('/').replace('/', '__')
    files = git('ls-files', '--cached', '--others', '--exclude-standard').splitlines()
    rows[name] = dict(head=git('rev-parse', 'HEAD'), status=git('status', '--short', '--branch'), sha256={})
    (OUT/(label+'.patch')).write_text(git('diff', '--binary'))
    for f in files:
        p = repo/f
        if p.is_file():
            rows[name]['sha256'][f] = hashlib.sha256(p.read_bytes()).hexdigest()
            if not name.startswith('/') and name != 'blitz-router/request-sim':
                dest = OUT/'source_before'/label/f
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
rows['historical_results'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'results').rglob('*') if p.is_file() and OUT not in p.parents}
(OUT/'baseline.json').write_text(json.dumps(rows, indent=2)+'\n')
for cmd, filename in [(['npu-smi','info'],'npu_before.txt'),(['lscpu'],'cpu.txt'),(['free','-h'],'memory.txt'),(['ps','-eo','pid,ppid,pgid,args'],'processes_before.txt')]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    (OUT/filename).write_text(p.stdout+p.stderr)
print(OUT)
