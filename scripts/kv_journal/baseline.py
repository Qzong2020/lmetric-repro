"""Snapshot all existing source work and historical acceptance artifacts."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile
ROOT=Path('/root/lmetric-repro')
OUT=Path(tempfile.mkdtemp(prefix='kv_journal_20261025_',dir=ROOT/'results'))
def run(repo,args):
 p=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,check=True)
 return p.stdout
rows={}
for name in ['engines/vllm','blitz-router','blitz-router/request-sim','/vllm-workspace/vllm','/vllm-workspace/vllm-ascend']:
 repo=ROOT/name;label=name.strip('/').replace('/','__')
 rows[name]={k:run(repo,v) for k,v in {'head':['rev-parse','HEAD'],'status':['status','--short','--branch'],'branches':['branch','-avv'],'files':['ls-files','--cached','--others','--exclude-standard']}.items()}
 (OUT/(label+'.patch')).write_text(run(repo,['diff','--binary']))
 files={}
 for f in rows[name]['files'].splitlines():
  p=repo/f
  if p.is_file():files[f]=hashlib.sha256(p.read_bytes()).hexdigest()
 rows[name]['sha256']=files
 if not name.startswith('/') and name!='blitz-router/request-sim':
  shutil.copytree(repo,OUT/'source_before'/label,ignore=shutil.ignore_patterns('.git','target','__pycache__','.venv'))
rows['historical_results']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'results/ascend_single_card_validation').rglob('*') if p.is_file()}
(OUT/'baseline.json').write_text(json.dumps(rows,indent=2)+'\n')
print(OUT)
