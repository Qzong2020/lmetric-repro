"""Check preservation and package raw evidence without modifying old results."""
import difflib,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/root/lmetric-repro');OUT=Path(sys.argv[1]);baseline=json.loads((OUT/'baseline.json').read_text())
history=baseline['historical_results'];changed=[];missing=[]
for name,old in history.items():
    p=ROOT/name
    if not p.is_file():missing.append(name)
    elif hashlib.sha256(p.read_bytes()).hexdigest()!=old:changed.append(name)
assert not missing and not changed,(missing,changed)
result={'historical_files_checked':len(history),'historical_files_changed':changed,'historical_files_missing':missing,'repositories':{}}
for name,record in baseline.items():
    if name=='historical_results':continue
    repo=ROOT/name;label=name.strip('/').replace('/','__')
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True)
    assert git('rev-parse','HEAD')==record['head']
    absent=[f for f in record['sha256'] if not (repo/f).is_file()]
    assert not absent,absent
    changes=[f for f,sha in record['sha256'].items() if hashlib.sha256((repo/f).read_bytes()).hexdigest()!=sha]
    if name.startswith('/') or name=='blitz-router/request-sim' or name=='engines/vllm':assert not changes,(name,changes)
    (OUT/(label+'_final.patch')).write_text(git('diff','--binary'))
    if name=='blitz-router':
        parts=[]
        incoming=set(record['sha256'])
        current=set(git('ls-files','--cached','--others','--exclude-standard').splitlines())
        for file in sorted(set(changes) | (current-incoming)):
            old=OUT/'source_before'/label/file;new=repo/file
            if new.is_file():
                a=old.read_text().splitlines(True) if old.is_file() else []
                b=new.read_text().splitlines(True)
                parts.extend(difflib.unified_diff(a,b,fromfile='a/'+file,tofile='b/'+file))
        (OUT/'blitz-router_this_round.patch').write_text(''.join(parts))

    result['repositories'][name]={'head':record['head'].strip(),'unchanged_head':True,'existing_files_missing':absent,'changed_existing_files':changes,'status':git('status','--short','--branch')}
(OUT/'preservation.json').write_text(json.dumps(result,indent=2)+'\n')
subprocess.run(['npu-smi','info'],stdout=(OUT/'npu_final.txt').open('w'),check=True,timeout=15)
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'}
(OUT/'artifact_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'historical_files_checked':len(history),'historical_files_unchanged':True,'heads_unchanged':True}))
