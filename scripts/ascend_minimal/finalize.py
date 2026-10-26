"""Record deltas against the saved baseline and verify user edits are retained."""
import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path
from support import ROOT,ENGINE

OUT=ROOT/'results/ascend_minimal_events'
base=json.loads((OUT/'baseline.json').read_text())
router=ROOT/'blitz-router'

def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo,text=True)
def sha(data):return hashlib.sha256(data).hexdigest()
def read(repo,path):
    p=repo/path
    return p.read_bytes() if p.is_file() else None

def delta(repo,prior,name):
    candidates=set(git(repo,'ls-files','-m','-o','--exclude-standard').splitlines())
    candidates.update(prior)
    changed=[];patch=[]
    for rel in sorted(candidates):
        new=read(repo,rel)
        if new is None:continue
        if rel in prior:old=(OUT/'baseline_files'/rel).read_bytes()
        else:
            p=subprocess.run(['git','show','HEAD:'+rel],cwd=repo,capture_output=True)
            old=p.stdout if p.returncode==0 else b''
        if new==old:continue
        changed.append(dict(file=rel,baseline_sha256=sha(old),current_sha256=sha(new),new=not bool(old)))
        patch.append(f'diff --git a/{rel} b/{rel}\n')
        patch.extend(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),
                                         fromfile='a/'+rel if old else '/dev/null',tofile='b/'+rel))
    (OUT/(name+'_this_round.patch')).write_text(''.join(patch))
    return changed

changes={'router':delta(router,base['files'],'router'),'engine':delta(ENGINE,{},'engine')}
(OUT/'changes.json').write_text(json.dumps(changes,indent=2)+'\n')
checks={}
for rel,oldhash in base['files'].items():
    p=router/rel
    data=p.read_bytes()
    if sha(data)==oldhash:checks[rel]='byte-identical'
    elif rel=='README.md':
        original=(OUT/'baseline_files'/rel).read_bytes()
        assert data.startswith(original),rel
        checks[rel]='original bytes retained; documentation appended'
    elif rel=='router/src/scheduler/policies/mod.rs':
        s=data.decode()
        a=s.index('        // A completion may contain multiple tokens.')
        b=s.index('        self.block_hash_state.append_tokens(new_tokens);',a)
        reverted=s[:a]+'        self.append_tbt_to_tpot(tbt);\n'+s[b:]
        assert sha(reverted.encode())==oldhash,rel
        checks[rel]='baseline reconstructed byte-for-byte by removing only new token accounting'
    elif rel=='router/src/engine/colocation.rs':
        s=data.decode()
        a=s.index('            // Never apply a request-v1 step using a different token/block unit.')
        b=s.index('            tracing::trace!',a);s=s[:a]+s[b:]
        a=s.index('                        // request-v1 excludes hits from reported prompt work.')
        b=s.index('                        if is_finished {',a);s=s[:a]+s[b:]
        marker='\n    #[cfg(all(test, feature = "lmetric-q", feature = "vllm-backend"))]\n    include!("ascend_request_tests.rs");\n'
        assert s.count(marker)==1;s=s.replace(marker,'')
        assert sha(s.encode())==oldhash,rel
        checks[rel]='baseline reconstructed byte-for-byte by removing only new protocol checks/correction/test include'
    else:raise AssertionError('Unexpected modification of pre-existing user edit: '+rel)
installed={}
for name,before in base['engines'].items():
    repo=Path('/vllm-workspace')/name
    after={'head':git(repo,'rev-parse','HEAD').strip(),'status':git(repo,'status','--porcelain')}
    assert after==before,name
    installed[name]={'unchanged':True,**after}
assert git(router,'branch','--show-current').strip()=='lmetric/ascend-minimal-events'
assert git(ENGINE,'branch','--show-current').strip()=='lmetric/ascend-minimal-events'
assert git(router,'diff','--cached')==''
assert git(ENGINE,'diff','--cached')==''
for repo in [router,ENGINE]:subprocess.run(['git','diff','--check'],cwd=repo,check=True)
preservation={'all_checks_passed':True,'preexisting_files':checks,'installed_sources':installed,
              'router_branch':git(router,'branch','--show-current').strip(),
              'engine_branch':git(ENGINE,'branch','--show-current').strip(),
              'original_submodule_status_retained':' ? request-sim' in git(router,'status','--short'),
              'no_staged_changes':True}
assert preservation['original_submodule_status_retained']
(OUT/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
files=[]
for name,repo in [('router',router),('engine',ENGINE)]:
    for entry in changes[name]:files.append(repo/entry['file'])
files.extend((ROOT/'scripts/ascend_minimal').glob('*.py'))
files.extend([ROOT/'scripts/run_ascend_minimal_tests.py',ROOT/'configs/ascend_minimal_events.json',ROOT/'docs/ascend_minimal_events.md'])
# Syntax checks do not execute or initialize an engine.
for p in files:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
manifest={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(files)}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'preserved':len(checks),'router_changed_files':len(changes['router']),
                  'engine_changed_files':len(changes['engine']),'installed_sources_unchanged':True}))
