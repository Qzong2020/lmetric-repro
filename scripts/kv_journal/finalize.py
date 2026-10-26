"""Audit this run, including the explicitly unresolved historical artifact issue."""
import difflib,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/root/lmetric-repro');OUT=ROOT/'results/kv_journal_20261025_j7bol44q'
base=json.loads((OUT/'baseline.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True)
allowed={
'engines/vllm':{'vllm/v1/engine/lmetric_events.py','vllm/entrypoints/openai/completion/serving.py','vllm/v1/core/block_pool.py','vllm/v1/core/kv_cache_manager.py','vllm/v1/engine/async_llm.py','vllm/v1/engine/core.py'},
'blitz-router':{'router/src/engine/ascend_request_tests.rs','router/src/scheduler/policies/paper_tests.rs','policy-dsl/src/lower.rs','radixtree/src/block_hash.rs','router/src/engine/client.rs','router/src/engine/colocation.rs','router/src/engine/mod.rs','router/src/engine/vllm_http.rs','router/src/gateway/server.rs','router/src/scheduler/infer.rs','router/src/scheduler/kvcache.rs','router/src/scheduler/policies/dsl_runtime.rs','router/src/scheduler/policies/lmetric.rs','router/src/scheduler/policies/preble/mod.rs','router/src/scheduler/simulator/mirror.rs','router/src/scheduler/simulator/mod.rs','router/src/scheduler/simulator/pctx.rs','router/src/scheduler/simulator/sched.rs','router/src/scheduler/state.rs'}}
audit={'repositories':{},'historical_single_card_results':{},'all_historical_results_preserved':False,'history_integrity_issue':json.loads((OUT/'history_integrity_issue.json').read_text())}
for name in ['engines/vllm','blitz-router','blitz-router/request-sim','/vllm-workspace/vllm','/vllm-workspace/vllm-ascend']:
 repo=ROOT/name;before=base[name];changed=[]
 assert git(repo,'rev-parse','HEAD')==before['head']
 for rel,old in before['sha256'].items():
  p=repo/rel;assert p.is_file(),('missing original file',name,rel)
  if sha(p)!=old:changed.append(rel)
 if name in allowed:assert set(changed)<=allowed[name],('unexpected user-source changes',name,changed)
 else:
  assert not changed,(name,changed)
  assert git(repo,'status','--short','--branch')==before['status']
 row={'head':before['head'].strip(),'head_unchanged':True,'branch':git(repo,'branch','--show-current').strip(),'status':git(repo,'status','--short','--branch'),'unchanged_original_files':len(before['sha256'])-len(changed),'modified_original_files':changed,'no_original_files_deleted':True}
 if name in allowed:
  patch=[];backup=OUT/'source_before'/name.replace('/','__')
  files=set(git(repo,'ls-files','--cached','--others','--exclude-standard').splitlines())
  for rel in sorted(set(changed) | (files-set(before["sha256"]))):
   p=repo/rel
   if not p.is_file():continue
   old=(backup/rel).read_text() if (backup/rel).is_file() else ''
   try:new=p.read_text()
   except UnicodeDecodeError:continue
   if old!=new:patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='before/'+rel,tofile='after/'+rel))
  (OUT/(name.replace('/','__')+'_this_round.patch')).write_text(''.join(patch))
  check=subprocess.run(['git','-C',str(repo),'diff','--check'],capture_output=True,text=True)
  assert check.returncode==0,check.stdout+check.stderr
  row['diff_check_passed']=True
 audit['repositories'][name]=row
for rel,old in base['historical_results'].items():
 assert sha(ROOT/rel)==old,rel
 audit['historical_single_card_results'][rel]=True
assert sha(ROOT/'results/ascend_minimal_events/synthetic_ipc_events.json')==audit['history_integrity_issue']['current_sha256']
audit['single_card_43_files_unchanged']=len(audit['historical_single_card_results'])==43
audit['note']='Source/submodule/install preservation checks pass; one historical synthetic IPC artifact was overwritten by an old test. Original random epoch unavailable; do not claim full history preservation.'
(OUT/'preservation.json').write_text(json.dumps(audit,indent=2)+'\n')
summary={'sync':json.loads((OUT/'sync_final/verified.json').read_text()),'async':json.loads((OUT/'async_release/verified.json').read_text()),'gateway':json.loads((OUT/'gateway_release/verified.json').read_text()) if (OUT/'gateway_release/verified.json').exists() else None,'history_integrity_issue':audit['history_integrity_issue']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15);(OUT/'npu_final.txt').write_text(p.stdout+p.stderr)
assert 'No running processes found' in p.stdout
print('Source/submodule/install and single-card history checks passed; historical synthetic IPC exception remains explicit.')
