"""Build final preservation and runtime evidence; report only observed values."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path('/root/lmetric-repro');OUT=ROOT/'results/ascend_single_card_validation'
initial=json.loads((OUT/'initial.json').read_text());final=json.loads((OUT/'final.json').read_text())
preservation={}
for key,before in initial['repositories'].items():
 after=final['repositories'][key]
 preservation[key]={'head_unchanged':before['head']==after['head'],'status_unchanged':before['status']==after['status'],'diff_unchanged':before['diff']==after['diff'],'captured_worktree_hashes_unchanged':before['hashes']==after['hashes']}
assert all(all(row.values()) for row in preservation.values())
(OUT/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
# Include the exact instrumentation delta for independent review.
import difflib
for rel,name in [('router/src/engine/client.rs','router_observation.patch'),('router/src/engine/colocation.rs','router_driver.patch')]:
 before=(ROOT/'blitz-router'/rel).read_text().splitlines(keepends=True);after=(Path('/tmp/lmetric-real-npu-router')/rel).read_text().splitlines(keepends=True)
 (OUT/name).write_text(''.join(difflib.unified_diff(before,after,fromfile='user-worktree/'+rel,tofile='isolated-driver/'+rel)))
summary={'scope':'real single NPU request/execution events only; no KV journal or performance comparison','ordinary_inference':json.loads((OUT/'baseline_response.json').read_text()),'off':json.loads((OUT/'off_result.json').read_text()),'sync':json.loads((OUT/'sync_final_evidence.json').read_text()),'async':json.loads((OUT/'async_evidence.json').read_text()) if (OUT/'async_evidence.json').exists() else None,'preservation':preservation}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [OUT,ROOT/'scripts/ascend_single_card'] for p in sorted(folder.glob('*')) if p.is_file() and p.name!='artifact_sha256.json'}
(OUT/'artifact_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Preservation passed; summary and artifact hashes saved')
