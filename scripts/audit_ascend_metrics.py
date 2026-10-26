#!/usr/bin/env python3
"""CPU-only audit in a disposable Router copy; all events are SIMULATED.
No production source writes, engine import, model execution or NPU access.
"""
import hashlib, json, os, re, shutil, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ascend_metric_audit'
def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'blitz-router'
    with tempfile.TemporaryDirectory(prefix='ascend-router-audit-') as tmp:
        copy=Path(tmp)/'blitz-router'
        shutil.copytree(source,copy,ignore=shutil.ignore_patterns('.git','target'))
        path=copy/'router/src/engine/colocation.rs'
        original=path.read_text()
        marker='\n}\n\nmod except_management {'
        assert original.count(marker)==1
        tests=(ROOT/'scripts/ascend_audit/router_replay.rs').read_text().replace('__FULL_JSON__',str(ROOT/'results/yaullm_audit/field_contract/full.json'))
        path.write_text(original.replace(marker,'\n'+tests+marker))
        client=copy/'router/src/engine/client.rs'
        s=client.read_text().replace('    fn vllm_metric_to_step_output(', '    pub(super) fn vllm_metric_to_step_output(')
        s+='\n#[cfg(test)]\npub(crate) fn ascend_audit_convert(v: serde_json::Value) -> EngineStepOutput {\n vllm_impl::vllm_metric_to_step_output(serde_json::from_value(v).unwrap())\n}\n'
        client.write_text(s)
        cmd=['cargo','+1.88.0','test','-p','router','--lib','--no-default-features','--features',
             'vllm-backend,radixtree-blockhash,default-hash-algo,determinent-schedule,lmetric-q',
             '--locked','--offline','--jobs','4','ascend_audit','--','--nocapture','--test-threads=1']
        libdir=Path(tmp)/'link-libs'; libdir.mkdir()
        for lib in ['ssl','crypto']:
            runtime=Path('/usr/lib64')/f'lib{lib}.so.3'
            if runtime.exists(): (libdir/f'lib{lib}.so').symlink_to(runtime)
        env=dict(os.environ,CARGO_TARGET_DIR=str(source/'target'),
                 LIBRARY_PATH=str(libdir)+':'+os.environ.get('LIBRARY_PATH',''))
        r=run(cmd,cwd=copy,env=env)
        (OUT/'router_replay.log').write_text(r.stdout)
        cases=[]
        for line in r.stdout.splitlines():
            if 'AUDIT_RESULT ' in line:
                cases.append(json.loads(line.split('AUDIT_RESULT ',1)[1]))
        report={'scope':'SIMULATED events in unchanged production completion_event_loop and real LmetricQ admission; visibility-only test hook in /tmp copy',
                'npu_used':False,'engine_executed':False,'rust_full_event_loop_executed':bool(cases),
                'command':cmd,'exit_code':r.returncode,'cases':cases,
                'test_summary':next((line for line in r.stdout.splitlines() if line.startswith('test result:')),None),
                'expected_panic_cases':['duplicate_prefill','unsolicited_abort_ack','radix_parent_removal','non16_partial_hash'],
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [source/'router/src/engine/colocation.rs',source/'router/src/engine/client.rs',source/'router/src/engine/vllm_http.rs',source/'router/src/scheduler/kvcache.rs',source/'router/src/scheduler/state.rs',ROOT/'scripts/ascend_audit/router_replay.rs']}}
        (OUT/'router_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(r.stdout[-6500:])
        raise SystemExit(r.returncode)
if __name__=='__main__':main()
