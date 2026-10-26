"""Run the old CPU suite without writing its fixed historical artifact path."""
import hashlib,importlib,json,sys,unittest
from pathlib import Path
ROOT=Path('/root/lmetric-repro');OUT=ROOT/'results/kv_journal_20261025_j7bol44q'
sys.path.insert(0,str(ROOT/'scripts/ascend_minimal'))

def main():
 legacy=ROOT/'results/ascend_minimal_events/synthetic_ipc_events.json'
 before=hashlib.sha256(legacy.read_bytes()).hexdigest()
 # A normal import plus a main guard keeps multiprocessing spawn importable.
 module=importlib.import_module('test_events')
 redirect=OUT/'request_regression_safe_output';(redirect/'results').mkdir(parents=True)
 module.ROOT=redirect
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
 after=hashlib.sha256(legacy.read_bytes()).hexdigest();assert before==after
 (OUT/'request_regression_safe.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'artifact_root':str(redirect),'historical_file_sha256_before':before,'historical_file_sha256_after':after,'historical_file_unchanged_during_this_safe_run':True},indent=2)+'\n')
 return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
