from pathlib import Path
R=Path('/root/lmetric-repro/blitz-router/router/src')
p=R/'engine/kv_journal_tests.rs'
s=p.read_text().replace('#[cfg(test)]','#[cfg(all(test, feature = "lmetric-q", feature = "vllm-backend"))]',1)
s=s.replace('    struct Observed {','    include!("dual_replica_tests.rs");\n\n    struct Observed {')
p.write_text(s)
