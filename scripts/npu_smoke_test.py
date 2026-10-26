#!/usr/bin/env python3
"""Bounded Ascend health smoke test; no model download or repository edits."""
import argparse
from datetime import datetime
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def compute(out):
    import torch
    import torch_npu
    torch.set_num_threads(2)
    assert torch.npu.is_available() and torch.npu.device_count() == 2
    torch.manual_seed(260315202)
    results = []
    for index in range(2):
        torch.npu.set_device(index)
        device = f'npu:{index}'
        for dtype, bound in [(torch.float16, 0.002), (torch.bfloat16, 0.015)]:
            a = (torch.randn(512, 512) / math.sqrt(512)).to(dtype)
            b = torch.randn(512, 512).to(dtype)
            expected = a.float() @ b.float()
            na, nb = a.to(device), b.to(device)
            for _ in range(10):
                actual = na @ nb
            torch.npu.synchronize(index)
            actual = actual.float().cpu()
            delta = actual - expected
            relative_l2 = float(delta.norm() / expected.norm())
            assert torch.isfinite(actual).all() and relative_l2 < bound, (index, dtype, relative_l2)
            results.append({'logical_device': index, 'name': torch.npu.get_device_name(index),
                'operation': 'matmul', 'shape': [512, 512], 'dtype': str(dtype),
                'iterations': 10, 'relative_l2_error': relative_l2,
                'max_abs_error': float(delta.abs().max()), 'relative_l2_limit': bound,
                'peak_allocated_bytes': torch.npu.max_memory_allocated(index), 'passed': True})
            del na, nb, actual
        q, k, v = [torch.randn(1, 4, 64, 64).half() for _ in range(3)]
        expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        actual = torch.nn.functional.scaled_dot_product_attention(q.to(device), k.to(device), v.to(device))
        torch.npu.synchronize(index)
        relative_l2 = float((actual.float().cpu()-expected).norm()/expected.norm())
        assert torch.isfinite(actual).all() and relative_l2 < 0.01, (index, relative_l2)
        results.append({'logical_device': index, 'operation': 'scaled_dot_product_attention',
            'shape': [1, 4, 64, 64], 'dtype': 'torch.float16', 'relative_l2_error': relative_l2,
            'relative_l2_limit': 0.01, 'passed': True})
        del actual
        torch.npu.empty_cache()
    save(out / 'compute.json', results)
    print(json.dumps(results), flush=True)


def hccl_worker(rank, init_method, out):
    import torch
    import torch_npu
    import torch.distributed as dist
    from datetime import timedelta
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    dist.init_process_group('hccl', init_method=init_method, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        tensor = torch.full((65536,), float(rank + 1), device=f'npu:{rank}')
        dist.all_reduce(tensor)
        torch.npu.synchronize(rank)
        assert bool((tensor.cpu() == 3).all()), 'all_reduce mismatch'
        tensor.fill_(42 if rank == 0 else 0)
        dist.broadcast(tensor, src=0)
        torch.npu.synchronize(rank)
        assert bool((tensor.cpu() == 42).all()), 'broadcast mismatch'
        local = torch.tensor([rank + 1.], device=f'npu:{rank}')
        gathered = [torch.empty_like(local) for _ in range(2)]
        dist.all_gather(gathered, local)
        torch.npu.synchronize(rank)
        assert [float(t.cpu().item()) for t in gathered] == [1., 2.]
        save(Path(out) / f'hccl_rank{rank}.json', {'rank': rank, 'all_reduce': True,
             'broadcast': True, 'all_gather': True, 'elements_all_reduce': 65536, 'passed': True})
    finally:
        dist.destroy_process_group()


def hccl(out):
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory(prefix='npu-smoke-') as tmp:
        mp.spawn(hccl_worker, args=(f'file://{tmp}/rendezvous', str(out)), nprocs=2, join=True)
    print('Both HCCL ranks passed', flush=True)


def engine(out):
    import vllm
    import vllm_ascend
    from vllm.platforms import current_platform
    from vllm import LLM, SamplingParams
    platform = type(current_platform).__module__ + '.' + type(current_platform).__name__
    assert 'ascend' in platform.lower() or 'npu' in platform.lower(), platform
    row = {'vllm': metadata.version('vllm'), 'vllm_ascend': metadata.version('vllm_ascend'),
           'platform': platform, 'LLM_import': True, 'SamplingParams_import': True,
           'scope': 'import and platform detection only; no model engine initialized', 'passed': True}
    save(out / 'engine_import.json', row)
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['compute', 'hccl', 'engine'])
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'results/npu_smoke')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase:
        globals()[args.phase](args.out)
        return
    summary = {'started_at': datetime.now().astimezone().isoformat(), 'phases': {},
               'scope': 'hardware, elementary math, HCCL and Python imports; not LLM serving validation'}
    for phase, limit in [('compute', 150), ('hccl', 100), ('engine', 90)]:
        start = time.monotonic()
        env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        try:
            with (args.out / f'{phase}.log').open('w') as log:
                process = subprocess.Popen([sys.executable, '-u', __file__, '--phase', phase,
                    '--out', str(args.out)], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                try:
                    code = process.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 'timeout'
            row = {'exit_code': code, 'passed': code == 0, 'elapsed_seconds': round(time.monotonic()-start, 2)}
        except Exception as exc:
            row = {'passed': False, 'error': repr(exc)}
        summary['phases'][phase] = row
        save(args.out / 'summary.json', summary)
        print(phase, json.dumps(row), flush=True)
    summary['all_passed'] = all(r['passed'] for r in summary['phases'].values())
    save(args.out / 'summary.json', summary)
    sys.exit(0 if summary['all_passed'] else 1)


if __name__ == '__main__':
    main()
