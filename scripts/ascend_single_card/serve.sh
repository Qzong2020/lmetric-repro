#!/usr/bin/env bash
set -eu
mode=${1:?mode: off sync async}
case "$mode" in
 off) event_flag=0; async_flag=--no-async-scheduling ;;
 sync) event_flag=1; async_flag=--no-async-scheduling ;;
 async) event_flag=1; async_flag=--async-scheduling ;;
 *) exit 2 ;;
esac
export ASCEND_RT_VISIBLE_DEVICES=0 VLLM_LMETRIC_EVENTS="$event_flag"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="/root/lmetric-repro/engines/vllm:${PYTHONPATH:-}"
exec /usr/local/python3.12.13/bin/python -u /root/lmetric-repro/scripts/ascend_single_card/launch.py \
 --model /models/Qwen3-0.6B --served-model-name ascend-small \
 --host 127.0.0.1 --port 18080 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
 --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 128 \
 --gpu-memory-utilization 0.25 --enforce-eager "$async_flag" \
 --enable-prefix-caching --enable-chunked-prefill --block-size 128
