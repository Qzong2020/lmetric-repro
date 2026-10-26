# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Sequence
from typing import Any, Optional, Union

import msgspec
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.inputs import PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
FINISH_REASON_STRINGS = ("stop", "length", "abort")


class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, or abort.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted for another reason

    """
    STOP = 0
    LENGTH = 1
    ABORT = 2

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


def split_hash_to_32bit_list(hash_int):
    mask = 0xFFFFFFFF
    hash_int = int(hash_int, 16)
    bit_len = hash_int.bit_length()
    if bit_len <= 64:
        num_chunks = 2
    else:
        num_chunks = 8
    result = []
    for i in range(num_chunks):
        chunk = (hash_int >> (i * 32)) & mask
        result.append(chunk)
    return result

def split_hash_to_64bit_list(hash_int):
    mask = 0xFFFFFFFFFFFFFFFF
    hash_int = int(hash_int, 16)
    bit_len = hash_int.bit_length()
    
    if bit_len <= 64:
        num_chunks = 1
    else:
        num_chunks = 4
    result = []
    for i in range(num_chunks):
        chunk = (hash_int >> (i * 64)) & mask
        result.append(chunk)
    return result

def simplified_engine_core_outputs_to_dict(outputs: "EngineCoreOutputs", mode: str = "full") -> dict[str, Any]:
    """
    Convert EngineCoreOutputs to a simplified dictionary for metrics reporting.

    This function extracts only the essential fields from EngineCoreOutputs
    and its nested structures, making it suitable for metrics reporting.

    Args:
        outputs (EngineCoreOutputs): The EngineCoreOutputs instance to convert.
        mode (str): Serialization mode. "full" includes all fields (block
            hashes, evicted IDs, etc.). "minimal" only includes step_id,
            latency, and per-request outputs.

    Returns:
        dict[str, Any]: A simplified dictionary representation of the outputs.
    """
    simplified_outputs = {
        "outputs": [],
        "step_id": outputs.step_id if outputs.step_id is not None else 0,
        "latency": int(outputs.latency * 1000) if outputs.latency is not None else 0,
    }

    if mode == "full":
        serilized_new_block_hashes = [split_hash_to_64bit_list(hs) for hs in outputs.new_block_hashes]
        serilized_evicted_block_hashes = [split_hash_to_64bit_list(hs) for hs in outputs.evicted_block_hashes]
        simplified_outputs.update({
            "prefill_token_budget": outputs.prefill_token_budget if outputs.prefill_token_budget is not None else 0,
            "prefill_tokens": outputs.prefill_tokens if outputs.prefill_tokens is not None else 0,
            "new_block_hashes": serilized_new_block_hashes,
            "evicted_block_hashes": serilized_evicted_block_hashes,
            "evicted_block_ids": outputs.evicted_block_ids,
            "cur_used_block_ids": outputs.used_block_ids,
            "new_block_hashes_ids": outputs.new_block_hashes_ids,
            "preempted_ids": outputs.preempted_ids,
            "aborted_requests": [int(id) for id in outputs.aborted_requests] if outputs.aborted_requests else []
        })
    else:
        # Minimal mode: empty defaults for block-level fields.
        simplified_outputs.update({
            "prefill_token_budget": 0,
            "prefill_tokens": 0,
            "new_block_hashes": [],
            "evicted_block_hashes": [],
            "evicted_block_ids": [],
            "cur_used_block_ids": {},
            "new_block_hashes_ids": {},
            "preempted_ids": [],
            "aborted_requests": []
        })

    for output in outputs.outputs:
        simplified_output = {
            "request_id": int(output.request_id),
            "new_token_ids": output.new_token_ids,
            "state": output.state,
            "is_finished": output.finished,
            "hit_token_cnt": output.num_cached_tokens,
            "prev_computed_tokens": output.num_computed_tokens,
        }
        if simplified_output["state"] == "PREFILL":
            simplified_output.update({
                "ttft": output.ttft
            })

        simplified_outputs["outputs"].append(simplified_output)
    return simplified_outputs

class EngineCoreRequest(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    prompt_token_ids: list[int]
    mm_inputs: Optional[Sequence[Optional[MultiModalKwargs]]]
    mm_hashes: Optional[list[str]]
    mm_placeholders: Optional[list[PlaceholderRange]]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]
    cache_salt: Optional[str]
    data_parallel_rank: Optional[int]

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0


class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""
    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamps and is used for by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.
    """
    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(cls,
                  event_type: EngineCoreEventType,
                  timestamp: Optional[float] = None) -> "EngineCoreEvent":
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    new_token_ids: list[int]

    new_logprobs: Optional[LogprobsLists] = None
    new_prompt_logprobs_tensors: Optional[LogprobsTensors] = None

    pooling_output: Optional[torch.Tensor] = None

    finish_reason: Optional[FinishReason] = None
    stop_reason: Union[int, str, None] = None
    events: Optional[list[EngineCoreEvent]] = None
    kv_transfer_params: Optional[dict[str, Any]] = None

    # The number of tokens with prefix cache hits.
    num_cached_tokens: int = 0
    num_computed_tokens: Optional[int] = None
    
    state: Optional[str] = None
    ttft: Optional[float] = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class UtilityOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    call_id: int

    # Non-None implies the call failed, result should be None.
    failure_message: Optional[str] = None
    result: Any = None


class EngineCoreOutputs(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    #NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout


    
    latency: Optional[float] = None
    step_id: Optional[int] = None
    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: Optional[SchedulerStats] = None
    timestamp: float = 0.0

    utility_output: Optional[UtilityOutput] = None
    finished_requests: Optional[set[str]] = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: Optional[int] = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: Optional[int] = None

    # prefill_token_budget
    prefill_token_budget: Optional[int] = None
    prefill_tokens: Optional[int] = None

    # evict block ids
    evicted_block_hashes: Optional[list[str]] = []
    new_block_hashes: Optional[list[str]] = []
    evicted_block_ids: Optional[list[int]] = []

    # a map req_id -> The block_id used by this request
    used_block_ids: Optional[dict[str, list[int]]] = None
    # a map req_id -> The block_ids generated new blocks
    new_block_hashes_ids: Optional[dict[str, list[int]]] = None

    preempted_ids: Optional[list[int]] = None

    # abonormal requests 
    aborted_requests: Optional[set[str]] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    ADD = b'\x00'
    ABORT = b'\x01'
    START_DP_WAVE = b'\x02'
    UTILITY = b'\x03'
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b'\x04'


class ReconfigureDistributedRequest(msgspec.Struct):
    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int


class ReconfigureRankType(enum.IntEnum):
    """
    Rank type for reconfiguring distributed request.
    """
    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2
