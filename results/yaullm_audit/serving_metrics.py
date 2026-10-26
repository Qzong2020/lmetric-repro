from typing import Optional

from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.serving_engine import EngineClient
from vllm.entrypoints.openai.serving_engine import ModelConfig
from fastapi import Request
from collections.abc import AsyncGenerator, AsyncIterator
from vllm.v1.engine import EngineCoreOutputs

from vllm.entrypoints.openai.protocol import CompletionRequest

class MetricsReporter:
    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
    ):
        self.engine_client = engine_client
        self.model_config = model_config
    
    async def report_metrics(self, request: CompletionRequest, raw_request: Optional[Request] = None) -> AsyncGenerator[EngineCoreOutputs, None]:
        return self.engine_client.report_metrics()
