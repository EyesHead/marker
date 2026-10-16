"""Native Marker enrichment using OpenAI and a per-PDF dollar ceiling."""

from typing import Annotated

from marker.cloud.openai_client import OpenAIResponsesClient
from marker.cloud.defaults import (DEFAULT_CONTEXT_TOKENS, DEFAULT_MAX_IMAGES,
                                   DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TIMEOUT)
from marker.services import BaseService


class OpenAIBudgetService(BaseService):
    openai_model: Annotated[str, "OpenAI vision model: gpt-6-luna or gpt-6.1-sol."] = "gpt-6-luna"
    openai_base_url: Annotated[str, "Official OpenAI API endpoint."] = "https://api.openai.com/v1"
    openai_reasoning_effort: Annotated[str, "Reasoning effort; low by default to control cost."] = "low"
    cloud_budget_usd: Annotated[str, "Maximum model-token USD cost per input PDF, up to 1.00."] = "1.00"
    cloud_budget_dir: Annotated[str, "Persistent shared PDF budget ledger directory."] = ""
    cloud_context_tokens: Annotated[int, "Conservative request sizing budget."] = DEFAULT_CONTEXT_TOKENS
    cloud_max_images: Annotated[int, "Maximum images per complete logical object."] = DEFAULT_MAX_IMAGES
    cloud_max_output_tokens: Annotated[int, "Output cap, including invisible reasoning tokens."] = DEFAULT_MAX_OUTPUT_TOKENS
    cloud_max_attempts: Annotated[int, "Total generation attempts, including repairs."] = 2
    cloud_max_concurrency: Annotated[int, "Concurrent cloud requests sharing one PDF budget."] = 2
    cloud_max_requests: Annotated[int, "Maximum generation attempts per PDF; zero is unlimited."] = 0
    timeout: Annotated[int, "Per-request timeout in seconds."] = DEFAULT_TIMEOUT

    def __init__(self, config=None, *, client=None):
        raw = config.model_dump() if hasattr(config, "model_dump") else dict(config or {})
        super().__init__(raw)
        resolved = {name: getattr(self, name) for name in (
            "cloud_context_tokens", "cloud_max_images", "cloud_max_output_tokens", "cloud_max_attempts",
            "cloud_max_concurrency", "cloud_max_requests", "cloud_budget_usd", "cloud_budget_dir",
            "openai_reasoning_effort")}
        resolved.update(cloud_model=self.openai_model, cloud_base_url=self.openai_base_url,
                        cloud_timeout=raw.get("cloud_timeout", self.timeout))
        self.client = OpenAIResponsesClient(resolved, client=client)

    def __call__(self, prompt, image, block, response_schema, max_retries=None, timeout=None):
        metadata = {} if block is not None else None
        try:
            return self.client.request(prompt, image, response_schema,
                                       max_attempts=None if max_retries is None else max_retries + 1,
                                       timeout=timeout, metadata=metadata)
        finally:
            if metadata:
                block.update_metadata(**metadata)

    def request_text(self, prompt, images, validator, *, block=None, max_retries=None, **kwargs):
        metadata = {} if block is not None else None
        try:
            return self.client.request_text(
                prompt, images, validator, metadata=metadata,
                max_attempts=None if max_retries is None else max_retries + 1, **kwargs,
            )
        finally:
            if metadata:
                block.update_metadata(**metadata)
