"""Ollama's cloud API with bounded attempts and local response validation."""

from typing import Annotated
from urllib.parse import urlsplit, urlunsplit

from PIL import Image
from pydantic import BaseModel

from marker.cloud.client import CloudClient
from marker.cloud.defaults import (
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_IMAGES,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TIMEOUT,
)
from marker.schema.blocks import Block
from marker.services import BaseService


def _api_base_url(value: str) -> str:
    """Accept an Ollama server origin as well as an explicit compatible API URL."""
    parts = urlsplit(value.rstrip("/"))
    if not parts.path:
        parts = parts._replace(path="/v1")
    return urlunsplit(parts)


class OllamaService(BaseService):
    ollama_base_url: Annotated[str, "Ollama OpenAI-compatible API endpoint."] = DEFAULT_CLOUD_BASE_URL
    ollama_model: Annotated[str, "Ollama HTTP model name; cloud models need no local installation."] = DEFAULT_CLOUD_MODEL
    timeout: Annotated[int, "Timeout in seconds for each Ollama HTTP request."] = DEFAULT_TIMEOUT
    max_retries: Annotated[int, "Retries after an initial Ollama request."] = DEFAULT_MAX_ATTEMPTS - 1
    max_output_tokens: Annotated[int, "Maximum response tokens for each Ollama request."] = DEFAULT_MAX_OUTPUT_TOKENS

    def __init__(self, config: BaseModel | dict | None = None, *, client=None):
        raw_config = config.model_dump() if isinstance(config, BaseModel) else dict(config or {})
        super().__init__(raw_config)

        def configured(name):
            return raw_config.get(f"{type(self).__name__}_{name}", raw_config.get(name))

        def setting(name, default):
            value = configured(name)
            return default if value is None else value

        # Explicit service settings win; shared cloud/OpenAI aliases remain
        # compatible with the document converters and existing configurations.
        client_config = dict(raw_config)
        self.ollama_model = next((value for value in (
            configured("ollama_model"), configured("cloud_model"),
            configured("openai_model"), self.ollama_model,
        ) if value is not None), self.ollama_model)
        self.ollama_base_url = _api_base_url(next((value for value in (
            configured("ollama_base_url"), configured("cloud_base_url"),
            configured("openai_base_url"), self.ollama_base_url,
        ) if value is not None), self.ollama_base_url))
        client_config["cloud_model"] = self.ollama_model
        client_config["cloud_base_url"] = self.ollama_base_url
        for name, default in (
            ("cloud_context_tokens", DEFAULT_CONTEXT_TOKENS),
            ("cloud_max_images", DEFAULT_MAX_IMAGES),
            ("cloud_max_concurrency", setting("max_concurrency", DEFAULT_MAX_CONCURRENCY)),
            ("cloud_max_requests", 0),
        ):
            client_config[name] = setting(name, default)
        client_config["cloud_timeout"] = setting("cloud_timeout", self.timeout)
        client_config["cloud_max_attempts"] = setting("cloud_max_attempts", self.max_retries + 1)
        client_config["cloud_max_output_tokens"] = setting("cloud_max_output_tokens", self.max_output_tokens)
        self.client = client if isinstance(client, CloudClient) else CloudClient(client_config, client=client)

    def process_images(self, images):
        return [self.img_to_base64(image, format="PNG") for image in images]

    def _options(self, block, max_retries, timeout, max_output_tokens):
        if max_retries is not None and max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        metadata = {} if block is not None else None
        return metadata, {
            "max_attempts": None if max_retries is None else max_retries + 1,
            "timeout": timeout,
            "max_output_tokens": max_output_tokens,
            "metadata": metadata,
        }

    def __call__(
        self,
        prompt: str,
        image: Image.Image | list[Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        metadata, options = self._options(block, max_retries, timeout, None)
        try:
            return self.client.request(prompt, image, response_schema, **options)
        finally:
            if metadata:
                block.update_metadata(**metadata)

    def request_text(
        self, prompt, images, validator, *, block=None, max_retries=None,
        timeout=None, max_output_tokens=None, image_long_edge=3072,
    ):
        """Recognize complete HTML/LaTeX using this service's shared budget."""
        metadata, options = self._options(block, max_retries, timeout, max_output_tokens)
        try:
            return self.client.request_text(
                prompt, images, validator, image_long_edge=image_long_edge, **options
            )
        finally:
            if metadata:
                block.update_metadata(**metadata)
