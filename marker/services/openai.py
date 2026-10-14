import os
import time
from typing import Annotated, List
from urllib.parse import urlparse

import openai
import PIL
from marker.logger import get_logger
from PIL import Image
from pydantic import BaseModel, ValidationError

from marker.cloud.client import (
    ResponseValidationError,
    provider_error_reason,
    repair_prompt,
    retryable_provider_error,
    schema_error_reason,
    schema_prompt,
    validate_response,
)
from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class OpenAIService(BaseService):
    openai_base_url: Annotated[
        str, "The base url to use for OpenAI-like models.  No trailing slash."
    ] = "https://api.openai.com/v1"
    openai_model: Annotated[str, "The model name to use for OpenAI-like model."] = (
        "gpt-5-mini"
    )
    openai_api_key: Annotated[
        str, "The API key to use for the OpenAI-like service."
    ] = os.environ.get("OPENAI_API_KEY")
    openai_image_format: Annotated[
        str,
        "The image format to use for the OpenAI-like service. Use 'png' for better compatability",
    ] = "webp"

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        """
        Generate the base-64 encoded message to send to an
        openAI-compatabile multimodal model.

        Args:
            images: Image or list of PIL images to include
            format: Format to use for the image; use "png" for better compatability.

        Returns:
            A list of OpenAI-compatbile multimodal messages containing the base64-encoded images.
        """
        if isinstance(images, Image.Image):
            images = [images]

        img_fmt = self.openai_image_format
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/{};base64,{}".format(
                        img_fmt, self.img_to_base64(img, format=img_fmt)
                    ),
                },
            }
            for img in images
        ]

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        try:
            client = self.get_client()
        except Exception:
            logger.error("OpenAI client initialization failed")
            return {}
        image_data = self.format_image_for_llm(image)
        is_cloud = urlparse(self.openai_base_url).hostname == "ollama.com"
        original_prompt = schema_prompt(prompt, response_schema) if is_cloud else prompt
        current_prompt = original_prompt

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            messages = [{"role": "user", "content": [
                *image_data, {"type": "text", "text": current_prompt},
            ]}]
            try:
                if block:
                    block.update_metadata(llm_request_count=1)
                # Ollama Cloud does not enforce JSON Schema / structured outputs.
                # Include the schema in the request and validate locally instead.
                if is_cloud:
                    options = {"max_tokens": self.max_output_tokens} if self.max_output_tokens else {}
                    response = client.chat.completions.create(
                        model=self.openai_model, messages=messages,
                        timeout=timeout, temperature=0, **options,
                    )
                else:
                    response = client.chat.completions.parse(
                        extra_headers={"X-Title": "Marker", "HTTP-Referer": "https://github.com/datalab-to/marker"},
                        model=self.openai_model, messages=messages, timeout=timeout,
                        response_format=response_schema,
                    )
                total_tokens = response.usage.total_tokens if response.usage else 0
                if block:
                    block.update_metadata(llm_tokens_used=total_tokens)
                if not response.choices:
                    raise ResponseValidationError("Response contains no completion")
                choice = response.choices[0]
                return validate_response(choice.message.content, response_schema, choice.finish_reason)
            except (ResponseValidationError, ValidationError, openai.LengthFinishReasonError) as error:
                if isinstance(error, ValidationError):
                    reason = schema_error_reason(error, response_schema)
                elif isinstance(error, openai.LengthFinishReasonError):
                    reason = "Response was truncated; regenerate complete JSON"
                else:
                    reason = str(error)
                logger.warning("LLM response failed validation: %s (attempt %d/%d)", reason, tries, total_tries)
                current_prompt = repair_prompt(original_prompt, reason)
            except Exception as error:
                reason = provider_error_reason(error)
                if not retryable_provider_error(error):
                    logger.error("OpenAI inference failed: %s", reason)
                    break
                if tries == total_tries:
                    logger.error("%s. Maximum attempts reached (%d/%d)", reason, tries, total_tries)
                    break
                else:
                    wait_time = tries * self.retry_wait_time
                    logger.warning("%s. Retrying in %s seconds (attempt %d/%d)", reason, wait_time, tries, total_tries)
                    time.sleep(wait_time)

        return {}

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url, max_retries=0)
