import time
from typing import List, Annotated

import PIL
from PIL import Image
import anthropic
from anthropic import RateLimitError, APITimeoutError
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class ClaudeService(BaseService):
    claude_model_name: Annotated[
        str, "The name of the Anthropic model to use for the service."
    ] = "claude-opus-4-8"
    claude_api_key: Annotated[str, "The Claude API key to use for the service."] = None
    max_claude_tokens: Annotated[
        int, "The maximum number of tokens to use for a single Claude request."
    ] = 8192

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/webp",
                    "data": self.img_to_base64(img),
                },
            }
            for img in images
        ]

    def get_client(self):
        return anthropic.Anthropic(
            api_key=self.claude_api_key,
            max_retries=0,
        )

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

        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        messages = [
            {
                "role": "user",
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                # Structured outputs constrain the response to the schema, so
                # we no longer hand-roll JSON instructions or repair the output.
                response = client.messages.parse(
                    model=self.claude_model_name,
                    max_tokens=self.max_claude_tokens,
                    messages=messages,
                    output_format=response_schema,
                    timeout=timeout,
                )
                if block and response.usage:
                    block.update_metadata(
                        llm_tokens_used=response.usage.input_tokens
                        + response.usage.output_tokens,
                        llm_request_count=1,
                    )
                parsed = response.parsed_output
                if parsed is None:
                    # e.g. a safety refusal - no schema-conformant output
                    logger.warning(
                        f"Claude returned no parsed output (stop_reason={response.stop_reason})"
                    )
                    return {}
                return parsed.model_dump()
            except (RateLimitError, APITimeoutError) as e:
                # Rate limit exceeded
                if tries == total_tries:
                    # Last attempt failed. Give up
                    logger.error(
                        f"Rate limit error: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                else:
                    wait_time = tries * self.retry_wait_time
                    logger.warning(
                        f"Rate limit error: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})",
                    )
                    time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error during Claude API call: {e}")
                break

        return {}
