"""Conservative client-side sizing; these estimates are not server token counts."""

from math import ceil
from typing import Sequence

from PIL import Image


class RecognitionFailure(RuntimeError):
    """A required recognition request failed with a safe, input-free reason."""

    def __init__(self, reason: str, attempts: int = 0):
        self.reason = reason
        self.attempts = attempts
        super().__init__(f"{reason} (attempts={attempts})")


def estimate_request_tokens(prompt: str, images: Sequence[Image.Image]) -> int:
    """Reserve one token per UTF-8 byte and one per 16-pixel image patch.

    Message/image overhead is included.  This deliberately conservative bound
    allows callers to split work without relying on an unavailable tokenizer.
    """
    return 128 + len(prompt.encode("utf-8")) + sum(
        256 + ceil(image.width / 16) * ceil(image.height / 16) for image in images
    )


def fit_request(
    prompt: str,
    images: Sequence[Image.Image],
    budget: int,
    max_output_tokens: int,
    *,
    max_image_edge: int = 1536,
    min_image_edge: int = 128,
) -> list[Image.Image]:
    """Return reduced request copies, retaining every source image untouched.

    Limit the initial long edge to 1536 pixels.  Further reductions stop after
    at most 12 steps or at a 128-pixel long edge, then fail explicitly.  Dropping
    an image or truncating source text would hide missing required content.
    """
    if budget <= 0 or max_output_tokens <= 0 or max_output_tokens >= budget:
        raise RecognitionFailure("Request context budget cannot reserve the response")
    if not 1 <= min_image_edge <= max_image_edge:
        raise RecognitionFailure("Request image bounds are invalid")
    if estimate_request_tokens(prompt, []) + max_output_tokens > budget:
        raise RecognitionFailure("Prompt exceeds the request context budget")

    copies = []
    for image in images:
        if not isinstance(image, Image.Image) or min(image.size) <= 0:
            raise RecognitionFailure("Request images must have valid pixel dimensions")
        copy = image.copy()
        copy.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
        copies.append(copy)

    for step in range(13):
        if estimate_request_tokens(prompt, copies) + max_output_tokens <= budget:
            return copies
        if step == 12:
            break
        reduced = []
        changed = False
        for image in copies:
            edge = max(image.size)
            target = max(min_image_edge, int(edge * 0.75))
            if target >= edge:
                reduced.append(image)
                continue
            scale = target / edge
            size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            reduced.append(image.resize(size, Image.Resampling.LANCZOS))
            changed = True
        if not changed:
            break
        copies = reduced
    raise RecognitionFailure("Required images cannot fit the request context budget")
