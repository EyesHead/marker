"""Bounded Ollama Cloud requests and locally validated JSON responses."""

import base64
import json
import logging
import os
import re
from functools import wraps
from io import BytesIO
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import get_args
from urllib.parse import urlparse

import openai
from PIL import Image
from pydantic import BaseModel, ValidationError
from pydantic_core.core_schema import ErrorType
from pydantic_core import PydanticCustomError

from marker.cloud.context import RecognitionFailure, fit_request
from marker.cloud.defaults import (
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT,
)


# Curated constraints only: arbitrary validator/provider text is never reflected.
SEMANTIC_ERRORS = {
    "layout_unresolved": "Required visible layout remains unresolved",
    "layout_source_bounds": "Layout region exceeds assigned source bounds",
    "layout_ownership_overlap": "Layout regions overlap source ownership",
    "layout_only_tables_own_cells": "Only tables can own cells",
    "layout_table_cells_missing": "Table requires source cells",
    "layout_cell_bounds": "Cell exceeds table bounds",
    "layout_line_bounds": "Baseline exceeds cell bounds",
    "layout_line_overlap": "Baseline atoms overlap",
    "layout_rectangular_grid": "Table source cells must cover a rectangular grid",
    "layout_full_table_coverage": "Table cells must cover the entire table extent",
    "math_unsupported": "Required math is unresolved or unsupported",
    "math_script_geometry": "Math script letters must agree with source baseline geometry",
    "table_cell_ownership": "Cell IDs must cover assigned cells once",
    "table_trusted_spans": "Trusted physical spans cannot change",
    "table_span_bounds": "Cell span exceeds table extent",
    "table_span_overlap": "Cell spans overlap",
    "table_line_ownership": "Source baseline lines must be covered once",
    "table_atom_ownership": "Line source atoms must be owned exactly once",
    "table_complete_expression": "Complete source expression must have one math owner",
    "table_content_ownership": "Output content has no source ownership",
    "table_selectable_text_figure": "Selectable text cannot disappear into a figure",
    "table_text_unresolved": "Required text remains unresolved",
    "content_unresolved": "Required content unresolved",
    "content_ocr_disabled": "Required text OCR is disabled",
    "content_expression_kind": "Required expression must be math",
}


def semantic_error(code: str):
    """Create one declared constraint; only its registry text may be emitted."""
    return PydanticCustomError(code, SEMANTIC_ERRORS[code])


class ResponseValidationError(ValueError):
    """A response is incomplete or invalid, with safe details for a repair."""


def schema_error_reason(error: ValidationError, schema: type[BaseModel]) -> str:
    """Describe declared fields and error types without reflecting input values."""
    names = set()

    def collect_names(node):
        if isinstance(node, dict):
            names.update(node.get("properties", {}))
            for value in node.values():
                collect_names(value)
        elif isinstance(node, list):
            for value in node:
                collect_names(value)

    collect_names(schema.model_json_schema())
    known_types = set(get_args(ErrorType))
    details = []
    for item in error.errors(include_input=False, include_context=False, include_url=False):
        location = ".".join(
            str(part) if isinstance(part, int) or part in names else "[field]"
            for part in item["loc"]
        ) or "JSON"
        error_type = item["type"] if item["type"] in known_types else "validation_error"
        detail = f"{location}: {error_type}"
        if item["type"] in SEMANTIC_ERRORS:
            detail = f"{location}: {SEMANTIC_ERRORS[item['type']]}"
        if error_type == "json_invalid":
            position = re.search(r"line (\d+) column (\d+)", item["msg"])
            if position:
                detail += f" at line {position[1]} column {position[2]}"
        details.append(detail)
    return "; ".join(details)


def validate_response(
    text: str | None,
    schema: type[BaseModel],
    finish_reason: str | None = None,
) -> dict:
    if finish_reason == "length":
        raise ResponseValidationError("Response was truncated; regenerate complete JSON")
    if not isinstance(text, str) or not text.strip():
        raise ResponseValidationError("Response contains no JSON")
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence[1].strip()
    try:
        result = schema.model_validate_json(text, strict=True).model_dump()
    except ValidationError as error:
        raise ResponseValidationError(schema_error_reason(error, schema)) from None
    if not isinstance(result, dict):
        raise ResponseValidationError("Response must be a JSON object")

    def has_control_character(value, field=None):
        if isinstance(value, str):
            if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", value):
                return True
            # JSON's legal \t, \r and \n escapes can corrupt \theta, \rho
            # and \nabla. Formula bodies use spaces for layout; ordinary OCR
            # text and HTML outside math retain their deliberate line breaks.
            formulas = [value] if field == "latex" else re.findall(
                r"<math\b[^>]*>(.*?)</math>", value, re.DOTALL | re.IGNORECASE
            )
            return any(re.search(r"[\x00-\x1f]", formula) for formula in formulas)
        if isinstance(value, dict):
            return any(has_control_character(item, name) for name, item in value.items())
        if isinstance(value, list):
            return any(has_control_character(item, field) for item in value)
        return False

    if has_control_character(result):
        raise ResponseValidationError("JSON string has a control escape; double every LaTeX backslash")
    return result


def schema_prompt(prompt: str, schema: type[BaseModel]) -> str:
    return (
        prompt + "\n\nReturn only complete JSON matching this schema. "
        "Encode every literal LaTeX backslash as a doubled JSON backslash. "
        "Use spaces for whitespace inside formula strings and HTML math bodies.\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


def repair_prompt(prompt: str, reason: str) -> str:
    return prompt + "\n\nThe previous response failed validation: " + reason + (
        ". Regenerate the complete JSON object from the source. "
        "Include every required field with its correct type; double literal backslashes."
    )


def context_overflow(error):
    """Classify locally; no provider body/message is returned to callers."""
    if not isinstance(error, openai.APIStatusError) or error.status_code != 400:
        return False
    body = error.body if isinstance(error.body, dict) else {}
    body = body.get("error", body)
    if not isinstance(body, dict):
        return False
    return body.get("code") in (
        "context_length_exceeded",
        "context_window_exceeded",
    ) or bool(
        re.search(
            r"(?:input length exceeds|exceed[^.]*context (?:window|length)|context length exceeded)",
            str(body.get("message", "")),
            re.I,
        )
    )


def retryable_provider_error(error: Exception) -> bool:
    return isinstance(error, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)) or (
        isinstance(error, openai.APIStatusError) and error.status_code >= 500
    )


def provider_error_reason(error: Exception) -> str:
    # Provider text can echo credentials or base64.  Only classify it locally.
    if isinstance(error, openai.APIStatusError):
        return f"Provider request failed with HTTP {error.status_code}"
    if isinstance(error, openai.APITimeoutError):
        return "Provider request timed out"
    if isinstance(error, openai.APIConnectionError):
        return "Provider connection failed"
    return "Provider request failed"


def _document_request(method):
    """Keep document resets outside whole calls, including retries and waits."""
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            self._active_requests += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            with self._lock:
                self._active_requests -= 1
    return call


class CloudClient:
    """One shared instance owns every attempt across a document's cloud stages."""

    def __init__(self, config=None, *, client=None):
        config = config.model_dump() if isinstance(config, BaseModel) else dict(config or {})
        # Keep the primitive window-planning defaults stable. Whole-object
        # callers, including OllamaService, supply their larger explicit budgets.
        self.cloud_context_tokens = int(config.get("cloud_context_tokens", 8192))
        self.cloud_max_output_tokens = int(config.get("cloud_max_output_tokens", 2048))
        self.cloud_max_images = int(config.get("cloud_max_images", 3))
        self.cloud_max_attempts = int(config.get("cloud_max_attempts", DEFAULT_MAX_ATTEMPTS))
        self.cloud_max_requests = int(config.get("cloud_max_requests", 0))
        self.cloud_max_concurrency = int(config.get("cloud_max_concurrency", 1))
        self.cloud_model = config.get("cloud_model", config.get("openai_model", DEFAULT_CLOUD_MODEL))
        self.cloud_base_url = config.get("cloud_base_url", config.get("openai_base_url", DEFAULT_CLOUD_BASE_URL))
        self.timeout = config.get("cloud_timeout", config.get("timeout", DEFAULT_TIMEOUT))
        if self.cloud_max_attempts < 1 or self.cloud_max_requests < 0 or self.cloud_max_images < 0:
            raise ValueError("Cloud attempts must be positive; image/request limits must be nonnegative")
        if not 1 <= self.cloud_max_concurrency <= 4:
            raise ValueError("Cloud concurrency must be between one and four")
        self._client = client.with_options(max_retries=0) if client is not None else None
        self._lock = Lock()
        self._concurrency = BoundedSemaphore(self.cloud_max_concurrency)
        self._active_requests = 0
        self.request_count = 0
        self.total_tokens = 0
        self.last_usage = None

    def start_document(self):
        """Reset one document's budget while retaining transport and settings.

        Call once before scheduling a new document. Resetting while a request
        is queued, in HTTP transport, validating, or retrying would split its
        accounting between documents, so such overlap fails explicitly.
        """
        with self._lock:
            if self._active_requests:
                raise RecognitionFailure(
                    "Cannot start a document while recognition requests are active"
                )
            self.request_count = 0
            self.total_tokens = 0
            self.last_usage = None

    def _get_client(self):
        with self._lock:
            if self._client is None:
                key = os.environ.get("OLLAMA_API_KEY")
                if not key:
                    if urlparse(self.cloud_base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
                        raise RecognitionFailure("OLLAMA_API_KEY is required for cloud recognition")
                    # The SDK requires a nonempty key even for unauthenticated
                    # local Ollama servers. This value is never a cloud credential.
                    key = "ollama"
                self._client = openai.OpenAI(api_key=key, base_url=self.cloud_base_url, max_retries=0)
            return self._client

    def _reserve_request(self, attempts):
        with self._lock:
            if self.cloud_max_requests and self.request_count >= self.cloud_max_requests:
                raise RecognitionFailure("Document cloud request budget exhausted", attempts=attempts)
            self.request_count += 1
            return self.request_count

    def _request_options(self, max_attempts, timeout, max_output_tokens):
        attempts = self.cloud_max_attempts if max_attempts is None else int(max_attempts)
        timeout = self.timeout if timeout is None else timeout
        output_tokens = self.cloud_max_output_tokens if max_output_tokens is None else int(max_output_tokens)
        if attempts < 1 or timeout <= 0 or output_tokens < 1:
            raise ValueError("Request attempts, timeout and output tokens must be positive")
        return attempts, timeout, output_tokens

    def _record_usage(self, usage, metadata):
        tokens = usage.total_tokens if usage else 0
        with self._lock:
            self.last_usage = usage.model_dump() if usage else None
            self.total_tokens += tokens
        if metadata is not None:
            metadata["llm_tokens_used"] = metadata.get("llm_tokens_used", 0) + tokens

    @_document_request
    def request_text(
        self, prompt, images, validator, *, max_output_tokens=None, image_long_edge=3072,
        max_attempts=None, timeout=None, metadata=None,
    ):
        """Recognize a complete object as HTML/LaTeX without a JSON wrapper.

        The caller supplies a local validator. The original request/attempt
        budget and credential protections apply; no thumbnail below 2048 pixels
        replaces a large table crop just to fit an undersized context budget.
        """
        images = [images] if isinstance(images, Image.Image) else list(images or [])
        if len(images) > self.cloud_max_images:
            raise RecognitionFailure("Required images exceed the cloud request image limit")
        attempt_limit, request_timeout, output_tokens = self._request_options(
            max_attempts, timeout, max_output_tokens
        )
        original_prompt = current_prompt = prompt
        reason = "Required content unresolved"
        attempts = 0
        for _ in range(attempt_limit):
            try:
                copies = fit_request(
                    current_prompt, images, self.cloud_context_tokens, output_tokens,
                    max_image_edge=image_long_edge,
                    min_image_edge=min(2048, image_long_edge),
                )
            except RecognitionFailure as error:
                raise RecognitionFailure(error.reason, attempts=attempts) from None
            content = []
            for image in copies:
                buffer = BytesIO()
                try:
                    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                        image = image.convert("RGB")
                    image.save(buffer, format="PNG")
                except Exception:
                    raise RecognitionFailure("Request image encoding failed", attempts=attempts) from None
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + encoded
                }})
            content.append({"type": "text", "text": current_prompt})
            client = self._get_client()
            request_number = self._reserve_request(attempts)
            attempts += 1
            if metadata is not None:
                metadata["llm_request_count"] = metadata.get("llm_request_count", 0) + 1
            started = monotonic()
            logging.getLogger("marker").info(
                "Ollama request %d, attempt %d/%d (whole object)",
                request_number, attempts, attempt_limit,
            )
            try:
                with self._concurrency:
                    response = client.chat.completions.create(
                        model=self.cloud_model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=output_tokens, temperature=0, timeout=request_timeout,
                    )
                self._record_usage(response.usage, metadata)
                if not response.choices:
                    raise ResponseValidationError("Response contains no completion")
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    raise ResponseValidationError("Response was truncated; return the complete object")
                text = choice.message.content
                if not isinstance(text, str) or not text.strip():
                    raise ResponseValidationError("Response contains no recognized content")
                result = validator(text)
                logging.getLogger("marker").info(
                    "Ollama request %d validated in %.1f s", request_number, monotonic() - started
                )
                return result
            except ResponseValidationError as error:
                reason = str(error)
                current_prompt = original_prompt + (
                    "\nThe previous response failed local validation: " + reason +
                    ". Return the complete corrected object from the same source. "
                    "Do not omit content, explain the answer, or return JSON."
                )
            except RecognitionFailure:
                raise
            except Exception as error:
                if context_overflow(error):
                    raise RecognitionFailure(
                        "Whole source exceeds provider context; increase the context budget",
                        attempts=attempts,
                    ) from None
                reason = provider_error_reason(error)
                if not retryable_provider_error(error):
                    raise RecognitionFailure(reason, attempts=attempts) from None
        raise RecognitionFailure(reason, attempts=attempts)

    @_document_request
    def request(
        self, prompt: str, images, schema: type[BaseModel], *, context_reflow=None,
        max_attempts=None, timeout=None, max_output_tokens=None, metadata=None,
    ) -> dict:
        """Recognize once with at most cloud_max_attempts, including owner reflow.

        context_reflow() may return a strictly smaller (prompt, images, schema)
        tuple after a server context error, or None for an indivisible region.
        """
        images = [images] if isinstance(images, Image.Image) else list(images or [])
        if len(images) > self.cloud_max_images:
            raise RecognitionFailure("Required images exceed the cloud request image limit")
        attempt_limit, request_timeout, output_tokens = self._request_options(
            max_attempts, timeout, max_output_tokens
        )
        original_prompt = schema_prompt(prompt, schema)
        current_prompt = original_prompt
        reason = "Cloud recognition failed"
        attempts = 0
        for _ in range(attempt_limit):
            try:
                copies = fit_request(current_prompt, images, self.cloud_context_tokens, output_tokens)
            except RecognitionFailure as error:
                raise RecognitionFailure(error.reason, attempts=attempts) from None
            client = self._get_client()
            content = []
            for image in copies:
                stream = BytesIO()
                try:
                    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                        image = image.convert("RGB")
                    image.save(stream, format="PNG")
                except Exception:
                    raise RecognitionFailure("Request image encoding failed", attempts=attempts) from None
                data = base64.b64encode(stream.getvalue()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}})
            content.append({"type": "text", "text": current_prompt})
            request_number = self._reserve_request(attempts)
            attempts += 1
            if metadata is not None:
                metadata["llm_request_count"] = metadata.get("llm_request_count", 0) + 1
            started = monotonic()
            logging.getLogger("marker").info(
                "Ollama request %d, attempt %d/%d", request_number, attempts, attempt_limit
            )
            try:
                with self._concurrency:
                    response = client.chat.completions.create(
                        model=self.cloud_model, messages=[{"role": "user", "content": content}],
                        max_tokens=output_tokens, temperature=0, timeout=request_timeout,
                    )
                self._record_usage(response.usage, metadata)
                if not response.choices:
                    raise ResponseValidationError("Response contains no completion")
                choice = response.choices[0]
                result = validate_response(choice.message.content, schema, choice.finish_reason)
                logging.getLogger("marker").info(
                    "Ollama request %d validated in %.1f s", request_number, monotonic() - started
                )
                return result
            except ResponseValidationError as error:
                reason = str(error)
                current_prompt = repair_prompt(original_prompt, reason)
            except Exception as error:
                if context_overflow(error):
                    reason = "Provider context limit exceeded; reduce the assigned source window"
                    if attempts >= attempt_limit:
                        raise RecognitionFailure(reason, attempts=attempts) from None
                    # The owner can reduce text/line ownership as well as pixels.
                    # Reflow remains inside this request's original attempt loop.
                    replacement = (
                        context_reflow() if context_reflow is not None else None
                    )
                    if replacement is not None:
                        prompt, replacement_images, replacement_schema = replacement
                        replacement_prompt = schema_prompt(prompt, replacement_schema)
                        replacement_images = list(replacement_images)
                        if len(replacement_images) > self.cloud_max_images:
                            raise RecognitionFailure(
                                "Required images exceed the cloud request image limit",
                                attempts=attempts,
                            )
                        if len(replacement_prompt.encode()) >= len(
                            current_prompt.encode()
                        ):
                            raise RecognitionFailure(
                                reason, attempts=attempts
                            ) from None
                        schema = replacement_schema
                        images = replacement_images
                        original_prompt = current_prompt = replacement_prompt
                        continue
                    # Atomic regions can still reduce disposable image copies.
                    # Never retry an identical oversized payload.
                    reduced = []
                    for image in copies:
                        target = max(1, max(image.size) // 2)
                        copy = image.copy()
                        copy.thumbnail((target, target), Image.Resampling.LANCZOS)
                        reduced.append(copy)
                    if not any(a.size != b.size for a, b in zip(copies, reduced)):
                        raise RecognitionFailure(reason, attempts=attempts) from None
                    images = reduced
                    continue
                reason = provider_error_reason(error)
                if not retryable_provider_error(error):
                    raise RecognitionFailure(reason, attempts=attempts) from None
        raise RecognitionFailure(reason, attempts=attempts)
