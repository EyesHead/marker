"""OpenAI Responses transport for native Marker, with a durable PDF budget."""

import hashlib
from collections import OrderedDict
from concurrent.futures import Future
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from threading import Lock
from urllib.parse import urlsplit

import openai

from marker.cloud.client import CloudClient, ResponseValidationError
from marker.cloud.context import RecognitionFailure
from marker.cloud.defaults import (DEFAULT_CONTEXT_TOKENS, DEFAULT_MAX_IMAGES,
                                   DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TIMEOUT)
from marker.cloud.openai_budget import BudgetError, DollarBudget


def budget_directory():
    if os.environ.get("MARKER_CACHE"):
        root = Path(os.environ["MARKER_CACHE"]).expanduser()
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "eyeshead-marker"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "eyeshead-marker"
        if sys.platform == "darwin" and not root.exists():
            root = Path.home() / "Library/Caches/eyeshead-marker"
    return root / "budgets"


class _Usage:
    def __init__(self, usage):
        self.total_tokens = usage.total_tokens
        self._value = usage.model_dump()

    def model_dump(self):
        return self._value


class OpenAIResponsesClient(CloudClient):
    """Reuse source sizing, validation, caching and retries; replace HTTP only."""

    provider_label = "OpenAI"

    def __init__(self, config=None, *, client=None):
        config = dict(config or {})
        shared = {"cloud_model": "gpt-6-luna", "cloud_base_url": "https://api.openai.com/v1",
                  "cloud_context_tokens": DEFAULT_CONTEXT_TOKENS,
                  "cloud_max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                  "cloud_max_images": DEFAULT_MAX_IMAGES,
                  "cloud_max_concurrency": 2, "cloud_max_attempts": 2,
                  "cloud_timeout": DEFAULT_TIMEOUT, **config}
        super().__init__(shared, client=client)
        parsed = urlsplit(self.cloud_base_url)
        if (parsed.scheme != "https" or parsed.netloc != "api.openai.com"
                or parsed.path.rstrip("/") != "/v1" or parsed.query or parsed.fragment):
            raise ValueError("Budgeted OpenAI recognition requires https://api.openai.com/v1")
        if self.cloud_model not in {"gpt-6-luna", "gpt-6.1-sol"}:
            raise ValueError("Verified budget pricing is available for gpt-6-luna and gpt-6.1-sol only")
        self.reasoning_effort = config.get("openai_reasoning_effort", "low")
        choices = {"low", "medium", "high", "xhigh", "max"}
        if self.cloud_model == "gpt-6-luna":
            choices.add("none")
        if self.reasoning_effort not in choices:
            raise ValueError("Unsupported reasoning effort for the selected OpenAI model")
        self.limit_usd = str(config.get("cloud_budget_usd", "1.00"))
        # Validate even before a PDF is bound, without creating persistent files.
        DollarBudget(None, self.limit_usd)
        self.budget_dir = Path(config.get("cloud_budget_dir") or budget_directory())
        self.budget = None
        self._memo_lock = Lock()
        self._text_cache = OrderedDict()
        self._text_pending = {}
        self._count_cache = OrderedDict()
        self.cache_hits = self.generation_calls = self.count_calls = 0

    def bind_document(self, filepath):
        digest = hashlib.sha256()
        with open(filepath, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with self._lock:
            if self._active_requests:
                raise RecognitionFailure("Cannot change the PDF budget while requests are active")
            try:
                self.budget = DollarBudget(self.budget_dir / (digest.hexdigest() + ".json"), self.limit_usd)
            except BudgetError as error:
                raise RecognitionFailure(str(error)) from None

    def budget_snapshot(self):
        if self.budget is None:
            return {}
        try:
            value = self.budget.snapshot()
        except BudgetError as error:
            raise RecognitionFailure(str(error)) from None
        with self._memo_lock:
            value.update(generation_calls=self.generation_calls,
                         token_count_calls=self.count_calls,
                         validated_cache_hits=self.cache_hits)
        return value

    def start_document(self):
        super().start_document()
        self.cache_hits = self.generation_calls = self.count_calls = 0

    def _text_key(self, prompt, images, options):
        digest = hashlib.sha256()
        digest.update(json.dumps((self.cloud_model, self.reasoning_effort, prompt,
                                 options.get("max_output_tokens", self.cloud_max_output_tokens),
                                 options.get("image_long_edge", 3072)), ensure_ascii=False).encode())
        from PIL import Image
        pictures = [images] if isinstance(images, Image.Image) else list(images or [])
        for picture in pictures:
            if not isinstance(picture, Image.Image) or min(picture.size) <= 0:
                raise RecognitionFailure("Request images must have valid pixel dimensions")
            digest.update(repr((picture.mode, picture.size)).encode())
            # Palette indices alone do not describe displayed colours.
            digest.update(picture.convert("RGBA").tobytes())
        return digest.hexdigest()

    def request_text(self, prompt, images, validator, **options):
        """Reuse only identical, successfully validated source transcriptions.

        All pixels, the complete prompt, model, reasoning and output limits
        participate in the key. The caller validates a cached answer again.
        Concurrent identical jobs share one request; tables are never batched.
        """
        from PIL import Image
        images = [images] if isinstance(images, Image.Image) else list(images or [])
        key = self._text_key(prompt, images, options)
        with self._memo_lock:
            raw = self._text_cache.get(key)
            future = self._text_pending.get(key)
            owner = raw is None and future is None
            if owner:
                future = self._text_pending[key] = Future()
        if not owner:
            try:
                answer = raw if raw is not None else future.result()
                result = validator(answer)
                with self._memo_lock:
                    self.cache_hits += 1
                return result
            except ResponseValidationError:
                # A different source-aware validator can require a fresh read.
                with self._memo_lock:
                    self._text_cache.pop(key, None)
                return super().request_text(prompt, images, validator, **options)
        accepted = []

        def capture(answer):
            result = validator(answer)
            accepted.append(answer)
            return result

        try:
            result = super().request_text(prompt, images, capture, **options)
            with self._memo_lock:
                self._text_cache[key] = accepted[-1]
                if len(self._text_cache) > 128:
                    self._text_cache.popitem(last=False)
            future.set_result(accepted[-1])
            return result
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self._memo_lock:
                self._text_pending.pop(key, None)

    def _get_client(self):
        with self._lock:
            if self._client is None:
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise RecognitionFailure("OPENAI_API_KEY is required for OpenAI recognition")
                self._client = openai.OpenAI(api_key=key, base_url=self.cloud_base_url, max_retries=0)
            return self._client

    def _create_completion(self, client, content, output_tokens, timeout):
        if self.budget is None:
            raise RecognitionFailure("Bind the input PDF before budgeted OpenAI recognition")
        parts = [{"type": "input_text", "text": item["text"]} if item["type"] == "text"
                 else {"type": "input_image", "image_url": item["image_url"]["url"], "detail": "high"}
                 for item in content]
        common = {"model": self.cloud_model, "input": [{"role": "user", "content": parts}],
                  "reasoning": {"effort": self.reasoning_effort},
                  "text": {"verbosity": "low"}, "truncation": "disabled"}
        # Exact multimodal count: never substitute character or image estimates
        # for dollar accounting. An unavailable count endpoint stops generation.
        fingerprint = hashlib.sha256()
        fingerprint.update(json.dumps((self.cloud_model, self.reasoning_effort), separators=(",", ":")).encode())
        for part in parts:
            payload = part.get("text", part.get("image_url", "")).encode()
            fingerprint.update(f"{part['type']}:{len(payload)}:".encode())
            fingerprint.update(payload)
        count_key = fingerprint.hexdigest()
        with self._memo_lock:
            input_tokens = self._count_cache.get(count_key)
        if input_tokens is None:
            counted = client.responses.input_tokens.count(**common, timeout=timeout)
            input_tokens = counted.input_tokens
            with self._memo_lock:
                self.count_calls += 1
                self._count_cache[count_key] = input_tokens
                if len(self._count_cache) > 128:
                    self._count_cache.popitem(last=False)
        if type(input_tokens) is not int or input_tokens < 0:
            raise RecognitionFailure("OpenAI returned an invalid input token count")
        if input_tokens + output_tokens > 1_050_000:
            raise RecognitionFailure("The whole source object exceeds the selected OpenAI model context")
        try:
            reservation = self.budget.reserve(input_tokens, output_tokens, self.cloud_model)
        except BudgetError as error:
            raise RecognitionFailure(str(error)) from None
        logging.getLogger("marker").info("OpenAI %s: input=%d, output cap=%d; USD budget=%s",
                                        self.cloud_model, input_tokens, output_tokens, self.budget_snapshot())
        # A lost response leaves its whole reservation charged/uncertain. A
        # restart or model switch cannot silently reset that commitment.
        with self._memo_lock:
            self.generation_calls += 1
        response = client.responses.create(**common, max_output_tokens=output_tokens,
                                           service_tier="default", store=False, timeout=timeout)
        if response.usage is not None:
            usage = response.usage
            details = usage.input_tokens_details
            cached = getattr(details, "cached_tokens", None)
            written = getattr(details, "cache_write_tokens", None)
            if type(cached) is not int or type(written) is not int:
                raise RecognitionFailure("OpenAI cache usage is incomplete; full request reservation remains held")
            try:
                self.budget.settle(reservation, usage.input_tokens, usage.output_tokens,
                                   cached_input_tokens=cached, cache_write_tokens=written)
            except BudgetError as error:
                raise RecognitionFailure(str(error)) from None
        else:
            raise RecognitionFailure("OpenAI returned no usage; full request reservation remains held")
        logging.getLogger("marker").info("OpenAI USD usage: %s", self.budget_snapshot())
        # Shared recognizers consume this small transport-neutral completion
        # shape. Reasoning items/refusals are never published as source text.
        finished = response.status == "completed"
        return SimpleNamespace(
            usage=_Usage(response.usage),
            choices=[SimpleNamespace(finish_reason="stop" if finished else "length",
                                     message=SimpleNamespace(content=response.output_text if finished else ""))],
        )
