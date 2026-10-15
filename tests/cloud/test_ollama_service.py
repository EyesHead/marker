"""Native Marker Ollama service contracts, using only an in-memory transport."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from threading import Event, Lock
from time import sleep
import unittest
from unittest.mock import patch

import httpx
import openai
from PIL import Image
from pydantic import BaseModel

from marker.cloud.client import CloudClient, ResponseValidationError
from marker.cloud.context import RecognitionFailure
from marker.cloud.defaults import (
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_MAX_IMAGES,
    DEFAULT_MAX_OUTPUT_TOKENS,
)
from marker.services.ollama import OllamaService


class Formula(BaseModel):
    latex: str


class NestedFormula(BaseModel):
    expression: Formula


class BlockMetadata:
    def __init__(self):
        self.values = defaultdict(int)

    def update_metadata(self, **values):
        for name, value in values.items():
            self.values[name] += value


def completion(text, finish_reason="stop"):
    return httpx.Response(200, json={
        "id": "offline", "object": "chat.completion", "created": 0,
        "model": DEFAULT_CLOUD_MODEL,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {
            "role": "assistant", "content": text,
        }}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    })


class OfflineTransport:
    def __init__(self, responses=(), *, handler=None):
        self.responses = iter(responses)
        self.requests = []
        self.handler = handler
        self.client = openai.OpenAI(
            api_key="offline-transport-key", base_url=DEFAULT_CLOUD_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self.respond)),
        )

    def respond(self, request):
        self.requests.append(request)
        if self.handler:
            return self.handler(request)
        value = next(self.responses)
        return value if isinstance(value, httpx.Response) else completion(value)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.client.close()


def complete_table(text):
    if not text.startswith("<table>") or not text.endswith("</table>"):
        raise ResponseValidationError("Complete HTML table required")
    return text


class OllamaServiceTests(unittest.TestCase):
    def test_defaults_use_cloud_vision_model_and_whole_object_budgets(self):
        service = OllamaService()
        self.assertEqual(service.ollama_model, "kimi-k3")
        self.assertEqual(service.client.cloud_model, DEFAULT_CLOUD_MODEL)
        self.assertEqual(service.client.cloud_base_url, DEFAULT_CLOUD_BASE_URL)
        self.assertEqual(service.client.cloud_context_tokens, DEFAULT_CONTEXT_TOKENS)
        self.assertEqual(service.client.cloud_max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertEqual(service.client.cloud_max_images, DEFAULT_MAX_IMAGES)
        self.assertEqual(service.client.cloud_max_concurrency, 2)
        self.assertEqual(service.client.cloud_max_attempts, 3)
        self.assertEqual(service.client.timeout, 180)

    def test_primitive_and_whole_object_service_keep_distinct_planning_budgets(self):
        direct = CloudClient()
        service = OllamaService()
        self.assertEqual(direct.cloud_model, service.client.cloud_model)
        self.assertEqual(direct.cloud_context_tokens, 8192)
        self.assertEqual(direct.cloud_max_output_tokens, 2048)
        self.assertEqual(direct.cloud_max_images, 3)
        self.assertEqual(direct.cloud_max_concurrency, 1)
        self.assertEqual(service.client.cloud_context_tokens, DEFAULT_CONTEXT_TOKENS)
        self.assertEqual(service.client.cloud_max_images, DEFAULT_MAX_IMAGES)

    def test_explicit_local_model_and_origin_remain_supported(self):
        service = OllamaService({
            "ollama_model": "custom-vision:small",
            "ollama_base_url": "http://127.0.0.1:11434/",
            "cloud_context_tokens": 32768, "cloud_max_images": 4,
            "max_output_tokens": 4096, "max_retries": 0, "timeout": 27,
        })
        self.assertEqual(service.client.cloud_model, "custom-vision:small")
        self.assertEqual(service.client.cloud_base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(service.client.cloud_max_output_tokens, 4096)
        self.assertEqual(service.client.cloud_max_attempts, 1)
        self.assertEqual(service.client.timeout, 27)
        with patch.dict(os.environ, {}, clear=True), patch("marker.cloud.client.openai.OpenAI") as factory:
            service.client._get_client()
        self.assertEqual(factory.call_args.kwargs["api_key"], "ollama")

    def test_explicit_cloud_aliases_and_service_specific_settings(self):
        service = OllamaService({
            "cloud_model": "custom-cloud", "cloud_base_url": "https://provider.example/compatible/v1",
            "cloud_timeout": 39, "cloud_max_attempts": 1,
            "OllamaService_ollama_model": "specific-cloud",
            "OllamaService_cloud_max_images": 5,
        })
        self.assertEqual(service.client.cloud_model, "specific-cloud")
        self.assertEqual(service.client.cloud_base_url, "https://provider.example/compatible/v1")
        self.assertEqual(service.client.cloud_max_images, 5)
        self.assertEqual(service.client.cloud_max_attempts, 1)
        self.assertEqual(service.client.timeout, 39)

    def test_schema_is_locally_validated_without_provider_schema_options(self):
        latex = r"0,42R_{\text{bun}}"
        with OfflineTransport([json.dumps({"latex": latex})]) as offline:
            service = OllamaService(client=offline.client)
            block = BlockMetadata()
            result = service("Read exact subscripts.", Image.new("RGB", (100, 60)), block, Formula)
            self.assertEqual(result, {"latex": latex})
            payload = json.loads(offline.requests[0].content)
            self.assertEqual(payload["model"], "kimi-k3")
            self.assertNotIn("format", payload)
            self.assertNotIn("response_format", payload)
            content = payload["messages"][0]["content"]
            self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertIn('"latex"', content[-1]["text"])
            self.assertEqual(dict(block.values), {"llm_request_count": 1, "llm_tokens_used": 18})

    def test_nested_definitions_are_kept_in_schema_and_checked_locally(self):
        with OfflineTransport(['{"expression":{}}', '{"expression":{"latex":"x"}}']) as offline:
            service = OllamaService(client=offline.client)
            self.assertEqual(service("formula", None, None, NestedFormula), {
                "expression": {"latex": "x"},
            })
            prompt = json.loads(offline.requests[0].content)["messages"][0]["content"][-1]["text"]
            repair = json.loads(offline.requests[1].content)["messages"][0]["content"][-1]["text"]
            self.assertIn('"$defs"', prompt)
            self.assertIn("expression.latex", repair)

    def test_whole_table_image_limit_rejects_without_sending_or_dropping_a_page(self):
        with OfflineTransport([]) as offline:
            service = OllamaService(client=offline.client)
            images = [Image.new("RGB", (16, 16))] * (DEFAULT_MAX_IMAGES + 1)
            with self.assertRaisesRegex(RecognitionFailure, "image limit"):
                service.request_text("whole table", images, complete_table)
            self.assertEqual(offline.requests, [])
            self.assertEqual(service.client.request_count, 0)

    def test_schema_repair_and_raw_html_share_client_budget_and_metadata(self):
        html = "<table><tr><td>diagram</td></tr></table>"
        with OfflineTransport(["{}", '{"latex":"x"}', html]) as offline:
            service = OllamaService({"cloud_max_requests": 3}, client=offline.client)
            block = BlockMetadata()
            self.assertEqual(service("formula", None, block, Formula), {"latex": "x"})
            self.assertEqual(service.request_text("whole table", [], complete_table, block=block), html)
            self.assertEqual(service.client.request_count, 3)
            self.assertEqual(dict(block.values), {"llm_request_count": 3, "llm_tokens_used": 54})
            with self.assertRaisesRegex(RecognitionFailure, "budget"):
                service("formula", None, block, Formula)
            self.assertEqual(dict(block.values), {"llm_request_count": 3, "llm_tokens_used": 54})

    def test_per_call_retry_timeout_and_output_overrides_do_not_mutate_shared_client(self):
        html = "<table></table>"
        with OfflineTransport(["{}", '{"latex":"x"}', html]) as offline:
            service = OllamaService(client=offline.client)
            block = BlockMetadata()
            with self.assertRaises(RecognitionFailure) as error:
                service("formula", None, block, Formula, max_retries=0, timeout=7)
            self.assertEqual(error.exception.attempts, 1)
            self.assertEqual(service.client.cloud_max_attempts, 3)
            self.assertEqual(service.client.timeout, 180)
            self.assertEqual(service("formula", None, None, Formula), {"latex": "x"})
            self.assertEqual(service.request_text(
                "table", [], complete_table, max_retries=0, timeout=11, max_output_tokens=512,
            ), html)
            self.assertEqual(offline.requests[0].extensions["timeout"]["read"], 7)
            self.assertEqual(offline.requests[1].extensions["timeout"]["read"], 180)
            self.assertEqual(offline.requests[2].extensions["timeout"]["read"], 11)
            self.assertEqual(json.loads(offline.requests[2].content)["max_tokens"], 512)
            self.assertEqual(service.client.cloud_max_output_tokens, 16384)
            self.assertEqual(dict(block.values), {"llm_request_count": 1, "llm_tokens_used": 18})

    def test_failure_is_explicit_and_provider_cannot_leak_credentials(self):
        secret = "only-environment-placeholder-key"
        response = httpx.Response(401, json={"error": {"message": secret}})
        with OfflineTransport([response]) as offline:
            service = OllamaService(client=offline.client)
            block = BlockMetadata()
            with self.assertLogs("marker", level="INFO") as logs:
                with self.assertRaises(RecognitionFailure) as raised:
                    service("Read.", None, block, Formula)
            self.assertNotIn(secret, str(raised.exception))
            self.assertNotIn(secret, " ".join(logs.output))
            self.assertNotIn(secret, offline.requests[0].content.decode())
            self.assertEqual(dict(block.values), {"llm_request_count": 1})

    def test_cloud_key_is_read_only_from_environment_at_first_request(self):
        service = OllamaService({"ollama_api_key": "ignored-config-key"})
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RecognitionFailure, "OLLAMA_API_KEY"):
                service.client._get_client()
        self.assertEqual(service.client.request_count, 0)
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "environment-test-key"}), \
                patch("marker.cloud.client.openai.OpenAI") as factory:
            service.client._get_client()
        self.assertEqual(factory.call_args.kwargs["api_key"], "environment-test-key")
        self.assertEqual(factory.call_args.kwargs["base_url"], "https://ollama.com/v1")
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)

    def test_concurrent_calls_have_separate_metadata_and_respect_transport_limit(self):
        counts = {"active": 0, "maximum": 0}
        lock = Lock()

        def respond(_request):
            with lock:
                counts["active"] += 1
                counts["maximum"] = max(counts["active"], counts["maximum"])
            sleep(0.02)
            with lock:
                counts["active"] -= 1
            return completion('{"latex":"x"}')

        with OfflineTransport(handler=respond) as offline:
            service = OllamaService({"cloud_max_concurrency": 2}, client=offline.client)
            blocks = [BlockMetadata() for _ in range(4)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda block: service("formula", None, block, Formula), blocks))
            self.assertEqual(results, [{"latex": "x"}] * 4)
            self.assertLessEqual(counts["maximum"], 2)
            self.assertEqual(service.client.request_count, 4)
            self.assertEqual(service.client.total_tokens, 72)
            for block in blocks:
                self.assertEqual(dict(block.values), {"llm_request_count": 1, "llm_tokens_used": 18})

    def test_invalid_zero_attempt_limit_is_not_replaced_by_default(self):
        with self.assertRaises(ValueError):
            OllamaService({"cloud_max_attempts": 0})

    def test_explicit_legacy_concurrency_is_used_unless_cloud_setting_overrides_it(self):
        for config, expected in (
            ({"max_concurrency": 1}, 1),
            ({"max_concurrency": 4}, 4),
            ({"max_concurrency": 1, "cloud_max_concurrency": 3}, 3),
            ({"max_concurrency": 4, "OllamaService_cloud_max_concurrency": 1}, 1),
            ({"OllamaService_max_concurrency": 3}, 3),
        ):
            with self.subTest(config=config):
                self.assertEqual(OllamaService(config).client.cloud_max_concurrency, expected)
        for value in (0, 5):
            with self.subTest(invalid_concurrency=value), self.assertRaises(ValueError):
                OllamaService({"max_concurrency": value})

    def test_each_document_gets_its_own_finite_cap_with_same_service_and_transport(self):
        with OfflineTransport(['{"latex":"first"}', '{"latex":"second"}']) as offline:
            service = OllamaService({"cloud_max_requests": 1}, client=offline.client)
            transport = service.client._get_client()
            for label in ("first", "second"):
                service.client.start_document()
                self.assertEqual(service.client.request_count, 0)
                self.assertEqual(service.client.total_tokens, 0)
                self.assertIsNone(service.client.last_usage)
                self.assertIs(service.client._get_client(), transport)
                self.assertEqual(service("formula", None, None, Formula), {"latex": label})
                with self.assertRaisesRegex(RecognitionFailure, "budget"):
                    service("formula", None, None, Formula)
                self.assertEqual(service.client.request_count, 1)
                self.assertEqual(service.client.total_tokens, 18)
            self.assertEqual(len(offline.requests), 2)
            self.assertEqual(service.client.cloud_max_requests, 1)

    def test_document_reset_is_rejected_while_transport_or_validator_is_active(self):
        entered = Event()
        release = Event()

        def respond(_request):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Offline transport was not released")
            return completion("<table></table>")

        with OfflineTransport(handler=respond) as offline:
            service = OllamaService(client=offline.client)

            def validate(text):
                # The response already arrived, but ownership remains with this
                # document until local validation and accounting have finished.
                with self.assertRaisesRegex(RecognitionFailure, "requests are active"):
                    service.client.start_document()
                return complete_table(text)

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(service.request_text, "table", [], validate)
                try:
                    self.assertTrue(entered.wait(5))
                    with self.assertRaisesRegex(RecognitionFailure, "requests are active"):
                        service.client.start_document()
                    self.assertEqual(service.client.request_count, 1)
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=5), "<table></table>")
            self.assertEqual(service.client.total_tokens, 18)
            service.client.start_document()
            self.assertEqual(service.client.request_count, 0)
            self.assertEqual(service.client.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
