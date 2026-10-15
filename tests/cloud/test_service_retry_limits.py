"""Provider SDK retries and output limits respect Marker's request budget."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import BaseModel

from marker.services.azure_openai import AzureOpenAIService
from marker.services.claude import ClaudeService
from marker.services.gemini import GoogleGeminiService
from marker.services.openrouter import OpenRouterService


class OutputSchema(BaseModel):
    value: str


def sdk_completion():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"value": "ok"})))],
        usage=SimpleNamespace(total_tokens=18),
    )


class ServiceRetryAndOutputLimitTests(unittest.TestCase):
    def test_claude_disables_sdk_retries_without_changing_outer_service_retries(self):
        service = ClaudeService({"claude_api_key": "offline-placeholder"})
        with patch("marker.services.claude.anthropic.Anthropic") as factory:
            self.assertIs(service.get_client(), factory.return_value)
        self.assertEqual(factory.call_args.kwargs, {
            "api_key": "offline-placeholder", "max_retries": 0,
        })
        self.assertEqual(service.max_retries, 2)
    def test_azure_forwards_explicit_cap_and_preserves_unset_legacy_default(self):
        for cap in (None, 1024):
            with self.subTest(cap=cap):
                config = {
                    "azure_endpoint": "https://offline.example",
                    "azure_api_version": "offline-version",
                    "azure_api_key": "offline-placeholder", "deployment_name": "offline-model",
                }
                if cap is not None:
                    config["max_output_tokens"] = cap
                service = AzureOpenAIService(config)
                client = Mock()
                client.chat.completions.parse.return_value = sdk_completion()
                with patch.object(service, "get_client", return_value=client):
                    self.assertEqual(service("Read.", None, None, OutputSchema, max_retries=0), {"value": "ok"})
                kwargs = client.chat.completions.parse.call_args.kwargs
                if cap is None:
                    self.assertNotIn("max_completion_tokens", kwargs)
                    self.assertNotIn("max_tokens", kwargs)
                else:
                    self.assertEqual(kwargs["max_completion_tokens"], cap)
                self.assertEqual(client.chat.completions.parse.call_count, 1)

    def test_openrouter_forwards_explicit_cap_and_preserves_unset_legacy_default(self):
        for cap in (None, 1024):
            with self.subTest(cap=cap):
                config = {"openrouter_api_key": "offline-placeholder"}
                if cap is not None:
                    config["max_output_tokens"] = cap
                service = OpenRouterService(config)
                client = Mock()
                client.chat.completions.parse.return_value = sdk_completion()
                with patch.object(service, "get_client", return_value=client):
                    self.assertEqual(service("Read.", None, None, OutputSchema, max_retries=0), {"value": "ok"})
                kwargs = client.chat.completions.parse.call_args.kwargs
                if cap is None:
                    self.assertNotIn("max_tokens", kwargs)
                    self.assertNotIn("max_completion_tokens", kwargs)
                else:
                    self.assertEqual(kwargs["max_tokens"], cap)
                self.assertEqual(client.chat.completions.parse.call_count, 1)

    def test_gemini_already_forwards_cap_without_changing_unset_legacy_default(self):
        for cap in (None, 1024):
            with self.subTest(cap=cap):
                config = {"gemini_api_key": "offline-placeholder"}
                if cap is not None:
                    config["max_output_tokens"] = cap
                service = GoogleGeminiService(config)
                client = Mock()
                client.models.generate_content.return_value = SimpleNamespace(
                    candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
                        SimpleNamespace(text=json.dumps({"value": "ok"})),
                    ]))], usage_metadata=SimpleNamespace(total_token_count=18),
                )
                with patch.object(service, "get_google_client", return_value=client):
                    self.assertEqual(service("Read.", None, None, OutputSchema, max_retries=0), {"value": "ok"})
                kwargs = client.models.generate_content.call_args.kwargs
                if cap is None:
                    self.assertNotIn("max_output_tokens", kwargs["config"])
                else:
                    self.assertEqual(kwargs["config"]["max_output_tokens"], cap)
                self.assertEqual(client.models.generate_content.call_count, 1)

    def test_azure_disables_sdk_retries_and_keeps_endpoint_version_and_key(self):
        service = AzureOpenAIService({
            "azure_endpoint": "https://offline.example",
            "azure_api_version": "offline-version",
            "azure_api_key": "offline-placeholder", "deployment_name": "offline-model",
        })
        with patch("marker.services.azure_openai.AzureOpenAI") as factory:
            self.assertIs(service.get_client(), factory.return_value)
        self.assertEqual(factory.call_args.kwargs, {
            "api_version": "offline-version", "azure_endpoint": "https://offline.example",
            "api_key": "offline-placeholder", "max_retries": 0,
        })
        self.assertEqual(service.max_retries, 2)

    def test_openrouter_disables_sdk_retries_and_preserves_custom_endpoint(self):
        service = OpenRouterService({
            "openrouter_api_key": "offline-placeholder",
            "openrouter_base_url": "https://offline.example/v1",
        })
        with patch("marker.services.openrouter.openai.OpenAI") as factory:
            self.assertIs(service.get_client(), factory.return_value)
        self.assertEqual(factory.call_args.kwargs, {
            "api_key": "offline-placeholder", "base_url": "https://offline.example/v1",
            "max_retries": 0,
        })
        self.assertEqual(service.max_retries, 2)


if __name__ == "__main__":
    unittest.main()
