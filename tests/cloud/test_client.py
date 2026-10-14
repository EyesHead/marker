"""Offline response reliability and request-budget contracts."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx
import openai
from PIL import Image
from pydantic import BaseModel, field_validator

from marker.cloud import CloudClient, RecognitionFailure
from marker.cloud.context import estimate_request_tokens, fit_request

from marker.services.openai import OpenAIService


class FormulaSchema(BaseModel):
    latex: str


class EntrySchema(BaseModel):
    label: str
    count: int


class RestrictedFormulaSchema(BaseModel):
    latex: str

    @field_validator('latex')
    @classmethod
    def reject_secret(cls, value):
        if value != 'x':
            raise ValueError('rejected input: ' + value)
        return value


class OfflineCloud:
    """Replace only HTTP transport; retain the SDK and production parser."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.client = openai.OpenAI(
            api_key='offline-placeholder', base_url='https://ollama.com/v1', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self.respond)))

    def respond(self, request):
        self.requests.append(json.loads(request.content))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, httpx.Response):
            return response
        text, finish_reason = response if isinstance(response, tuple) else (response, 'stop')
        return httpx.Response(200, json={
            'id': 'offline', 'object': 'chat.completion', 'created': 0,
            'model': 'gemma4:31b',
            'choices': [{'index': 0, 'finish_reason': finish_reason, 'message': {
                'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()


class CloudClientTests(unittest.TestCase):
    def cloud(self, offline, **config):
        return CloudClient(config, client=offline.client)

    def test_schema_failure_is_repaired_with_field_specific_errors(self):
        with OfflineCloud(['{"label":"one","count":"many"}', '{"label":"one","count":1}']) as offline:
            cloud = self.cloud(offline)
            self.assertEqual(cloud.request('Read the entry.', [], EntrySchema), {'label': 'one', 'count': 1})
            repair = offline.requests[1]['messages'][0]['content'][-1]['text']
            self.assertIn('count', repair)
            self.assertIn('int_type', repair)
            self.assertEqual(cloud.request_count, 2)
            self.assertEqual(cloud.total_tokens, 36)
            self.assertEqual(cloud.last_usage['total_tokens'], 18)

    def test_missing_schema_field_is_named_in_the_repair(self):
        with OfflineCloud(['{"label":"one"}', '{"label":"one","count":1}']) as offline:
            cloud = self.cloud(offline)
            self.assertEqual(cloud.request('Read the entry.', [], EntrySchema)['count'], 1)
            repair = offline.requests[1]['messages'][0]['content'][-1]['text']
            self.assertIn('count', repair)
            self.assertIn('missing', repair)

    def test_valid_json_preserves_latex_backslashes(self):
        latex = r'\varphi = \frac{a_1}{b^2} + \theta'
        with OfflineCloud([json.dumps({'latex': latex})]) as offline:
            cloud = self.cloud(offline)
            self.assertEqual(cloud.request('Read the formula.', [], FormulaSchema), {'latex': latex})
            self.assertEqual(cloud.request_count, 1)

    def test_invalid_json_backslash_requires_a_new_response(self):
        latex = r'\varphi = x'
        with OfflineCloud([r'{"latex":"\varphi = x"}', json.dumps({'latex': latex})]) as offline:
            cloud = self.cloud(offline)
            self.assertEqual(cloud.request('Read the formula.', [], FormulaSchema)['latex'], latex)
            self.assertEqual(cloud.request_count, 2)
            repair = offline.requests[1]['messages'][0]['content'][-1]['text']
            self.assertIn('JSON', repair)
            self.assertIn('backslash', repair.lower())

    def test_latex_control_escape_is_repaired_instead_of_silently_corrupted(self):
        for latex in (r'\frac{a}{b}', r'\theta', r'\rho', r'\nu', r'\beta', r'\nabla'):
            with self.subTest(latex=latex), OfflineCloud([
                '{"latex":"' + latex + '"}', json.dumps({'latex': latex})
            ]) as offline:
                cloud = self.cloud(offline)
                self.assertEqual(cloud.request('Read the formula.', [], FormulaSchema)['latex'], latex)
                self.assertEqual(cloud.request_count, 2)

    def test_html_math_control_escapes_require_repair(self):
        label = r'<math>\theta + \rho</math>'
        with OfflineCloud([
            r'{"label":"<math>\theta + \rho</math>","count":1}',
            json.dumps({'label': label, 'count': 1})
        ]) as offline:
            cloud = self.cloud(offline)
            self.assertEqual(cloud.request('Read.', [], EntrySchema)['label'], label)
            self.assertEqual(cloud.request_count, 2)

    def test_ordinary_ocr_line_breaks_and_latex_row_separators_survive(self):
        label = '<p>first line\nsecond line</p>'
        latex = r'\begin{matrix}a & b \\ c & d\end{matrix}'
        with OfflineCloud([json.dumps({'label': label, 'count': 1}), json.dumps({'latex': latex})]) as offline:
            cloud = self.cloud(offline)
            self.assertEqual(cloud.request('Read.', [], EntrySchema)['label'], label)
            self.assertEqual(cloud.request('Read.', [], FormulaSchema)['latex'], latex)

    def test_complete_json_code_fences_are_accepted(self):
        for fence in ('```json\n', '```\n', '```JSON\r\n'):
            with self.subTest(fence=fence), OfflineCloud([fence + '{"latex":"x"}\n```']) as offline:
                self.assertEqual(self.cloud(offline).request('Read.', [], FormulaSchema), {'latex': 'x'})

    def test_truncated_response_is_repaired_even_if_its_json_parses(self):
        with OfflineCloud([('{"latex":"partial"}', 'length'), '{"latex":"complete"}']) as offline:
            cloud = self.cloud(offline)
            self.assertEqual(cloud.request('Read.', [], FormulaSchema), {'latex': 'complete'})
            self.assertEqual(cloud.request_count, 2)

    def test_attempt_limit_returns_clear_required_recognition_failure(self):
        with OfflineCloud(['{}', '{}', '{}']) as offline:
            cloud = self.cloud(offline)
            with self.assertRaises(RecognitionFailure) as raised:
                cloud.request('Read.', [], FormulaSchema)
            self.assertIn('latex', raised.exception.reason)
            self.assertEqual(raised.exception.attempts, 3)
            self.assertEqual(cloud.request_count, 3)

    def test_document_cap_counts_repairs_and_subsequent_requests(self):
        with OfflineCloud(['{}', '{"latex":"x"}']) as offline:
            cloud = self.cloud(offline, cloud_max_requests=1)
            with self.assertRaises(RecognitionFailure) as raised:
                cloud.request('Read.', [], FormulaSchema)
            self.assertIn('budget', raised.exception.reason.lower())
            self.assertEqual(raised.exception.attempts, 1)
            with self.assertRaises(RecognitionFailure):
                cloud.request('Read again.', [], FormulaSchema)
            self.assertEqual(len(offline.requests), 1)
            self.assertEqual(cloud.request_count, 1)

    def test_unlimited_document_cap_still_bounds_each_request(self):
        with OfflineCloud(['{}', '{}', '{}', '{"latex":"x"}']) as offline:
            cloud = self.cloud(offline, cloud_max_requests=0)
            with self.assertRaises(RecognitionFailure):
                cloud.request('Read.', [], FormulaSchema)
            self.assertEqual(cloud.request('Read again.', [], FormulaSchema), {'latex': 'x'})
            self.assertEqual(cloud.request_count, 4)

    def test_network_retries_count_towards_the_same_document_cap(self):
        failure = httpx.Response(503, json={'error': {'message': 'offline unavailable', 'type': 'server_error'}})
        with OfflineCloud([failure, '{"latex":"x"}']) as offline:
            cloud = self.cloud(offline, cloud_max_requests=1)
            with self.assertRaises(RecognitionFailure):
                cloud.request('Read.', [], FormulaSchema)
            self.assertEqual(len(offline.requests), 1)
            self.assertEqual(cloud.request_count, 1)

    def test_injected_sdk_retries_cannot_bypass_the_document_cap(self):
        failure = httpx.Response(503, json={'error': {'message': 'offline unavailable', 'type': 'server_error'}})
        with OfflineCloud([failure, '{"latex":"x"}']) as offline:
            cloud = CloudClient({'cloud_max_requests': 1}, client=offline.client.with_options(max_retries=2))
            with self.assertRaises(RecognitionFailure):
                cloud.request('Read.', [], FormulaSchema)
            self.assertEqual(len(offline.requests), 1)
            self.assertEqual(cloud.request_count, 1)

    def test_concurrent_requests_cannot_bypass_the_document_cap(self):
        with OfflineCloud(['{"latex":"x"}'] * 4) as offline:
            cloud = self.cloud(offline, cloud_max_requests=2)
            def recognize(_):
                try:
                    return cloud.request('Read.', [], FormulaSchema)
                except RecognitionFailure:
                    return None
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(recognize, range(4)))
            self.assertEqual(sum(result is not None for result in results), 2)
            self.assertEqual(cloud.request_count, 2)
            self.assertEqual(len(offline.requests), 2)

    def test_provider_errors_do_not_echo_secret_values(self):
        secret = 'dummy-secret-must-stay-in-memory-789'
        failure = httpx.Response(400, json={'error': {'message': secret, 'type': 'invalid_request_error'}})
        with OfflineCloud([failure]) as offline:
            cloud = self.cloud(offline)
            with self.assertRaises(RecognitionFailure) as raised:
                cloud.request('Read.', [], FormulaSchema)
            self.assertNotIn(secret, str(raised.exception))
            self.assertNotIn(secret, raised.exception.reason)
            self.assertEqual(cloud.request_count, 1)

    def test_validator_input_values_do_not_enter_failure_or_repair_prompt(self):
        secret = 'dummy-secret-rejected-by-validator-123'
        with OfflineCloud([json.dumps({'latex': secret}), '{"latex":"x"}']) as offline:
            cloud = self.cloud(offline)
            self.assertEqual(cloud.request('Read.', [], RestrictedFormulaSchema), {'latex': 'x'})
            self.assertNotIn(secret, offline.requests[1]['messages'][0]['content'][-1]['text'])
        with OfflineCloud([json.dumps({'latex': secret})]) as offline:
            cloud = self.cloud(offline, cloud_max_attempts=1)
            with self.assertRaises(RecognitionFailure) as raised:
                cloud.request('Read.', [], RestrictedFormulaSchema)
            self.assertNotIn(secret, str(raised.exception))

    def test_oversized_prompt_fails_before_using_a_request(self):
        with OfflineCloud([]) as offline:
            cloud = self.cloud(offline, cloud_context_tokens=100, cloud_max_output_tokens=90)
            with self.assertRaises(RecognitionFailure):
                cloud.request('Long source text ' * 20, [], FormulaSchema)
            self.assertEqual(cloud.request_count, 0)
            self.assertEqual(offline.requests, [])

    def test_too_many_images_fail_without_dropping_required_sources(self):
        with OfflineCloud([]) as offline:
            cloud = self.cloud(offline)
            with self.assertRaises(RecognitionFailure):
                cloud.request('Read.', [Image.new('RGB', (16, 16))] * 4, FormulaSchema)
            self.assertEqual(cloud.request_count, 0)

    def test_fitted_request_copies_are_bounded_and_original_pixels_survive(self):
        original = Image.new('RGB', (4000, 2500), (31, 47, 59))
        images = fit_request('Read the formula.', [original], 8192, 2048)
        self.assertLessEqual(max(images[0].size), 1536)
        self.assertLessEqual(estimate_request_tokens('Read the formula.', images) + 2048, 8192)
        self.assertIsNot(images[0], original)
        self.assertEqual(original.size, (4000, 2500))
        self.assertEqual(original.getpixel((3999, 2499)), (31, 47, 59))

    def test_image_fitting_reduces_copies_to_fit_and_rejects_impossible_budget(self):
        images = [Image.new('RGB', (2000, 2000)) for _ in range(3)]
        fitted = fit_request('Read.', images, 2500, 500)
        self.assertEqual(len(fitted), 3)
        self.assertLessEqual(estimate_request_tokens('Read.', fitted) + 500, 2500)
        with self.assertRaises(RecognitionFailure):
            fit_request('x' * 100, images, 100, 100)
        with self.assertRaisesRegex(RecognitionFailure, 'images'):
            fit_request('Read.', images, 1600, 1000)

    def test_native_cmyk_source_is_encoded_as_a_request_copy(self):
        original = Image.new('CMYK', (32, 32), (1, 2, 3, 4))
        with OfflineCloud(['{"latex":"x"}']) as offline:
            self.assertEqual(self.cloud(offline).request('Read.', [original], FormulaSchema), {'latex': 'x'})
            self.assertEqual(original.mode, 'CMYK')
            self.assertEqual(original.getpixel((0, 0)), (1, 2, 3, 4))


class OpenAIServiceReliabilityTests(unittest.TestCase):
    def service(self):
        return OpenAIService({'openai_base_url': 'https://ollama.com/v1',
                              'openai_model': 'gemma4:31b', 'openai_api_key': 'offline-placeholder',
                              'retry_wait_time': 0})

    def test_existing_service_repairs_schema_errors(self):
        with OfflineCloud(['{}', '{"latex":"x"}']) as offline:
            service = self.service()
            with patch.object(service, 'get_client', return_value=offline.client), self.assertLogs('marker', level='WARNING'):
                self.assertEqual(service('Read.', None, None, FormulaSchema), {'latex': 'x'})
            self.assertEqual(len(offline.requests), 2)

    def test_existing_service_repairs_truncated_json(self):
        with OfflineCloud([('{"latex":"partial"}', 'length'), '{"latex":"complete"}']) as offline:
            service = self.service()
            with patch.object(service, 'get_client', return_value=offline.client), self.assertLogs('marker', level='WARNING'):
                self.assertEqual(service('Read.', None, None, FormulaSchema), {'latex': 'complete'})

    def test_existing_structured_output_service_repairs_sdk_truncation(self):
        with OfflineCloud([('{"latex":"partial"}', 'length'), '{"latex":"complete"}']) as offline:
            service = self.service()
            service.openai_base_url = 'https://api.openai.com/v1'
            with patch.object(service, 'get_client', return_value=offline.client), self.assertLogs('marker', level='WARNING'):
                self.assertEqual(service('Read.', None, None, FormulaSchema), {'latex': 'complete'})

    def test_existing_service_logs_safe_failure_details(self):
        secret = 'dummy-secret-must-not-enter-logs-456'
        failure = httpx.Response(400, json={'error': {'message': secret, 'type': 'invalid_request_error'}})
        with OfflineCloud([failure]) as offline:
            service = self.service()
            with patch.object(service, 'get_client', return_value=offline.client), self.assertLogs('marker', level='ERROR') as logs:
                self.assertEqual(service('Read.', None, None, FormulaSchema), {})
            self.assertNotIn(secret, '\n'.join(logs.output))


if __name__ == '__main__':
    unittest.main()
