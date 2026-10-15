"""Whole-object text responses use the same bounded cloud transport."""

import unittest

from PIL import Image

from marker.cloud.client import CloudClient, ResponseValidationError
from marker.cloud.context import RecognitionFailure
from test_client import OfflineCloud


def table(text):
    if not text.startswith("<table>") or not text.endswith("</table>"):
        raise ResponseValidationError("A complete HTML table is required")
    return text


class TextRequestTests(unittest.TestCase):
    def test_raw_html_does_not_receive_a_json_schema(self):
        html = "<table><tr><td>0,42R_{bun}</td></tr></table>"
        with OfflineCloud([html]) as offline:
            client = CloudClient(client=offline.client)
            self.assertEqual(client.request_text("Return one table.", [], table), html)
            payload = offline.requests[0]
            self.assertNotIn("schema", payload["messages"][0]["content"][-1]["text"])
            self.assertEqual(client.request_count, 1)
            self.assertEqual(client.total_tokens, 18)

    def test_invalid_and_truncated_html_use_the_original_attempt_cap(self):
        html = "<table><tr><td>value</td></tr></table>"
        with OfflineCloud(["not a table", (html, "length"), html]) as offline:
            client = CloudClient(client=offline.client)
            self.assertEqual(client.request_text("Return one table.", [], table), html)
            self.assertEqual(client.request_count, 3)
            self.assertIn(
                "complete", offline.requests[-1]["messages"][0]["content"][-1]["text"]
            )

    def test_json_and_html_share_one_document_request_cap(self):
        from test_client import FormulaSchema

        with OfflineCloud(['{"latex":"x"}', "<table></table>"]) as offline:
            client = CloudClient({"cloud_max_requests": 1}, client=offline.client)
            client.request("formula", [], FormulaSchema)
            with self.assertRaises(RecognitionFailure):
                client.request_text("table", [], table)
            self.assertEqual(len(offline.requests), 1)

    def test_whole_table_crops_stay_readable_and_originals_unchanged(self):
        html = "<table><tr><td>value</td></tr></table>"
        image = Image.new("RGB", (2200, 2800), "white")
        with OfflineCloud([html]) as offline:
            client = CloudClient({"cloud_context_tokens": 65536}, client=offline.client)
            client.request_text("table", [image], table, image_long_edge=3072)
            import base64
            from io import BytesIO

            url = offline.requests[0]["messages"][0]["content"][0]["image_url"]["url"]
            with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as sent:
                self.assertEqual(sent.size, (2200, 2800))
            self.assertEqual(image.size, (2200, 2800))

    def test_tight_budget_fails_before_sending_an_unreadable_thumbnail(self):
        image = Image.new("RGB", (2200, 2800), "white")
        with OfflineCloud([]) as offline:
            client = CloudClient({"cloud_context_tokens": 8192}, client=offline.client)
            with self.assertRaises(RecognitionFailure):
                client.request_text("table", [image], table, image_long_edge=3072)
            self.assertEqual(offline.requests, [])


if __name__ == "__main__":
    unittest.main()
