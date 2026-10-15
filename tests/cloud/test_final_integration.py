"""Offline regressions for the final source/client integration review."""

import base64
import json
import tempfile
import unittest
import zlib
from io import BytesIO
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from pydantic import ValidationError

from marker.cloud.client import CloudClient, schema_error_reason
from marker.cloud.content import resource_name
from marker.cloud.source import PdfSource
from marker.cloud.tables import (
    LogicalTable,
    TablePart,
    detect_tables,
    group_tables,
    recognize_table,
)
from marker.converters.ollama import OllamaPdfConverter, _layout_schema
from test_client import OfflineCloud, FormulaSchema
from test_content import FIXTURES, math_font_pdf, offline_client, table_echo
from test_launcher import load_launcher


def simple_pdf(path, sentence=None, raster=False, inline_raster=False, overlay=False):
    pixels = Image.new("RGB", (2048, 1024), "white")
    draw = ImageDraw.Draw(pixels)
    # Raster equation with high-density glyphs and operators.
    draw.text((100, 300), "x = 0", fill="black", font_size=180)
    if inline_raster:
        pixels = Image.new("RGB", (96, 96), "white")
        draw = ImageDraw.Draw(pixels)
        draw.line([(20, 20), (70, 44), (20, 68)], fill="black", width=7)
        draw.line([(20, 80), (70, 80)], fill="black", width=7)
    stream = (
        b"q 160 0 0 80 10 10 cm /Im Do Q"
        if raster
        else b"BT /F1 10 Tf 10 50 Td (" + sentence.encode() + b") Tj ET"
    )
    if inline_raster:
        stream = (
            b"BT /F1 10 Tf 10 50 Td (If) Tj 15 0 Td (x) Tj ET q 12 0 0 12 35 45 cm /Im Do Q BT /F1 10 Tf 50 50 Td ("
            + (sentence or "0, then accepted").encode()
            + b") Tj ET"
        )
    if overlay:
        stream += b" BT /F1 10 Tf 30 40 Td (x = 0) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] /Resources << /Font << /F1 4 0 R >> /XObject << /Im 6 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    data = zlib.compress(pixels.tobytes())
    objects.append(
        f"<< /Type /XObject /Subtype /Image /Width {pixels.width} /Height {pixels.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(data)} >>\nstream\n".encode()
        + data
        + b"\nendstream"
    )
    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(result)
    result.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        result.extend(f"{offset:010} 00000 n \n".encode())
    result.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )
    Path(path).write_bytes(result)


class FinalIntegrationTests(unittest.TestCase):
    def test_launcher_overrides_reach_effective_client_with_cloud_precedence(self):
        from marker.config.parser import ConfigParser

        launcher = load_launcher()
        args = launcher.parse_arguments(
            ["--model", "review-custom", "--base-url", "https://offline.invalid/v1"]
        )
        command = launcher.conversion_arguments(args, "python", "source.pdf", "output")
        options = {
            command[i][2:]: command[i + 1]
            for i in range(len(command) - 1)
            if command[i]
            in (
                "--openai_model",
                "--openai_base_url",
                "--cloud_model",
                "--cloud_base_url",
            )
        }
        config = ConfigParser(options).generate_config_dict()
        converter = OllamaPdfConverter(config)
        self.assertEqual(converter.client.cloud_model, "review-custom")
        self.assertEqual(converter.client.cloud_base_url, "https://offline.invalid/v1")
        converter = OllamaPdfConverter(
            config
            | {
                "cloud_model": "explicit-cloud",
                "cloud_base_url": "https://cloud.invalid/v1",
            }
        )
        self.assertEqual(converter.client.cloud_model, "explicit-cloud")
        self.assertEqual(converter.client.cloud_base_url, "https://cloud.invalid/v1")
        converter = OllamaPdfConverter(
            config
            | {
                "OllamaPdfConverter_cloud_model": "class-cloud",
                "OllamaPdfConverter_cloud_base_url": "https://class.invalid/v1",
            }
        )
        self.assertEqual(converter.client.cloud_model, "class-cloud")
        self.assertEqual(converter.client.cloud_base_url, "https://class.invalid/v1")

    def test_sentence_preserves_prose_and_two_separate_inline_expressions(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "sentence.pdf"
            simple_pdf(filename, "If x = 0, then y = 1, accepted")
            client, observed = offline_client(
                lambda s, n: {
                    "kind": "math",
                    "resolved": True,
                    "latex": "x = 0" if n == 1 else "y = 1",
                }
            )
            output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                filename
            )
            self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
            self.assertEqual(output.markdown, "If $x = 0$, then $y = 1$, accepted")
            self.assertEqual(
                [[a["text"] for a in s["atoms"]] for s in observed],
                [["x", "=", "0,"], ["y", "=", "1,"]],
            )

    def test_attached_delimiters_keep_single_letter_prose_and_expression_owners(self):
        for separator in (",", ";", ":"):
            for sentence, latex, expected, owners in (
                (
                    f"When x = 0{separator} I agree",
                    ["x = 0"],
                    f"When $x = 0${separator} I agree",
                    [["x", "=", f"0{separator}"]],
                ),
                (
                    f"If x = 0{separator} y = 1{separator} accepted",
                    ["x = 0", "y = 1"],
                    f"If $x = 0${separator} $y = 1${separator} accepted",
                    [["x", "=", f"0{separator}"], ["y", "=", f"1{separator}"]],
                ),
            ):
                with (
                    self.subTest(sentence=sentence),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    filename = Path(directory) / "punctuation.pdf"
                    simple_pdf(filename, sentence)
                    client, observed = offline_client(
                        lambda s, n: {
                            "kind": "math",
                            "resolved": True,
                            "latex": latex[n - 1],
                        }
                    )
                    output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                        filename
                    )
                    self.assertEqual(
                        output.metadata["cloud_conversion"]["status"], "complete"
                    )
                    self.assertEqual(output.markdown, expected)
                    self.assertEqual(
                        [[a["text"] for a in s["atoms"]] for s in observed], owners
                    )
                    self.assertEqual(client.request_count, len(owners))

    def test_bracketed_expression_owns_internal_separators_and_nested_operands(self):
        from copy import deepcopy
        from marker.cloud.content import group_expression_regions
        from marker.cloud.regions import analyze_page

        cases = [
            (
                "For x = [ 0, 1 ] accepted",
                "x = [0,1]",
                ["x", "=", "[", "0,", "1", "]"],
            ),
            (
                "For x = [ ( 0, 1 ) ; { 2 : 3 } ] accepted",
                r"x = [(0,1); \{2:3\}]",
                ["x", "=", "[", "(", "0,", "1", ")", ";", "{", "2", ":", "3", "}", "]"],
            ),
        ]
        for sentence, latex, owner in cases:
            with (
                self.subTest(sentence=sentence),
                tempfile.TemporaryDirectory() as directory,
            ):
                filename = Path(directory) / "bracketed-expression.pdf"
                simple_pdf(filename, sentence)
                with PdfSource(filename) as source:
                    regions = analyze_page(source, 0)
                    original = deepcopy([r.atoms for r in regions])
                    grouped = group_expression_regions(regions)
                    self.assertEqual([r.atoms for r in regions], original)
                    self.assertEqual(
                        [
                            [a["text"] for a in r.atoms]
                            for r in grouped
                            if r.kind == "math"
                        ],
                        [owner],
                    )
                client, observed = offline_client(
                    lambda s, n: {"kind": "math", "resolved": True, "latex": latex}
                )
                output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                    filename
                )
                self.assertEqual(
                    output.metadata["cloud_conversion"]["status"], "complete"
                )
                self.assertEqual(output.markdown, f"For ${latex}$ accepted")
                self.assertEqual(
                    [[a["text"] for a in s["atoms"]] for s in observed], [owner]
                )
                self.assertEqual(client.request_count, 1)

    def test_closing_bracket_separator_ends_owner_before_prose_or_next_expression(self):
        cases = [
            (
                "When x = [ 0, 1 ], I agree",
                ["x = [0,1]"],
                "When $x = [0,1]$, I agree",
                [["x", "=", "[", "0,", "1", "],"]],
            ),
            (
                "When x = ( 0; 1 ): I agree",
                ["x = (0;1)"],
                "When $x = (0;1)$: I agree",
                [["x", "=", "(", "0;", "1", "):"]],
            ),
            (
                "When x = [0,5; 1], y = {2: 3}; I agree",
                ["x = [0,5;1]", r"y = \{2:3\}"],
                r"When $x = [0,5;1]$, $y = \{2:3\}$; I agree",
                [["x", "=", "[0,5;", "1],"], ["y", "=", "{2:", "3};"]],
            ),
        ]
        for sentence, latex, expected, owners in cases:
            with (
                self.subTest(sentence=sentence),
                tempfile.TemporaryDirectory() as directory,
            ):
                filename = Path(directory) / "closed-bracket.pdf"
                simple_pdf(filename, sentence)
                client, observed = offline_client(
                    lambda s, n: {
                        "kind": "math",
                        "resolved": True,
                        "latex": latex[n - 1],
                    }
                )
                output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                    filename
                )
                self.assertEqual(
                    output.metadata["cloud_conversion"]["status"], "complete"
                )
                self.assertEqual(output.markdown, expected)
                self.assertEqual(
                    [[a["text"] for a in s["atoms"]] for s in observed], owners
                )
                self.assertEqual(client.request_count, len(owners))

    def test_decimal_commas_remain_inside_math_and_sentence_punctuation_occurs_once(
        self,
    ):
        for separator in (",", ";", ":"):
            for recognized_separator in ("", separator):
                with (
                    self.subTest(separator=separator, recognized=recognized_separator),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    filename = Path(directory) / "decimal-comma.pdf"
                    simple_pdf(filename, f"When x = 0,5{separator} I agree")
                    latex = "x = 0,5" + recognized_separator
                    client, observed = offline_client(
                        lambda s, n: {"kind": "math", "resolved": True, "latex": latex}
                    )
                    output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                        filename
                    )
                    expected = (
                        f"When ${latex}$"
                        + ("" if recognized_separator else separator)
                        + " I agree"
                    )
                    self.assertEqual(
                        output.metadata["cloud_conversion"]["status"], "complete"
                    )
                    self.assertEqual(output.markdown, expected)
                    self.assertEqual(
                        [a["text"] for a in observed[0]["atoms"]],
                        ["x", "=", f"0,5{separator}"],
                    )
                    self.assertEqual(client.request_count, 1)

    def test_raster_operator_attached_delimiter_keeps_single_letter_prose(self):
        for separator in (",", ";", ":"):
            with (
                self.subTest(separator=separator),
                tempfile.TemporaryDirectory() as directory,
            ):
                filename = Path(directory) / "raster-punctuation.pdf"
                simple_pdf(
                    filename, sentence=f"0{separator} I agree", inline_raster=True
                )
                client, observed = offline_client(
                    lambda s, n: {
                        "kind": "math",
                        "resolved": True,
                        "latex": r"x \geq 0",
                    }
                )
                output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                    filename
                )
                self.assertEqual(
                    output.metadata["cloud_conversion"]["status"], "complete"
                )
                self.assertEqual(
                    output.markdown, r"If $x \geq 0$" + separator + " I agree"
                )
                self.assertEqual(
                    [a["text"] for a in observed[0]["atoms"] if a["kind"] == "text"],
                    ["x", f"0{separator}"],
                )
                self.assertEqual(
                    [a["kind"] for a in observed[0]["atoms"]], ["text", "image", "text"]
                )

    def test_math_font_attached_delimiter_keeps_single_letter_prose(self):
        for separator in (",", ";", ":"):
            with (
                self.subTest(separator=separator),
                tempfile.TemporaryDirectory() as directory,
            ):
                filename = Path(directory) / "font-punctuation.pdf"
                math_font_pdf(filename, separator=separator, suffix="I agree")
                client, observed = offline_client(
                    lambda s, n: {
                        "kind": "math",
                        "resolved": True,
                        "latex": r"xy \geq 0",
                    }
                )
                output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                    filename
                )
                self.assertEqual(
                    output.metadata["cloud_conversion"]["status"], "complete"
                )
                self.assertEqual(
                    output.markdown, r"For $xy \geq 0$" + separator + " I agree"
                )
                self.assertEqual(
                    [a["text"] for a in observed[0]["atoms"]],
                    ["xy", "≥", f"0{separator}"],
                )

    def test_sentence_math_font_operand_keeps_prose_and_inline_order(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "font.pdf"
            math_font_pdf(filename)
            client, observed = offline_client(
                lambda s, n: {"kind": "math", "resolved": True, "latex": r"xy \geq 0"}
            )
            output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                filename
            )
            self.assertEqual(output.markdown, r"For $xy \geq 0$ where")
            self.assertEqual(
                [a["text"] for a in observed[0]["atoms"]], ["xy", "≥", "0"]
            )

    def test_sentence_raster_operator_keeps_whole_formula_and_source_prose(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "raster-inline.pdf"
            simple_pdf(filename, sentence="", inline_raster=True)
            client, observed = offline_client(
                lambda s, n: {"kind": "math", "resolved": True, "latex": r"x \geq 0"}
            )
            output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                filename
            )
            self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
            self.assertEqual(output.markdown, r"If $x \geq 0$, then accepted")
            self.assertEqual(
                [a["kind"] for a in observed[0]["atoms"]], ["text", "image", "text"]
            )
            self.assertEqual(
                [a["text"] for a in observed[0]["atoms"] if a["kind"] == "text"],
                ["x", "0,"],
            )

    def test_server_context_overflow_shrinks_actual_pdf_payload_with_counted_attempts(
        self,
    ):
        from marker.cloud.client import CloudClient

        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            image = source.crop(0, (20, 20, 180, 100))
            overflow = httpx.Response(
                400,
                json={
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "SECRET provider text",
                    }
                },
            )
            with OfflineCloud([overflow, '{"latex":"x"}']) as offline:
                cloud = CloudClient(client=offline.client)
                self.assertEqual(
                    cloud.request("Read formula", [image], FormulaSchema),
                    {"latex": "x"},
                )
                sizes = []
                for body in offline.requests:
                    uri = body["messages"][0]["content"][0]["image_url"]["url"]
                    with Image.open(
                        BytesIO(base64.b64decode(uri.split(",")[1]))
                    ) as pixels:
                        sizes.append(pixels.size)
                self.assertLess(max(sizes[1]), max(sizes[0]))
                self.assertEqual(cloud.request_count, 2)
            with OfflineCloud([overflow] * 3) as offline:
                cloud = CloudClient(client=offline.client)
                from marker.cloud.context import RecognitionFailure

                with self.assertRaises(RecognitionFailure) as raised:
                    cloud.request("Read formula", [image], FormulaSchema)
                self.assertEqual(raised.exception.attempts, 3)
                self.assertEqual(cloud.request_count, 3)
                self.assertNotIn("SECRET", str(raised.exception))

    def test_context_overflow_without_reducible_images_stops_without_duplicate_retry(
        self,
    ):
        from marker.cloud.context import RecognitionFailure

        overflow = httpx.Response(
            400,
            json={"error": {"code": "context_length_exceeded", "message": "PRIVATE"}},
        )
        with OfflineCloud([overflow, '{"latex":"x"}']) as offline:
            cloud = CloudClient(client=offline.client)
            with self.assertRaises(RecognitionFailure) as raised:
                cloud.request("Read source", [], FormulaSchema)
            self.assertEqual(raised.exception.attempts, 1)
            self.assertEqual(cloud.request_count, 1)
            self.assertIn("context limit", raised.exception.reason)
            self.assertNotIn("PRIVATE", str(raised.exception))

    def test_text_dominant_table_context_reflows_owned_lines_inside_attempt_limit(self):
        class ServerWindowLimit(OfflineCloud):
            def respond(self, request):
                body = json.loads(request.content)
                self.requests.append(body)
                prompt = body["messages"][0]["content"][-1]["text"]
                source = json.JSONDecoder().raw_decode(prompt.split("SOURCE:\n", 1)[1])[
                    0
                ]
                lines = sum(max(1, len(c["lines"])) for c in source["cells"])
                if lines > 1:
                    return httpx.Response(
                        400,
                        json={
                            "error": {
                                "code": "context_length_exceeded",
                                "message": "PRIVATE",
                            }
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "id": "offline",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "offline",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(table_echo(source, 1)),
                                },
                            }
                        ],
                    },
                )

        with (
            PdfSource(FIXTURES / "unrelated_tables.pdf") as source,
            ServerWindowLimit([]) as offline,
        ):
            table = group_tables(detect_tables(source), None)[0]
            cloud = CloudClient(client=offline.client)
            result = recognize_table(source, table, cloud)
            self.assertTrue(result.complete, result.reasons)
            prompts = [
                r["messages"][0]["content"][-1]["text"] for r in offline.requests
            ]
            payloads = [
                json.JSONDecoder().raw_decode(p.split("SOURCE:\n", 1)[1])[0]
                for p in prompts
            ]
            counts = [
                sum(max(1, len(c["lines"])) for c in p["cells"]) for p in payloads
            ]
            self.assertGreater(counts[0], counts[1])
            self.assertLess(len(prompts[1]), len(prompts[0]))
            self.assertEqual(cloud.request_count, len(offline.requests))
            # Every successful owner has one line. Coverage must include all cells
            # once despite the rejected parents and queued disjoint remainders.
            accepted_ids = [
                p["cells"][0]["id"] for p, count in zip(payloads, counts) if count == 1
            ]
            self.assertEqual(len(accepted_ids), len(set(accepted_ids)))
            self.assertEqual(len(accepted_ids), len(table.grid["cells"]))
            limited = CloudClient({"cloud_max_attempts": 1}, client=offline.client)
            failed = recognize_table(source, table, limited)
            self.assertFalse(failed.complete)
            self.assertEqual(limited.request_count, 1)

    def test_reflow_and_semantic_repairs_share_original_three_attempt_cap(self):
        overflow = httpx.Response(
            400,
            json={"error": {"code": "context_length_exceeded", "message": "PRIVATE"}},
        )
        for responses in ([overflow] * 3, [overflow, "{}", "{}"]):
            with (
                self.subTest(responses=len(responses)),
                PdfSource(FIXTURES / "unrelated_tables.pdf") as source,
                OfflineCloud(responses) as offline,
            ):
                table = group_tables(detect_tables(source), None)[0]
                cloud = CloudClient(client=offline.client)
                result = recognize_table(source, table, cloud)
                self.assertFalse(result.complete)
                self.assertEqual(cloud.request_count, 3)
                self.assertEqual(len(offline.requests), 3)
                self.assertNotIn("PRIVATE", str(result.reasons))
                self.assertIn(
                    resource_name(
                        "unresolved-table", table.parts[0].page, table.parts[0].bbox
                    ),
                    result.resources,
                )

    def test_semantic_layout_errors_have_distinct_safe_repair_constraints(self):
        schema = _layout_schema((0, 0, 100, 100))
        cases = [
            (
                {"complete": False, "regions": []},
                "Required visible layout remains unresolved",
            ),
            (
                {
                    "complete": True,
                    "regions": [
                        {"kind": "text", "bbox": [0.0, 0.0, 101.0, 50.0], "cells": []}
                    ],
                },
                "Layout region exceeds assigned source bounds",
            ),
            (
                {
                    "complete": True,
                    "regions": [
                        {"kind": "table", "bbox": [0.0, 0.0, 100.0, 50.0], "cells": []}
                    ],
                },
                "Table requires source cells",
            ),
        ]
        for response, reason in cases:
            with self.subTest(reason=reason):
                try:
                    schema.model_validate(response)
                except ValidationError as error:
                    self.assertIn(reason, schema_error_reason(error, schema))
                else:
                    self.fail("invalid layout accepted")

    def test_partial_long_and_continued_tables_keep_every_physical_part(self):
        for filename in ("rows_125.pdf", "three_page_continuation.pdf"):
            with (
                self.subTest(filename=filename),
                PdfSource(FIXTURES / filename) as source,
            ):
                table = group_tables(detect_tables(source), None)[0]
                client, _ = offline_client(table_echo, {"cloud_max_requests": 1})
                result = recognize_table(source, table, client)
                self.assertFalse(result.complete)
                for part in table.parts:
                    name = resource_name("unresolved-table", part.page, part.bbox)
                    self.assertIn(name, result.resources)
                    expected = source.resource(part.page, part.bbox)
                    self.assertEqual(result.resources[name].size, expected.size)
                    self.assertEqual(
                        result.resources[name].tobytes(), expected.tobytes()
                    )
                    self.assertIn(name, result.html)

    def test_raster_table_formula_audit_and_fallback_keep_native_pixels(self):
        for overlay in (False, True):
            with (
                self.subTest(overlay=overlay),
                tempfile.TemporaryDirectory() as directory,
            ):
                filename = Path(directory) / "raster-equation.pdf"
                simple_pdf(filename, raster=True, overlay=overlay)
                with PdfSource(filename) as source:
                    image = source.page_images(0)[0]
                    bbox = image.bbox
                    atom = {
                        "kind": "unknown" if overlay else "image",
                        "content_kind": "math",
                        "bbox": bbox,
                        "source_page": 0,
                    }
                    grid = {
                        "page": 0,
                        "bbox": bbox,
                        "trusted": False,
                        "cells": [{"bbox": bbox, "source_page": 0, "atoms": [atom]}],
                    }
                    table = LogicalTable(
                        [TablePart(0, bbox, grid, source.page_size(0), source=source)],
                        grid,
                    )
                    client, _ = offline_client(table_echo)
                    result = recognize_table(source, table, client)
                    self.assertTrue(result.complete, result.reasons)
                    audit = next(
                        v
                        for k, v in result.resources.items()
                        if k.startswith("formula")
                    )
                    expected = source.resource(0, bbox)
                    self.assertEqual(audit.size, (2048, 1024))
                    self.assertEqual(audit.tobytes(), expected.tobytes())
                    client, _ = offline_client(
                        lambda s, n: {}, {"cloud_max_attempts": 1}
                    )
                    failed = recognize_table(source, table, client)
                    fallback = failed.resources[
                        resource_name("unresolved-table", 0, bbox)
                    ]
                    self.assertEqual(fallback.tobytes(), expected.tobytes())

    def test_automatic_ruled_table_keeps_math_font_and_repairs_partial_operand(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "ruled-font.pdf"
            math_font_pdf(filename, ruled=True)
            with PdfSource(filename) as source:
                parts = detect_tables(source)
                self.assertEqual(len(parts), 1)
                table = group_tables(parts, None)[0]
                xy = next(
                    a
                    for c in table.grid["cells"]
                    for a in c["atoms"]
                    if a.get("text") == "xy"
                )
                self.assertEqual(xy.get("fontname"), "CMMI10")

                def respond(s, n):
                    response = table_echo(s, n)
                    for original, cell in zip(s["cells"], response["cells"]):
                        for original_line, line in zip(
                            original["lines"], cell["lines"]
                        ):
                            atoms = original_line["atoms"]
                            groups = original_line["expression_groups"]
                            grouped = {i for g in groups for i in g}
                            line["parts"] = [
                                {
                                    "kind": "text",
                                    "text": a["text"],
                                    "atom_ids": [a["id"]],
                                }
                                for a in atoms
                                if a["id"] not in grouped
                            ]
                            for group in groups:
                                owner = {
                                    "kind": "math",
                                    "latex": r"xy \geq 0",
                                    "atom_ids": group,
                                }
                                if n == 1:
                                    operand = next(
                                        a for a in atoms if a["text"] == "xy"
                                    )
                                    line["parts"].append(
                                        {
                                            "kind": "text",
                                            "text": "xy",
                                            "atom_ids": [operand["id"]],
                                        }
                                    )
                                    owner["atom_ids"] = [
                                        i for i in group if i != operand["id"]
                                    ]
                                    owner["latex"] = r"\geq 0"
                                line["parts"].append(owner)
                    return response

                client, _ = offline_client(respond)
                result = recognize_table(source, table, client)
                self.assertTrue(result.complete, result.reasons)
                self.assertEqual(client.request_count, 2)
                self.assertEqual(
                    BeautifulSoup(result.html, "html.parser").find("annotation").text,
                    r"xy \geq 0",
                )
                audit = next(
                    v for k, v in result.resources.items() if k.startswith("formula")
                )
                self.assertGreater(audit.width, 240)
