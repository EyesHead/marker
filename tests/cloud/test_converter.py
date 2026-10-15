"""Offline adaptive conversion against actual PDF fixtures and crop pixels."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from test_content import offline_client, table_echo

FIXTURES = Path(__file__).parent / "fixtures"


def text_with_vector_pdf(path):
    content = (
        b"BT /F1 10 Tf 20 50 Td (Visible underlined text) Tj ET 20 50 m 100 50 l S"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    data, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(data)
    data.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )
    Path(path).write_bytes(data)


def respond(source, attempt):
    if source.get("stage") == "layout":
        if source["bbox"] != [0.0, 0.0, 360.0, 400.0]:
            return {
                "complete": True,
                "regions": [{"kind": "figure", "bbox": source["bbox"], "cells": []}],
            }
        xs, ys = [30.0, 130.0, 230.0, 330.0], [60.0, 260 / 3, 340 / 3, 140.0]
        cells = []
        for row in range(3):
            for col in range(3):
                lines = (
                    [{"bbox": [33.0, 66.0, 90.0, 72.0], "kind": "text"}]
                    if row == col == 0
                    else []
                )
                cells.append(
                    {
                        "bbox": [xs[col], ys[row], xs[col + 1], ys[row + 1]],
                        "lines": lines,
                    }
                )
        return {
            "complete": True,
            "regions": [
                {"kind": "text", "bbox": [30.0, 28.0, 120.0, 36.0], "cells": []},
                {"kind": "table", "bbox": [30.0, 60.0, 330.0, 140.0], "cells": cells},
            ],
        }
    if "cells" in source:
        output = table_echo(source, attempt)
        for cell in output["cells"]:
            for line in cell["lines"]:
                for part in line["parts"]:
                    if part["kind"] == "text" and not part["text"]:
                        part["text"] = "Scanned baseline " + line["id"]
        return output
    if source.get("required_kind") == "math":
        return {"kind": "math", "resolved": True, "latex": r"x \geq 0"}
    if source.get("required_kind") in ("unknown", "image", "figure"):
        return {"kind": "figure", "resolved": True}
    return {"kind": "text", "resolved": True, "text": "Recovered scanned paragraph"}


def rasterized_table(path):
    from marker.cloud.content import math_required, source_lines
    from marker.cloud.regions import analyze_page
    from marker.cloud.source import PdfSource

    with PdfSource(str(FIXTURES / "content_lines.pdf")) as source:
        items = []
        for region in analyze_page(source, 0):
            item = {
                "kind": (
                    region.kind
                    if region.kind in ("text", "math", "table")
                    else "figure"
                ),
                "bbox": list(region.bbox),
                "cells": [],
            }
            if region.grid:
                for cell in region.grid["cells"]:
                    lines = []
                    for line in source_lines(cell["atoms"]):
                        lines.append(
                            {
                                "bbox": [
                                    min(a["bbox"][0] for a in line),
                                    min(a["bbox"][1] for a in line),
                                    max(a["bbox"][2] for a in line),
                                    max(a["bbox"][3] for a in line),
                                ],
                                "kind": "math" if math_required(line) else "text",
                            }
                        )
                    item["cells"].append({"bbox": list(cell["bbox"]), "lines": lines})
            items.append(item)
        source.crop(0, (0, 0, *source.page_size(0)), dpi=144).save(
            path, "PDF", resolution=144
        )
    return {"complete": True, "regions": items}


class AdaptiveConverterTests(unittest.TestCase):
    def convert(self, fixture, responder=respond, config=None):
        from marker.converters.ollama import OllamaPdfConverter

        client, observed = offline_client(responder, config)
        converter = OllamaPdfConverter(
            config=config, artifact_dict={"cloud_client": client}
        )
        return converter(str(FIXTURES / fixture)), observed

    def test_digital_multiple_tables_and_document_order(self):
        output, observed = self.convert("unrelated_tables.pdf")
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertEqual(output.markdown.count("<table"), 3)
        self.assertIn("Table 1", output.markdown)
        self.assertLess(
            output.markdown.index("Table 1"), output.markdown.index("<table")
        )
        self.assertEqual(output.metadata["cloud_conversion"]["requests"], len(observed))

    def test_scan_supplies_untrusted_cells_and_separate_baselines(self):
        output, observed = self.convert("scan.pdf")
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        tables = [item for item in observed if "cells" in item]
        self.assertTrue(tables)
        self.assertFalse(tables[0]["trusted"])
        self.assertEqual(
            sum(len(c["lines"]) for item in tables for c in item["cells"]), 1
        )
        self.assertEqual(sum(len(item["cells"]) for item in tables), 9)
        self.assertEqual(output.markdown.count("<table"), 1)
        self.assertIn("Recovered scanned paragraph", output.markdown)
        self.assertTrue(
            all(
                a["page"] == 0
                for c in tables[0]["cells"]
                for l in c["lines"]
                for a in l["atoms"]
            )
        )

    def test_rasterized_actual_merged_table_preserves_all_baselines(self):
        from marker.converters.ollama import OllamaPdfConverter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raster-table.pdf"
            layout = rasterized_table(path)
            client, observed = offline_client(
                lambda data, n: (
                    layout if data.get("stage") == "layout" else respond(data, n)
                )
            )
            output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                str(path)
            )
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        tables = [item for item in observed if "cells" in item]
        line_ids = {}
        for item in tables:
            for cell in item["cells"]:
                line_ids.setdefault(cell["id"], set()).update(
                    line["id"] for line in cell["lines"]
                )
        self.assertTrue(any(len(lines) == 3 for lines in line_ids.values()))
        self.assertEqual(output.markdown.count("<table"), 1)

    def test_three_page_continuation_is_one_logical_table_plus_independent(self):
        output, observed = self.convert("three_page_continuation.pdf")
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertEqual(output.markdown.count("<table"), 2)
        pages = {
            a["page"]
            for item in observed
            for cell in item.get("cells", [])
            for line in cell["lines"]
            for a in line["atoms"]
        }
        self.assertEqual(pages, {0, 1, 2})

    def test_mixed_actual_digital_and_scanned_pages(self):
        import pypdfium2 as pdfium
        from marker.converters.ollama import OllamaPdfConverter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.pdf"
            combined = pdfium.PdfDocument.new()
            for name in ("damaged_text.pdf", "scan.pdf"):
                with pdfium.PdfDocument(FIXTURES / name) as part:
                    combined.import_pages(part)
            combined.save(path)
            combined.close()
            client, observed = offline_client(respond)
            output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                str(path)
            )
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertLess(
            output.markdown.index("Trustworthy"), output.markdown.index("<table")
        )
        self.assertEqual(output.metadata["cloud_conversion"]["requests"], len(observed))

    def test_invalid_layout_geometry_is_repaired_with_same_counter(self):
        def invalid_then_valid(source, attempt):
            result = respond(source, attempt)
            if attempt == 1:
                result["regions"][0]["bbox"][0] = -1.0
            return result

        output, observed = self.convert("scan.pdf", responder=invalid_then_valid)
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertEqual(sum(item.get("stage") == "layout" for item in observed), 2)
        self.assertEqual(output.metadata["cloud_conversion"]["requests"], len(observed))

    def test_scanned_formula_cannot_be_classified_as_text_or_figure(self):
        from marker.converters.ollama import OllamaPdfConverter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rasterized-formulas.pdf"
            layout = rasterized_table(path)

            def formula_layout(source, attempt):
                if source.get("stage") == "layout":
                    return layout
                if "cells" in source:
                    output = respond(source, attempt)
                    for cell in output["cells"]:
                        for line in cell["lines"]:
                            for part in line["parts"]:
                                part["kind"] = "text"
                                part["text"] = "discarded formula"
                                part.pop("latex", None)
                    return output
                return respond(source, attempt)

            client, observed = offline_client(formula_layout)
            output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                str(path)
            )
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "incomplete")
        self.assertTrue(output.images)
        self.assertNotIn("discarded formula", output.markdown)
        self.assertTrue(
            any(
                line["math_required"]
                for item in observed
                for cell in item.get("cells", [])
                for line in cell["lines"]
            )
        )

    def test_damaged_text_recovers_only_failed_region(self):
        output, observed = self.convert("damaged_text.pdf")
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertIn("Trustworthy", output.markdown)
        self.assertIn("Recovered scanned paragraph", output.markdown)
        self.assertFalse(any("Trustworthy" in str(s) for s in observed))

    def test_vector_inside_trusted_text_rectangle_is_not_silently_owned(self):
        from marker.converters.ollama import OllamaPdfConverter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text-vector.pdf"
            text_with_vector_pdf(path)
            client, observed = offline_client(respond)
            output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                str(path)
            )
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertIn("Visible underlined text", output.markdown)
        self.assertTrue(
            any(
                atom["kind"] == "vector"
                for item in observed
                for atom in item.get("atoms", [])
            )
        )
        self.assertTrue(output.images)

    def test_native_raster_and_unowned_strokes_are_retained(self):
        output, observed = self.convert("native_figure.pdf")
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertIn((2048, 1024), [im.size for im in output.images.values()])
        self.assertTrue(any(item.get("stage") == "layout" for item in observed))
        vectors = [
            a for s in observed for a in s.get("atoms", []) if a["kind"] == "vector"
        ]
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(a["bbox"][1] == a["bbox"][3] for a in vectors))

    def test_formula_and_table_content_share_one_counter(self):
        output, observed = self.convert("split_formula.pdf")
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertIn(r"x \geq 0", output.markdown)
        self.assertIn("<table", output.markdown)
        self.assertEqual(output.metadata["cloud_conversion"]["requests"], len(observed))

    def test_exhausted_budget_keeps_partial_text_and_source(self):
        output, observed = self.convert(
            "damaged_text.pdf",
            config={"cloud_max_requests": 1},
            responder=lambda s, n: {"bad": "response"},
        )
        metadata = output.metadata["cloud_conversion"]
        self.assertEqual(metadata["status"], "incomplete")
        self.assertEqual(metadata["requests"], 1)
        self.assertTrue(metadata["reasons"])
        self.assertIn("Trustworthy", output.markdown)
        self.assertTrue(output.images)

    def test_disable_ocr_keeps_required_formula_transcription(self):
        output, observed = self.convert(
            "split_formula.pdf", config={"disable_ocr": True}
        )
        self.assertIn(r"x \geq 0", output.markdown)
        self.assertTrue(
            any(s.get("required_kind") == "math" or "cells" in s for s in observed)
        )
        damaged, calls = self.convert("damaged_text.pdf", config={"disable_ocr": True})
        self.assertEqual(damaged.metadata["cloud_conversion"]["status"], "incomplete")
        self.assertEqual(calls, [])

    def test_disabled_scan_text_keeps_math_transcription_without_table_ocr(self):
        from marker.converters.ollama import OllamaPdfConverter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "disabled-table.pdf"
            layout = rasterized_table(path)
            client, observed = offline_client(
                lambda source, n: (
                    layout if source.get("stage") == "layout" else respond(source, n)
                )
            )
            output = OllamaPdfConverter(
                config={"disable_ocr": True}, artifact_dict={"cloud_client": client}
            )(str(path))
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "incomplete")
        self.assertFalse(any("cells" in item for item in observed))
        self.assertTrue(any(item.get("required_kind") == "math" for item in observed))
        self.assertIn(r"x \geq 0", output.markdown)

    def test_disabled_ocr_rejects_unknown_vector_text_classification(self):
        def ordinary_text(source, attempt):
            if source.get("required_kind") == "unknown":
                return {
                    "kind": "text",
                    "resolved": True,
                    "text": "Forbidden vector OCR",
                }
            return respond(source, attempt)

        output, observed = self.convert(
            "native_figure.pdf", responder=ordinary_text, config={"disable_ocr": True}
        )
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "incomplete")
        self.assertNotIn("Forbidden vector OCR", output.markdown)
        self.assertTrue(any(s.get("allow_text") is False for s in observed))

    def test_force_ocr_withholds_table_source_spelling_and_preserves_math(self):
        output, observed = self.convert("content_lines.pdf", config={"force_ocr": True})
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertIn("Scanned baseline", output.markdown)
        self.assertIn("<math", output.markdown)
        self.assertTrue(
            any(
                a["kind"] == "unknown"
                for item in observed
                for c in item.get("cells", [])
                for l in c["lines"]
                for a in l["atoms"]
            )
        )

    def test_borderless_digital_tables_are_retained(self):
        output, observed = self.convert("borderless_tables.pdf")
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertEqual(output.markdown.count("<table"), 2)
        self.assertTrue(
            all(not item["trusted"] for item in observed if "cells" in item)
        )

    def test_force_ocr_and_conflicting_controls(self):
        output, observed = self.convert("damaged_text.pdf", config={"force_ocr": True})
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertTrue(any("Trustworthy" in str(s) for s in observed))
        from marker.converters.ollama import OllamaPdfConverter

        with self.assertRaises(ValueError):
            OllamaPdfConverter(config={"force_ocr": True, "disable_ocr": True})

    def test_source_analysis_failure_keeps_previous_results_and_page_crop(self):
        from marker.cloud.context import RecognitionFailure
        from marker.cloud.regions import analyze_page
        from marker.converters.ollama import OllamaPdfConverter

        def analyze(source, page):
            if page == 1:
                raise RecognitionFailure("Required source region could not be analyzed")
            return analyze_page(source, page)

        client, observed = offline_client(respond)
        with patch("marker.converters.ollama.analyze_page", side_effect=analyze):
            output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                str(FIXTURES / "unrelated_tables.pdf")
            )
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "incomplete")
        self.assertIn("Table 10", output.markdown)
        self.assertTrue(output.images)
        self.assertTrue(
            any(
                "Page 1" in reason
                for reason in output.metadata["cloud_conversion"]["reasons"]
            )
        )

    def test_layout_failure_is_incomplete_and_keeps_original_scan(self):
        output, observed = self.convert(
            "scan.pdf", responder=lambda s, n: {"complete": True, "regions": []}
        )
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "incomplete")
        self.assertTrue(output.images)
        self.assertEqual(len(observed), 3)

    def test_blank_actual_pdf_requires_no_cloud_request(self):
        from marker.converters.ollama import OllamaPdfConverter
        import pypdfium2 as pdfium

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.pdf"
            document = pdfium.PdfDocument.new()
            document.new_page(200, 200)
            document.save(path)
            document.close()
            client, observed = offline_client(respond)
            output = OllamaPdfConverter(artifact_dict={"cloud_client": client})(
                str(path)
            )
        self.assertEqual(output.metadata["cloud_conversion"]["status"], "complete")
        self.assertEqual(output.markdown, "")
        self.assertEqual(observed, [])


class AdaptiveCliTests(unittest.TestCase):
    def test_cli_saves_partial_and_skips_model_factory(self):
        from marker.scripts.convert_single import convert_single_cli
        from marker.converters.ollama import OllamaPdfConverter

        client, observed = offline_client(
            lambda s, n: {"bad": True}, {"cloud_max_attempts": 1}
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("marker.scripts.convert_single.create_model_dict") as factory,
            patch("marker.converters.ollama.CloudClient", return_value=client),
        ):
            result = CliRunner().invoke(
                convert_single_cli,
                [
                    str(FIXTURES / "damaged_text.pdf"),
                    "--converter_cls",
                    "marker.converters.ollama.OllamaPdfConverter",
                    "--output_dir",
                    directory,
                ],
            )
            self.assertEqual(result.exit_code, 1, result.output)
            factory.assert_not_called()
            self.assertTrue(list(Path(directory).rglob("*.md")))
            self.assertTrue(list(Path(directory).rglob("*.png")))
        self.assertFalse(OllamaPdfConverter.requires_model_dict)

    def test_absent_or_invalid_adaptive_completion_is_failure_after_saving(self):
        from marker.scripts.convert_single import convert_single_cli
        from marker.renderers.markdown import MarkdownOutput

        for metadata in ({}, {"cloud_conversion": {"status": "unknown"}}):

            class UncertainConverter:
                requires_model_dict = False

                def __init__(self, **kwargs):
                    pass

                def __call__(self, filepath):
                    return MarkdownOutput(
                        markdown="Partial", images={}, metadata=metadata
                    )

            with (
                tempfile.TemporaryDirectory() as directory,
                patch("marker.scripts.convert_single.create_model_dict") as factory,
                patch(
                    "marker.scripts.convert_single.ConfigParser.get_converter_cls",
                    return_value=UncertainConverter,
                ),
            ):
                result = CliRunner().invoke(
                    convert_single_cli,
                    [str(FIXTURES / "scan.pdf"), "--output_dir", directory],
                )
                self.assertEqual(result.exit_code, 1, result.output)
                factory.assert_not_called()
                self.assertEqual(
                    next(Path(directory).rglob("*.md")).read_text(), "Partial"
                )

    def test_module_entrypoint_exposes_cloud_options(self):
        result = subprocess.run(
            [sys.executable, "-m", "marker.scripts.convert_single", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cloud_max_requests", result.stdout)
        self.assertIn("--cloud_context_tokens", result.stdout)


if __name__ == "__main__":
    unittest.main()
