"""A local placeholder document precedes one job per logical object."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from marker.cloud.content import ContentResult
from marker.cloud.tables import TableResult
from marker.converters.ollama_document import StagedOllamaPdfConverter
from test_content import offline_client
from test_converter import respond

FIXTURES = Path(__file__).parent / "fixtures"


def whole_table(source, table, client):
    from bs4 import BeautifulSoup
    from marker.cloud.content import source_lines, latex_to_mathml
    from marker.cloud.whole_tables import _trusted_structure
    from marker.processors.table_geometry import grid_positions

    expected, template = _trusted_structure(table)
    if expected is not None:
        soup = BeautifulSoup(template, "html.parser")
        ordered = sorted(grid_positions(table.grid)[2], key=lambda item: item[:2])
        for index, (_, _, _, _, original) in enumerate(ordered):
            cell = soup.find(attrs={"data-cell-id": f"c{index}"})
            cell.clear()
            for line in source_lines(original.get("atoms", [])):
                paragraph = soup.new_tag("p")
                text = " ".join(atom.get("text", "") for atom in line).strip()
                if text:
                    paragraph.string = text
                else:
                    paragraph.append(BeautifulSoup(latex_to_mathml("x"), "html.parser"))
                cell.append(paragraph)
        html = str(soup)
        return TableResult("table", html, html)
    text = " ".join(
        a.get("text", "")
        for cell in (table.grid or {}).get("cells", [])
        for a in cell["atoms"]
    )
    html = "<table><tr><td>" + (text or "Scanned table") + "</td></tr></table>"
    return TableResult("table", html, html)


class StagedConverterTests(unittest.TestCase):
    def converter(self, directory, fixture="unrelated_tables.pdf"):
        client, observed = offline_client(
            respond, {"cloud_context_tokens": 131072, "cloud_max_output_tokens": 16384}
        )
        converter = StagedOllamaPdfConverter(
            config={"cloud_work_dir": directory}, artifact_dict={"cloud_client": client}
        )
        # The focused pipeline tests exercise routing/cache/publication; raw
        # transport and HTML recognition have their own HTTP-shaped tests.
        converter._recognize_object = lambda source, region: converter._recognize(
            source, region
        )
        return converter, client, observed, str(FIXTURES / fixture)

    def test_each_table_has_one_job_and_completed_jobs_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            converter, client, observed, path = self.converter(directory)
            with patch(
                "marker.converters.ollama_document.recognize_whole_table",
                side_effect=whole_table,
            ) as table:
                first = converter(path)
                self.assertEqual(table.call_count, 3)
                self.assertEqual(
                    first.metadata["cloud_conversion"]["status"], "complete"
                )
                self.assertEqual(first.markdown.count("<table"), 3)
                self.assertTrue(list(Path(directory).rglob("draft.md")))
                requests = client.request_count
                second = converter(path)
                self.assertEqual(table.call_count, 3)
                self.assertEqual(client.request_count, requests)
                self.assertEqual(first.markdown, second.markdown)
                self.assertGreaterEqual(
                    second.metadata["cloud_conversion"]["cached_jobs"], 3
                )

    def test_failed_table_is_pending_without_png_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            converter, _, _, path = self.converter(directory)
            pending = TableResult(
                "table", complete=False, reasons=("Table recognition failed",)
            )
            with patch(
                "marker.converters.ollama_document.recognize_whole_table",
                return_value=pending,
            ):
                output = converter(path)
            status = output.metadata["cloud_conversion"]
            self.assertEqual(status["status"], "incomplete")
            self.assertFalse(status["publishable"])
            self.assertEqual(status["tables_pending"], 3)
            self.assertNotIn("![", output.markdown)
            self.assertNotIn("<img", output.markdown)
            self.assertIn(
                "Pending table", next(Path(directory).rglob("draft.md")).read_text()
            )

    def test_cached_table_with_unusable_source_layout_reconverts_only_that_table(self):
        import hashlib
        from marker.cloud.stage_store import _canonical

        with tempfile.TemporaryDirectory() as directory:
            converter, client, _, path = self.converter(directory)
            with patch(
                "marker.converters.ollama_document.recognize_whole_table",
                side_effect=whole_table,
            ) as table:
                first = converter(path)
                self.assertTrue(first.metadata["cloud_conversion"]["publishable"])
                cache = next(
                    item
                    for item in Path(directory).rglob("result.json")
                    if json.loads(item.read_text())["record"]["kind"] == "table"
                )
                envelope = json.loads(cache.read_text())
                record = envelope["record"]
                record["html"] = record["html"].replace('data-cell-id="c0"', "")
                record["markdown"] = record["html"]
                envelope["sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
                cache.write_text(json.dumps(envelope))
                second = converter(path)
                self.assertEqual(table.call_count, 4)
                self.assertEqual(second.metadata["cloud_conversion"]["cached_jobs"], 2)
                self.assertTrue(second.metadata["cloud_conversion"]["publishable"])
                self.assertEqual(first.markdown, second.markdown)

    def test_raster_math_does_not_request_coordinate_layout(self):
        from test_client import OfflineCloud
        from marker.cloud.client import CloudClient

        with tempfile.TemporaryDirectory() as directory:
            path = str(FIXTURES / "split_formula.pdf")
            with OfflineCloud(
                [r"<math>x \geq 0</math>", "<math>x^2</math>"]
            ) as offline:
                client = CloudClient(
                    {"cloud_context_tokens": 245760, "cloud_max_output_tokens": 16384},
                    client=offline.client,
                )
                converter = StagedOllamaPdfConverter(
                    config={"cloud_work_dir": directory},
                    artifact_dict={"cloud_client": client},
                )
                with patch(
                    "marker.converters.ollama_document.recognize_whole_table",
                    side_effect=whole_table,
                ):
                    output = converter(path)
                self.assertEqual(client.request_count, 2)
                self.assertTrue(
                    all(
                        "raw HTML, never JSON"
                        in item["messages"][0]["content"][-1]["text"]
                        for item in offline.requests
                    )
                )
            self.assertIn(r"x \geq 0", output.markdown)
            self.assertFalse(any(name.startswith("formula-") for name in output.images))
            self.assertIn(r"Limit: $x \geq 0$", output.markdown)

    def test_known_article_math_and_prose_cannot_be_discarded(self):
        from marker.converters.ollama_document import _article_validator
        from marker.cloud.client import ResponseValidationError

        for text in (
            "<article><p>unrelated</p></article>",
            "<article><p>Русский текст</p></article>",
        ):
            with self.assertRaises(ResponseValidationError):
                _article_validator(
                    text, {}, expected_text=["Русский текст"], require_math=True
                )
        self.assertIn(
            "Русский",
            _article_validator(
                "<article><p>Русский текст <math>x^2</math></p></article>",
                {},
                expected_text=["Русский текст"],
                require_math=True,
            ),
        )

    def test_raw_formula_backslashes_and_repair_keep_latex_exact(self):
        from test_client import OfflineCloud
        from marker.cloud.client import CloudClient
        from marker.cloud.regions import Region, assess_text_quality
        from PIL import Image

        class Source:
            def resource(self, page, box):
                return Image.new("RGB", (64, 32), "white")

        region = Region(
            0, (0, 0, 20, 10), "math", (), assess_text_quality([], (0, 0, 20, 10))
        )
        with OfflineCloud(
            ["<figure/>", r"<math>\theta + \rho + \nabla</math>"]
        ) as offline:
            client = CloudClient(client=offline.client)
            converter = StagedOllamaPdfConverter(artifact_dict={"cloud_client": client})
            result = converter._recognize_object(Source(), region)
            self.assertTrue(result.complete)
            self.assertEqual(result.latex, r"\theta + \rho + \nabla")
            self.assertEqual(result.resources, {})
            prompt = offline.requests[0]["messages"][0]["content"][-1]["text"]
            self.assertNotIn("latex field", prompt)

    def test_searchable_scan_keeps_one_native_table_and_outside_owner(self):
        from marker.converters.ollama_document import _discover_scans
        from marker.cloud.regions import Region, assess_text_quality
        from marker.cloud.tables import TablePart

        class Source:
            def page_size(self, page):
                return (400, 500)

        def region(box, kind):
            return Region(0, box, kind, (), assess_text_quality([], box))

        parent = region((0, 0, 300, 400), "image")
        prose = region((10, 10, 200, 30), "text")
        outside = region((320, 10, 390, 30), "text")
        native = TablePart(0, (20, 80, 280, 300), {"trusted": True}, (400, 500))
        discovered = TablePart(0, (18, 78, 282, 302), None, (400, 500))
        found = [region((10, 10, 200, 30), "text"), region(discovered.bbox, "table")]
        with patch(
            "marker.cloud.scan_regions.discover_scan",
            return_value=(found, [discovered]),
        ):
            owners, parts, reasons = _discover_scans(
                Source(), [parent, prose, outside], [native], None, None
            )
        self.assertEqual(reasons, [])
        self.assertEqual(parts, [native])
        self.assertIn(outside, owners)
        self.assertNotIn(parent, owners)
        self.assertEqual(sum(owner.bbox == prose.bbox for owner in owners), 1)

    def test_scan_crossing_existing_owner_is_incomplete_without_losing_outside_text(
        self,
    ):
        from marker.converters.ollama_document import _discover_scans
        from marker.cloud.regions import Region, assess_text_quality

        class Source:
            def page_size(self, page):
                return (400, 500)

        def region(box, kind):
            return Region(0, box, kind, (), assess_text_quality([], box))

        parent = region((0, 0, 300, 400), "image")
        crossing = region((250, 10, 390, 30), "text")
        with patch("marker.cloud.scan_regions.discover_scan") as discover:
            owners, parts, reasons = _discover_scans(
                Source(), [parent, crossing], [], None, None
            )
        discover.assert_not_called()
        self.assertIn(crossing, owners)
        self.assertTrue(reasons)
        self.assertTrue(any(owner.kind == "scan_unresolved" for owner in owners))

    def test_partial_native_grid_cannot_hide_a_larger_discovered_table(self):
        from marker.converters.ollama_document import _discover_scans
        from marker.cloud.regions import Region, assess_text_quality
        from marker.cloud.tables import TablePart

        class Source:
            def page_size(self, page):
                return (400, 500)

        box = (0, 0, 400, 500)
        parent = Region(0, box, "image", (), assess_text_quality([], box))
        native = TablePart(0, (20, 80, 80, 120), {"trusted": True}, (400, 500))
        discovered = TablePart(0, (20, 80, 380, 400), None, (400, 500))
        with patch(
            "marker.cloud.scan_regions.discover_scan", return_value=([], [discovered])
        ):
            owners, parts, reasons = _discover_scans(
                Source(), [parent], [native], None, None
            )
        self.assertTrue(reasons)
        self.assertEqual(parts, [native])
        self.assertTrue(any(owner.kind == "scan_unresolved" for owner in owners))

    def test_two_scanned_pages_become_one_table_and_resume_without_cloud(self):
        import pypdfium2 as pdfium
        from test_client import OfflineCloud
        from marker.cloud.client import CloudClient

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan-two-pages.pdf"
            with (
                pdfium.PdfDocument(str(FIXTURES / "scan.pdf")) as original,
                pdfium.PdfDocument.new() as doubled,
            ):
                doubled.import_pages(original, [0])
                doubled.import_pages(original, [0])
                doubled.save(str(path))

            def layout(continuation):
                return {
                    "complete": True,
                    "regions": [
                        {
                            "kind": "table",
                            "bbox": [
                                50,
                                0 if continuation else 600,
                                950,
                                350 if continuation else 1000,
                            ],
                            "caption": "Table A.1",
                            "continuation": continuation,
                        }
                    ],
                }

            with OfflineCloud(
                [json.dumps(layout(False)), json.dumps(layout(True))]
            ) as offline:
                client = CloudClient(
                    {"cloud_context_tokens": 245760, "cloud_max_output_tokens": 16384},
                    client=offline.client,
                )
                converter = StagedOllamaPdfConverter(
                    config={"cloud_work_dir": str(Path(directory) / "work")},
                    artifact_dict={"cloud_client": client},
                )
                with (
                    patch(
                        "marker.converters.ollama_document.detect_tables",
                        return_value=[],
                    ),
                    patch(
                        "marker.converters.ollama_document.recognize_whole_table",
                        side_effect=whole_table,
                    ) as table,
                ):
                    first = converter(str(path))
                    second = converter(str(path))
                self.assertEqual(table.call_count, 1)
                self.assertEqual(len(table.call_args.args[1].parts), 2)
                self.assertEqual(client.request_count, 2)
                self.assertEqual(
                    first.metadata["cloud_conversion"]["tables_expected"], 1
                )
                self.assertEqual(
                    first.metadata["cloud_conversion"]["scan_layout_requests"], 2
                )
                self.assertEqual(
                    second.metadata["cloud_conversion"]["scan_layout_requests"], 0
                )

    def test_small_raster_table_can_be_html_and_disable_ocr_remains_effective(self):
        from test_client import OfflineCloud
        from marker.cloud.client import CloudClient
        from marker.cloud.regions import Region, assess_text_quality
        from PIL import Image

        class Source:
            def resource(self, page, box):
                return Image.new("RGB", (300, 80), "white")

        region = Region(
            0, (0, 0, 100, 25), "image", (), assess_text_quality([], (0, 0, 100, 25))
        )
        for disabled in (False, True):
            with (
                self.subTest(disable_ocr=disabled),
                OfflineCloud(["<table><tr><td>Text</td></tr></table>"]) as offline,
            ):
                client = CloudClient({"cloud_max_attempts": 1}, client=offline.client)
                converter = StagedOllamaPdfConverter(
                    config={"disable_ocr": disabled},
                    artifact_dict={"cloud_client": client},
                )
                result = converter._recognize_object(Source(), region)
                self.assertEqual(result.complete, not disabled)
                if not disabled:
                    self.assertIn("<table", result.markdown)

    def test_scanned_paragraph_with_inline_math_uses_article_recognition(self):
        from marker.cloud.regions import Region, assess_text_quality

        region = Region(
            0, (0, 0, 100, 25), "text", (), assess_text_quality([], (0, 0, 100, 25))
        )
        converter = StagedOllamaPdfConverter()
        with patch.object(
            converter,
            "_recognize_article",
            return_value=ContentResult("text", "value $x^2$"),
        ) as article:
            result = converter._recognize_job(None, {"kind": "text", "region": region})
        article.assert_called_once()
        self.assertIn("$x^2$", result.markdown)


if __name__ == "__main__":
    unittest.main()
