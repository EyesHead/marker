"""Offline performance contracts preserve source ownership and crop pixels."""

from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from marker.cloud.content import ContentResult
from marker.cloud.client import CloudClient, schema_prompt
from marker.cloud.context import fit_request
from marker.cloud.regions import analyze_page
from marker.cloud.source import PdfSource
from marker.cloud.tables import (
    LogicalTable,
    _window_images,
    detect_tables,
)


FIXTURES = Path(__file__).parent / "fixtures"


class SourcePlanningPerformanceTests(unittest.TestCase):
    def test_page_words_extract_once_with_independent_returned_records(self):
        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            page = source.pdf_page(0)
            with patch.object(page, "extract_words", wraps=page.extract_words) as extract:
                expected = source.page_words(0)
                self.assertTrue(expected)
                original = deepcopy(expected)
                expected[0]["text"] = "mutated caller spelling"
                expected[0]["bbox"] = (0, 0, 1, 1)
                self.assertEqual(source.page_words(0), original)
                box = original[0]["bbox"]
                filtered = source.page_words(0, box)
                self.assertIn(original[0], filtered)
                self.assertEqual(extract.call_count, 1)

    def test_cached_word_metadata_is_deeply_independent(self):
        word = {
            "text": "value", "bbox": (10, 20, 30, 40), "x0": 10, "top": 20,
            "width": 20, "height": 20, "metadata": {"ids": [1, 2]},
        }
        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            with patch.object(source.pdf_page(0), "extract_words", return_value=[word]) as extract:
                first = source.page_words(0)
                first[0]["metadata"]["ids"].append(3)
                self.assertEqual(source.page_words(0)[0]["metadata"], {"ids": [1, 2]})
                self.assertEqual(word["metadata"], {"ids": [1, 2]})
                self.assertEqual(extract.call_count, 1)

    def test_precomputed_tables_preserve_regions_without_rediscovery(self):
        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            expected = analyze_page(source, 0)
            parts = detect_tables(source, [0])
            self.assertTrue(parts)
            with patch("marker.cloud.tables.detect_tables", side_effect=AssertionError("duplicate discovery")):
                actual = analyze_page(source, 0, table_parts=parts)
            self.assertEqual([r.__dict__ for r in actual], [r.__dict__ for r in expected])

    def test_explicit_empty_tables_do_not_trigger_detection(self):
        with PdfSource(FIXTURES / "native_figure.pdf") as source:
            with patch("marker.cloud.tables.detect_tables", side_effect=AssertionError("duplicate discovery")):
                self.assertTrue(analyze_page(source, 0, table_parts=[]))

    def test_converter_discovers_each_page_once_and_reports_safe_stages(self):
        import marker.cloud.tables as tables
        from marker.converters.ollama import OllamaPdfConverter

        counts = Counter()
        original = tables.detect_tables

        def observed(source, page_ids=None):
            counts.update(page_ids)
            return original(source, page_ids)

        def separate_tables(parts, _client):
            return [LogicalTable([part], deepcopy(part.grid)) for part in parts]

        converter = OllamaPdfConverter(artifact_dict={"cloud_client": SimpleNamespace(request_count=0)})
        with patch("marker.converters.ollama.detect_tables", side_effect=observed), \
                patch.object(tables, "detect_tables", side_effect=observed), \
                patch("marker.converters.ollama.group_tables", side_effect=separate_tables), \
                patch("marker.converters.ollama.recognize_table", return_value=ContentResult("table", "table")), \
                patch.object(converter, "_recognize", return_value=ContentResult("text", "text")), \
                self.assertLogs("marker", level="INFO") as progress:
            output = converter(str(FIXTURES / "unrelated_tables.pdf"))
        self.assertTrue(output.markdown)
        self.assertTrue(counts)
        self.assertTrue(all(count == 1 for count in counts.values()), counts)
        self.assertIn("source page", " ".join(progress.output).lower())
        self.assertIn("requests=0", " ".join(progress.output))
        self.assertNotIn("Table 1", " ".join(progress.output))


class WindowCropCacheTests(unittest.TestCase):
    def test_all_window_prompts_schemas_and_request_pngs_match_uncached_planning(self):
        import marker.cloud.tables as tables
        from marker.processors.table_geometry import grid_positions

        def signature(windows, client):
            result = []
            for cells, prompt, images, schema in windows:
                full_prompt = schema_prompt(prompt, schema)
                pngs = []
                for image in fit_request(full_prompt, images, client.cloud_context_tokens, client.cloud_max_output_tokens):
                    stream = BytesIO()
                    image.save(stream, format="PNG")
                    pngs.append(hashlib.sha256(stream.getvalue()).hexdigest())
                result.append((cells, full_prompt, pngs))
            return result

        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            part = detect_tables(source, [0])[0]
            xs, ys, positions = grid_positions(part.grid)
            described, originals = tables._describe_cells(part.grid, positions)
            client = CloudClient()
            image_builder = tables._window_images

            def uncached(*args, **_kwargs):
                return image_builder(*args)

            args = (source, described, originals, client, part.grid.get("trusted", True), len(ys) - 1, len(xs) - 1)
            with patch.object(tables, "_window_images", side_effect=uncached), patch.object(source, "crop", wraps=source.crop) as before:
                expected = list(tables._table_windows(*args))
            with patch.object(source, "crop", wraps=source.crop) as after:
                actual = list(tables._table_windows(*args))
            self.assertGreater(len(actual), 1)
            self.assertLess(after.call_count, before.call_count)
            self.assertEqual(signature(actual, client), signature(expected, client))
            self.assertEqual(client.request_count, 0)

    def test_repeated_crops_reuse_rendering_with_identical_independent_pixels(self):
        cells = [{"id": "c0", "lines": [{"page": 0, "bbox": (40, 70, 110, 90)}]}]
        headers = [{"page": 0, "bbox": (40, 30, 110, 50), "text": "header"}]
        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            expected = _window_images(source, cells, {}, headers, 3)
            cache = OrderedDict()
            with patch.object(source, "crop", wraps=source.crop) as crop:
                first = _window_images(source, cells, {}, headers, 3, crop_cache=cache)
                second = _window_images(source, cells, {}, headers, 3, crop_cache=cache)
            self.assertEqual(crop.call_count, 2)
            for original, a, b in zip(expected, first, second):
                self.assertEqual((a.mode, a.size, a.tobytes()), (original.mode, original.size, original.tobytes()))
                self.assertEqual(b.tobytes(), original.tobytes())
                a.paste((1, 2, 3), (0, 0, a.width, a.height))
                self.assertEqual(b.tobytes(), original.tobytes())

    def test_crop_memo_stays_bounded_across_many_unique_windows(self):
        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            cache = OrderedDict()
            for index in range(30):
                cells = [{"id": "c0", "lines": [{"page": 0, "bbox": (20, 20 + index, 100, 30 + index)}]}]
                _window_images(source, cells, {}, [], 3, crop_cache=cache)
            self.assertLessEqual(len(cache), 8)


if __name__ == "__main__":
    unittest.main()
