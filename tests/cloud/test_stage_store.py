"""Intermediate document state resumes only validated, complete jobs."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from marker.cloud.content import ContentResult
from marker.cloud.context import RecognitionFailure
from marker.cloud.stage_store import StageStore


class StageStoreTests(unittest.TestCase):
    def test_complete_job_resumes_without_source_image_lifetime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"PDF-one")
            store = StageStore(root / "work", pdf, {"model": "test"})
            result = ContentResult(
                "figure",
                "![Figure](figure.png)",
                resources={"figure.png": Image.new("RGBA", (12, 9), (1, 2, 3, 4))},
            )
            store.save("a0", result)
            loaded = StageStore(root / "work", pdf, {"model": "test"}).load("a0")
            self.assertEqual(loaded.markdown, result.markdown)
            self.assertEqual(
                loaded.resources["figure.png"].getpixel((0, 0)), (1, 2, 3, 4)
            )

    def test_failed_jobs_are_not_cached_and_changed_inputs_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"one")
            store = StageStore(root / "work", pdf, {"model": "test"})
            store.save(
                "a0", ContentResult("math", complete=False, reasons=("unreadable",))
            )
            self.assertIsNone(store.load("a0"))
            self.assertNotEqual(
                store.root, StageStore(root / "work", pdf, {"model": "different"}).root
            )
            pdf.write_bytes(b"two")
            self.assertNotEqual(
                store.root, StageStore(root / "work", pdf, {"model": "test"}).root
            )

    def test_credentials_never_reach_cache_or_intermediate_files(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"OLLAMA_API_KEY": "TEST_SECRET_NOT_TO_WRITE"}),
        ):
            root = Path(temporary)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"one")
            store = StageStore(root / "work", pdf, {})
            for write in (
                lambda: store.write_text("draft.md", "TEST_SECRET_NOT_TO_WRITE"),
                lambda: store.save(
                    "a0", ContentResult("text", "TEST_SECRET_NOT_TO_WRITE")
                ),
            ):
                with self.assertRaises(RecognitionFailure):
                    write()
            self.assertFalse(
                any(
                    b"TEST_SECRET_NOT_TO_WRITE" in p.read_bytes()
                    for p in store.root.rglob("*")
                    if p.is_file()
                )
            )

    def test_missing_or_corrupt_cached_asset_requires_reconversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"one")
            store = StageStore(root / "work", pdf, {})
            store.save(
                "a0",
                ContentResult(
                    "figure",
                    "![Figure](figure.png)",
                    resources={"figure.png": Image.new("RGB", (3, 4))},
                ),
            )
            next(store.root.rglob("figure.png")).write_bytes(b"not a PNG")
            self.assertIsNone(store.load("a0"))

    def test_changed_cached_markup_cannot_bypass_recognition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"one")
            store = StageStore(root / "work", pdf, {})
            html = "<table><tr><td>value</td></tr></table>"
            store.save("a0", ContentResult("table", html, html))
            path = next(store.root.rglob("result.json"))
            envelope = json.loads(path.read_text())
            envelope["record"]["markdown"] = "<script>alert(1)</script>"
            path.write_text(json.dumps(envelope))
            self.assertIsNone(store.load("a0"))

    def test_malformed_cache_container_is_a_miss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"one")
            store = StageStore(root / "work", pdf, {})
            for value in ([], {"record": []}, {"record": {"assets": []}}):
                path = store._job("a0") / "result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                self.assertIsNone(store.load("a0"))

    def test_generated_column_widths_and_exact_math_survive_cache(self):
        from bs4 import BeautifulSoup
        from marker.cloud.whole_tables import normalize_table_html

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"one")
            store = StageStore(root / "work", pdf, {})
            html = normalize_table_html(
                r"<table><tr><td>Value</td><td><math>0,42R_{\text{bun}}</math></td></tr></table>",
                column_widths=[20, 80],
            )
            store.save("a0", ContentResult("table", html, html))
            loaded = store.load("a0")
            self.assertIsNotNone(loaded)
            soup = BeautifulSoup(loaded.html, "html.parser")
            self.assertEqual(
                [float(col["width"][:-1]) for col in soup.find_all("col")], [20, 80]
            )
            self.assertEqual(soup.find("annotation").get_text(), r"0,42R_{\text{bun}}")
            self.assertEqual(loaded.markdown, loaded.html)


if __name__ == "__main__":
    unittest.main()
