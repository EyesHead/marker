"""Native --use_llm wiring and whole-object contracts, with no model/API calls."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from bs4 import BeautifulSoup
from PIL import Image

from marker.cloud.context import RecognitionFailure
from marker.cloud.client import ResponseValidationError
from marker.cloud.source import NativeImage, intersection, PdfSource
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.processors.llm import BaseLLMProcessor
from marker.processors.llm.llm_native import NativeLLMProcessor, ServiceClient
from marker.processors.table import TableProcessor
from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.schema.blocks import Equation, Picture, Table, Text
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.services.ollama import OllamaService
from marker.scripts.convert import needs_local_vlm


class Source:
    def __init__(self, words=(), pictures=()):
        self.words, self.pictures = list(words), list(pictures)
        self.native = Image.new("RGBA", (800, 400), (11, 23, 37, 123))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def page_size(self, page):
        return 600, 800

    def page_words(self, page, box=None):
        return deepcopy([word for word in self.words if word.get("source_page", 0) == page
                         and (box is None or intersection(word["bbox"], box) > 0)])

    def page_images(self, page, bbox=None):
        return [image for image in self.pictures if image.page == page and
                (bbox is None or intersection(image.bbox, bbox) > 0)]

    def crop(self, page, bbox, dpi=None):
        return self.native.copy()

    def resource(self, page, bbox, dpi=None):
        return self.native.copy()


class Client:
    cloud_model = "kimi-k3"
    cloud_base_url = "https://ollama.com/v1"
    cloud_context_tokens = 245760
    cloud_max_output_tokens = 16384
    cloud_max_images = 16
    cloud_max_concurrency = 2
    cloud_max_requests = 0
    cloud_max_attempts = 1
    request_count = 0
    total_tokens = 0

    def __init__(self, answer):
        self.answer, self.calls = answer, []
        self.layout = ({"complete": True, "regions": [{"kind": "math", "bbox": [0.0, 0.0, 1000.0, 1000.0]}]}
                       if isinstance(answer, str) and answer.startswith("<math>") else None)

    def request(self, prompt, images, schema):
        self.request_count += 1
        if "Locate ALL visible content" in prompt:
            return self.layout
        return {"merge": True}

    def request_text(self, prompt, images, validator, **kwargs):
        self.request_count += 1
        self.calls.append((prompt, images))
        if isinstance(self.answer, Exception):
            raise self.answer
        try:
            return validator(self.answer(prompt) if callable(self.answer) else self.answer)
        except ResponseValidationError as error:
            raise RecognitionFailure(str(error)) from None


def page(page_id=0):
    return PageGroup(page_id=page_id, polygon=PolygonBox.from_bbox((0, 0, 600, 800)),
                     children=[], structure=[], lowres_image=Image.new("RGB", (600, 800), "white"))


def add(owner, cls, box, **kwargs):
    block = owner.add_full_block(cls(page_id=owner.page_id, polygon=PolygonBox.from_bbox(box), **kwargs))
    owner.add_structure(block)
    return block


class NativePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pdf = Path(self.temp.name) / "source.pdf"
        self.pdf.write_bytes(b"offline source scope")

    def processor(self, client, **kwargs):
        service = Mock(spec=OllamaService)
        service.client = client
        return NativeLLMProcessor(service, {"use_llm": True, "cloud_work_dir": self.temp.name,
                                           **kwargs})

    def document(self, pages):
        return Document(filepath=str(self.pdf), pages=pages)

    def test_explicit_provider_cache_identity_tracks_actual_model_and_endpoint(self):
        for model_attr, endpoint_attr in (("openai_model", "openai_base_url"),
                                          ("openrouter_model", "openrouter_base_url"),
                                          ("deployment_name", "azure_endpoint"),
                                          ("claude_model_name", "provider_endpoint")):
            with self.subTest(provider=model_attr):
                service = SimpleNamespace(**{model_attr: "first-model", endpoint_attr: "https://first.example"})
                first = ServiceClient(service, {})
                setattr(service, model_attr, "second-model")
                second = ServiceClient(service, {})
                self.assertNotEqual(first.cloud_model, second.cloud_model)
                if endpoint_attr != "provider_endpoint":
                    setattr(service, endpoint_attr, "https://second.example")
                    third = ServiceClient(service, {})
                    self.assertNotEqual(second.cloud_base_url, third.cloud_base_url)
        claude = SimpleNamespace(claude_model_name="offline", max_claude_tokens=8192)
        adapter = ServiceClient(claude, {"cloud_max_output_tokens": 12000})
        self.assertEqual(adapter.service.max_claude_tokens, 12000)
        self.assertEqual(claude.max_claude_tokens, 8192)

    def test_native_pdf_converter_uses_ollama_and_one_whole_object_processor(self):
        artifacts = {"recognition_model": Mock(), "ocr_error_model": Mock(),
                     "layout_model": Mock(), "fast_layout_model": Mock()}
        with patch("marker.converters.download_font"):
            converter = PdfConverter(artifacts, config={"use_llm": True})
        self.assertIsInstance(converter.llm_service, OllamaService)
        self.assertIsInstance(converter.native_llm_processor, NativeLLMProcessor)
        self.assertFalse(any(isinstance(item, BaseLLMProcessor) for item in converter.processor_list))
        self.assertTrue(next(p for p in converter.processor_list if isinstance(p, TableProcessor)).disable_ocr)
        self.assertEqual(converter.source_config()["mode"], "fast")
        self.assertFalse(converter.source_config()["force_ocr"])
        self.assertFalse(converter.config.get("disable_ocr", False))
        self.assertEqual(ConfigParser({"use_llm": True}).get_llm_service(),
                         "marker.services.ollama.OllamaService")

    def test_batch_cli_does_not_start_a_local_vlm_for_native_cloud_recognition(self):
        self.assertFalse(needs_local_vlm({"use_llm": True}))
        self.assertFalse(needs_local_vlm({"PdfConverter_use_llm": True}))
        self.assertTrue(needs_local_vlm({"use_llm": True, "llm_native_pipeline": False}))
        self.assertTrue(needs_local_vlm({"use_llm": True, "processors": "marker.processors.table.TableProcessor"}))
        self.assertTrue(needs_local_vlm({}))
        self.assertFalse(needs_local_vlm({"disable_ocr": True}))

    def test_namespaced_use_llm_enables_the_same_service_and_processor(self):
        with patch("marker.converters.download_font"):
            converter = PdfConverter({"recognition_model": Mock()},
                                     config={"PdfConverter_use_llm": True})
        self.assertTrue(converter.native_llm)
        self.assertIsInstance(converter.llm_service, OllamaService)
        self.assertTrue(converter.native_llm_processor.use_llm)

    def test_no_llm_and_explicit_custom_processors_keep_original_path(self):
        artifacts = {"recognition_model": Mock(), "ocr_error_model": Mock()}
        for config, processors in (({}, []), ({"use_llm": True}, []),
                                   ({"use_llm": True, "llm_native_pipeline": False}, [])):
            with self.subTest(config=config), patch("marker.converters.download_font"):
                converter = PdfConverter(artifacts.copy(), processor_list=processors, config=config)
                self.assertFalse(converter.native_llm)
                self.assertIsNone(converter.native_llm_processor)

    def test_namespaced_layout_overrides_cannot_skip_the_source_detector(self):
        with patch("marker.converters.download_font"):
            converter = PdfConverter({"recognition_model": Mock()}, processor_list=None,
                config={"use_llm": True, "LayoutBuilder_mode": "balanced", "LayoutBuilder_force_ocr": True})
        self.assertEqual(converter.source_config()["LayoutBuilder_mode"], "fast")
        self.assertFalse(converter.source_config()["LayoutBuilder_force_ocr"])

    def test_real_pdf_uses_native_cells_and_one_request_per_independent_table(self):
        fixture = Path(__file__).parent / "fixtures/unrelated_tables.pdf"
        with PdfSource(fixture) as source:
            pages = [PageGroup(page_id=i, polygon=PolygonBox.from_bbox((0, 0, *source.page_size(i))),
                               children=[], structure=[]) for i in range(len(source))]
        doc = Document(filepath=str(fixture), pages=pages)
        TableProcessor(Mock(), {"disable_ocr": True, "extract_images": False})(doc)
        physical = [block for current in pages for block in current.children if block.block_type == BlockTypes.Table]
        self.assertEqual(len(physical), 3)
        def fill_template(prompt):
            payload = json.loads(prompt.split("SOURCE:\n", 1)[1])
            soup = BeautifulSoup(payload["html_template"], "html.parser")
            text = {}
            for part in payload["parts"]:
                for cell in part["cells"]:
                    text.setdefault(cell["id"], []).extend(cell["trusted_text"].splitlines())
            for cell in soup.find_all(["td", "th"]):
                for node, line in zip(cell.find_all("p"), text.get(cell["data-cell-id"], [])):
                    node.string = line
            return str(soup)
        client = Client(fill_template)
        self.processor(client)(doc)
        output = MarkdownRenderer()(doc)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all(len(images) == 1 for _, images in client.calls))
        self.assertEqual(len(BeautifulSoup(output.markdown, "html.parser").find_all("table")), 3)
        self.assertEqual(output.metadata["cloud_conversion"]["logical_tables"], 3)
        self.assertEqual(output.metadata["cloud_conversion"]["reasons"], [])

    def test_one_table_request_for_three_continuations_and_cached_rerun(self):
        pages = [page(i) for i in range(3)]
        for i, current in enumerate(pages):
            add(current, Table, (10, 550 if i == 0 else 10, 590, 790 if i < 2 else 200))
        client = Client('<table><tr><td><math>0,42R_{\\text{bun}}</math></td></tr></table>')
        processor = self.processor(client)
        with patch("marker.processors.llm.llm_native.PdfSource", return_value=Source()):
            doc = self.document(pages)
            processor(doc)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0][1]), 3)
        markdown = MarkdownRenderer()(doc)
        soup = BeautifulSoup(markdown.markdown, "html.parser")
        self.assertEqual(len(soup.find_all("table")), 1)
        self.assertEqual(soup.annotation.get_text(), r"0,42R_{\text{bun}}")
        self.assertFalse(markdown.images)
        rerun = [page(i) for i in range(3)]
        for i, current in enumerate(rerun):
            add(current, Table, (10, 550 if i == 0 else 10, 590, 790 if i < 2 else 200))
        with patch("marker.processors.llm.llm_native.PdfSource", return_value=Source()):
            processor(self.document(rerun))
        self.assertEqual(len(client.calls), 1)

    def test_picture_recognized_as_math_is_latex_and_has_no_bitmap(self):
        current = page()
        picture = add(current, Picture, (40, 40, 140, 80))
        client = Client(r"<math>R_{\text{un}}</math>")
        with patch("marker.processors.llm.llm_native.PdfSource", return_value=Source()):
            doc = self.document([current])
            self.processor(client)(doc)
        output = MarkdownRenderer()(doc)
        self.assertIn(r"R_{\text{un}}", output.markdown)
        self.assertNotIn("![", output.markdown)
        self.assertEqual(output.images, {})
        self.assertTrue(picture.ignore_for_output)
        self.assertTrue(any(block.llm_processed for block in current.children if block.block_type == BlockTypes.Equation))

    def test_source_diagram_in_table_retains_native_pixels(self):
        current = page()
        add(current, Table, (0, 0, 600, 400))
        box = (20, 20, 120, 100)
        image = NativeImage(0, 0, box, 800, 400, box, 0)
        source = Source(pictures=[image])
        def respond(prompt):
            payload = json.loads(prompt.split("SOURCE:\n", 1)[1])
            identity = payload["figures"][0]["id"]
            return f'<table><tr><td><img data-source-image="{identity}" alt="Diagram"></td><td><math>\\varphi</math></td></tr></table>'
        client = Client(respond)
        with patch("marker.processors.llm.llm_native.PdfSource", return_value=source):
            doc = self.document([current])
            self.processor(client)(doc)
        output = MarkdownRenderer()(doc)
        self.assertEqual(len(output.images), 1)
        retained = next(iter(output.images.values()))
        self.assertEqual(retained.mode, source.native.mode)
        self.assertEqual(retained.size, source.native.size)
        self.assertEqual(retained.tobytes(), source.native.tobytes())
        self.assertIn(r"\varphi", output.markdown)

    def test_failed_table_is_not_published_or_replaced_with_a_source_png(self):
        current = page()
        table = add(current, Table, (0, 0, 600, 400), html="original source placeholder")
        client = Client(RecognitionFailure("offline failure"))
        with patch("marker.processors.llm.llm_native.PdfSource", return_value=Source()):
            doc = self.document([current])
            with self.assertRaises(RecognitionFailure):
                self.processor(client)(doc)
        self.assertEqual(table.html, "original source placeholder")
        self.assertFalse(table.llm_processed)
        self.assertFalse(doc.cloud_conversion["publishable"])
        self.assertEqual(doc.cloud_conversion["status"], "incomplete")

    def test_scanned_full_page_uses_article_recognition_not_a_figure_placeholder(self):
        current = page()
        add(current, Picture, (0, 0, 600, 800))
        client = Client("<table><tr><td>1</td><td><math>x_1</math></td></tr></table>")
        client.layout = {"complete": True, "regions": [{"kind": "table", "bbox": [0.0, 0.0, 1000.0, 1000.0],
                                                      "caption": "Table 1", "continuation": False}]}
        with patch("marker.processors.llm.llm_native.PdfSource", return_value=Source()):
            doc = self.document([current])
            self.processor(client)(doc)
        output = MarkdownRenderer()(doc)
        self.assertEqual(len(BeautifulSoup(output.markdown, "html.parser").find_all("table")), 1)
        self.assertFalse(output.images)

    def test_large_raster_equation_and_inline_math_cannot_be_plain_text(self):
        from marker.schema.blocks import InlineMath
        box = (10, 10, 160, 65)
        picture = NativeImage(0, 0, box, 900, 330, box, 0)
        for cls in (Equation, InlineMath):
            with self.subTest(cls=cls.__name__):
                current = page()
                add(current, cls, box)
                client = Client("<article><p>Run</p></article>")
                with patch("marker.processors.llm.llm_native.PdfSource", return_value=Source(pictures=[picture])):
                    doc = self.document([current])
                    with self.assertRaises(RecognitionFailure):
                        self.processor(client)(doc)
                self.assertEqual(doc.cloud_conversion["status"], "incomplete")

    def test_selectable_equation_always_requires_latex(self):
        current = page()
        box = (10, 10, 160, 65)
        equation = add(current, Equation, box)
        word = {"kind": "text", "text": "Run", "fontname": "Arial", "size": 12,
                "bbox": (20, 20, 50, 35), "source_page": 0}
        source = Source(words=[word])
        client = Client(r"<math>R_{\text{un}}</math>")
        processor = self.processor(client)
        region = processor._block_region(self.document([current]), source, equation)
        self.assertIsNotNone(region)
        self.assertEqual(region.kind, "math")

    def test_inline_math_paragraph_keeps_prose_and_requires_math(self):
        from marker.schema.blocks import InlineMath
        current = page()
        box = (10, 10, 160, 65)
        block = add(current, InlineMath, box)
        words = [{"kind": "text", "text": text, "fontname": "Arial", "size": 12,
                  "bbox": bounds, "source_page": 0} for text, bounds in
                 (("Where", (20, 20, 50, 35)), ("R", (55, 20, 65, 35)))]
        source = Source(words=words)
        client = Client(r"<article><p>Where <math>R</math></p></article>")
        processor = self.processor(client)
        region = processor._block_region(self.document([current]), source, block)
        self.assertEqual(region.kind, "text")
        with patch("marker.processors.llm.llm_native.PdfSource", return_value=source):
            doc = self.document([current])
            processor(doc)
        output = MarkdownRenderer()(doc)
        self.assertIn("Where", output.markdown)
        self.assertIn("$R$", output.markdown)


if __name__ == "__main__":
    unittest.main()
