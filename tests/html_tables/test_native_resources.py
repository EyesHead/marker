"""Native LLM replacements preserve semantic HTML and source image resources."""

import base64
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
from PIL import Image

from marker.cloud.resources import retain_png
from marker.output import save_output
from marker.renderers.html import HTMLRenderer
from marker.renderers.chunk import ChunkRenderer
from marker.renderers.json import JSONRenderer
from marker.renderers.markdown import MarkdownRenderer
from marker.schema.blocks import Diagram, Figure, Picture, Table, TableCell, Text
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox


MATH = (r'<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">'
        r'<semantics><msub><mi>R</mi><mi>bun</mi></msub>'
        r'<annotation encoding="application/x-tex">0,42R_{\text{bun}}</annotation>'
        r'</semantics></math>')


def document_with(*blocks):
    page_image = Image.new("RGB", (100, 100), "white")
    page = PageGroup(
        page_id=0, polygon=PolygonBox.from_bbox([0, 0, 100, 100]),
        children=[], structure=[], lowres_image=page_image, highres_image=page_image,
    )
    for block in blocks:
        page.add_structure(page.add_full_block(block))
    return Document(filepath="offline.pdf", pages=[page])


def block(cls, **kwargs):
    return cls(page_id=0, polygon=PolygonBox.from_bbox([10, 10, 90, 90]), **kwargs)


class NativeResourceTests(unittest.TestCase):
    def test_image_class_replaced_by_formula_never_readds_original_png(self):
        for cls in (Picture, Figure, Diagram):
            owner = block(cls, html=MATH, llm_processed=True)
            document = document_with(owner)
            with self.subTest(cls=cls.__name__):
                with patch.object(
                    cls, "get_image", side_effect=AssertionError("original image re-added")
                ):
                    rendered = MarkdownRenderer()(document)
                self.assertIn(r"$0,42R_{\text{bun}}$", rendered.markdown)
                self.assertNotIn("![", rendered.markdown)
                self.assertEqual(rendered.images, {})

    def test_genuine_diagram_keeps_exactly_one_source_resource(self):
        resource = retain_png(Image.new("RGBA", (17, 23), (10, 20, 30, 100)))
        owner = block(Diagram, html='<p><img src="diagram.png" alt="Схема"></p>',
                      llm_processed=True, embedded_images={"diagram.png": resource})
        document = document_with(owner)
        with patch.object(resource, "load", side_effect=AssertionError("resource decoded")):
            rendered = HTMLRenderer()(document)
            self.assertEqual(len(BeautifulSoup(rendered.html, "html.parser").find_all("img")), 1)
            self.assertEqual(set(rendered.images), {"diagram.png"})
            self.assertIs(rendered.images["diagram.png"], resource)
            with tempfile.TemporaryDirectory() as temporary:
                save_output(rendered, temporary, "document")
                self.assertTrue(Path(temporary, "diagram.png").is_file())

    def test_embedded_resources_belong_to_any_block(self):
        resource = Image.new("RGB", (21, 7), "red")
        owner = block(Text, html='<p>Текст <img src="inline.png"></p>',
                      llm_processed=True, embedded_images={"inline.png": resource})
        rendered = MarkdownRenderer()(document_with(owner))
        self.assertEqual(set(rendered.images), {"inline.png"})
        self.assertIs(rendered.images["inline.png"], resource)
        self.assertIn("inline.png", rendered.markdown)
        other = block(Text)
        self.assertEqual(other.embedded_images, {})
        self.assertFalse(other.llm_processed)

    def test_ignored_continuation_never_renders_or_exports_its_assets(self):
        first = block(Table, html='<table><tr><td><img src="kept.png"></td></tr></table>',
                      embedded_images={"kept.png": Image.new("RGB", (3, 3))})
        hidden = block(Table, html='<table><tr><td><img src="hidden.png"></td></tr></table>',
                       ignore_for_output=True,
                       embedded_images={"hidden.png": Image.new("RGB", (4, 4))})
        rendered = MarkdownRenderer()(document_with(first, hidden))
        self.assertEqual(set(rendered.images), {"kept.png"})
        self.assertEqual(rendered.markdown.count("<table>"), 1)
        self.assertNotIn("hidden.png", rendered.markdown)

    def test_unprocessed_picture_keeps_high_resolution_source(self):
        source = Image.new("RGBA", (217, 123), (10, 20, 30, 40))
        owner = block(Picture, highres_image=source)
        rendered = MarkdownRenderer()(document_with(owner))
        self.assertEqual(len(rendered.images), 1)
        self.assertIs(next(iter(rendered.images.values())), source)
        self.assertIn("![", rendered.markdown)

    def test_mathml_in_tables_preserves_tex_annotation(self):
        owner = block(Table, html=f'<table><tr><td>{MATH}</td></tr></table>',
                      llm_processed=True)
        rendered = MarkdownRenderer()(document_with(owner))
        soup = BeautifulSoup(rendered.markdown, "html.parser")
        self.assertEqual(soup.find("annotation").get_text(), r"0,42R_{\text{bun}}")
        self.assertIsNotNone(soup.find("semantics"))
        self.assertEqual(rendered.images, {})

    def test_cloud_metadata_is_optional_and_independent(self):
        document = document_with(block(Text, html="<p>text</p>"))
        self.assertNotIn("cloud_conversion", MarkdownRenderer()(document).metadata)
        report = {"complete": True, "tables": {"count": 1}}
        expected = deepcopy(report)
        document.cloud_conversion = report
        metadata = MarkdownRenderer()(document).metadata
        self.assertEqual(report, expected)
        self.assertEqual(metadata["cloud_conversion"], expected)
        metadata["cloud_conversion"]["tables"]["count"] = 99
        self.assertEqual(document.cloud_conversion, expected)

    def test_json_replacement_excludes_original_crop_and_retains_embedded_png(self):
        source = retain_png(Image.new("RGBA", (12, 9), (1, 2, 3, 4)))
        owner = block(Diagram, html='<img src="diagram.png">', llm_processed=True,
                      embedded_images={"diagram.png": source})
        document = document_with(owner)
        with patch.object(Diagram, "get_image", side_effect=AssertionError("original crop")), \
                patch.object(source, "load", side_effect=AssertionError("decoded PNG")):
            output = JSONRenderer()(document)
        images = output.children[0].children[0].images
        self.assertEqual(set(images), {"diagram.png"})
        self.assertTrue(base64.b64decode(images["diagram.png"]).startswith(b"\x89PNG\r\n"))

    def test_chunk_replacement_formula_never_readds_original_crop(self):
        owner = block(Picture, html=MATH, llm_processed=True)
        document = document_with(owner)
        with patch.object(Picture, "get_image", side_effect=AssertionError("original crop")):
            output = ChunkRenderer()(document)
        self.assertEqual(len(output.blocks), 1)
        self.assertNotIn("<img", output.blocks[0].html)
        self.assertIn("application/x-tex", output.blocks[0].html)
        self.assertEqual(output.blocks[0].images, {})

    def test_native_json_table_keeps_cell_tree_and_embedded_resources(self):
        owner = block(Table, embedded_images={"scheme.png": Image.new("RGB", (3, 3))})
        document = document_with(owner)
        page = document.pages[0]
        cell = page.add_full_block(block(
            TableCell, rowspan=1, colspan=1, row_id=0, col_id=0,
            is_header=False, text_lines=['<img src="scheme.png">'],
        ))
        owner.add_structure(cell)
        output = JSONRenderer()(document)
        table_output = output.children[0].children[0]
        self.assertEqual(len(table_output.children), 1)
        self.assertEqual(set(table_output.images), {"scheme.png"})


if __name__ == "__main__":
    unittest.main()
