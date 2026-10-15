"""Complete-expression zooms share one table request and never become assets."""

from copy import deepcopy
import unittest

from PIL import Image

from marker.cloud.whole_tables import _table_math_closeups, recognize_whole_table
from test_whole_tables import FakeClient, FakeSource, logical


class CloseupClient(FakeClient):
    cloud_context_tokens = 245760
    cloud_max_output_tokens = 16384
    cloud_max_images = 3


def formula():
    atoms = [
        {"kind": "text", "text": "R", "fontname": "Times-Italic", "size": 14,
         "bbox": (10, 20, 22, 38), "source_page": 0},
        {"kind": "image", "bbox": (22, 29, 42, 39), "source_page": 0},
    ]
    return logical(atoms)


class TableMathCloseupTests(unittest.TestCase):
    def test_zoom_keeps_the_whole_expression_and_original_source_pixels(self):
        table, source = formula(), FakeSource()
        original = deepcopy(table.parts[0].grid)
        picture = Image.new("RGB", (600, 400), (10, 20, 30))
        before = picture.tobytes(), source.native.tobytes()
        client = CloseupClient("")
        images, descriptions = _table_math_closeups(source, table.parts, [picture], client)
        self.assertEqual(len(images), 2)
        self.assertIs(images[0], picture)
        self.assertEqual(source.resources, [(0, (10, 20, 42, 39))])
        self.assertEqual(descriptions[0]["part_index"], 0)
        self.assertEqual((picture.tobytes(), source.native.tobytes()), before)
        self.assertEqual(table.parts[0].grid, original)

    def test_image_cap_keeps_overview_and_adds_no_request(self):
        table, source = formula(), FakeSource()
        client = CloseupClient("")
        client.cloud_max_images = 1
        overview = Image.new("RGB", (600, 400))
        images, descriptions = _table_math_closeups(source, table.parts, [overview], client)
        self.assertEqual(images, [overview])
        self.assertEqual(descriptions, [])
        self.assertEqual(source.resources, [])
        self.assertEqual(client.calls, [])

    def test_extra_formula_image_is_in_the_same_call_and_not_in_published_pngs(self):
        client = CloseupClient(r"<table><tr><td><math>R_{\text{bun}}</math></td></tr></table>")
        result = recognize_whole_table(FakeSource(), formula(), client)
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0][1]), 2)
        self.assertIn("MATH FOCUS", client.calls[0][0])
        self.assertIn(r"R_{\text{bun}}", result.html)
        self.assertFalse(result.resources)


if __name__ == "__main__":
    unittest.main()
