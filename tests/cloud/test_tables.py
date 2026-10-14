"""Physical provenance, logical continuations and bounded visual decisions."""

import base64
import json
import unittest
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import httpx
import openai
import pdfplumber
from bs4 import BeautifulSoup
from PIL import Image

from marker.cloud.client import CloudClient
from marker.processors.table_geometry import extract_grids, grid_positions, join_grids, render_grid


FIXTURES = Path(__file__).parent / "fixtures"


def cloud_client(merge, *, max_requests=0):
    def respond(request):
        body = json.loads(request.content)
        images = [item for item in body["messages"][0]["content"] if item["type"] == "image_url"]
        if len(images) != 2:
            raise AssertionError("A continuation decision requires both actual source crops")
        for item in images:
            data = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
            with Image.open(BytesIO(data)) as image:
                if image.convert("L").getextrema()[0] >= 100:
                    raise AssertionError("The decision crop lost the source table")
                if max(image.size) > 1536:
                    raise AssertionError("The decision bypassed bounded request copies")
        return httpx.Response(200, json={
            "id": "offline-table-decision", "object": "chat.completion", "created": 0,
            "model": "offline", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({"merge": merge})}}],
        })
    sdk = openai.OpenAI(api_key="offline-dummy", base_url="https://offline.invalid/v1",
                       http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    return CloudClient({"cloud_max_requests": max_requests}, client=sdk)


class TableGeometryContinuationTests(unittest.TestCase):
    def test_joined_atoms_and_cells_keep_physical_coordinates_after_three_parts(self):
        with pdfplumber.open(FIXTURES / "three_page_continuation.pdf") as pdf:
            grids = [extract_grids(page)[0] for page in pdf.pages]
        physical = deepcopy(grids)
        first_join = join_grids(grids[0], grids[1])
        self.assertIsNotNone(first_join)
        merged = join_grids(first_join, grids[2])
        self.assertIsNotNone(merged)
        self.assertEqual(grids, physical, "Joining must not mutate physical parts")
        atoms = [atom for cell in merged["cells"] for atom in cell["atoms"]]
        expected = [(atom["source_page"], tuple(atom["bbox"]))
                    for grid in physical for cell in grid["cells"] for atom in cell["atoms"]]
        self.assertCountEqual([(atom["source_page"], tuple(atom["bbox"])) for atom in atoms], expected)
        self.assertTrue(all(atom["source_bbox"] == tuple(atom["bbox"]) for atom in atoms))
        self.assertEqual(merged["page"], 0)
        self.assertEqual(merged["last_page"], 2)
        self.assertEqual(merged["bbox"], (30, 280, 330, 380))
        self.assertEqual(merged["layout_bbox"], (30, 280, 330, 870))
        self.assertIsNotNone(grid_positions(merged))
        text = BeautifulSoup(render_grid(merged), 'html.parser').get_text(' ', strip=True)
        self.assertIn("state continues", text)
        self.assertIn("ends", text)


class LogicalTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from marker.cloud.source import PdfSource
        from marker.cloud.tables import detect_tables, group_tables
        cls.PdfSource = PdfSource
        cls.detect_tables = staticmethod(detect_tables)
        cls.group_tables = staticmethod(group_tables)

    def test_split_header_and_three_page_continuation_keep_all_source_parts(self):
        with self.PdfSource(FIXTURES / "three_page_continuation.pdf") as source:
            parts = self.detect_tables(source)
            self.assertEqual([part.page for part in parts], [0, 1, 2, 2])
            self.assertEqual([part.table_id for part in parts], ["7", "7", "7", "8"])
            before = deepcopy([part.grid for part in parts])
            tables = self.group_tables(parts, None)
            self.assertEqual([len(table.parts) for table in tables], [3, 1])
            self.assertEqual([part.grid for part in parts], before)
            table = tables[0]
            self.assertFalse(table.warnings)
            header = next(cell for cell in table.grid["cells"] if cell["bbox"][:2] == (30, 280))
            self.assertEqual(len(header["source_boxes"]), 3)
            self.assertEqual(header["layout_bbox"], (30, 280, 130, 770))
            for part in table.parts:
                self.assertEqual(part.page_size, (360, 400))
                for cell in part.grid["cells"]:
                    for atom in cell["atoms"]:
                        crop = source.crop(atom["source_page"], atom["source_bbox"], dpi=144)
                        self.assertGreater(min(crop.size), 0)
                        self.assertLess(crop.convert("L").getextrema()[0], 240)

    def test_unrelated_captioned_tables_stay_separate_despite_identical_columns(self):
        with self.PdfSource(FIXTURES / "unrelated_tables.pdf") as source:
            parts = self.detect_tables(source)
            self.assertEqual([part.page for part in parts], [0, 1, 1])
            self.assertEqual([part.table_id for part in parts], ["10", "11", "12"])
            client = cloud_client(True)
            tables = self.group_tables(parts, client)
            self.assertEqual([len(table.parts) for table in tables], [1, 1, 1])
            self.assertEqual(client.request_count, 0)

    def test_finer_header_columns_on_following_page_do_not_break_continuation(self):
        with self.PdfSource(FIXTURES / "refined_header.pdf") as source:
            parts = self.detect_tables(source)
            self.assertEqual([len(grid_positions(part.grid)[0])-1 for part in parts], [2, 3])
            tables = self.group_tables(parts, None)
            self.assertEqual([len(table.parts) for table in tables], [2])
            self.assertEqual(len(grid_positions(tables[0].grid)[0])-1, 3)
            html = BeautifulSoup(render_grid(tables[0].grid), 'html.parser')
            self.assertIsNotNone(html.find('td', colspan='2'))
            self.assertIn('Header begins Header ends', html.get_text(' ', strip=True))

    def test_reused_numbers_with_conflicting_caption_titles_stay_separate(self):
        with self.PdfSource(FIXTURES / "conflicting_captions.pdf") as source:
            parts = self.detect_tables(source)
            client = cloud_client(True)
            tables = self.group_tables(parts, client)
            self.assertEqual([len(table.parts) for table in tables], [1, 1])
            self.assertEqual(client.request_count, 0)

    def test_multiple_borderless_tables_are_enumerated_with_untrusted_inferred_grids(self):
        with self.PdfSource(FIXTURES / "borderless_tables.pdf") as source:
            parts = self.detect_tables(source)
            self.assertEqual(len(parts), 2)
            self.assertEqual([part.table_id for part in parts], ["21", "22"])
            for part in parts:
                self.assertFalse(part.grid["trusted"])
                xs, ys, positions = grid_positions(part.grid)
                self.assertEqual((len(xs)-1, len(ys)-1), (3, 4))
                self.assertEqual(len(positions), 12)
                self.assertTrue(all(cell["atoms"] for cell in part.grid["cells"]))

    def test_adjacent_open_tables_cannot_borrow_each_others_connected_sides(self):
        with self.PdfSource(FIXTURES / "open_neighbours.pdf") as source:
            parts = self.detect_tables(source)
            self.assertEqual([part.bbox for part in parts], [(20, 80, 100, 180), (150, 80, 230, 180)])
            self.assertEqual([len(grid_positions(part.grid)[0])-1 for part in parts], [2, 2])

    def test_ambiguous_closed_tables_require_bounded_visual_evidence(self):
        for merge, expected in ((True, [2]), (False, [1, 1])):
            with self.subTest(merge=merge), self.PdfSource(FIXTURES / "ambiguous_tables.pdf") as source:
                parts = self.detect_tables(source)
                client = cloud_client(merge)
                tables = self.group_tables(parts, client)
                self.assertEqual([len(table.parts) for table in tables], expected)
                self.assertEqual(client.request_count, 1)
                self.assertTrue(all(not table.warnings for table in tables))

    def test_continuation_checks_obey_shared_document_request_cap(self):
        with self.PdfSource(FIXTURES / "ambiguous_tables.pdf") as source:
            client = cloud_client(True, max_requests=1)
            parts = self.detect_tables(source)
            self.assertEqual([len(t.parts) for t in self.group_tables(parts, client)], [2])
            exhausted = self.group_tables(parts, client)
            self.assertEqual([len(t.parts) for t in exhausted], [1, 1])
            self.assertEqual(client.request_count, 1)
            self.assertTrue(all(t.warnings for t in exhausted))


if __name__ == "__main__":
    unittest.main()
