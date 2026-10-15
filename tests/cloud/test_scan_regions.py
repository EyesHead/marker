"""One compact normalized discovery job per scanned source rectangle."""

from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from marker.cloud.context import RecognitionFailure
from marker.cloud.regions import Region, assess_text_quality
from marker.cloud.scan_regions import discover_scan
from marker.cloud.tables import TablePart, group_tables


class Source:
    def __init__(self, size=(360, 400)):
        self.size = size
        self.crops = []

    def page_size(self, _page):
        return self.size

    def crop(self, page, box, dpi=None):
        self.crops.append((page, tuple(box), dpi))
        return Image.new("RGB", (240, 280), (17, 31, 49))


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, prompt, images, schema):
        self.calls.append((prompt, images, schema))
        return self.responses.pop(0)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.writes = []

    def write_text(self, name, text):
        if "DUMMY_CREDENTIAL" in text:
            raise RecognitionFailure("Fixture storage guard rejected a credential")
        self.writes.append(name)
        (self.root / name).write_text(text, encoding="utf-8")


def parent(page=0, box=(0, 0, 360, 400)):
    atom = {
        "kind": "image",
        "bbox": box,
        "source_bbox": box,
        "source_page": page,
        "image_index": 0,
    }
    return Region(page, box, "image", (atom,), assess_text_quality([], box))


def response(
    kind="table", box=(100, 600, 900, 1000), *, caption="Table A.1", continuation=False
):
    return {
        "complete": True,
        "regions": [
            {
                "kind": kind,
                "bbox": list(box),
                "caption": caption,
                "continuation": continuation,
            }
        ],
    }


class ScanRegionTests(unittest.TestCase):
    def test_two_raster_parts_scale_to_same_table_and_group_without_cells(self):
        source = Source()
        client = Client(
            [response(), response(box=(100, 0, 900, 350), continuation=True)]
        )
        first, first_parts = discover_scan(source, parent(0), client)
        second, second_parts = discover_scan(source, parent(1), client)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(first_parts[0].bbox, (36, 240, 324, 400))
        self.assertEqual(second_parts[0].bbox, (36, 0, 324, 140))
        self.assertEqual(first_parts[0].table_id, second_parts[0].table_id)
        self.assertEqual(first_parts[0].table_id, "a.1")
        self.assertTrue(second_parts[0].is_continuation)
        self.assertIsNone(first_parts[0].grid)
        self.assertIs(first_parts[0].source, source)
        groups = group_tables(first_parts + second_parts, Client([]))
        self.assertEqual([group.page_ids for group in groups], [(0, 1)])
        self.assertTrue(
            all(
                region.kind == "table" and region.grid is None
                for region in first + second
            )
        )

    def test_unlabelled_raster_continuations_use_shared_fake_visual_client(self):
        source = Source()
        client = Client(
            [
                response(caption=""),
                response(box=(100, 0, 900, 350), caption="", continuation=True),
                {"merge": True},
            ]
        )
        _, first = discover_scan(source, parent(0), client)
        _, second = discover_scan(source, parent(1), client)
        groups = group_tables(first + second, client)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(groups[0].page_ids, (0, 1))
        self.assertEqual(len(client.calls[-1][1]), 2)

    def test_offset_crop_preserves_all_region_kinds_and_source_coordinates(self):
        source = Source((600, 800))
        original = parent(3, (100, 200, 500, 700))
        snapshot = deepcopy(original.__dict__)
        client = Client(
            [
                {
                    "complete": True,
                    "regions": [
                        {"kind": "text", "bbox": [0, 0, 200, 200]},
                        {"kind": "math", "bbox": [250, 200, 750, 400]},
                        {"kind": "figure", "bbox": [0, 600, 1000, 1000]},
                    ],
                }
            ]
        )
        regions, tables = discover_scan(source, original, client)
        self.assertEqual(original.__dict__, snapshot)
        self.assertEqual(tables, [])
        self.assertEqual(
            [region.kind for region in regions], ["text", "math", "figure"]
        )
        self.assertEqual(
            [region.bbox for region in regions],
            [(100, 200, 180, 300), (200, 300, 400, 400), (100, 500, 500, 700)],
        )
        self.assertFalse(regions[0].quality.trusted)
        for region in regions:
            atom = region.atoms[0]
            self.assertEqual(
                (atom["kind"], atom["content_kind"]), ("unknown", region.kind)
            )
            self.assertEqual(
                (atom["source_page"], atom["source_bbox"]), (3, region.bbox)
            )
        self.assertEqual(source.crops, [(3, original.bbox, 200)])

    def test_prompt_and_schema_have_only_normalized_region_layout(self):
        client = Client([response()])
        discover_scan(Source(), parent(), client)
        prompt, images, schema = client.calls[0]
        self.assertIn("0..1000", prompt)
        self.assertIn("normalized", prompt.lower())
        self.assertNotIn('"page"', prompt)
        self.assertNotIn("360", prompt)
        self.assertNotIn("400", prompt)
        definition = schema.model_json_schema()
        fields = definition["$defs"][next(iter(definition["$defs"]))]["properties"]
        self.assertEqual(set(fields), {"kind", "bbox", "caption", "continuation"})
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].getpixel((0, 0)), (17, 31, 49))

    def test_incomplete_overlap_outside_and_extra_geometry_are_rejected(self):
        invalid = []
        item = response()
        item["complete"] = False
        invalid.append(item)
        item = response()
        item["complete"] = "true"
        invalid.append(item)
        item = response()
        item["regions"] = []
        invalid.append(item)
        item = response()
        item["regions"].append(deepcopy(item["regions"][0]))
        invalid.append(item)
        item = response(box=(-1, 0, 1000, 1000))
        invalid.append(item)
        item = response(box=(0, 0, 1001, 1000))
        invalid.append(item)
        item = response(box=(1, 0, 1, 1000))
        invalid.append(item)
        item = response(box=(0, 0, float("nan"), 1000))
        invalid.append(item)
        item = response()
        item["regions"][0]["cells"] = []
        invalid.append(item)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(RecognitionFailure):
                discover_scan(Source(), parent(), Client([item]))

    def test_touching_regions_are_valid(self):
        item = {
            "complete": True,
            "regions": [
                {"kind": "text", "bbox": [0, 0, 500, 1000]},
                {"kind": "figure", "bbox": [500, 0, 1000, 1000]},
            ],
        }
        regions, _ = discover_scan(Source(), parent(), Client([item]))
        self.assertEqual(len(regions), 2)

    def test_full_normalized_crop_keeps_exact_parent_edges(self):
        box = (32.38327648331624, 56.78, 183.23245040781813, 700.2)
        regions, _ = discover_scan(
            Source((600, 800)),
            parent(0, box),
            Client([response(kind="text", box=(0, 0, 1000, 1000))]),
        )
        self.assertEqual(regions[0].bbox, box)

    def test_actual_scanned_pdf_uses_its_source_pixels_and_untrusted_text(self):
        from marker.cloud.source import PdfSource

        fixture = Path(__file__).parent / "fixtures" / "scan.pdf"
        with PdfSource(fixture) as source:
            box = (0, 0, *source.page_size(0))
            expected = source.crop(0, box, dpi=200)
            client = Client([response(kind="text", box=(0, 0, 1000, 1000))])
            regions, parts = discover_scan(source, parent(0, box), client)
            self.assertEqual(parts, [])
            self.assertEqual(client.calls[0][1][0].tobytes(), expected.tobytes())
            self.assertLess(expected.convert("L").getextrema()[0], 240)
        self.assertEqual(regions[0].bbox, box)
        self.assertFalse(regions[0].quality.trusted)
        self.assertEqual(
            client.calls[0][1][0].getpixel((0, 0)), expected.getpixel((0, 0))
        )

    def test_valid_cache_skips_request_and_render_and_keeps_other_parent_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Source()
            store = Store(temporary)
            client = Client(
                [response(), response(box=(100, 0, 900, 350), continuation=True)]
            )
            expected, _ = discover_scan(source, parent(0), client, store=store)
            discover_scan(source, parent(1), client, store=store)
            cache = json.loads((store.root / "scan-layout.json").read_text())
            self.assertEqual(len(cache), 2)
            before = len(source.crops)
            actual, _ = discover_scan(source, parent(0), Client([]), store=store)
            self.assertEqual(
                [r.__dict__ for r in actual], [r.__dict__ for r in expected]
            )
            self.assertEqual(len(source.crops), before)
            self.assertEqual(store.writes, ["scan-layout.json", "scan-layout.json"])

    def test_malformed_cache_containers_and_cached_geometry_are_recovered(self):
        for corrupt in ([], True, "not a map", {"bad": []}, {"bad": response()}):
            with (
                self.subTest(corrupt=corrupt),
                tempfile.TemporaryDirectory() as temporary,
            ):
                store = Store(temporary)
                (store.root / "scan-layout.json").write_text(json.dumps(corrupt))
                client = Client([response()])
                discover_scan(Source(), parent(), client, store=store)
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(
                    len(json.loads((store.root / "scan-layout.json").read_text())), 1
                )
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            discover_scan(Source(), parent(), Client([response()]), store=store)
            cached = json.loads((store.root / "scan-layout.json").read_text())
            cached[next(iter(cached))]["complete"] = False
            (store.root / "scan-layout.json").write_text(json.dumps(cached))
            client = Client([response()])
            discover_scan(Source(), parent(), client, store=store)
            self.assertEqual(len(client.calls), 1)

    def test_cache_write_uses_storage_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            with self.assertRaisesRegex(RecognitionFailure, "storage guard"):
                discover_scan(
                    Source(),
                    parent(),
                    Client([response(caption="DUMMY_CREDENTIAL")]),
                    store=store,
                )
            self.assertFalse((store.root / "scan-layout.json").exists())


class ScanOwnerReconciliationTests(unittest.TestCase):
    @staticmethod
    def owner(page, box, kind, atoms=(), grid=None):
        return Region(page, box, kind, tuple(atoms), assess_text_quality([], box), grid=grid)

    @staticmethod
    def word(page, box, text):
        return {"kind": "text", "bbox": box, "source_bbox": box,
                "source_page": page, "text": text}

    @staticmethod
    def frame(page, box):
        left, top, right, bottom = box
        lines = [(left, top, left, bottom), (right, top, right, bottom)]
        lines.extend((left, top + (bottom - top) * index / 52, right,
                      top + (bottom - top) * index / 52) for index in range(53))
        return tuple({"kind": "vector", "bbox": line, "source_bbox": line,
                      "source_page": page,
                      "crop_bbox": (line[0] - .5, line[1] - .5,
                                    line[2] + .5, line[3] + .5)} for line in lines)

    def test_larger_ruled_table_replaces_partial_grid_and_owns_native_atoms_once(self):
        from marker.converters.ollama_document import _discover_scans

        # Bounds and saved normalized layout from the failed document's page 60.
        source = Source((842.04, 595.32))
        page = 60
        box = (84.244, 56.16, 753.61997, 373.89)
        vectors = self.frame(page, (84.744, 56.66, 753.11997, 373.39))
        vector_owner = self.owner(page, box, "unknown", vectors)
        image_box = (113.05, 62.3, 169.3, 192.85)
        image_atom = {"kind": "image", "bbox": image_box, "source_bbox": image_box,
                      "source_page": page, "image_index": 0}
        image_owner = self.owner(page, image_box, "image", (image_atom,))
        title = self.word(page, (218.78, 63.71, 742.606, 72.71), "Коэффициенты")
        title_owner = self.owner(page, title["bbox"], "text", (title,))
        cell = self.word(page, (225, 210, 250, 219), "0,42")
        grid = {"bbox": (219.3, 203.58, 744.606, 368.88),
                "cells": [{"id": "c0", "atoms": [cell]}]}
        native = TablePart(page, grid["bbox"], grid, source.size)
        native_owner = self.owner(page, native.bbox, "table", (cell,), native.grid)
        outside = self.word(page, (100, 390, 200, 400), "Следующая таблица")
        outside_owner = self.owner(page, outside["bbox"], "text", (outside,))
        original = [vector_owner, image_owner, title_owner, native_owner, outside_owner]
        snapshot = deepcopy([(region.bbox, region.atoms) for region in original])
        client = Client([{"complete": True, "regions": [
            {"kind": "figure", "bbox": [48, 44, 135, 325]},
            {"kind": "table", "bbox": [201, 44, 993, 965]},
        ]}])

        owners, parts, reasons = _discover_scans(source, original, [native], client, None)

        self.assertEqual(reasons, [])
        self.assertEqual(len(client.calls), 1, "The diagram is already owned by its whole table")
        self.assertEqual(len(parts), 1)
        self.assertIsNone(parts[0].grid, "A partial native grid cannot be stretched into a full grid")
        for actual, expected in zip(parts[0].bbox, box):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertEqual(native.grid, grid)
        self.assertIn(outside_owner, owners)
        table = next(region for region in owners if region.kind == "table")
        self.assertIsNone(table.grid)
        self.assertTrue(all(atom in table.atoms for atom in (*vectors, image_atom, title, cell)))
        original_atoms = [atom for region in original for atom in region.atoms]
        counts = Counter(id(atom) for region in owners for atom in region.atoms)
        self.assertTrue(all(counts[id(atom)] == 1 for atom in original_atoms))
        self.assertEqual([(region.bbox, region.atoms) for region in original], snapshot)

    def test_saved_page_eight_bounds_keep_partial_native_cell_beyond_model_edge(self):
        from marker.converters.ollama_document import _discover_scans

        source = Source((842.04, 595.32))
        page = 8
        box = (84.244, 56.16, 794.78003, 532.1)
        vectors = self.frame(page, (84.744, 56.66, 794.28003, 531.6))
        broad = self.owner(page, box, "unknown", vectors)
        cell = self.word(page, (740, 400, 751.566, 409), "ячейка")
        native = TablePart(page, (106.98, 367.022, 751.566, 423.51),
                           {"cells": [{"atoms": [cell]}]}, source.size)
        old = self.owner(page, native.bbox, "table", (cell,), native.grid)
        client = Client([response(box=(0, 0, 935, 993), caption="Колонны")])

        owners, parts, reasons = _discover_scans(source, [broad, old], [native], client, None)

        self.assertEqual(reasons, [])
        self.assertEqual(len(parts), 1)
        for actual, expected in zip(parts[0].bbox, box):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertIsNone(parts[0].grid)
        self.assertEqual(sum(atom is cell for region in owners for atom in region.atoms), 1)

    def test_disjoint_vector_support_does_not_make_its_envelope_a_crossing_owner(self):
        from marker.converters.ollama_document import _discover_scans

        source = Source((600, 800))
        bitmap = parent(0, (100, 100, 300, 300))
        atoms = self.frame(0, (10, 10, 20, 20))[:2] + self.frame(0, (380, 380, 390, 390))[:2]
        envelope = self.owner(0, (9.5, 9.5, 390.5, 390.5), "unknown", atoms)
        client = Client([response(kind="figure", box=(0, 0, 1000, 1000), caption="")])

        owners, parts, reasons = _discover_scans(source, [envelope, bitmap], [], client, None)

        self.assertEqual(reasons, [])
        self.assertEqual(parts, [])
        self.assertEqual(len(client.calls), 1)
        self.assertIn(envelope, owners)
        self.assertTrue(any(owner.kind == "figure" and owner.bbox == bitmap.bbox for owner in owners))

    def test_crossing_paragraph_partitions_whole_atoms_without_changing_coordinates(self):
        from marker.converters.ollama_document import _discover_scans

        source = Source((400, 500))
        bitmap = parent(0, (100, 100, 300, 300))
        inside = self.word(0, (110, 110, 180, 120), "подпись")
        outside = self.word(0, (50, 210, 90, 220), "снаружи")
        prose = self.owner(0, (50, 110, 180, 220), "text", (inside, outside))
        client = Client([response(kind="figure", box=(0, 0, 1000, 1000), caption="")])

        owners, _, reasons = _discover_scans(source, [bitmap, prose], [], client, None)

        self.assertEqual(reasons, [])
        figure = next(region for region in owners if region.kind == "figure")
        self.assertIn(inside, figure.atoms)
        residual = next(region for region in owners if region.kind == "text")
        self.assertEqual(residual.atoms, (outside,))
        self.assertEqual(residual.bbox, outside["bbox"])
        self.assertIs(residual.atoms[0], outside)
        self.assertEqual(prose.atoms, (inside, outside))

    def test_embedded_bitmap_layout_validation_failure_routes_original_whole_object(self):
        from marker.converters.ollama_document import _discover_scans

        source = Source((842.04, 595.32))
        bitmap = parent(5, (254.05, 202.7, 630.6, 455.4))
        with patch("marker.cloud.scan_regions.discover_scan", side_effect=RecognitionFailure("JSON: value_error")):
            owners, parts, reasons = _discover_scans(source, [bitmap], [], None, None)

        self.assertEqual(reasons, [])
        self.assertEqual(parts, [])
        self.assertEqual(owners, [bitmap])
        self.assertIs(owners[0], bitmap)


if __name__ == "__main__":
    unittest.main()
