"""One compact normalized discovery job per scanned source rectangle."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from marker.cloud.context import RecognitionFailure
from marker.cloud.regions import Region, assess_text_quality
from marker.cloud.scan_regions import discover_scan
from marker.cloud.tables import group_tables


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


if __name__ == "__main__":
    unittest.main()
