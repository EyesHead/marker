"""Source ownership discovery contracts, using offline clients and real PDFs."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image

from marker.cloud.context import RecognitionFailure
from marker.cloud.native_discovery import discover_native_owners
from marker.cloud.source import NativeImage, PdfSource, intersection
from marker.cloud.tables import group_tables
from marker.schema.blocks import Equation, PageFooter, PageHeader, Picture, Table, Text
from marker.schema.text.line import Line
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox


FIXTURES = Path(__file__).parent / "fixtures"


class Source:
    def __init__(self, words=(), images=(), *, size=(360, 400), blank=True):
        self.words, self.images, self.size = list(words), list(images), size
        self.blank = blank

    def page_size(self, page):
        return self.size

    def page_words(self, page, bbox=None):
        return deepcopy([word for word in self.words if word.get("source_page", 0) == page
                         and (bbox is None or intersection(word["bbox"], bbox) > 0)])

    def page_images(self, page):
        return [image for image in self.images if image.page == page]

    def crop(self, page, box, dpi=None):
        return Image.new("RGB", (360, 400), "white" if self.blank else "black")


class Client:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def request(self, prompt, images, schema):
        self.calls.append((prompt, images, schema))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def layout(*items):
    return {"complete": True, "regions": [
        {"kind": kind, "bbox": list(box), "caption": caption, "continuation": continuation}
        for kind, box, caption, continuation in items
    ]}


def page(number=0, *, native=(0, 0, 360, 400)):
    return PageGroup(page_id=number, polygon=PolygonBox.from_bbox(native),
                     children=[], structure=[], lowres_image=Image.new("RGB", (360, 400), "white"))


def add(page, cls, box, **kwargs):
    result = page.add_full_block(cls(page_id=page.page_id, polygon=PolygonBox.from_bbox(box), **kwargs))
    page.add_structure(result)
    return result


def picture(box, page=0, index=0):
    return NativeImage(page, index, tuple(box), 120, 80, tuple(box), index)


def word(value, box):
    return {"kind": "text", "text": value, "bbox": tuple(box), "size": 10, "fontname": "ArialMT"}


def document(*pages):
    return Document(filepath="offline.pdf", pages=list(pages))


class NativeDiscoveryTests(unittest.TestCase):
    def test_orphan_operator_and_operand_extend_native_equation_as_math(self):
        current = page()
        equation = add(current, Equation, (20, 30, 70, 48))
        operator = (75, 30, 91, 46)
        source = Source([word("Run", (20, 30, 60, 45)), word("0", (98, 30, 106, 45))],
                        [picture(operator)])
        original = deepcopy(source.words)
        owners, _, regions = discover_native_owners(document(current), source, [equation], Client(), None)
        self.assertEqual(len(owners), 1)
        self.assertIsInstance(owners[0], Equation)
        region = regions[owners[0].id]
        self.assertEqual(region.kind, "math")
        self.assertCountEqual([a.get("text") for a in region.atoms if a["kind"] == "text"], ["Run", "0"])
        self.assertEqual(next(a["source_bbox"] for a in region.atoms if a["kind"] == "image"), operator)
        self.assertEqual(source.words, original)

    def test_orphan_prose_cannot_inherit_neighboring_equation_semantics(self):
        current = page()
        equation = add(current, Equation, (20, 30, 70, 48))
        source = Source([word("Run", (20, 30, 60, 45)), word("значение", (98, 30, 150, 45))],
                        [picture((75, 30, 91, 46))])
        owners, _, regions = discover_native_owners(document(current), source, [equation], Client(), None)
        self.assertEqual(len(owners), 1)
        self.assertIsInstance(owners[0], Text)
        self.assertEqual(regions[owners[0].id].kind, "text")
        self.assertIn("значение", [a.get("text") for a in regions[owners[0].id].atoms])

    def test_actual_page_26_table_note_falsely_labelled_footer_is_kept(self):
        current = page(native=(0, 0, 842.04, 595.32))
        footer = add(current, PageFooter, (81.57164578966362, 522.0431663188144,
                                         96.84358960145991, 532.1221895274322))
        body = add(current, Text, (111.06599065249178, 506.19972043872457,
                                  767.7277635000613, 520.4965300564032))
        note_box = (86.424, 523.0500000000001, 96.95400000000001, 532.0500000000001)
        source = Source([word("типа", (745.225, 509.13, 764.143, 518.13)),
                         word("2).", note_box)], size=(842.04, 595.32))
        source.pdf_page = lambda _: SimpleNamespace(edges=[
            {"bbox": (85.464, 56.66, 765.244, 56.66)},
            {"bbox": (85.464, 537.84, 765.244, 537.84)},
            {"bbox": (84.744, 63.14, 84.744, 537.84)},
            {"bbox": (765.96, 63.14, 765.96, 537.84)},
        ], curves=[])
        before = footer.polygon.model_dump()
        client = Client(layout(("table", (0, 0, 1000, 1000), "Table 8", True)))
        owners, parts, regions = discover_native_owners(document(current), source, [body], client, None)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(parts), 1)
        self.assertIsInstance(owners[0], Table)
        self.assertCountEqual([a["text"] for a in regions[owners[0].id].atoms if a["kind"] == "text"],
                              ["типа", "2)."])
        self.assertEqual(next(a["source_bbox"] for a in regions[owners[0].id].atoms
                              if a.get("text") == "2)."), note_box)
        self.assertEqual(footer.polygon.model_dump(), before)

    def test_actual_footer_outside_table_frame_remains_excluded(self):
        current = page()
        footer = add(current, PageFooter, (150, 370, 180, 390))
        native = add(current, Table, (40, 40, 330, 330))
        source = Source([word("Page 1", (150, 370, 180, 390)), word("Body", (80, 80, 120, 95))])
        source.pdf_page = lambda _: SimpleNamespace(edges=[
            {"bbox": (40, 40, 330, 40)}, {"bbox": (40, 330, 330, 330)},
            {"bbox": (40, 40, 40, 330)}, {"bbox": (330, 40, 330, 330)},
        ], curves=[])
        owners, parts, regions = discover_native_owners(document(current), source, [native], Client(), None)
        self.assertEqual(owners, [native])
        self.assertFalse(regions)
        self.assertIn(native.id, parts)
        self.assertIn(footer.id, current.structure)

    def test_actual_duplicate_text_equation_keeps_one_math_owner(self):
        current = page()
        box = (20, 30, 200, 50)
        prose = add(current, Text, box)
        equation = add(current, Equation, box)
        line = current.add_full_block(Line(page_id=0, polygon=PolygonBox.from_bbox(box), formats=["math"]))
        equation.add_structure(line)
        source = Source([word("Кн", (25, 33, 45, 45))])
        owners, _, regions = discover_native_owners(document(current), source, [prose, equation], Client(), None)
        self.assertEqual(len(owners), 1)
        self.assertIsInstance(owners[0], Equation)
        self.assertEqual(regions[owners[0].id].kind, "math")
        self.assertEqual([a["text"] for a in regions[owners[0].id].atoms], ["Кн"])
        self.assertIn(line.id, owners[0].structure)
        self.assertTrue(prose.ignore_for_output)
        self.assertTrue(equation.ignore_for_output)

    def test_coalesced_equation_and_prose_keep_whole_article_scope(self):
        current = page()
        box = (20, 30, 200, 50)
        prose = add(current, Text, box)
        equation = add(current, Equation, (50, 30, 200, 50))
        source = Source([word("для", (25, 33, 45, 45)), word("x", (50, 33, 60, 45))])
        owners, _, regions = discover_native_owners(document(current), source, [prose, equation], Client(), None)
        self.assertEqual(len(owners), 1)
        self.assertIsInstance(owners[0], Equation)
        self.assertEqual(regions[owners[0].id].kind, "text")
        self.assertEqual([a["text"] for a in regions[owners[0].id].atoms], ["для", "x"])

    def test_actual_page_12_duplicate_first_line_joins_its_paragraph(self):
        current = page(native=(0, 0, 842.04, 595.32))
        first = add(current, Text, (108.8693186987203, 285.3025872297092, 736.3776269319599, 296.466196514211))
        following = add(current, Text, (80.39021782094333, 289.04959947067067, 800.3287019842136, 321.43445298533527))
        source = Source([word("5.9", (113.54, 288.24160000000006, 127.3844, 298.2016000000001)),
                         word("условия", (85, 310, 130, 320))], size=(842.04, 595.32))
        originals = [owner.polygon.model_dump() for owner in (first, following)]
        owners, _, regions = discover_native_owners(document(current), source, [first, following], Client(), None)
        self.assertEqual(len(owners), 1)
        self.assertEqual([a["text"] for a in regions[owners[0].id].atoms], ["5.9", "условия"])
        self.assertEqual([owner.polygon.model_dump() for owner in (first, following)], originals)

    def test_actual_page_55_tall_raster_formula_belongs_to_its_short_text_row(self):
        current = page(native=(0, 0, 842.04, 595.32))
        native = add(current, Text, (216.49317848530276, 498.75530342347827, 417.8216091656653, 507.2865591291307))
        box = (259.3, 492.70000000000005, 276.6, 510.70000000000005)
        source = Source([word("Значение", (216.74, 498.186, 256.79, 507.186))],
                        [picture(box)], size=(842.04, 595.32))
        owners, _, regions = discover_native_owners(document(current), source, [native], Client(), None)
        self.assertEqual(owners, [native])
        self.assertEqual(regions[native.id].bbox[1:4:2], (box[1], box[3]))
        self.assertEqual(next(a for a in regions[native.id].atoms if a["kind"] == "image")["source_bbox"], box)

    def test_actual_page_27_diagram_crop_includes_complete_source_bitmap(self):
        current = page(native=(0, 0, 842.04, 595.32))
        native = add(current, Picture, (338.8592704852562, 205.0181836132618, 536.6870140810215, 485.6063904008083))
        box = (344.75, 217.45000000000005, 539.75, 478.45003051800006)
        source = Source(images=[picture(box)], size=(842.04, 595.32))
        client = Client(layout(("figure", (0, 0, 1000, 1000), "", False)))
        owners, _, regions = discover_native_owners(document(current), source, [native], client, None)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(regions[owners[0].id].bbox[2], box[2])
        self.assertTrue(native.ignore_for_output)
        self.assertEqual(source.images[0].bbox, box)

    def test_actual_page_58_coarse_table_crop_keeps_complete_connected_frame(self):
        current = page(native=(0, 0, 842.04, 595.32))
        native = add(current, Table, (80.75183531353109, 51.83865684772119, 819.5451309380859, 249.7598361270361))
        original = native.polygon.model_dump()
        source = Source(size=(842.04, 595.32))
        source.pdf_page = lambda _: SimpleNamespace(edges=[
            {"bbox": (84.744, 52, 820.55997, 52)}, {"bbox": (84.744, 253.25, 820.55997, 253.25)},
            {"bbox": (84.744, 52, 84.744, 253.25)}, {"bbox": (820.55997, 52, 820.55997, 253.25)},
        ], curves=[])
        owners, parts, _ = discover_native_owners(document(current), source, [native], Client(), None)
        self.assertEqual(owners, [native])
        self.assertGreaterEqual(parts[native.id].bbox[3], 253.5)
        self.assertIsNone(parts[native.id].grid)
        self.assertEqual(native.polygon.model_dump(), original)

    def test_long_unowned_frame_crossing_text_becomes_one_discovered_table(self):
        current = page()
        first = add(current, Text, (50, 100, 300, 120))
        second = add(current, Text, (50, 160, 300, 180))
        source = Source([word("first", (50, 100, 80, 115)), word("second", (50, 160, 90, 175))])
        source.pdf_page = lambda _: SimpleNamespace(edges=[
            {"bbox": (50, 40, 330, 40)}, {"bbox": (50, 220, 330, 220)},
            {"bbox": (50, 40, 50, 220)}, {"bbox": (330, 40, 330, 220)},
        ], curves=[])
        client = Client(layout(("table", (0, 0, 1000, 1000), "Table 6", False)))
        owners, parts, regions = discover_native_owners(document(current), source, [first, second], client, None)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(owners), 1)
        self.assertIsInstance(owners[0], Table)
        self.assertEqual(len(parts), 1)
        self.assertEqual([a["text"] for a in regions[owners[0].id].atoms if a["kind"] == "text"], ["first", "second"])

    def test_actual_page_11_formula_overhang_expands_job_not_source_atom(self):
        current = page(native=(0, 0, 842.04, 595.32))
        native_box = (81.27852126907139, 205.3961741114176, 803.0183997709812, 232.47606641003836)
        native = add(current, Text, native_box)
        bitmap = (261.35, 217.70000000000005, 278.65000000000003, 234.95000000000005)
        source = Source([word("текст", (82, 206, 120, 216))], [picture(bitmap)], size=(842.04, 595.32))
        original = native.polygon.model_dump()
        owners, _, regions = discover_native_owners(document(current), source, [native], Client(), None)
        self.assertEqual(owners, [native])
        self.assertEqual(regions[native.id].bbox[3], bitmap[3])
        self.assertEqual(next(a for a in regions[native.id].atoms if a["kind"] == "image")["source_bbox"], bitmap)
        self.assertEqual(native.polygon.model_dump(), original)

    def test_actual_page_16_formula_descender_does_not_enter_next_paragraph(self):
        current = page(native=(0, 0, 842.04, 595.32))
        equation = add(current, Equation, (534.9227316864664, 422.02769027478166, 802.5102257387164, 438.8103771304334))
        prose = add(current, Text, (109.81741977787394, 439.3524692854527, 648.6131707447367, 453.67340500932653))
        bitmap = (538.7, 423.25000000000006, 603.95, 441.95000000000005)
        source = Source([word("(1)", (787.32, 428.6716, 799.64052, 438.63160000000005)),
                         word("Значения", (113.54, 443.5516, 157.83212000000003, 453.51160000000004))],
                        [picture(bitmap)], size=(842.04, 595.32))
        _, _, regions = discover_native_owners(document(current), source, [equation, prose], Client(), None)
        self.assertEqual(regions[equation.id].kind, "math")
        self.assertTrue(any(a["kind"] == "image" for a in regions[equation.id].atoms))
        self.assertFalse(any(a["kind"] == "image" for a in regions[prose.id].atoms))
        self.assertEqual([a["text"] for a in regions[prose.id].atoms], ["Значения"])

    def test_actual_page_64_overlapping_layout_headers_keep_unique_words(self):
        current = page(native=(0, 0, 842.04, 595.32))
        first = add(current, Text, (352.84908787117826, 396.6001517511716, 531.7724416830919, 410.1630910001195))
        second = add(current, Text, (80.94907653807245, 407.97081360747796, 428.3159310583993, 421.6699517851136))
        source = Source([word("К", (357.77, 399.99160000000006, 363.8456, 409.95160000000004)),
                         word("расчету", (366.64436, 399.99160000000006, 405.61784, 409.95160000000004)),
                         word("Таблица", (85.104, 411.51160000000004, 125.04360000000001, 421.4716000000001))],
                        size=(842.04, 595.32))
        original = deepcopy(source.words)
        _, _, regions = discover_native_owners(document(current), source, [first, second], Client(), None)
        self.assertEqual([a["text"] for a in regions[first.id].atoms], ["К", "расчету"])
        self.assertEqual([a["text"] for a in regions[second.id].atoms], ["Таблица"])
        self.assertEqual(source.words, original)

    def test_table_grid_already_in_pdf_frame_is_not_scaled_twice(self):
        current = page(native=(0, 0, 843, 596))
        source_bbox = (85.104, 57.02, 793.920015, 492.46)
        native = add(current, Table, source_bbox, table_grid={"bbox": list(source_bbox), "cells": []})
        source = Source([word("cell", (90, 60, 120, 70))], size=(842.0399780273438, 595.3200073242188))
        original_grid = deepcopy(native.table_grid)
        _, parts, _ = discover_native_owners(document(current), source, [native], Client(), None)
        self.assertEqual(parts[native.id].bbox, source_bbox)
        self.assertEqual(native.table_grid, original_grid)

    def test_vectors_inside_native_text_are_handed_to_whole_object_job(self):
        current = page()
        native = add(current, Text, (20, 30, 200, 60))
        source = Source([word("Limit", (20, 31, 80, 45))])
        vector = (90, 40, 105, 40)
        source.pdf_page = lambda _: SimpleNamespace(edges=[{"bbox": vector}], curves=[])
        _, _, regions = discover_native_owners(document(current), source, [native], Client(), None)
        self.assertEqual(regions[native.id].kind, "text")
        self.assertEqual([atom["source_bbox"] for atom in regions[native.id].atoms if atom["kind"] == "vector"], [vector])
        self.assertEqual([atom["text"] for atom in regions[native.id].atoms if atom["kind"] == "text"], ["Limit"])

    def test_blank_no_layout_page_needs_no_model_call(self):
        current = page()
        client = Client()
        owners, parts, regions = discover_native_owners(document(current), Source(), [], client, None)
        self.assertEqual((owners, parts, regions), ([], {}, {}))
        self.assertFalse(client.calls)

    def test_unowned_raster_is_not_silently_marked_complete(self):
        current = page()
        text = add(current, Text, (10, 10, 180, 30), html="<p>Native paragraph</p>")
        source = Source([word("Native paragraph", (10, 10, 180, 30))],
                        [picture((260, 200, 280, 218))])
        owners, parts, regions = discover_native_owners(document(current), source, [text], Client(), None)
        self.assertEqual(len(owners), 2)
        orphan = next(owner for owner in owners if owner is not text)
        self.assertEqual(regions[orphan.id].bbox, (260, 200, 280, 218))
        self.assertEqual(len(regions[orphan.id].atoms), 1)
        self.assertFalse(parts)

    def test_adjacent_orphan_glyphs_are_one_whole_expression_owner(self):
        current = page()
        source = Source([word("x", (60, 30, 70, 45)), word("0", (99, 30, 106, 45))],
                        [picture((75, 30, 91, 46))])
        # A tiny existing prose block avoids full-page scan discovery. All
        # operands of the orphan expression must still be grouped together.
        native = add(current, Text, (10, 100, 30, 115), html="<p>ok</p>")
        owners, _, regions = discover_native_owners(document(current), source, [native], Client(), None)
        new = [owner for owner in owners if owner is not native]
        self.assertEqual(len(new), 1)
        self.assertIsInstance(new[0], Equation)
        self.assertEqual([a.get("text") for a in regions[new[0].id].atoms if a["kind"] == "text"], ["x", "0"])

    def test_orphan_operator_absorbs_entire_nearby_native_paragraph(self):
        current = page()
        native = add(current, Text, (20, 30, 70, 48), html="<p>Limit x</p>")
        source = Source([word("Limit", (20, 30, 50, 45)), word("x", (60, 30, 70, 45)),
                         word("0", (98, 30, 106, 45))], [picture((75, 30, 91, 46))])
        owners, _, regions = discover_native_owners(document(current), source, [native], Client(), None)
        self.assertEqual(len(owners), 1)
        owner = owners[0]
        self.assertIsNot(owner, native)
        self.assertTrue(native.ignore_for_output)
        self.assertEqual(regions[owner.id].kind, "text")
        self.assertCountEqual([a.get("text") for a in regions[owner.id].atoms if a["kind"] == "text"], ["Limit", "x", "0"])

    def test_empty_layout_scan_is_classified_before_table_grouping(self):
        current = page()
        source = Source(images=[picture((0, 0, 360, 400))])
        client = Client(layout(("table", (100, 100, 900, 400), "Table 6", False)))
        owners, parts, regions = discover_native_owners(document(current), source, [], client, None)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(owners), 1)
        self.assertIsInstance(owners[0], Table)
        self.assertEqual(parts[owners[0].id].bbox, (36, 40, 324, 160))
        self.assertEqual(parts[owners[0].id].table_id, "6")
        self.assertEqual(regions[owners[0].id].kind, "table")

    def test_full_page_owner_is_detached_only_after_successful_discovery(self):
        current = page()
        native = add(current, Picture, (0, 0, 360, 400))
        source = Source(images=[picture((0, 0, 360, 400))])
        before = list(current.structure)
        with self.assertRaises(RecognitionFailure):
            discover_native_owners(document(current), source, [native],
                                   Client(RecognitionFailure("offline discovery failed")), None)
        self.assertEqual(current.structure, before)
        self.assertFalse(native.ignore_for_output)

    def test_discovered_continuations_join_as_one_logical_table(self):
        first, second = page(0), page(1)
        original = [add(first, Picture, (0, 0, 360, 400)), add(second, Picture, (0, 0, 360, 400))]
        source = Source(images=[picture((0, 0, 360, 400), 0), picture((0, 0, 360, 400), 1)])
        client = Client(layout(("table", (100, 600, 900, 1000), "Table A.1", False)),
                        layout(("table", (100, 0, 900, 350), "Table A.1", True)))
        owners, parts, _ = discover_native_owners(document(first, second), source, original, client, None)
        self.assertTrue(all(owner.ignore_for_output for owner in original))
        grouped = group_tables(list(parts.values()), Client())
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].page_ids, (0, 1))
        self.assertEqual(len(owners), 2)

    def test_subpage_native_picture_continuations_join_as_one_table(self):
        first, second = page(0), page(1)
        boxes = [(36, 250, 324, 400), (36, 0, 324, 150)]
        native = [add(current, Picture, box) for current, box in zip((first, second), boxes)]
        source = Source(images=[picture(box, index) for index, box in enumerate(boxes)])
        client = Client(layout(("table", (0, 0, 1000, 1000), "Table 9", False)),
                        layout(("table", (0, 0, 1000, 1000), "Table 9", True)))
        owners, parts, _ = discover_native_owners(document(first, second), source, native, client, None)
        self.assertTrue(all(owner.ignore_for_output for owner in native))
        self.assertTrue(all(isinstance(owner, Table) for owner in owners))
        grouped = group_tables(list(parts.values()), Client())
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].page_ids, (0, 1))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual([part.bbox for part in grouped[0].parts], boxes)

    def test_multiple_native_pictures_are_discovered_sequentially_without_prose_loss(self):
        current = page()
        prose = add(current, Text, (10, 10, 300, 30), html="<p>Body</p>")
        first = add(current, Picture, (20, 100, 160, 220))
        second = add(current, Picture, (200, 100, 340, 220))
        source = Source([word("Body", (10, 10, 300, 30))],
                        [picture((20, 100, 160, 220)), picture((200, 100, 340, 220), index=1)])
        client = Client(layout(("table", (0, 0, 1000, 1000), "Table 2", False)),
                        layout(("figure", (0, 0, 1000, 1000), "", False)))
        owners, parts, regions = discover_native_owners(document(current), source,
            [prose, first, second], client, None)
        self.assertEqual(len(client.calls), 2)
        self.assertIn(prose, owners)
        self.assertFalse(prose.ignore_for_output)
        self.assertEqual(len(parts), 1)
        self.assertEqual(sorted(region.kind for region in regions.values()), ["figure", "table"])
        self.assertTrue(first.ignore_for_output)
        self.assertTrue(second.ignore_for_output)

    def test_genuine_full_page_figure_is_positive_classification(self):
        current = page()
        native = add(current, Picture, (0, 0, 360, 400))
        source = Source(images=[picture((0, 0, 360, 400))])
        owners, _, regions = discover_native_owners(document(current), source, [native],
            Client(layout(("figure", (0, 0, 1000, 1000), "", False))), None)
        self.assertEqual(regions[owners[0].id].kind, "figure")
        self.assertEqual(regions[owners[0].id].bbox, (0, 0, 360, 400))

    def test_matching_native_table_is_preserved_during_scan_discovery(self):
        current = page()
        native = add(current, Table, (36, 80, 324, 240),
                     table_grid={"bbox": [36, 80, 324, 240], "cells": []})
        source = Source(images=[picture((0, 0, 360, 400))])
        client = Client(layout(("table", (100, 200, 900, 600), "Table 4", False)))
        owners, parts, _ = discover_native_owners(document(current), source, [native], client, None)
        self.assertEqual(owners, [native])
        self.assertFalse(native.ignore_for_output)
        self.assertEqual(parts[native.id].grid, native.table_grid)
        self.assertEqual(parts[native.id].caption, "Table 4")

    def test_conflicting_discovery_cannot_replace_native_table(self):
        current = page()
        full = add(current, Picture, (0, 0, 360, 400))
        native = add(current, Table, (36, 80, 324, 240))
        source = Source(images=[picture((0, 0, 360, 400))])
        client = Client(layout(("table", (100, 100, 900, 650), "Table 4", False)))
        original = list(current.structure)
        with self.assertRaisesRegex(RecognitionFailure, "conflicts"):
            discover_native_owners(document(current), source, [full, native], client, None)
        self.assertEqual(current.structure, original)
        self.assertFalse(full.ignore_for_output)
        self.assertFalse(native.ignore_for_output)

    def test_missing_vector_table_is_discovered_before_transcription(self):
        current = page()
        native = add(current, Text, (10, 300, 120, 320), html="<p>Native</p>")
        source = Source()
        source.pdf_page = lambda _: SimpleNamespace(edges=[
            {"bbox": (40, 40, 200, 40)}, {"bbox": (40, 160, 200, 160)},
            {"bbox": (40, 40, 40, 160)}, {"bbox": (200, 40, 200, 160)},
        ], curves=[])
        client = Client(layout(("table", (0, 0, 1000, 1000), "Table 8", False)))
        owners, parts, _ = discover_native_owners(document(current), source, [native], client, None)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(parts), 1)
        self.assertEqual(sum(isinstance(owner, Table) for owner in owners), 1)

    def test_later_page_failure_rolls_back_all_previous_native_mutations(self):
        first, second = page(0), page(1)
        original = [add(first, Picture, (0, 0, 360, 400)), add(second, Picture, (0, 0, 360, 400))]
        source = Source(images=[picture((0, 0, 360, 400), 0), picture((0, 0, 360, 400), 1)])
        client = Client(layout(("table", (100, 600, 900, 1000), "Table A.1", False)),
                        RecognitionFailure("second page failed"))
        before = [(list(p.structure), list(p.children), p.block_id) for p in (first, second)]
        with self.assertRaises(RecognitionFailure):
            discover_native_owners(document(first, second), source, original, client, None)
        for current, (structure, children, identity) in zip((first, second), before):
            self.assertEqual(current.structure, structure)
            self.assertEqual(current.children, children)
            self.assertEqual(current.block_id, identity)
        self.assertTrue(all(not owner.ignore_for_output for owner in original))

    def test_intentional_page_header_is_excluded_from_coverage(self):
        current = page()
        add(current, PageHeader, (10, 5, 170, 20), html="<p>Header</p>")
        native = add(current, Text, (10, 100, 200, 120), html="<p>Body</p>")
        source = Source([word("Header", (10, 5, 170, 20)), word("Body", (10, 100, 200, 120))])
        owners, parts, regions = discover_native_owners(document(current), source, [native], Client(), None)
        self.assertEqual(owners, [native])
        self.assertFalse(parts)
        self.assertFalse(regions)

    def test_source_atom_crossing_native_owner_fails_closed(self):
        current = page()
        native = add(current, Text, (10, 10, 40, 30), html="<p>clipped</p>")
        source = Source([word("clipped", (30, 10, 80, 30))])
        with self.assertRaisesRegex(RecognitionFailure, "clips"):
            discover_native_owners(document(current), source, [native], Client(), None)
        self.assertFalse(native.ignore_for_output)

    def test_inverse_coordinates_for_new_owner_do_not_mutate_source(self):
        current = page(native=(20, 30, 740, 830))
        native = add(current, Text, (40, 50, 200, 90), html="<p>Native</p>")
        source = Source(images=[picture((260, 200, 280, 218))])
        owners, _, regions = discover_native_owners(document(current), source, [native], Client(), None)
        new = next(owner for owner in owners if owner is not native)
        self.assertEqual(new.polygon.bbox, [540, 430, 580, 466])
        self.assertEqual(regions[new.id].atoms[0]["source_bbox"], (260, 200, 280, 218))

    def test_real_scan_pdf_gets_a_table_owner_without_local_ocr(self):
        current = page()
        client = Client(layout(("text", (70, 65, 700, 120), "", False),
                               ("table", (70, 140, 930, 370), "Table 1", False)))
        with PdfSource(FIXTURES / "scan.pdf") as source:
            owners, parts, regions = discover_native_owners(document(current), source, [], client, None)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(parts), 1)
        self.assertEqual(sum(isinstance(owner, Table) for owner in owners), 1)
        self.assertFalse(any(a.get("kind") == "image" for region in regions.values() for a in region.atoms))

    def test_real_split_formula_preserves_all_unowned_glyphs_in_whole_jobs(self):
        current = page()
        # Native table still owns every bitmap/line in its cells. Only the
        # two display expressions and their prose are missing from layout.
        native = add(current, Table, (30, 100, 330, 180))
        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            owners, _, regions = discover_native_owners(document(current), source, [native], Client(), None)
            original = source.page_images(0)
        self.assertIn(native, owners)
        raster = [atom for region in regions.values() for atom in region.atoms if atom.get("kind") == "image"]
        outside = [image for image in original if image.bbox[1] < 100]
        self.assertEqual(len(raster), len(outside))
        expression = next(region for region in regions.values() if any(a.get("kind") == "image" for a in region.atoms))
        self.assertCountEqual([a.get("text") for a in expression.atoms if a.get("kind") == "text"], ["x", "0"])
        self.assertTrue(any(a.get("text") == "Limit:" for region in regions.values() for a in region.atoms))


if __name__ == "__main__":
    unittest.main()
