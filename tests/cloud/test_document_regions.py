"""Native prose and whole raster expressions keep disjoint source owners."""

from collections import Counter
from copy import deepcopy
import unittest

from marker.cloud.content import atom_id, bounds
from marker.cloud.document_regions import document_regions
from marker.cloud.regions import Region, assess_text_quality


def word(text, x, y=10, *, width=None, page=0, font="Helvetica", size=10):
    box = (x, y, x + (width or len(text) * 5), y + size)
    return {
        "kind": "text",
        "text": text,
        "bbox": box,
        "source_bbox": box,
        "source_page": page,
        "fontname": font,
        "size": size,
    }


def raster(box, *, page=0, index=0):
    return {
        "kind": "image",
        "bbox": box,
        "source_bbox": box,
        "source_page": page,
        "image_index": index,
    }


def region(kind, atoms, *, page=0, inline_group=None):
    box = bounds(atoms)
    return Region(
        page,
        box,
        kind,
        tuple(atoms),
        assess_text_quality(
            [atom for atom in atoms if atom.get("kind") == "text"], box
        ),
        inline_group=inline_group,
    )


class DocumentRegionPartitionTests(unittest.TestCase):
    def partition(self, regions):
        before = deepcopy([item.__dict__ for item in regions])
        expected = {atom_id(atom, item.page) for item in regions for atom in item.atoms}
        result = document_regions(regions)
        self.assertEqual([item.__dict__ for item in regions], before)
        observed = [atom_id(atom, item.page) for item in result for atom in item.atoms]
        self.assertEqual(set(observed), expected)
        self.assertTrue(
            all(count == 1 for count in Counter(observed).values()), observed
        )
        return result

    def test_large_formula_bitmap_stays_whole_and_separate_from_prose(self):
        prefix, suffix = word("где", 10), word("стержня", 165)
        formula = raster((35, 7, 160, 37))
        result = self.partition(
            [region("text", [prefix, suffix]), region("image", [formula])]
        )
        self.assertEqual([item.kind for item in result], ["text", "image", "text"])
        self.assertEqual(
            [list(item.atoms) for item in result], [[prefix], [formula], [suffix]]
        )
        self.assertEqual(result[1].bbox, formula["source_bbox"])
        self.assertTrue(all(item.inline_group is not None for item in result))
        self.assertEqual(len({item.inline_group for item in result}), 1)

    def test_small_raster_symbol_and_neighboring_operands_have_one_owner(self):
        x, zero = word("x", 10), word("0", 31)
        operator = raster((19, 10, 29, 20))
        result = self.partition(
            [region("text", [x, zero]), region("image", [operator])]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].kind, "image")
        self.assertEqual(list(result[0].atoms), [x, operator, zero])

    def test_explicit_native_math_evidence_is_kept(self):
        variable = word("x", 10, font="CMMI10")
        formula = raster((20, 8, 145, 38))
        formula["content_kind"] = "math"
        result = self.partition(
            [region("text", [variable]), region("image", [formula])]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].kind, "math")

    def test_tall_diagram_does_not_absorb_its_prose_caption(self):
        diagram = raster((20, 10, 270, 110))
        caption = word("Рисунок", 20, 115, width=60)
        result = self.partition([region("image", [diagram, caption])])
        self.assertEqual([item.kind for item in result], ["image", "text"])
        self.assertEqual(result[0].bbox, diagram["bbox"])
        self.assertEqual(list(result[0].atoms), [diagram])
        self.assertEqual(list(result[1].atoms), [caption])

    def test_short_diagram_is_a_candidate_without_being_declared_math(self):
        diagram = raster((20, 10, 270, 80))
        caption = word("Схема", 20, 90)
        result = self.partition([region("image", [diagram]), region("text", [caption])])
        self.assertEqual([item.kind for item in result], ["image", "text"])
        self.assertEqual(result[0].bbox, diagram["bbox"])

    def test_greek_overlay_label_does_not_declare_a_short_diagram_math(self):
        diagram = raster((20, 10, 270, 80))
        label = word("α", 100, 35, font="CMMI10")
        result = self.partition([region("image", [diagram, label])])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].kind, "image")
        self.assertEqual(
            set(atom_id(atom) for atom in result[0].atoms),
            {atom_id(diagram), atom_id(label)},
        )

    def test_tall_diagram_keeps_genuinely_overlaid_labels(self):
        diagram = raster((20, 10, 270, 110))
        label, caption = word("Label", 40, 30), word("Caption", 20, 115)
        result = self.partition([region("image", [diagram, label, caption])])
        self.assertEqual([item.kind for item in result], ["image", "text"])
        self.assertEqual(list(result[0].atoms), [diagram, label])
        self.assertEqual(list(result[1].atoms), [caption])

    def test_explicitly_known_tall_formula_keeps_its_math_kind(self):
        formula = raster((20, 10, 170, 100))
        formula["content_kind"] = "math"
        result = self.partition([region("math", [formula])])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].kind, "math")

    def test_untouched_native_paragraph_is_preserved(self):
        paragraph = region(
            "text",
            [
                word("Первое", 10),
                word("предложение", 50),
                word("Второе", 10, 25),
                word("предложение", 50, 25),
            ],
        )
        result = self.partition([paragraph])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], paragraph)

    def test_pages_do_not_mix_and_original_records_are_returned(self):
        regions = []
        originals = []
        for page in (0, 1):
            prefix = word("где", 10, page=page)
            formula = raster((35, 7, 160, 37), page=page)
            originals.extend([prefix, formula])
            regions.extend(
                [
                    region("text", [prefix], page=page),
                    region("image", [formula], page=page),
                ]
            )
        result = self.partition(regions)
        self.assertEqual([item.page for item in result], [0, 0, 1, 1])
        self.assertTrue(
            all(
                any(atom is original for original in originals)
                for item in result
                for atom in item.atoms
            )
        )
        self.assertTrue(all("content_kind" not in atom for atom in originals))

    def test_existing_inline_group_survives_partitioning(self):
        prefix = word("где", 10)
        formula = raster((35, 7, 160, 37))
        inline = (0, (0, 5, 200, 40))
        result = self.partition(
            [
                region("text", [prefix], inline_group=inline),
                region("image", [formula], inline_group=inline),
            ]
        )
        self.assertEqual({item.inline_group for item in result}, {inline})

    def test_tables_and_unknown_source_regions_remain_unchanged(self):
        table = region("table", [word("Cell", 10)])
        unknown = region("unknown", [word("Visible", 10, 50)])
        result = self.partition([table, unknown])
        self.assertIs(result[0], table)
        self.assertIs(result[1], unknown)

    def test_duplicate_physical_atoms_are_owned_once(self):
        formula = raster((35, 7, 160, 37))
        result = self.partition(
            [region("image", [formula]), region("image", [deepcopy(formula)])]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result[0].atoms), [formula])

    def test_duplicate_text_atoms_do_not_reappear_when_paragraph_is_restored(self):
        atom = word("Visible", 10)
        result = self.partition([region("text", [atom, deepcopy(atom)])])
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result[0].atoms), [atom])


if __name__ == "__main__":
    unittest.main()
