"""Whole logical tables use one raw HTML request and retain real formulae."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup
from PIL import Image

from marker.cloud.context import RecognitionFailure
from marker.cloud.client import ResponseValidationError
from marker.cloud.tables import LogicalTable, TablePart
from marker.cloud.whole_tables import normalize_table_html, recognize_whole_table


FIXTURES = Path(__file__).parent / "fixtures"
NS = {"m": "http://www.w3.org/1998/Math/MathML"}


class FakeClient:
    cloud_max_images = 8

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def request_text(
        self, prompt, images, validator, *, max_output_tokens=None, image_long_edge=2048
    ):
        self.calls.append((prompt, list(images), max_output_tokens, image_long_edge))
        if isinstance(self.answer, Exception):
            raise self.answer
        answer = self.answer(prompt, images) if callable(self.answer) else self.answer
        return validator(answer)


class FakeSource:
    def __init__(self, images=()):
        self.images = list(images)
        self.crops = []
        self.crop_dpis = []
        self.resources = []
        self.native = Image.new("RGBA", (1000, 700), (17, 29, 41, 127))

    def page_size(self, page):
        return 600, 500

    def crop(self, page, bbox, dpi=None):
        self.crops.append((page, tuple(bbox)))
        self.crop_dpis.append(dpi)
        return Image.new("RGB", (600, 400), (240 - page, 245, 250))

    def page_images(self, page, bbox=None):
        return [image for image in self.images if image.page == page]

    def resource(self, page, bbox):
        self.resources.append((page, tuple(bbox)))
        return self.native.copy()


def word(text, x=10, page=0):
    return {
        "kind": "text",
        "text": text,
        "size": 12,
        "bbox": (x, 20, x + max(10, len(text) * 6), 32),
        "source_page": page,
    }


def logical(words=(), *, pages=(0,), warnings=(), trusted=False):
    parts = []
    for page in pages:
        box = (0, 0, 600, 450)
        atoms = [dict(atom, source_page=page) for atom in words]
        grid = {
            "page": page,
            "bbox": box,
            "trusted": trusted,
            "cells": [{"bbox": box, "source_page": page, "atoms": atoms}],
        }
        parts.append(TablePart(page, box, grid, (600, 500)))
    return LogicalTable(parts, deepcopy(parts[0].grid), warnings)


def trusted_two_by_two():
    cells = []
    for index, value in enumerate("ABCD"):
        row, column = divmod(index, 2)
        box = (column * 100, row * 40, (column + 1) * 100, (row + 1) * 40)
        atom = word(value)
        atom["bbox"] = (box[0] + 10, box[1] + 10, box[0] + 20, box[1] + 22)
        cells.append({"bbox": box, "source_page": 0, "atoms": [atom]})
    grid = {"page": 0, "bbox": (0, 0, 200, 80), "trusted": True, "cells": cells}
    return LogicalTable([TablePart(0, grid["bbox"], grid, (600, 500))], grid)


class WholeTableTests(unittest.TestCase):
    def recognize(self, answer, words=(), *, source=None, table=None):
        source = source or FakeSource()
        client = FakeClient(answer)
        result = recognize_whole_table(source, table or logical(words), client)
        return result, client, source

    def test_missing_leaf_math_closer_at_its_parent_end_preserves_exact_latex(self):
        for body in (r"<p>до <math>45^\circ</p>", r"<math>0,42R_{\text{bun}}"):
            with self.subTest(body=body):
                html = normalize_table_html(
                    "<table><tr><td>" + body + "</td></tr></table>"
                )
                annotation = BeautifulSoup(html, "html.parser").find("annotation")
                expected = (
                    r"45^\circ" if body.startswith("<p>") else r"0,42R_{\text{bun}}"
                )
                self.assertEqual(annotation.get_text(), expected)

    def test_math_closer_repair_cannot_hide_missing_cells_or_truncated_table(self):
        for malformed in (
            "<table><tr><td><math>x</tr></table>",
            "<table><tr><td><math>x",
            "<table><tr><td><math><mrow><mi>x</mi></mrow></td></tr></table>",
        ):
            with (
                self.subTest(malformed=malformed),
                self.assertRaises(ResponseValidationError),
            ):
                normalize_table_html(malformed)

    def test_raw_html_russian_decimals_and_entire_index_become_native_mathml(self):
        latex = r"0,42R_{\text{bun}}"
        answer = (
            "<table><tr><th>Прочность</th><td><math>"
            + latex
            + "</math></td></tr></table>"
        )
        result, client, source = self.recognize(answer, [word("Прочность")])
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(result.markdown, result.html)
        soup = BeautifulSoup(result.html, "html.parser")
        tree = ET.fromstring(str(soup.find("math")))
        script = tree.find(".//m:msub", NS)
        self.assertEqual("".join(script[0].itertext()), "R")
        self.assertEqual("".join(script[1].itertext()), "bun")
        self.assertEqual(tree.find(".//m:annotation", NS).text, latex)
        self.assertEqual(result.resources, {})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][2:], (None, 3072))
        self.assertEqual(source.crops, [(0, (0, 0, 600, 450))])
        self.assertEqual(source.crop_dpis, [288])
        self.assertNotIn("atom_ids", client.calls[0][0])
        self.assertNotIn("expression_groups", client.calls[0][0])

    def test_data_latex_regenerates_presentation_and_preserves_phi_variants(self):
        result, _, _ = self.recognize(
            r'<table><tr><td><span data-latex="\phi+\varphi">wrong presentation</span></td></tr></table>'
        )
        self.assertTrue(result.complete, result.reasons)
        self.assertNotIn("wrong presentation", result.html)
        soup = BeautifulSoup(result.html, "html.parser")
        self.assertEqual(soup.find("annotation").get_text(), r"\phi+\varphi")
        self.assertIn("ϕ", soup.find("math").get_text())
        self.assertIn("φ", soup.find("math").get_text())

    def test_model_mathml_is_rebuilt_from_its_annotation(self):
        result, _, _ = self.recognize(
            "<table><tr><td><math><semantics><mrow><mi>R</mi><msub><mi>b</mi><mi>un</mi></msub></mrow>"
            '<annotation encoding="application/x-tex">R_{bun}</annotation></semantics></math></td></tr></table>'
        )
        self.assertTrue(result.complete, result.reasons)
        tree = ET.fromstring(
            str(BeautifulSoup(result.html, "html.parser").find("math"))
        )
        script = tree.find(".//m:msub", NS)
        self.assertEqual("".join(script[0].itertext()), "R")
        self.assertEqual("".join(script[1].itertext()), "bun")

    def test_html_fence_and_rectangular_rowspan_colspan_are_supported(self):
        result, _, _ = self.recognize(
            '```html\n<table><thead><tr><th rowspan="2">A</th><th colspan="2">B</th></tr>'
            "<tr><th>C</th><th>D</th></tr></thead><tbody><tr><td>E</td><td>F</td><td>G</td></tr>"
            "</tbody></table>\n```"
        )
        self.assertTrue(result.complete, result.reasons)
        self.assertIn('rowspan="2"', result.html)
        self.assertIn('colspan="2"', result.html)

    def test_invalid_spans_or_ragged_rows_are_incomplete_without_substitution(self):
        answers = (
            '<table><tr><td rowspan="0">A</td></tr></table>',
            '<table><tr><td colspan="-1">A</td></tr></table>',
            '<table><tr><td colspan="many">A</td></tr></table>',
            '<table><tr><td rowspan="2">A</td></tr></table>',
            "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>",
            '<table><tr><td>A</td><td rowspan="2">B</td></tr><tr><td colspan="2">C</td></tr></table>',
        )
        for answer in answers:
            with self.subTest(answer=answer):
                result, _, _ = self.recognize(answer)
                self.assertFalse(result.complete)
                self.assertTrue(result.reasons)
                self.assertEqual(
                    (result.markdown, result.html, result.resources), ("", "", {})
                )

    def test_executable_external_and_nested_markup_is_rejected(self):
        answers = (
            "<table><tr><td><script>SECRET_PROVIDER_ECHO</script>A</td></tr></table>",
            '<table onclick="SECRET_PROVIDER_ECHO"><tr><td>A</td></tr></table>',
            '<table><tr><td><a href="https://outside.invalid">A</a></td></tr></table>',
            '<table><tr><td><img src="data:image/png;base64,SECRET_PROVIDER_ECHO"></td></tr></table>',
            '<table><tr><td><img src="https://outside.invalid/table.png"></td></tr></table>',
            '<table><tr><td style="display:none">A</td></tr></table>',
            "<table><tr><td><table><tr><td>A</td></tr></table></td></tr></table>",
            "<table><tr><td>A</td></tr></table><table><tr><td>B</td></tr></table>",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                result, _, _ = self.recognize(answer)
                self.assertFalse(result.complete)
                self.assertEqual(result.markdown, "")
                self.assertNotIn("SECRET_PROVIDER_ECHO", str(result.reasons))

    def test_missing_closing_tags_empty_tables_and_json_are_rejected(self):
        for answer in (
            "<table><tr><td>A",
            "<table><tr><td>A</tr></table>",
            "<table><tr><td> </td></tr></table>",
            '{"table":"A"}',
            "Here is the table: <table><tr><td>A</td></tr></table>",
        ):
            with self.subTest(answer=answer):
                result, _, _ = self.recognize(answer)
                self.assertFalse(result.complete)

    def test_ordinary_trusted_source_text_cannot_disappear(self):
        words = [word("Расчётное"), word("сопротивление", 90), word("0,42", 200)]
        result, _, _ = self.recognize(
            "<table><tr><td>Расчётное 0,42</td></tr></table>", words
        )
        self.assertFalse(result.complete)
        self.assertIn("trusted", " ".join(result.reasons).lower())
        complete, _, _ = self.recognize(
            "<table><tr><td>Расчётное сопротивление <math>0,42</math></td></tr></table>",
            words,
        )
        self.assertTrue(complete.complete, complete.reasons)

    def test_repeated_numeric_values_are_not_silently_dropped(self):
        result, _, _ = self.recognize(
            "<table><tr><td>0,42</td></tr></table>", [word("0,42"), word("0,42", 100)]
        )
        self.assertFalse(result.complete)

    def test_trusted_numeric_signs_percentages_and_fraction_values_survive(self):
        for original, changed in (("-1", "1"), ("42%", "42"), ("1/2", "1 2")):
            with self.subTest(original=original):
                result, _, _ = self.recognize(
                    "<table><tr><td>" + changed + "</td></tr></table>", [word(original)]
                )
                self.assertFalse(result.complete)
        right, _, _ = self.recognize(
            r"<table><tr><td><math>-1</math> <math>42\%</math> <math>\frac{1}{2}</math></td></tr></table>",
            [word("-1"), word("42%", 100), word("1/2", 200)],
        )
        self.assertTrue(right.complete, right.reasons)

    def test_block_text_boundaries_are_visible_whitespace(self):
        result, _, _ = self.recognize(
            "<table><tr><td><p>Расчётное</p><div>сопротивление</div><br>0,42</td></tr></table>",
            [word("Расчётное"), word("сопротивление", 100), word("0,42", 200)],
        )
        self.assertTrue(result.complete, result.reasons)

    def test_russian_line_hyphenation_compares_joined_and_preserved_headers(self):
        source_text = "Услов-\nное\nобо-\nзначе-\nние"
        for visible in (
            "Условное обозначение",
            "<p>Услов-</p><p>ное</p><p>обо-</p><p>значе-</p><p>ние</p>",
        ):
            with self.subTest(visible=visible):
                html = normalize_table_html(
                    "<table><tr><td>" + visible + "</td></tr></table>",
                    expected_text=source_text,
                )
                self.assertIn(visible, html)

    def test_cross_page_header_fragments_with_separate_hyphen_words_compare_without_changing_html(
        self,
    ):
        expected = {
            "c0": {
                "row": 0,
                "column": 0,
                "rowspan": 1,
                "colspan": 1,
                "trusted_text": ["Услов -", "ное\nобо -\nзначе -\nние"],
            }
        }
        for visible in (
            "Условное обозначение",
            "<p>Услов-</p><p>ное</p><p>обо-</p><p>значе-</p><p>ние</p>",
        ):
            with self.subTest(visible=visible):
                html = normalize_table_html(
                    '<table><tr><td data-cell-id="c0">'
                    + visible
                    + "</td></tr></table>",
                    expected_grid=expected,
                )
                self.assertIn(visible, html)

    def test_unmapped_trusted_source_fragment_cannot_bypass_per_cell_coverage(self):
        table = trusted_two_by_two()
        table.parts[0].grid["cells"][0]["source_bbox"] = (10, 10, 80, 30)
        result, client, _ = self.recognize(
            "<table><tr><td>omitted</td></tr></table>", table=table
        )
        self.assertFalse(result.complete)
        self.assertIn("no corresponding logical table owner", result.reasons[0])
        self.assertEqual(client.calls, [])

    def test_trusted_small_raster_formula_requires_latex_in_its_physical_cell(self):
        table = trusted_two_by_two()
        table.grid["cells"][1]["atoms"] = [
            {
                "kind": "image",
                "bbox": (110, 10, 155, 28),
                "source_page": 0,
            }
        ]
        table.parts[0].grid["cells"] = deepcopy(table.grid["cells"])
        template = (
            '<table><tr><td data-cell-id="c0">A</td><td data-cell-id="c1">{}</td></tr>'
            '<tr><td data-cell-id="c2">C</td><td data-cell-id="c3">D</td></tr></table>'
        )
        for value in ("", "unreadable", '<img data-source-crop="0:550,125,775,350">'):
            with self.subTest(value=value):
                result, _, source = self.recognize(template.format(value), table=table)
                self.assertFalse(result.complete)
                self.assertEqual(
                    (result.markdown, result.resources, source.resources), ("", {}, [])
                )
        result, _, source = self.recognize(
            template.format(r"<math>R_{bun}</math>"), table=table
        )
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual((result.resources, source.resources), ({}, []))

    def test_trusted_raster_formula_baselines_cannot_collapse_into_one_math(self):
        table = logical(trusted=True)
        table.grid["cells"][0]["atoms"] = [
            {"kind": "image", "bbox": (10, y, 55, y + 18), "source_page": 0}
            for y in (20, 60)
        ]
        table.parts[0].grid["cells"] = deepcopy(table.grid["cells"])
        template = '<table><tr><td data-cell-id="c0">{}</td></tr></table>'
        omitted, _, _ = self.recognize(template.format("<math>x</math>"), table=table)
        self.assertFalse(omitted.complete)
        complete, _, _ = self.recognize(
            template.format("<p><math>x</math></p><p><math>y</math></p>"),
            table=table,
        )
        self.assertTrue(complete.complete, complete.reasons)

    def test_native_figure_axis_scripts_do_not_require_separate_symbol_ocr(self):
        from types import SimpleNamespace

        source = FakeSource(
            [SimpleNamespace(page=0, index=7, bbox=(100, 100, 300, 220))]
        )
        labels = [
            {
                "kind": "text",
                "text": "x",
                "size": 12,
                "source_page": 0,
                "bbox": (120, 120, 130, 132),
            },
            {
                "kind": "text",
                "text": "2",
                "size": 8,
                "source_page": 0,
                "bbox": (130, 115, 135, 123),
            },
        ]

        def answer(prompt, _images):
            payload = json.loads(prompt.split("SOURCE:\n", 1)[1])
            return (
                '<table><tr><td><img data-source-image="'
                + payload["figures"][0]["id"]
                + '"></td></tr></table>'
            )

        result, _, _ = self.recognize(answer, labels, source=source)
        self.assertTrue(result.complete, result.reasons)

    def test_all_ordered_page_crops_are_sent_together_without_source_mutation(self):
        table = logical(pages=(2, 0, 1))
        before = deepcopy([part.grid for part in table.parts])
        result, client, source = self.recognize(
            "<table><tr><td>A</td></tr></table>", table=table
        )
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual([page for page, _ in source.crops], [0, 1, 2])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0][1]), 3)
        self.assertEqual([part.grid for part in table.parts], before)

    def test_prompt_geometry_is_normalized_to_offset_part_without_source_mutation(self):
        from types import SimpleNamespace

        box = (85, 135, 485, 435)
        figure_box = (125, 195, 205, 255)
        cells = []
        for bounds, text, x in (
            ((85, 135, 285, 435), "A", 100),
            ((285, 135, 485, 435), "B", 300),
        ):
            atom = word(text, x)
            atom["bbox"] = (x, 145, x + 10, 157)
            cells.append({"bbox": bounds, "source_page": 0, "atoms": [atom]})
        grid = {"page": 0, "bbox": box, "trusted": True, "cells": cells}
        table = LogicalTable([TablePart(0, box, grid, (600, 500))], grid)
        before = deepcopy((table.grid, table.parts[0].grid, table.parts[0].bbox))
        source = FakeSource([SimpleNamespace(page=0, index=7, bbox=figure_box)])

        def answer(prompt, _images):
            payload = json.loads(prompt.split("SOURCE:\n", 1)[1])
            self.assertEqual(payload["parts"][0]["bbox"], [0, 0, 1000, 1000])
            self.assertEqual(
                [cell["bbox"] for cell in payload["parts"][0]["cells"]],
                [[0, 0, 500, 1000], [500, 0, 1000, 1000]],
            )
            self.assertEqual(payload["figures"][0]["part_index"], 0)
            self.assertEqual(payload["figures"][0]["bbox"], [100, 200, 300, 400])
            identity = payload["figures"][0]["id"]
            return (
                '<table><tr><td data-cell-id="c0">A<img data-source-image="'
                + identity
                + '"></td>'
                '<td data-cell-id="c1">B</td></tr></table>'
            )

        result, client, _ = self.recognize(answer, table=table, source=source)
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(source.crops, [(0, box)])
        self.assertEqual(source.resources, [(0, figure_box)])
        self.assertEqual((table.grid, table.parts[0].grid, table.parts[0].bbox), before)

    def test_eight_page_logical_table_remains_one_complete_request(self):
        result, client, source = self.recognize(
            "<table><tr><td>A</td></tr></table>", table=logical(pages=tuple(range(8)))
        )
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0][1]), 8)
        self.assertEqual([page for page, _ in source.crops], list(range(8)))

    def test_eight_page_trusted_grid_keeps_all_physical_cells_in_one_request(self):
        from marker.processors.table_geometry import stack_grids

        table = logical([word("A")], pages=tuple(range(8)), trusted=True)
        for part in table.parts[1:]:
            table.grid = stack_grids(table.grid, part.grid)

        def answer(prompt, _images):
            payload = json.loads(prompt.split("SOURCE:\n", 1)[1])
            html = BeautifulSoup(payload["html_template"], "html.parser")
            for cell in html.find_all("td"):
                cell.string = "A"
            return str(html)

        result, client, source = self.recognize(answer, table=table)
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0][1]), 8)
        self.assertEqual(
            len(BeautifulSoup(result.html, "html.parser").find_all("td")), 8
        )
        self.assertEqual(source.crop_dpis, [288] * 8)

    def test_public_pure_normalizer_accepts_explicit_assets_and_trusted_text(self):
        result = normalize_table_html(
            '<table><tr><td>A <math>R_{bun}</math><img data-source-image="diagram"></td></tr></table>',
            source_images={"diagram": "diagram.png"},
            expected_text="A",
            require_math=True,
        )
        soup = BeautifulSoup(result, "html.parser")
        self.assertEqual(soup.find("img")["src"], "diagram.png")
        self.assertEqual(soup.find("annotation").get_text(), "R_{bun}")

    def test_visible_caption_counts_toward_trusted_source_text(self):
        result = normalize_table_html(
            "<table><caption>Таблица 1</caption><tr><td>Прочность</td></tr></table>",
            expected_text=["Таблица 1", "Прочность"],
        )
        self.assertIn("Таблица 1", result)

    def test_native_mathml_can_be_normalized_again_for_whole_page_tables(self):
        latex = r"x\quad y+\frac{a}{b}+\begin{matrix}1&2\\3&4\end{matrix}"
        first = normalize_table_html(
            "<table><tr><td><math>"
            + latex.replace("&", "&amp;")
            + "</math></td></tr></table>"
        )
        second = normalize_table_html(first)
        self.assertEqual(second, first)

    def test_wrong_multi_letter_index_is_rejected_without_rewriting(self):
        words = [
            word("R"),
            {
                "kind": "text",
                "text": "bun",
                "size": 8,
                "source_page": 0,
                "bbox": (20, 27, 35, 35),
            },
        ]
        wrong, _, _ = self.recognize(
            "<table><tr><td><math>Rb_{un}</math></td></tr></table>", words
        )
        self.assertFalse(wrong.complete)
        right, _, _ = self.recognize(
            "<table><tr><td><math>R_{bun}</math></td></tr></table>", words
        )
        self.assertTrue(right.complete, right.reasons)

    def test_trusted_grid_rejects_flattening_wrong_spans_or_cell_order(self):
        table = trusted_two_by_two()
        answers = (
            '<table><tr><td data-cell-id="c0">A B C D</td></tr></table>',
            '<table><tr><td data-cell-id="c0" rowspan="2" colspan="2">A B C D</td></tr><tr></tr></table>',
            '<table><tr><td data-cell-id="c0">A</td><td data-cell-id="c1">B</td></tr>'
            '<tr><td data-cell-id="c3">D</td><td data-cell-id="c2">C</td></tr></table>',
        )
        for answer in answers:
            with self.subTest(answer=answer):
                result, _, _ = self.recognize(answer, table=table)
                self.assertFalse(result.complete)

    def test_complete_trusted_grid_copies_cell_ids_and_keeps_contents_in_their_cells(
        self,
    ):
        valid = (
            '<table><tr><td data-cell-id="c0">A</td><td data-cell-id="c1">B</td></tr>'
            '<tr><td data-cell-id="c2">C</td><td data-cell-id="c3">D</td></tr></table>'
        )
        result, client, _ = self.recognize(valid, table=trusted_two_by_two())
        self.assertTrue(result.complete, result.reasons)
        self.assertIn("html_template", client.calls[0][0])
        self.assertNotIn("atom_ids", client.calls[0][0])
        swapped = valid.replace('c0">A', 'c0">B').replace('c1">B', 'c1">A')
        result, _, _ = self.recognize(swapped, table=trusted_two_by_two())
        self.assertFalse(result.complete)

    def test_trusted_source_column_proportions_replace_model_widths_and_roundtrip(self):
        table = trusted_two_by_two()
        for grid in (table.grid, table.parts[0].grid):
            for index, cell in enumerate(grid["cells"]):
                left = 0 if index % 2 == 0 else 40
                right = 40 if index % 2 == 0 else 200
                cell["bbox"] = (left, cell["bbox"][1], right, cell["bbox"][3])
        answer = (
            '<table><colgroup><col width="90%"><col width="10%"></colgroup>'
            '<tr><td data-cell-id="c0">A</td><td data-cell-id="c1">B</td></tr>'
            '<tr><td data-cell-id="c2">C</td><td data-cell-id="c3">D</td></tr></table>'
        )
        result, _, _ = self.recognize(answer, table=table)
        self.assertTrue(result.complete, result.reasons)
        columns = BeautifulSoup(result.html, "html.parser").find_all("col")
        self.assertEqual(
            [column["width"] for column in columns], ["20.000%", "80.000%"]
        )
        self.assertEqual(normalize_table_html(result.html), result.html)

    def test_column_percentages_reject_unsafe_or_out_of_range_widths(self):
        for width in (
            "0%",
            "-1%",
            "100.001%",
            "NaN%",
            "10px",
            "calc(100%)",
            "1%;color:red",
        ):
            with self.subTest(width=width), self.assertRaises(ResponseValidationError):
                normalize_table_html(
                    '<table><colgroup><col width="' + width + '"></colgroup>'
                    "<tr><td>A</td></tr></table>"
                )

    def test_native_math_coefficients_cannot_be_omitted_or_changed(self):
        words = []
        for row, value in enumerate(("0,42", "0,41")):
            for x, text in ((10, value), (40, "R")):
                words.append(
                    {
                        "kind": "text",
                        "text": text,
                        "size": 12,
                        "fontname": "CambriaMath",
                        "source_page": 0,
                        "bbox": (x, 10 + row * 30, x + 25, 22 + row * 30),
                    }
                )
        for body in ("<math>x</math>", "<math>0,42R</math><math>0,42R</math>"):
            with self.subTest(body=body):
                result, _, _ = self.recognize(
                    "<table><tr><td>" + body + "</td></tr></table>", words
                )
                self.assertFalse(result.complete)
        result, _, _ = self.recognize(
            "<table><tr><td><math>0,42R</math><br><math>0,41R</math></td></tr></table>",
            words,
        )
        self.assertTrue(result.complete, result.reasons)

    def test_an_explicit_native_negative_math_coefficient_keeps_its_sign(self):
        words = [
            {
                "kind": "text",
                "text": "-0,42",
                "size": 12,
                "fontname": "CambriaMath",
                "source_page": 0,
                "bbox": (10, 10, 40, 22),
            },
            {
                "kind": "text",
                "text": "R",
                "size": 12,
                "fontname": "CambriaMath",
                "source_page": 0,
                "bbox": (40, 10, 50, 22),
            },
        ]
        wrong, _, _ = self.recognize(
            "<table><tr><td><math>0,42R</math></td></tr></table>", words
        )
        self.assertFalse(wrong.complete)
        right, _, _ = self.recognize(
            "<table><tr><td><math>-0,42R</math></td></tr></table>", words
        )
        self.assertTrue(right.complete, right.reasons)

    def test_genuine_large_source_figure_id_resolves_to_native_pixels(self):
        from types import SimpleNamespace

        source = FakeSource(
            [SimpleNamespace(page=0, index=7, bbox=(100, 100, 300, 220))]
        )
        expected = source.native.tobytes()

        def answer(prompt, _images):
            payload = json.loads(prompt.split("SOURCE:\n", 1)[1])
            identity = payload["figures"][0]["id"]
            return (
                '<table><tr><td>A<img data-source-image="'
                + identity
                + '"></td></tr></table>'
            )

        result, _, _ = self.recognize(answer, source=source)
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(len(result.resources), 1)
        self.assertEqual(next(iter(result.resources.values())).tobytes(), expected)
        image = BeautifulSoup(result.html, "html.parser").find("img")
        self.assertIn(image["src"], result.resources)
        self.assertNotIn("data-source-image", image.attrs)
        self.assertEqual(source.resources, [(0, (100, 100, 300, 220))])

    def test_moderate_native_diagram_80_by_40_points_keeps_native_pixels(self):
        from types import SimpleNamespace

        source = FakeSource(
            [SimpleNamespace(page=0, index=7, bbox=(100, 100, 180, 140))]
        )

        def answer(prompt, _images):
            payload = json.loads(prompt.split("SOURCE:\n", 1)[1])
            identity = (
                payload["figures"][0]["id"] if payload["figures"] else "image-p0-i7"
            )
            return (
                '<table><tr><td><img data-source-image="'
                + identity
                + '"></td></tr></table>'
            )

        result, client, _ = self.recognize(answer, source=source)
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(source.resources, [(0, (100, 100, 180, 140))])
        self.assertEqual(
            next(iter(result.resources.values())).tobytes(), source.native.tobytes()
        )

    def test_moderate_scan_diagram_crop_80_by_40_points_keeps_native_pixels(self):
        table = LogicalTable([TablePart(0, (0, 0, 400, 400), None, (600, 500))], None)
        result, client, source = self.recognize(
            '<table><tr><td><img data-source-crop="0:250,250,450,350"></td></tr></table>',
            table=table,
        )
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(source.resources, [(0, (100, 100, 180, 140))])
        self.assertEqual(
            next(iter(result.resources.values())).tobytes(), source.native.tobytes()
        )

    def test_trusted_labels_inside_a_genuine_figure_remain_visible_in_its_native_asset(
        self,
    ):
        from types import SimpleNamespace

        source = FakeSource(
            [SimpleNamespace(page=0, index=7, bbox=(100, 100, 300, 220))]
        )
        labels = [
            {
                "kind": "text",
                "text": "Диаграмма",
                "size": 12,
                "source_page": 0,
                "bbox": (120, 120, 180, 132),
            }
        ]

        def answer(prompt, _images):
            payload = json.loads(prompt.split("SOURCE:\n", 1)[1])
            return (
                '<table><tr><td><img data-source-image="'
                + payload["figures"][0]["id"]
                + '"></td></tr></table>'
            )

        result, _, _ = self.recognize(answer, labels, source=source)
        self.assertTrue(result.complete, result.reasons)
        self.assertEqual(len(result.resources), 1)

    def test_unknown_image_ids_and_formula_or_whole_table_pngs_are_rejected(self):
        from types import SimpleNamespace

        for box in ((10, 10, 50, 30), (0, 0, 600, 450)):
            source = FakeSource([SimpleNamespace(page=0, index=7, bbox=box)])
            result, _, _ = self.recognize(
                '<table><tr><td><img data-source-image="image-p0-i7"></td></tr></table>',
                source=source,
            )
            self.assertFalse(result.complete)
            self.assertEqual(source.resources, [])
        result, _, _ = self.recognize(
            '<table><tr><td><img data-source-image="image-p99-i7"></td></tr></table>'
        )
        self.assertFalse(result.complete)

    def test_raster_table_figure_crop_resolves_to_original_source_pixels(self):
        scanned = LogicalTable([TablePart(0, (0, 0, 600, 450), None, (600, 500))], None)
        for table in (logical(), scanned):
            with self.subTest(scanned=table.grid is None):
                result, client, source = self.recognize(
                    '<table><tr><td><img data-source-crop="0:200,200,600,600"></td></tr></table>',
                    table=table,
                )
                self.assertTrue(result.complete, result.reasons)
                self.assertEqual(len(client.calls), 1)
                self.assertIn("data-source-crop=", client.calls[0][0])
                self.assertEqual(source.resources, [(0, (120, 90, 360, 270))])
                self.assertEqual(
                    next(iter(result.resources.values())).tobytes(),
                    source.native.tobytes(),
                )
                self.assertNotIn("data-source-crop", result.html)

    def test_raster_table_figure_crop_rejects_tiny_whole_part_and_bad_coordinates(self):
        for value in (
            "0:0,0,1000,1000",
            "0:100,100,130,130",
            "8:200,200,600,600",
            "0:-1,200,600,600",
            "0:200,200,1001,600",
            "0:700,200,600,600",
            "0:200.5,200,600,600",
            "javascript:SECRET_PROVIDER_ECHO",
        ):
            with self.subTest(value=value):
                result, _, source = self.recognize(
                    '<table><tr><td><img data-source-crop="'
                    + value
                    + '"></td></tr></table>'
                )
                self.assertFalse(result.complete)
                self.assertEqual(source.resources, [])
                self.assertNotIn("SECRET_PROVIDER_ECHO", str(result.reasons))

    def test_raster_figure_crop_must_stay_in_one_trusted_physical_cell(self):
        boxes = [(0, 0, 300, 450), (300, 0, 600, 450)]
        grid = {
            "page": 0,
            "bbox": (0, 0, 600, 450),
            "trusted": True,
            "cells": [{"bbox": box, "source_page": 0, "atoms": []} for box in boxes],
        }
        table = LogicalTable([TablePart(0, grid["bbox"], grid, (600, 500))], grid)
        result, _, source = self.recognize(
            '<table><tr><td data-cell-id="c0"><img data-source-crop="0:200,200,600,600"></td>'
            '<td data-cell-id="c1">A</td></tr></table>',
            table=table,
        )
        self.assertFalse(result.complete)
        self.assertEqual(source.resources, [])

    def test_known_math_source_cannot_be_replaced_with_a_padded_figure_crop(self):
        table = logical()
        table.parts[0].grid["cells"][0]["atoms"] = [
            {
                "kind": "unknown",
                "content_kind": "math",
                "bbox": (150, 110, 250, 190),
                "source_page": 0,
            }
        ]
        result, _, source = self.recognize(
            '<table><tr><td><img data-source-crop="0:200,200,600,600"></td></tr></table>',
            table=table,
        )
        self.assertFalse(result.complete)
        self.assertEqual(source.resources, [])

    def test_crop_cannot_cover_known_native_expression_without_content_kind_metadata(
        self,
    ):
        words = [
            {
                "kind": "text",
                "text": text,
                "fontname": "CambriaMath",
                "size": 12,
                "bbox": (150 + index * 14, 110, 160 + index * 14, 122),
                "source_page": 0,
            }
            for index, text in enumerate(("x", "=", "y"))
        ]
        result, _, source = self.recognize(
            '<table><tr><td><img data-source-crop="0:200,200,600,600"><math>x=y</math></td></tr></table>',
            words,
        )
        self.assertFalse(result.complete)
        self.assertEqual(source.resources, [])

    def test_crop_can_retain_native_diagram_axis_labels_as_part_of_figure(self):
        from types import SimpleNamespace

        source = FakeSource(
            [SimpleNamespace(page=0, index=7, bbox=(100, 100, 300, 220))]
        )
        labels = [
            {
                "kind": "text",
                "text": text,
                "fontname": "CambriaMath",
                "size": 12,
                "bbox": (120 + index * 14, 120, 130 + index * 14, 132),
                "source_page": 0,
            }
            for index, text in enumerate(("x", "=", "y"))
        ]
        result, _, _ = self.recognize(
            '<table><tr><td><img data-source-crop="0:160,200,520,520"></td></tr></table>',
            labels,
            source=source,
        )
        self.assertTrue(result.complete, result.reasons)

    def test_required_source_formula_cannot_be_ordinary_placeholder_text(self):
        words = [word("x"), word("=", 20), word("0", 30)]
        result, _, _ = self.recognize("<table><tr><td>x = 0</td></tr></table>", words)
        self.assertFalse(result.complete)

    def test_unusable_math_and_formula_images_are_rejected(self):
        for answer in (
            r"<table><tr><td><math>\unknowncommand{x}</math></td></tr></table>",
            "<table><tr><td><math><mrow><mi>x</mi></mrow></math></td></tr></table>",
            "<table><tr><td>R<sub>bun</sub></td></tr></table>",
        ):
            with self.subTest(answer=answer):
                result, _, _ = self.recognize(answer)
                self.assertFalse(result.complete)

    def test_figures_cannot_be_hidden_inside_a_formula_or_nested_math(self):
        for body in (
            '<span data-latex="R"><img data-source-image="diagram"></span>',
            '<math><semantics><img data-source-image="diagram"><annotation encoding="application/x-tex">R</annotation></semantics></math>',
            "<math><math>R</math></math>",
        ):
            with self.subTest(body=body), self.assertRaises(ResponseValidationError):
                normalize_table_html(
                    "<table><tr><td>" + body + "</td></tr></table>",
                    source_images={
                        "diagram": {"name": "diagram.png", "trusted_text": "Label"}
                    },
                    expected_text="Label",
                )

    def test_image_limit_fails_explicitly_before_any_request_or_crop(self):
        source = FakeSource()
        client = FakeClient("<table><tr><td>A</td></tr></table>")
        client.cloud_max_images = 2
        result = recognize_whole_table(source, logical(pages=(0, 1, 2)), client)
        self.assertFalse(result.complete)
        self.assertEqual((client.calls, source.crops, result.markdown), ([], [], ""))

    def test_provider_failure_and_table_warnings_remain_incomplete(self):
        failed, _, _ = self.recognize(RecognitionFailure("Cloud budget exhausted"))
        self.assertFalse(failed.complete)
        self.assertEqual((failed.markdown, failed.html), ("", ""))
        warned, _, _ = self.recognize(
            "<table><tr><td>A</td></tr></table>",
            table=logical(warnings=("Uncertain continuation",)),
        )
        self.assertFalse(warned.complete)
        self.assertIn("Uncertain continuation", warned.reasons)

    def test_three_page_real_pdf_uses_physical_parts_in_one_request(self):
        from marker.cloud.source import PdfSource
        from marker.cloud.tables import detect_tables, group_tables

        with PdfSource(FIXTURES / "three_page_continuation.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            before = deepcopy([part.grid for part in table.parts])

            def answer(prompt, _images):
                payload = json.loads(prompt.split("SOURCE:\n", 1)[1])
                continuations = payload["cell_continuations"]
                self.assertTrue(continuations)
                self.assertTrue(
                    all(len(fragments) > 1 for fragments in continuations.values())
                )
                self.assertIn("inside that same cell", prompt)
                html = BeautifulSoup(payload["html_template"], "html.parser")
                for cell in html.find_all(["td", "th"]):
                    from marker.cloud.content import source_lines
                    from marker.processors.table_geometry import grid_positions

                    index = int(cell["data-cell-id"][1:])
                    ordered = sorted(
                        grid_positions(table.grid)[2], key=lambda item: item[:2]
                    )
                    original = ordered[index][4]
                    cell.clear()
                    for line in source_lines(original.get("atoms", [])):
                        paragraph = html.new_tag("p")
                        text = " ".join(atom.get("text", "") for atom in line).strip()
                        if text:
                            paragraph.string = text
                        if any(atom.get("kind") == "image" for atom in line):
                            paragraph.append(
                                BeautifulSoup("<math>x</math>", "html.parser").math
                            )
                        cell.append(paragraph)
                return str(html)

            client = FakeClient(answer)
            result = recognize_whole_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(len(client.calls[0][1]), 3)
            self.assertTrue(
                all(
                    image.convert("L").getextrema()[0] < 240
                    for image in client.calls[0][1]
                )
            )
            self.assertEqual([part.grid for part in table.parts], before)


if __name__ == "__main__":
    unittest.main()
