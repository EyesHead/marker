"""Source geometry aligns whole recognized paragraphs, never individual glyphs."""

from copy import deepcopy
import re
import unittest

from bs4 import BeautifulSoup

from marker.cloud.content import latex_to_mathml
from marker.cloud.table_layout import TableLayoutError, apply_source_table_layout


def atom(text, x, y, *, page=0, layout_y=None):
    result = {
        "kind": "text",
        "text": text,
        "size": 9,
        "fontname": "ArialMT",
        "source_page": page,
        "bbox": (x, y, x + max(5, len(text) * 4), y + 9),
    }
    if layout_y is not None:
        result["layout_bbox"] = (x, layout_y, result["bbox"][2], layout_y + 9)
    return result


def alignment_grid():
    return {
        "page": 0,
        "bbox": (0, 0, 260, 120),
        "trusted": True,
        "cells": [
            {
                "bbox": (0, 0, 200, 120),
                "atoms": [
                    atom(text, 4, y)
                    for text, y in (
                        ("Introduction one", 10),
                        ("Introduction two", 22),
                        ("200", 85),
                        ("10", 105),
                    )
                ],
            },
            {
                "bbox": (200, 0, 260, 120),
                "atoms": [atom("1", 225, 85), atom("4", 225, 105)],
            },
        ],
    }


HTML = (
    '<table><tr><td data-cell-id="c0"><p>Introduction one</p><p>Introduction two</p>'
    '<p>200</p><p>10</p></td><td data-cell-id="c1"><p>1</p><p>4</p></td></tr></table>'
)


def point(style, name):
    return float(re.search(r"(?:^|;)" + name + r":([0-9.-]+)pt(?:;|$)", style)[1])


class SourceTableLayoutTests(unittest.TestCase):
    def test_values_share_source_y_even_when_neighbor_has_extra_intro_paragraphs(self):
        grid = alignment_grid()
        before = deepcopy(grid)
        html = apply_source_table_layout(HTML, grid)
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = {p.get_text(): p for p in soup.find_all("p")}
        for left, right, y in (("200", "1", 85), ("10", "4", 105)):
            self.assertEqual(point(paragraphs[left]["style"], "top"), y)
            self.assertEqual(point(paragraphs[right]["style"], "top"), y)
        self.assertEqual(point(paragraphs["Introduction one"]["style"], "top"), 10)
        self.assertIn("vertical-align:top", soup.find("td")["style"])
        self.assertEqual(point(soup.table["style"], "width"), 260)
        self.assertEqual(grid, before)

    def test_joined_pages_use_layout_bbox_without_mutating_physical_points(self):
        grid = alignment_grid()
        for cell in grid["cells"]:
            cell["layout_bbox"] = (cell["bbox"][0], 700, cell["bbox"][2], 820)
            for source in cell["atoms"]:
                source["layout_bbox"] = (
                    source["bbox"][0],
                    source["bbox"][1] + 700,
                    source["bbox"][2],
                    source["bbox"][3] + 700,
                )
        before = deepcopy(grid)
        soup = BeautifulSoup(apply_source_table_layout(HTML, grid), "html.parser")
        self.assertEqual(point(soup.find("p", string="1")["style"], "top"), 85)
        self.assertEqual(grid, before)

    def test_math_annotations_and_figure_sources_are_unchanged(self):
        grid = {
            "page": 0,
            "bbox": (0, 0, 100, 80),
            "trusted": True,
            "cells": [
                {
                    "bbox": (0, 0, 100, 80),
                    "atoms": [
                        {"kind": "image", "bbox": (10, 12, 55, 30), "source_page": 0},
                        {"kind": "image", "bbox": (10, 45, 90, 75), "source_page": 0},
                    ],
                }
            ],
        }
        body = (
            r'<table><tr><td data-cell-id="c0"><p><math><semantics><mi>R</mi>'
            r'<annotation encoding="application/x-tex">0,42R_{bun}</annotation></semantics></math></p>'
            '<p><img src="figure.png" style="width:80.000pt;height:30.000pt"></p></td></tr></table>'
        )
        soup = BeautifulSoup(apply_source_table_layout(body, grid), "html.parser")
        self.assertEqual(soup.find("annotation").get_text(), r"0,42R_{bun}")
        self.assertEqual(soup.find("img")["src"], "figure.png")
        self.assertEqual(soup.find("img")["style"], "width:80.000pt;height:30.000pt")

    def test_direct_one_baseline_contents_work_without_new_model_slots(self):
        grid = {
            "page": 0,
            "bbox": (0, 0, 50, 30),
            "trusted": True,
            "cells": [{"bbox": (0, 0, 50, 30), "atoms": [atom("Value", 5, 8)]}],
        }
        soup = BeautifulSoup(
            apply_source_table_layout(
                '<table><tr><td data-cell-id="c0">Value</td></tr></table>', grid
            ),
            "html.parser",
        )
        self.assertEqual(soup.find("p").get_text(), "Value")
        self.assertEqual(point(soup.find("p")["style"], "top"), 8)

    def test_mathml_decimal_nodes_remain_one_native_numeric_value(self):
        grid = {
            "page": 0,
            "bbox": (0, 0, 50, 30),
            "trusted": True,
            "cells": [{"bbox": (0, 0, 50, 30), "atoms": [atom("1,35", 5, 8)]}],
        }
        body = (
            '<table><tr><td data-cell-id="c0"><p><math><semantics><mrow>'
            "<mn>1</mn><mo>,</mo><mn>35</mn></mrow>"
            '<annotation encoding="application/x-tex">1,35</annotation></semantics></math>'
            "</p></td></tr></table>"
        )
        soup = BeautifulSoup(apply_source_table_layout(body, grid), "html.parser")
        self.assertEqual(soup.find("annotation").get_text(), "1,35")

    def test_native_fraction_is_not_concatenated_during_baseline_comparison(self):
        grid = {
            "page": 0,
            "bbox": (0, 0, 50, 30),
            "trusted": True,
            "cells": [{"bbox": (0, 0, 50, 30), "atoms": [atom("1/2", 5, 8)]}],
        }
        body = (
            '<table><tr><td data-cell-id="c0"><p><math><semantics>'
            "<mfrac><mn>1</mn><mn>2</mn></mfrac>"
            '<annotation encoding="application/x-tex">\\frac{1}{2}</annotation>'
            "</semantics></math></p></td></tr></table>"
        )
        soup = BeautifulSoup(apply_source_table_layout(body, grid), "html.parser")
        self.assertEqual(soup.find("annotation").get_text(), r"\frac{1}{2}")

    def test_stacked_native_fraction_remains_one_complete_math_object(self):
        atoms = [
            dict(
                atom(text, 20, top),
                bbox=(20, top, 30, top + 10),
                fontname="CambriaMath",
                size=10,
            )
            for text, top in (("x", 10), ("y", 30))
        ]
        grid = {
            "page": 0,
            "bbox": (0, 0, 60, 60),
            "trusted": True,
            "cells": [{"bbox": (0, 0, 60, 60), "atoms": atoms}],
        }
        before = deepcopy(grid)
        latex = r"\frac{x}{y}"
        body = (
            '<table><tr><td data-cell-id="c0"><p>'
            + latex_to_mathml(latex)
            + "</p></td></tr></table>"
        )
        soup = BeautifulSoup(apply_source_table_layout(body, grid), "html.parser")
        self.assertEqual(len(soup.find_all("p")), 1)
        self.assertEqual(len(soup.find_all("mfrac")), 1)
        self.assertEqual(soup.find("annotation").get_text(), latex)
        self.assertEqual(point(soup.p["style"], "top"), 10)
        self.assertEqual(point(soup.p["style"], "left"), 20)
        self.assertEqual(grid, before)

    def test_native_matrix_rows_remain_one_complete_math_object(self):
        atoms = [
            dict(atom(text, x, y), fontname="CambriaMath")
            for text, x, y in (
                ("a", 10, 10),
                ("b", 30, 10),
                ("c", 10, 30),
                ("d", 30, 30),
            )
        ]
        grid = {
            "page": 0,
            "bbox": (0, 0, 60, 60),
            "trusted": True,
            "cells": [{"bbox": (0, 0, 60, 60), "atoms": atoms}],
        }
        latex = r"\begin{matrix}a&b\\c&d\end{matrix}"
        body = (
            '<table><tr><td data-cell-id="c0"><p>'
            + latex_to_mathml(latex)
            + "</p></td></tr></table>"
        )
        soup = BeautifulSoup(apply_source_table_layout(body, grid), "html.parser")
        self.assertEqual(len(soup.find_all("p")), 1)
        self.assertEqual(len(soup.find_all("mtr")), 2)
        self.assertEqual(soup.find("annotation").get_text(), latex)
        self.assertEqual(point(soup.p["style"], "top"), 10)

    def test_multiline_math_exception_never_combines_ordinary_source_rows(self):
        for texts in (("1", "4"), ("Introduction", "y")):
            with self.subTest(texts=texts):
                atoms = [atom(text, 10, y) for text, y in zip(texts, (10, 30))]
                grid = {
                    "page": 0,
                    "bbox": (0, 0, 90, 60),
                    "trusted": True,
                    "cells": [{"bbox": (0, 0, 90, 60), "atoms": atoms}],
                }
                body = (
                    '<table><tr><td data-cell-id="c0"><p>'
                    + latex_to_mathml(r"\frac{1}{4}")
                    + "</p></td></tr></table>"
                )
                with self.assertRaises(TableLayoutError):
                    apply_source_table_layout(body, grid)

    def test_multiline_math_exception_requires_one_pure_vertical_math_object(self):
        atoms = [
            dict(atom(text, 20, top), fontname="CambriaMath")
            for text, top in (("x", 10), ("y", 30))
        ]
        grid = {
            "page": 0,
            "bbox": (0, 0, 60, 60),
            "trusted": True,
            "cells": [{"bbox": (0, 0, 60, 60), "atoms": atoms}],
        }
        for content in (
            latex_to_mathml("x+y"),
            latex_to_mathml(r"\frac{x}{y}") + "ordinary text",
            latex_to_mathml(r"\frac{x}{y}") + latex_to_mathml("z"),
        ):
            with self.subTest(content=content), self.assertRaises(TableLayoutError):
                apply_source_table_layout(
                    '<table><tr><td data-cell-id="c0"><p>'
                    + content
                    + "</p></td></tr></table>",
                    grid,
                )

    def test_explicit_br_slots_are_aligned_as_complete_lines(self):
        grid = {
            "page": 0,
            "bbox": (0, 0, 100, 50),
            "trusted": True,
            "cells": [
                {
                    "bbox": (0, 0, 100, 50),
                    "atoms": [atom("One", 5, 5), atom("Two", 5, 35)],
                }
            ],
        }
        soup = BeautifulSoup(
            apply_source_table_layout(
                '<table><tr><td data-cell-id="c0"><p>One<br>Two</p></td></tr></table>',
                grid,
            ),
            "html.parser",
        )
        self.assertEqual([p.get_text() for p in soup.find_all("p")], ["One", "Two"])
        self.assertEqual(
            [point(p["style"], "top") for p in soup.find_all("p")], [5, 35]
        )

    def test_ambiguous_paragraph_count_fails_with_static_cell_reason(self):
        changed = HTML.replace("<p>1</p><p>4</p>", "<p>SECRET_SOURCE_WORD 1 4</p>")
        with self.assertRaises(TableLayoutError) as error:
            apply_source_table_layout(changed, alignment_grid())
        self.assertIn("c1", str(error.exception))
        self.assertNotIn("SECRET_SOURCE_WORD", str(error.exception))
        self.assertEqual(error.exception.reason, str(error.exception))

    def test_trusted_native_paragraph_order_cannot_silently_swap_values(self):
        changed = HTML.replace("<p>1</p><p>4</p>", "<p>4</p><p>1</p>")
        with self.assertRaises(TableLayoutError):
            apply_source_table_layout(changed, alignment_grid())

    def test_missing_or_duplicate_source_cell_ids_fail(self):
        for changed in (
            HTML.replace('data-cell-id="c1"', ""),
            HTML.replace('data-cell-id="c1"', 'data-cell-id="c0"'),
        ):
            with self.subTest(changed=changed), self.assertRaises(TableLayoutError):
                apply_source_table_layout(changed, alignment_grid())

    def test_cached_html_cannot_swap_physical_source_rows(self):
        grid = {
            "page": 0,
            "bbox": (0, 0, 100, 60),
            "trusted": True,
            "cells": [
                {"bbox": (0, 0, 100, 30), "atoms": [atom("A", 10, 5)]},
                {"bbox": (0, 30, 100, 60), "atoms": [atom("B", 10, 35)]},
            ],
        }
        body = (
            '<table><tr><td data-cell-id="c1"><p>B</p></td></tr>'
            '<tr><td data-cell-id="c0"><p>A</p></td></tr></table>'
        )
        with self.assertRaises(TableLayoutError):
            apply_source_table_layout(body, grid)

    def test_cached_html_cannot_replace_source_rowspan_with_colspan(self):
        grid = {
            "page": 0,
            "bbox": (0, 0, 100, 60),
            "trusted": True,
            "cells": [
                {"bbox": (0, 0, 50, 60), "atoms": [atom("A", 10, 5)]},
                {"bbox": (50, 0, 100, 30), "atoms": [atom("B", 60, 5)]},
                {"bbox": (50, 30, 100, 60), "atoms": [atom("C", 60, 35)]},
            ],
        }
        body = (
            '<table><tr><td data-cell-id="c0" colspan="2"><p>A</p></td></tr>'
            '<tr><td data-cell-id="c1"><p>B</p></td><td data-cell-id="c2"><p>C</p></td></tr></table>'
        )
        with self.assertRaises(TableLayoutError):
            apply_source_table_layout(body, grid)

    def test_untrusted_and_absent_grids_preserve_normal_flow(self):
        grid = alignment_grid()
        grid["trusted"] = False
        self.assertEqual(apply_source_table_layout(HTML, grid), HTML)
        self.assertEqual(apply_source_table_layout(HTML, None), HTML)

    def test_model_css_is_never_accepted_by_postprocessor(self):
        with self.assertRaises(TableLayoutError) as error:
            apply_source_table_layout(
                HTML.replace("<table>", '<table style="background:url(SECRET)">'),
                alignment_grid(),
            )
        self.assertNotIn("SECRET", str(error.exception))


if __name__ == "__main__":
    unittest.main()
