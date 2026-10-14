"""Actual PDF pixels plus deterministic HTTP responses for visible content."""

import base64
import json
import tempfile
import unittest
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import httpx
import openai
from bs4 import BeautifulSoup
from PIL import Image

from marker.cloud.client import CloudClient
from marker.cloud.regions import analyze_page
from marker.cloud.source import PdfSource
from marker.cloud.tables import detect_tables, group_tables
from marker.output import save_output
from marker.renderers.markdown import MarkdownOutput, Markdownify

FIXTURES = Path(__file__).parent / "fixtures"


def offline_client(responder, config=None):
    observed = []

    def respond(request):
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        images = [item for item in content if item["type"] == "image_url"]
        assert images, "Recognition must inspect actual PDF crop pixels"
        for item in images:
            with Image.open(
                BytesIO(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]))
            ) as image:
                assert max(image.size) <= 1536
                assert image.convert("L").getextrema()[0] < 240
        prompt = next(item["text"] for item in content if item["type"] == "text")
        source = json.JSONDecoder().raw_decode(prompt.split("SOURCE:\n", 1)[1])[0]
        observed.append(source)
        result = responder(source, len(observed))
        return httpx.Response(
            200,
            json={
                "id": "offline-content",
                "object": "chat.completion",
                "created": 0,
                "model": "offline",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
            },
        )

    sdk = openai.OpenAI(
        api_key="offline-dummy",
        base_url="https://offline.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    return CloudClient(config, client=sdk), observed


def table_echo(source, _):
    cells = []
    for cell in source["cells"]:
        lines = []
        for line in cell["lines"]:
            atoms = line["atoms"]
            parts = []
            if line["math_required"]:
                parts.append(
                    {
                        "kind": "math",
                        "latex": r"x \geq 0",
                        "atom_ids": [a["id"] for a in atoms],
                    }
                )
            else:
                for atom in atoms:
                    parts.append(
                        {
                            "kind": "figure" if atom["kind"] == "image" else "text",
                            "text": atom.get("text", ""),
                            "atom_ids": [atom["id"]],
                        }
                    )
            lines.append({"id": line["id"], "parts": parts})
        cells.append(
            {key: cell[key] for key in ("id", "row", "column", "rowspan", "colspan")}
            | {"lines": lines}
        )
    return {"cells": cells}


def markdown_converter():
    return Markdownify(
        False,
        "-",
        ("$", "$"),
        ("$$", "$$"),
        True,
        escape_dollars=True,
        escape_underscores=True,
        escape_asterisks=True,
    )


class ContentTests(unittest.TestCase):
    def content(self):
        from marker.cloud import content

        return content

    def test_split_glyphs_and_text_layer_superscript_become_whole_expressions(self):
        content = self.content()
        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            original = analyze_page(source, 0)
            regions = content.group_expression_regions(original)
            math = [r for r in regions if r.kind == "math"]
            self.assertEqual(len(math), 2)
            inequality = next(
                r for r in math if any(a["kind"] == "image" for a in r.atoms)
            )
            self.assertEqual(
                [a.get("text") for a in inequality.atoms if a["kind"] == "text"],
                ["x", "0"],
            )
            original_atoms = [
                content.atom_id(a, r.page) for r in original for a in r.atoms
            ]
            grouped_atoms = [
                content.atom_id(a, r.page) for r in regions for a in r.atoms
            ]
            self.assertCountEqual(grouped_atoms, original_atoms)
            self.assertEqual(len(grouped_atoms), len(set(grouped_atoms)))
            client, observed = offline_client(
                lambda s, n: {
                    "kind": "math",
                    "latex": r"x \geq 0" if n == 1 else "x^{2}",
                    "resolved": True,
                }
            )
            results = [
                content.recognize_region(source, region, client)
                for region in sorted(math, key=lambda r: r.bbox[1])
            ]
            self.assertTrue(all(r.complete for r in results))
            self.assertIn(r"\geq", results[0].markdown)
            self.assertEqual(results[1].markdown, "$$x^{2}$$")
            self.assertEqual(client.request_count, 2)
            self.assertEqual(len(results[0].resources), 1)
            audit = next(iter(results[0].resources.values()))
            self.assertGreater(audit.width, 250)
            self.assertGreaterEqual(len(observed[0]["atoms"]), 3)

    def test_native_figure_is_linked_at_native_resolution(self):
        content = self.content()
        with PdfSource(FIXTURES / "native_figure.pdf") as source:
            region = next(r for r in analyze_page(source, 0) if r.kind == "image")
            client, _ = offline_client(
                lambda s, n: {"kind": "figure", "resolved": True}
            )
            result = content.recognize_region(source, region, client)
            self.assertTrue(result.complete)
            self.assertEqual(next(iter(result.resources.values())).size, (2048, 1024))
            self.assertIn(".png)", result.markdown)

    def test_table_math_is_native_mathml_and_diagram_is_a_resource(self):
        content = self.content()
        from marker.cloud.tables import recognize_table

        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            original = deepcopy(table.grid)
            client, _ = offline_client(table_echo)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            self.assertEqual(table.grid, original)
            soup = BeautifulSoup(result.html, "html.parser")
            annotation = soup.find("annotation", encoding="application/x-tex")
            self.assertEqual(annotation.text, r"x \geq 0")
            self.assertIsNotNone(soup.find("math").find("semantics").find("mrow"))
            diagram = soup.find("img")
            self.assertIsNotNone(diagram)
            self.assertEqual(result.resources[diagram["src"]].size, (240, 160))
            self.assertGreater(
                len(result.resources),
                1,
                "The complete formula audit crop must remain available",
            )
            converted = markdown_converter().convert(result.html)
            self.assertIsNotNone(
                BeautifulSoup(converted, "html.parser").find("semantics")
            )
            self.assertNotIn("$x", converted)

    def test_phi_variants_have_exact_annotations_and_distinct_glyphs(self):
        content = self.content()
        variants = [
            BeautifulSoup(content.latex_to_mathml(latex), "html.parser")
            for latex in (r"\phi", r"\varphi")
        ]
        self.assertEqual(
            [s.find("annotation").text for s in variants], [r"\phi", r"\varphi"]
        )
        self.assertEqual([s.find("mi").text for s in variants], ["ϕ", "φ"])
        for soup in variants:
            soup.find("math")["display"] = "inline"
            converted = markdown_converter().convert(
                "<table><tr><td>" + str(soup) + "</td></tr></table>"
            )
            self.assertIn("<semantics>", converted)
        self.assertIn("$x$", markdown_converter().convert("<p><math>x</math></p>"))
        self.assertIn(
            "$$x$$", markdown_converter().convert('<math display="block">x</math>')
        )
        self.assertIn(
            r"$\phi$", markdown_converter().convert(content.latex_to_mathml(r"\phi"))
        )
        self.assertIn(
            r"$$\varphi$$",
            markdown_converter().convert(
                content.latex_to_mathml(r"\varphi", display="block")
            ),
        )

    def test_125_rows_use_bounded_windows_without_duplicate_headers_or_rows(self):
        self.content()
        from marker.cloud.tables import recognize_table

        with PdfSource(FIXTURES / "rows_125.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            client, observed = offline_client(table_echo)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            rows = BeautifulSoup(result.html, "html.parser").find_all("tr")
            self.assertEqual(len(rows), 125)
            self.assertEqual(
                [r.find("td").get_text(strip=True) for r in rows],
                [str(i) for i in range(125)],
            )
            self.assertGreater(client.request_count, 1)
            owned = [c["id"] for window in observed for c in window["cells"]]
            self.assertEqual(len(owned), 250)
            self.assertEqual(len(owned), len(set(owned)))
            self.assertTrue(all(window["headers"] for window in observed[1:]))

    def test_invalid_trusted_spans_and_unresolved_math_remain_incomplete(self):
        self.content()
        from marker.cloud.tables import recognize_table

        for failure in ("span", "unresolved", "ownership"):
            with (
                self.subTest(failure=failure),
                PdfSource(FIXTURES / "split_formula.pdf") as source,
            ):
                table = group_tables(detect_tables(source), None)[0]

                def invalid(s, n):
                    result = table_echo(s, n)
                    if failure == "span":
                        result["cells"][0]["colspan"] = 99
                    else:
                        target = next(
                            (
                                part
                                for c in result["cells"]
                                for line in c["lines"]
                                for part in line["parts"]
                                if part["kind"] == "math"
                            ),
                            None,
                        )
                        if target:
                            if failure == "unresolved":
                                target["latex"] = ""
                            else:
                                target["atom_ids"] = ["foreign-atom"]
                    return result

                client, _ = offline_client(invalid, {"cloud_max_attempts": 2})
                result = recognize_table(source, table, client)
                self.assertFalse(result.complete)
                self.assertTrue(result.reasons)
                self.assertTrue(result.resources)

    def test_untrusted_grid_overlaps_and_missing_ids_are_rejected(self):
        self.content()
        from marker.cloud.tables import recognize_table

        with PdfSource(FIXTURES / "borderless_tables.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]

            def invalid(s, n):
                result = table_echo(s, n)
                if len(result["cells"]) > 1:
                    result["cells"][1]["row"] = result["cells"][0]["row"]
                    result["cells"][1]["column"] = result["cells"][0]["column"]
                else:
                    result["cells"][0]["id"] = "unknown-cell"
                return result

            client, _ = offline_client(invalid, {"cloud_max_attempts": 1})
            result = recognize_table(source, table, client)
            self.assertFalse(result.complete)

    def test_multi_expression_cell_preserves_source_baselines_and_neighbor_values(self):
        self.content()
        from marker.cloud.tables import recognize_table

        with PdfSource(FIXTURES / "rows_125.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            grid = deepcopy(table.grid)
            merged = []
            for column in (0, 1):
                cells = sorted(
                    [c for c in grid["cells"] if c["bbox"][0] == (50, 300)[column]],
                    key=lambda c: c["bbox"][1],
                )[:3]
                cell = deepcopy(cells[0])
                cell["bbox"] = (cell["bbox"][0], 50, cell["bbox"][2], 104)
                cell["source_bbox"] = cell["bbox"]
                cell["source_boxes"] = ((0, cell["bbox"]),)
                cell["atoms"] = [a for c in cells for a in c["atoms"]]
                merged.append(cell)
            grid["cells"] = merged
            grid["bbox"] = (50, 50, 550, 104)
            table.grid = grid

            def formulas(s, n):
                result = table_echo(s, n)
                for cell in result["cells"]:
                    if cell["column"] == 0:
                        assigned = next(c for c in s["cells"] if c["id"] == cell["id"])
                        for line in cell["lines"]:
                            owned = next(
                                l for l in assigned["lines"] if l["id"] == line["id"]
                            )
                            latex = {"0": r"\alpha_1", "1": r"\alpha_2", "2": r"\beta"}[
                                owned["atoms"][0]["text"]
                            ]
                            for part in line["parts"]:
                                part.update(kind="math", latex=latex)
                return result

            client, _ = offline_client(formulas)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            soup = BeautifulSoup(result.html, "html.parser")
            cells = soup.find_all("td")
            mathlines = cells[0].find_all(attrs={"data-source-line": True})
            values = cells[1].find_all(attrs={"data-source-line": True})
            self.assertEqual(
                [line.find("annotation").text for line in mathlines],
                [r"\alpha_1", r"\alpha_2", r"\beta"],
            )
            self.assertEqual(
                [line.get_text(strip=True) for line in values],
                ["value-000", "value-001", "value-002"],
            )
            self.assertEqual(
                [line["data-source-y"] for line in mathlines],
                [line["data-source-y"] for line in values],
            )

    def test_broken_mathml_conversion_is_incomplete_with_audit(self):
        content = self.content()
        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            region = next(
                r
                for r in content.group_expression_regions(analyze_page(source, 0))
                if r.kind == "math"
            )
            client, _ = offline_client(
                lambda s, n: {"kind": "math", "latex": r"\frac{", "resolved": True}
            )
            result = content.recognize_region(source, region, client)
            self.assertFalse(result.complete)
            self.assertTrue(result.resources)

    def test_required_region_math_is_repaired_within_shared_attempt_cap(self):
        content = self.content()
        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            region = next(
                r
                for r in content.group_expression_regions(analyze_page(source, 0))
                if r.kind == "math"
            )

            def response(s, n):
                return (
                    {"kind": "figure", "resolved": True}
                    if n == 1
                    else {"kind": "math", "latex": r"x \geq 0", "resolved": True}
                )

            client, _ = offline_client(response, {"cloud_max_attempts": 2})
            result = content.recognize_region(source, region, client)
            self.assertTrue(result.complete, result.reasons)
            self.assertEqual(client.request_count, 2)

    def test_one_logical_table_keeps_three_pages_and_split_cell_lines(self):
        self.content()
        from marker.cloud.tables import recognize_table

        with PdfSource(FIXTURES / "three_page_continuation.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            before = deepcopy(table.grid)
            client, _ = offline_client(table_echo)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            soup = BeautifulSoup(result.html, "html.parser")
            self.assertEqual(len(soup.find_all("table")), 1)
            self.assertEqual(table.grid, before)
            self.assertEqual(
                {
                    int(line["data-source-page"])
                    for line in soup.find_all(attrs={"data-source-line": True})
                },
                {0, 1, 2},
            )
            self.assertIsNotNone(soup.find("td", rowspan="5"))
            self.assertEqual(soup.get_text(" ", strip=True).count("state continues"), 1)

    def test_missing_grid_keeps_full_density_table_source_and_incomplete_status(self):
        self.content()
        from marker.cloud.tables import recognize_table

        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            table.grid = None
            client, _ = offline_client(table_echo)
            result = recognize_table(source, table, client)
            self.assertFalse(result.complete)
            self.assertEqual(client.request_count, 0)
            self.assertGreater(next(iter(result.resources.values())).width, 1536)

    def test_scan_layout_cells_can_transcribe_raster_text_and_math(self):
        self.content()
        from marker.cloud.tables import LogicalTable, TablePart, recognize_table

        with PdfSource(FIXTURES / "scan.pdf") as source:
            cells = []
            for column, bbox in enumerate(
                ((30, 60, 130, 87), (130, 60, 230, 87), (230, 60, 330, 87))
            ):
                atom = {
                    "kind": "unknown",
                    "bbox": bbox,
                    "source_bbox": bbox,
                    "source_page": 0,
                }
                cells.append(
                    {
                        "bbox": bbox,
                        "source_bbox": bbox,
                        "source_page": 0,
                        "source_boxes": ((0, bbox),),
                        "atoms": [atom],
                    }
                )
            grid = {
                "page": 0,
                "bbox": (30, 60, 330, 87),
                "cells": cells,
                "trusted": False,
            }
            table = LogicalTable(
                [TablePart(0, grid["bbox"], grid, source.page_size(0), source=source)],
                grid,
            )

            def response(s, n):
                result = table_echo(s, n)
                for cell in result["cells"]:
                    for line in cell["lines"]:
                        for part in line["parts"]:
                            if cell["column"] == 0:
                                part.update(kind="math", latex=r"\alpha")
                            else:
                                part.update(
                                    kind="text", text=("12", "24")[cell["column"] - 1]
                                )
                return result

            client, _ = offline_client(response)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            soup = BeautifulSoup(result.html, "html.parser")
            self.assertEqual(soup.find("annotation").text, r"\alpha")
            self.assertEqual(
                [c.get_text(strip=True) for c in soup.find_all("td")[1:]], ["12", "24"]
            )

    def test_seven_column_coefficient_ink_keeps_subscripts_and_three_matching_lines(
        self,
    ):
        content = self.content()
        from marker.cloud.tables import recognize_table

        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            body = sorted(
                [c for c in table.grid["cells"] if c["bbox"][1] == 90],
                key=lambda c: c["bbox"][0],
            )
            self.assertEqual(len(body), 7)
            self.assertEqual(
                [
                    [a["text"] for a in line]
                    for line in content.source_lines(body[0]["atoms"])
                ],
                [["α", "1"], ["α", "2"], ["β"]],
            )

            def response(s, n):
                result = table_echo(s, n)
                lookup = {c["id"]: c for c in s["cells"]}
                for cell in result["cells"]:
                    if cell["row"] == 1 and cell["column"] == 0:
                        for line in cell["lines"]:
                            source_line = next(
                                l
                                for l in lookup[cell["id"]]["lines"]
                                if l["id"] == line["id"]
                            )
                            text = "".join(a["text"] for a in source_line["atoms"])
                            latex = {
                                "α1": r"\alpha_1",
                                "α2": r"\alpha_2",
                                "β": r"\beta",
                            }[text]
                            line["parts"] = [
                                {
                                    "kind": "math",
                                    "latex": latex,
                                    "atom_ids": [a["id"] for a in source_line["atoms"]],
                                }
                            ]
                return result

            client, _ = offline_client(response)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            soup = BeautifulSoup(result.html, "html.parser")
            cells = soup.find_all("tr")[1].find_all("td")
            coefficients = cells[0].find_all(attrs={"data-source-line": True})
            self.assertEqual(
                [line.find("annotation").text for line in coefficients],
                [r"\alpha_1", r"\alpha_2", r"\beta"],
            )
            coefficient_y = [float(line["data-source-y"]) for line in coefficients]
            for column, cell in enumerate(cells[1:], start=1):
                lines = cell.find_all(attrs={"data-source-line": True})
                self.assertEqual(
                    [line.get_text(strip=True) for line in lines],
                    [f"{row},{column}" for row in (1, 2, 3)],
                )
                self.assertTrue(
                    all(
                        abs(float(line["data-source-y"]) - y) < 3
                        for line, y in zip(lines, coefficient_y)
                    )
                )
            self.assertEqual(
                len([name for name in result.resources if name.startswith("formula")]),
                3,
            )

    def test_trusted_vector_grid_can_ocr_ordinary_raster_cell_text(self):
        self.content()
        from marker.cloud.tables import recognize_table

        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            self.assertTrue(table.grid["trusted"])
            header = next(
                cell for cell in table.grid["cells"] if cell["bbox"][:2] == (464, 60)
            )
            self.assertEqual([a["kind"] for a in header["atoms"]], ["image"])

            def response(s, n):
                result = table_echo(s, n)
                for cell in result["cells"]:
                    for line in cell["lines"]:
                        for part in line["parts"]:
                            if part["kind"] == "figure":
                                part.update(kind="text", text="Value 6")
                return result

            client, _ = offline_client(response)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            self.assertIn(
                "Value 6", BeautifulSoup(result.html, "html.parser").get_text()
            )

    def test_png_output_preserves_alpha(self):
        pixels = Image.new("RGBA", (3, 2), (10, 20, 30, 0))
        pixels.putpixel((1, 1), (40, 50, 60, 127))
        output = MarkdownOutput(
            markdown="![figure](figure.png)", images={"figure.png": pixels}, metadata={}
        )
        with tempfile.TemporaryDirectory() as directory:
            save_output(output, directory, "doc")
            with Image.open(Path(directory) / "figure.png") as written:
                self.assertEqual(written.mode, "RGBA")
                self.assertEqual(written.tobytes(), pixels.tobytes())


if __name__ == "__main__":
    unittest.main()
