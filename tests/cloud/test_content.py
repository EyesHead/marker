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


def math_font_pdf(path, *, ruled=False, separator="", suffix="where"):
    """Small real PDF with explicit math font metadata and no optional test deps."""
    widths = " ".join(["600"] * 256)
    fonts = []
    for name, encoding in [
        ("CMMI10", "/WinAnsiEncoding"),
        ("CMSY10", "<< /Differences [97 /greaterequal] >>"),
        ("CMR10", "/WinAnsiEncoding"),
    ]:
        fonts.append(
            (
                f"<< /Type /Font /Subtype /Type1 /BaseFont /{name} /Encoding {encoding} "
                f"/FirstChar 0 /LastChar 255 /Widths [{widths}] "
                f"/FontDescriptor << /Type /FontDescriptor /FontName /{name} /Flags 4 /FontBBox [0 -200 600 800] "
                "/ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 >> >>"
            ).encode()
        )
    stream = b"BT /F4 10 Tf 20 50 Td (For) Tj ET\nBT /F1 10 Tf 45 50 Td (xy) Tj ET\nBT /F2 10 Tf 60 50 Td (a) Tj ET\nBT /F3 10 Tf 70 50 Td (0) Tj ET\nBT /F4 10 Tf 90 50 Td (where) Tj ET"
    stream = stream.replace(b"(0)", f"(0{separator})".encode()).replace(
        b"(where)", f"({suffix})".encode()
    )
    if ruled:
        stream += b"\n10 30 m 170 30 l S 10 70 m 170 70 l S 10 30 m 10 70 l S 140 30 m 140 70 l S 170 30 m 170 70 l S"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 180 100] /Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R /F4 7 0 R >> >> /Contents 8 0 R >>",
        *fonts,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(data)
    data.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )
    Path(path).write_bytes(data)


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

    def test_table_isolated_operator_response_is_repaired_to_whole_expression(self):
        self.content()
        from marker.cloud.tables import recognize_table

        rejected = []

        def response(s, n):
            result = table_echo(s, n)
            for cell in result["cells"]:
                assigned = next(c for c in s["cells"] if c["id"] == cell["id"])
                for line in cell["lines"]:
                    owned = next(l for l in assigned["lines"] if l["id"] == line["id"])
                    if not owned["math_required"]:
                        continue
                    if not rejected:
                        rejected.append(True)
                        line["parts"] = [
                            (
                                {
                                    "kind": "math",
                                    "latex": r"\geq",
                                    "atom_ids": [a["id"]],
                                }
                                if a["kind"] == "image"
                                else {
                                    "kind": "text",
                                    "text": a["text"],
                                    "atom_ids": [a["id"]],
                                }
                            )
                            for a in owned["atoms"]
                        ]
            return result

        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            client, observed = offline_client(response, {"cloud_max_attempts": 2})
            result = recognize_table(
                source, group_tables(detect_tables(source), None)[0], client
            )
            self.assertTrue(result.complete, result.reasons)
            annotation = BeautifulSoup(result.html, "html.parser").find("annotation")
            self.assertEqual(annotation.text, r"x \geq 0")
            math_requests = [
                window
                for window in observed
                if any(l["math_required"] for c in window["cells"] for l in c["lines"])
            ]
            self.assertEqual(
                len(math_requests),
                2,
                "Isolated operator must require a counted schema repair",
            )
            audits = [
                image
                for name, image in result.resources.items()
                if name.startswith("formula")
            ]
            self.assertEqual(len(audits), 1)
            self.assertGreater(
                audits[0].width,
                300,
                "Audit must cover selectable x and 0 plus raster operator",
            )

    def test_table_isolated_operator_cannot_exhaust_repairs_and_report_complete(self):
        self.content()
        from marker.cloud.tables import recognize_table

        def response(s, n):
            result = table_echo(s, n)
            for cell in result["cells"]:
                assigned = next(c for c in s["cells"] if c["id"] == cell["id"])
                for line in cell["lines"]:
                    owned = next(l for l in assigned["lines"] if l["id"] == line["id"])
                    if owned["math_required"]:
                        line["parts"] = [
                            (
                                {
                                    "kind": "math",
                                    "latex": r"\geq",
                                    "atom_ids": [a["id"]],
                                }
                                if a["kind"] == "image"
                                else {
                                    "kind": "text",
                                    "text": a["text"],
                                    "atom_ids": [a["id"]],
                                }
                            )
                            for a in owned["atoms"]
                        ]
            return result

        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            client, observed = offline_client(response, {"cloud_max_attempts": 2})
            result = recognize_table(
                source, group_tables(detect_tables(source), None)[0], client
            )
            self.assertFalse(result.complete)
            self.assertTrue(result.reasons)
            self.assertTrue(result.resources)
            self.assertEqual(
                len(
                    [
                        w
                        for w in observed
                        if any(
                            l["math_required"] for c in w["cells"] for l in c["lines"]
                        )
                    ]
                ),
                2,
            )

    def test_mixed_prose_keeps_only_complete_expression_under_math_ownership(self):
        self.content()
        from marker.cloud.tables import LogicalTable, TablePart, recognize_table

        with PdfSource(FIXTURES / "split_formula.pdf") as source:
            bbox = (30, 40, 112, 56)
            atoms = source.page_words(0, bbox)
            image = next(
                image
                for image in source.page_images(0)
                if image.bbox == (85, 40, 101, 56)
            )
            atoms.append(
                {
                    "kind": "image",
                    "bbox": image.bbox,
                    "source_bbox": image.bbox,
                    "source_page": 0,
                    "image_index": image.index,
                }
            )
            grid = {
                "page": 0,
                "bbox": bbox,
                "cells": [
                    {
                        "bbox": bbox,
                        "source_bbox": bbox,
                        "source_page": 0,
                        "atoms": atoms,
                    }
                ],
                "trusted": False,
            }
            table = LogicalTable(
                [TablePart(0, bbox, grid, source.page_size(0), source=source)], grid
            )

            def response(s, n):
                cell = s["cells"][0]
                line = cell["lines"][0]
                label = next(a for a in line["atoms"] if a["text"] == "Limit:")
                expression = [a["id"] for a in line["atoms"] if a["id"] != label["id"]]
                self.assertEqual(line["expression_groups"], [expression])
                return {
                    "cells": [
                        {
                            key: cell[key]
                            for key in ("id", "row", "column", "rowspan", "colspan")
                        }
                        | {
                            "lines": [
                                {
                                    "id": line["id"],
                                    "parts": [
                                        {
                                            "kind": "text",
                                            "text": "Limit:",
                                            "atom_ids": [label["id"]],
                                        },
                                        {
                                            "kind": "math",
                                            "latex": r"x \geq 0",
                                            "atom_ids": expression,
                                        },
                                    ],
                                }
                            ]
                        }
                    ]
                }

            client, _ = offline_client(response)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            soup = BeautifulSoup(result.html, "html.parser")
            self.assertEqual(soup.find("annotation").text, r"x \geq 0")
            self.assertIn("Limit:", soup.find("td").get_text())
            self.assertNotIn("Limit", soup.find("annotation").text)

    def test_expression_groups_keep_subscripts_and_independent_expressions(self):
        from marker.cloud.content import expression_groups

        atoms = [
            {"kind": "text", "text": "For", "bbox": (0, 0, 14, 10), "size": 10},
            {"kind": "text", "text": "x", "bbox": (17, 0, 22, 10), "size": 10},
            {"kind": "text", "text": "2", "bbox": (23, -2, 27, 4), "size": 6},
            {"kind": "text", "text": "≥", "bbox": (30, 0, 38, 10), "size": 10},
            {"kind": "text", "text": "0", "bbox": (40, 0, 46, 10), "size": 10},
            {"kind": "text", "text": "and", "bbox": (49, 0, 66, 10), "size": 10},
            {"kind": "text", "text": "α", "bbox": (69, 0, 76, 10), "size": 10},
            {"kind": "text", "text": "1", "bbox": (77, 6, 81, 12), "size": 6},
            {"kind": "text", "text": "=", "bbox": (84, 0, 92, 10), "size": 10},
            {"kind": "text", "text": "1,5", "bbox": (94, 0, 111, 10), "size": 10},
        ]
        groups = expression_groups(atoms)
        self.assertEqual(
            [[atom["text"] for atom in group] for group in groups],
            [["x", "2", "≥", "0"], ["α", "1", "=", "1,5"]],
        )
        spaced = deepcopy(atoms[1:5] + atoms[6:])
        for atom in spaced[4:]:
            box = atom["bbox"]
            atom["bbox"] = (box[0] + 40, box[1], box[2] + 40, box[3])
        self.assertEqual(
            [[a["text"] for a in g] for g in expression_groups(spaced)],
            [["x", "2", "≥", "0"], ["α", "1", "=", "1,5"]],
        )

    def test_raster_symbols_separated_by_prose_keep_independent_math_owners(self):
        from marker.cloud.content import atom_id, expression_groups
        from marker.cloud.tables import _describe_cells, _table_schema
        from marker.processors.table_geometry import grid_positions

        for connective in ("и", "and", "or"):
            for hinted in (False, True):
                with self.subTest(connective=connective, hinted=hinted):
                    # These are the source roles in «α₂ и β», including the
                    # selectable conjunction between two independent rasters.
                    atoms = [
                        {"kind": "image", "bbox": (0, 0, 15, 17), "image_index": 0},
                        {
                            "kind": "text", "text": connective, "size": 9,
                            "fontname": "TimesNewRoman", "bbox": (16, 5, 30, 14),
                        },
                        {"kind": "image", "bbox": (32, 0, 42, 17), "image_index": 1},
                    ]
                    if hinted:
                        for atom in (atoms[0], atoms[2]):
                            atom["content_kind"] = "math"
                    groups = expression_groups(atoms)
                    self.assertFalse(any(atoms[1] in group for group in groups))
                    if hinted:
                        self.assertEqual(groups, [[atoms[0]], [atoms[2]]])
                    grid = {
                        "page": 0,
                        "cells": [{"bbox": (0, 0, 50, 20), "atoms": atoms}],
                    }
                    xs, ys, positions = grid_positions(grid)
                    described, _ = _describe_cells(grid, positions)
                    cell = described[0]
                    parts = [
                        {
                            "kind": "math", "atom_ids": [atom_id(atoms[0], 0)],
                            "latex": r"\alpha_2",
                        },
                        {
                            "kind": "text", "atom_ids": [atom_id(atoms[1], 0)],
                            "text": connective,
                        },
                        {
                            "kind": "math", "atom_ids": [atom_id(atoms[2], 0)],
                            "latex": r"\beta",
                        },
                    ]
                    response_cell = {
                        key: cell[key]
                        for key in ("id", "row", "column", "rowspan", "colspan")
                    }
                    response_cell["lines"] = [{"id": "c0l0", "parts": parts}]
                    schema = _table_schema(
                        described, True, len(ys) - 1, len(xs) - 1
                    )
                    schema.model_validate({"cells": [response_cell]})

    def test_wide_known_math_raster_and_unhinted_diagram_keep_distinct_roles(self):
        from marker.cloud.content import expression_groups

        equation = {
            "kind": "image", "content_kind": "math", "bbox": (0, 0, 80, 18)
        }
        diagram = {"kind": "image", "bbox": (100, 0, 200, 60)}
        self.assertEqual(expression_groups([equation, diagram]), [[equation]])

    def test_table_subscript_split_requires_counted_complete_expression_repair(self):
        self.content()
        from marker.cloud.tables import recognize_table

        rejected = []
        target = []

        def response(s, n):
            result = table_echo(s, n)
            for cell in result["cells"]:
                assigned = next(c for c in s["cells"] if c["id"] == cell["id"])
                if cell["column"] != 0 or cell["row"] != 1:
                    continue
                for line in cell["lines"]:
                    owned = next(l for l in assigned["lines"] if l["id"] == line["id"])
                    text = "".join(a["text"] for a in owned["atoms"])
                    latex = {"α1": r"\alpha_1", "α2": r"\alpha_2", "β": r"\beta"}[text]
                    line["parts"] = [
                        {
                            "kind": "math",
                            "latex": latex,
                            "atom_ids": [a["id"] for a in owned["atoms"]],
                        }
                    ]
                    if text == "α1" and not rejected:
                        target.append(line["id"])
                        rejected.append(True)
                        alpha = next(a for a in owned["atoms"] if a["text"] == "α")
                        subscript = next(a for a in owned["atoms"] if a["text"] == "1")
                        line["parts"] = [
                            {
                                "kind": "math",
                                "latex": r"\alpha",
                                "atom_ids": [alpha["id"]],
                            },
                            {
                                "kind": "text",
                                "text": "1",
                                "atom_ids": [subscript["id"]],
                            },
                        ]
            return result

        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            client, observed = offline_client(response, {"cloud_max_attempts": 2})
            result = recognize_table(
                source, group_tables(detect_tables(source), None)[0], client
            )
            self.assertTrue(result.complete, result.reasons)
            soup = BeautifulSoup(result.html, "html.parser")
            self.assertEqual(
                [a.text for a in soup.find_all("annotation")],
                [r"\alpha_1", r"\alpha_2", r"\beta"],
            )
            self.assertEqual(
                len(
                    [
                        w
                        for w in observed
                        if any(
                            l["id"] == target[0] for c in w["cells"] for l in c["lines"]
                        )
                    ]
                ),
                2,
            )
            audits = [
                image
                for name, image in result.resources.items()
                if name.startswith("formula")
            ]
            self.assertGreater(
                audits[0].width,
                90,
                "Complete alpha1 crop must retain lowered subscript ink",
            )

    def test_two_independent_expression_owners_preserve_prose_in_counted_schema(self):
        from marker.cloud.content import atom_id, expression_groups
        from marker.cloud.tables import _table_schema

        atoms = [
            {"kind": "text", "text": text, "bbox": box, "size": 10}
            for text, box in [
                ("For", (0, 0, 14, 10)),
                ("x", (17, 0, 22, 10)),
                ("≥", (24, 0, 32, 10)),
                ("0", (34, 0, 40, 10)),
                ("and", (43, 0, 60, 10)),
                ("α", (63, 0, 70, 10)),
                ("=", (72, 0, 80, 10)),
                ("1,5", (82, 0, 99, 10)),
            ]
        ]
        described = [dict(atom, id=atom_id(atom, 0)) for atom in atoms]
        groups = [[atom_id(a, 0) for a in group] for group in expression_groups(atoms)]
        line = {
            "id": "line",
            "atoms": described,
            "expression_groups": groups,
            "math_required": True,
        }
        cell = {
            "id": "cell",
            "row": 0,
            "column": 0,
            "rowspan": 1,
            "colspan": 1,
            "lines": [line],
        }
        schema = _table_schema([cell], True, 1, 1)
        parts = [
            {"kind": "text", "text": "For", "atom_ids": [described[0]["id"]]},
            {"kind": "math", "latex": r"x \geq 0", "atom_ids": groups[0]},
            {"kind": "text", "text": "and", "atom_ids": [described[4]["id"]]},
            {"kind": "math", "latex": r"\alpha=1{,}5", "atom_ids": groups[1]},
        ]
        response = {
            "cells": [
                {
                    key: cell[key]
                    for key in ("id", "row", "column", "rowspan", "colspan")
                }
                | {"lines": [{"id": "line", "parts": parts}]}
            ]
        }
        pixel = Image.new("RGB", (100, 20), "black")

        def respond(s, n):
            if n == 1:
                collapsed = deepcopy(response)
                collapsed["cells"][0]["lines"][0]["parts"] = [
                    {
                        "kind": "math",
                        "latex": r"x \geq 0 \quad \alpha=1{,}5",
                        "atom_ids": [a["id"] for a in described],
                    }
                ]
                return collapsed
            return response

        client, _ = offline_client(respond, {"cloud_max_attempts": 2})
        actual = client.request(
            "Recognize assigned expressions. SOURCE:\n" + json.dumps({"cells": [cell]}),
            [pixel],
            schema,
        )
        self.assertEqual(
            [part["kind"] for part in actual["cells"][0]["lines"][0]["parts"]],
            ["text", "math", "text", "math"],
        )
        self.assertEqual(
            client.request_count,
            2,
            "Combining independent expressions and prose must require repair",
        )

    def test_table_source_width_fonts_and_horizontal_offsets_are_intrinsic(self):
        self.content()
        from marker.cloud.tables import recognize_table

        with PdfSource(FIXTURES / "content_lines.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            client, _ = offline_client(table_echo)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            soup = BeautifulSoup(result.html, "html.parser")
            self.assertIsNotNone(
                soup.find("colgroup"),
                "Absolute children need intrinsic physical column widths",
            )
            self.assertEqual(
                [col["style"] for col in soup.find_all("col")], ["width:74.000pt"] * 7
            )
            self.assertIn("width:518.000pt", soup.table["style"])
            self.assertIn("min-width:518.000pt", soup.table["style"])
            header = soup.find("td")
            self.assertIn("padding:0", header["style"])
            line = header.find(attrs={"data-source-line": True})
            self.assertIn("font-size:10.000pt", line["style"])
            self.assertIn("left:8.000pt", line["style"])
            self.assertIn("top:8.070pt", line["style"])
            self.assertEqual(line.get_text(strip=True), "Coefficient")
            cell = header.find("div", recursive=False)
            self.assertIn("min-width:74.000pt", cell["style"])
        with PdfSource(FIXTURES / "refined_header.pdf") as source:
            table = group_tables(detect_tables(source), None)[0]
            client, _ = offline_client(table_echo)
            result = recognize_table(source, table, client)
            self.assertTrue(result.complete, result.reasons)
            soup = BeautifulSoup(result.html, "html.parser")
            merged = soup.find("td", colspan="2")
            self.assertIsNotNone(merged)
            self.assertIn(
                "min-width:200.000pt", merged.find("div", recursive=False)["style"]
            )

    def test_extracted_math_font_multiletter_operand_keeps_ordinary_prose_separate(
        self,
    ):
        from marker.cloud.content import expression_groups

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "math-font.pdf"
            math_font_pdf(filename)
            with PdfSource(filename) as source:
                atoms = source.page_words(0)
                xy = next(a for a in atoms if a["text"] == "xy")
                self.assertEqual(xy["fontname"], "CMMI10")
                self.assertEqual(xy["size"], 10)
                self.assertNotIn(
                    "italic",
                    xy,
                    "Grouping must use actual PDF metadata, not a manufactured style flag",
                )
                self.assertEqual(
                    [[a["text"] for a in group] for group in expression_groups(atoms)],
                    [["xy", "≥", "0"]],
                )

    def test_extracted_math_font_operand_partial_rejected_and_whole_accepted(self):
        from marker.cloud.content import atom_id
        from marker.cloud.tables import _describe_cells, _table_schema
        from marker.processors.table_geometry import grid_positions
        from pydantic import ValidationError

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "math-font.pdf"
            math_font_pdf(filename)
            with PdfSource(filename) as source:
                atoms = source.page_words(0)
                bbox = (15, 35, 140, 65)
                grid = {
                    "page": 0,
                    "bbox": bbox,
                    "trusted": False,
                    "cells": [{"bbox": bbox, "source_page": 0, "atoms": atoms}],
                }
                described, _ = _describe_cells(grid, grid_positions(grid)[2])
                cell = described[0]
                line = cell["lines"][0]
                lookup = {a["text"]: atom_id(a, 0) for a in atoms}
                whole = {
                    "cells": [
                        {
                            key: cell[key]
                            for key in ("id", "row", "column", "rowspan", "colspan")
                        }
                        | {
                            "lines": [
                                {
                                    "id": line["id"],
                                    "parts": [
                                        {
                                            "kind": "text",
                                            "text": "For",
                                            "atom_ids": [lookup["For"]],
                                        },
                                        {
                                            "kind": "math",
                                            "latex": r"xy \geq 0",
                                            "atom_ids": [
                                                lookup["xy"],
                                                lookup["≥"],
                                                lookup["0"],
                                            ],
                                        },
                                        {
                                            "kind": "text",
                                            "text": "where",
                                            "atom_ids": [lookup["where"]],
                                        },
                                    ],
                                }
                            ]
                        }
                    ]
                }
                partial = deepcopy(whole)
                partial["cells"][0]["lines"][0]["parts"][1:2] = [
                    {"kind": "text", "text": "xy", "atom_ids": [lookup["xy"]]},
                    {
                        "kind": "math",
                        "latex": r"\geq 0",
                        "atom_ids": [lookup["≥"], lookup["0"]],
                    },
                ]
                schema = _table_schema(described, False, 1, 1)
                with self.assertRaises(ValidationError):
                    schema.model_validate(partial)
                valid = schema.model_validate(whole).model_dump()
                self.assertEqual(
                    valid["cells"][0]["lines"][0]["parts"][1]["latex"], r"xy \geq 0"
                )

    def test_real_pdf_math_font_operand_repairs_to_complete_math_and_audit(self):
        from marker.cloud.tables import LogicalTable, TablePart, recognize_table

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "math-font.pdf"
            math_font_pdf(filename)
            with PdfSource(filename) as source:
                bbox = (15, 35, 140, 65)
                grid = {
                    "page": 0,
                    "bbox": bbox,
                    "trusted": False,
                    "cells": [
                        {"bbox": bbox, "source_page": 0, "atoms": source.page_words(0)}
                    ],
                }
                table = LogicalTable(
                    [TablePart(0, bbox, grid, source.page_size(0), source=source)], grid
                )

                def respond(s, n):
                    cell = s["cells"][0]
                    line = cell["lines"][0]
                    ids = {a["text"]: a["id"] for a in line["atoms"]}
                    parts = [{"kind": "text", "text": "For", "atom_ids": [ids["For"]]}]
                    if n == 1:
                        parts.extend(
                            [
                                {"kind": "text", "text": "xy", "atom_ids": [ids["xy"]]},
                                {
                                    "kind": "math",
                                    "latex": r"\geq 0",
                                    "atom_ids": [ids["≥"], ids["0"]],
                                },
                            ]
                        )
                    else:
                        parts.append(
                            {
                                "kind": "math",
                                "latex": r"xy \geq 0",
                                "atom_ids": [ids["xy"], ids["≥"], ids["0"]],
                            }
                        )
                    parts.append(
                        {"kind": "text", "text": "where", "atom_ids": [ids["where"]]}
                    )
                    return {
                        "cells": [
                            {
                                key: cell[key]
                                for key in ("id", "row", "column", "rowspan", "colspan")
                            }
                            | {"lines": [{"id": line["id"], "parts": parts}]}
                        ]
                    }

                client, _ = offline_client(respond, {"cloud_max_attempts": 2})
                result = recognize_table(source, table, client)
                self.assertTrue(result.complete, result.reasons)
                self.assertEqual(client.request_count, 2)
                soup = BeautifulSoup(result.html, "html.parser")
                self.assertEqual(soup.find("annotation").text, r"xy \geq 0")
                self.assertIn("For", soup.td.get_text())
                self.assertIn("where", soup.td.get_text())
                audit = next(
                    image
                    for name, image in result.resources.items()
                    if name.startswith("formula")
                )
                self.assertGreater(
                    audit.width,
                    240,
                    "Audit must include the xy operand from its actual PDF pixels",
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
