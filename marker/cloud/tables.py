"""Physical table discovery and bounded grouping before content recognition."""

import json
import re
from html import escape
from typing import Literal
from copy import deepcopy
from dataclasses import dataclass, field
from statistics import median

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marker.cloud.client import schema_prompt
from marker.cloud.content import (
    ContentResult, atom_id, bounds, context_box, latex_to_mathml,
    math_required, resource_name, source_lines,
)
from marker.cloud.context import RecognitionFailure, fit_request
from marker.cloud.regions import assess_text_quality, word_lines
from marker.cloud.source import intersection
from marker.processors.table_geometry import (
    can_join_grids, extract_grids, grid_positions, join_grids, stack_grids,
)


TABLE_LABEL = re.compile(r"(?:table|таблиц[аы])\s*(?:№\s*)?(\d+(?:\.\d+)*(?:[a-zа-я])?)", re.IGNORECASE)


@dataclass(frozen=True)
class TablePart:
    """One physical part. Its page, bbox and grid never use virtual offsets.

    ``source`` is needed only for an ambiguous visual decision while its
    PdfSource is open. Parts supplied by scanned layout may have grid=None.
    """

    page: int
    bbox: tuple[float, float, float, float]
    grid: dict | None = field(default=None, compare=False)
    page_size: tuple[float, float] = (0, 0)
    caption: str = ""
    source: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "bbox", tuple(self.bbox))
        object.__setattr__(self, "page_size", tuple(self.page_size))
        object.__setattr__(self, "grid", deepcopy(self.grid))

    @property
    def table_id(self):
        match = TABLE_LABEL.search(self.caption)
        return match[1].lower() if match else None

    @property
    def is_continuation(self):
        return bool(re.search(r"continued|continuation|продолжени", self.caption, re.IGNORECASE))


@dataclass
class LogicalTable:
    """All source parts and, when available, a separate joined display grid.

    Warnings mean a plausible grouping could not be established. Keeping both
    parts separately then exposes uncertainty to conversion completion status.
    """

    parts: list[TablePart]
    grid: dict | None = None
    warnings: tuple[str, ...] = ()

    @property
    def caption(self):
        return next((part.caption for part in self.parts if part.caption), "")

    @property
    def page_ids(self):
        return tuple(part.page for part in self.parts)


class ContinuationDecision(BaseModel):
    merge: bool


def _caption(source, page, bbox):
    width, _ = source.page_size(page)
    area = (max(0, bbox[0]-10), max(0, bbox[1]-50), min(width, bbox[2]+10), bbox[1])
    rows = word_lines(source.page_words(page, area))
    captions = [" ".join(word["text"] for word in row) for row in rows]
    return next((text for text in reversed(captions) if TABLE_LABEL.search(text) or
                 re.search(r"continued|продолжени", text, re.IGNORECASE)), "")


def _segments(row):
    result = []
    for word in row:
        if result and word["bbox"][0]-result[-1][-1]["bbox"][2] <= 12:
            result[-1].append(word)
        else:
            result.append([word])
    return result


def _text_grid(rows, page):
    segments = [_segments(row) for row in rows]
    count = len(segments[0])
    starts = [median(row[col][0]["bbox"][0] for row in segments) for col in range(count)]
    if any(abs(row[col][0]["bbox"][0]-starts[col]) > 4 for row in segments for col in range(count)):
        return None
    ends = [max(row[col][-1]["bbox"][2] for row in segments) for col in range(count)]
    if any(ends[col]+12 >= starts[col+1] for col in range(count-1)):
        return None
    xs = [starts[0]-2] + [(ends[col]+starts[col+1])/2 for col in range(count-1)] + [ends[-1]+2]
    tops = [min(word["bbox"][1] for word in row) for row in rows]
    bottoms = [max(word["bbox"][3] for word in row) for row in rows]
    ys = [tops[0]-2] + [(bottoms[row]+tops[row+1])/2 for row in range(len(rows)-1)] + [bottoms[-1]+2]
    cells = []
    for row, columns in enumerate(segments):
        for col, atoms in enumerate(columns):
            bbox = (xs[col], ys[row], xs[col+1], ys[row+1])
            copied = [dict(atom, kind="text", size=float(atom["size"]),
                           bold="bold" in atom["fontname"].lower(),
                           italic=any(style in atom["fontname"].lower() for style in ("italic", "oblique")))
                      for atom in atoms]
            cells.append({"bbox": bbox, "source_bbox": bbox, "source_page": page,
                          "source_boxes": ((page, bbox),), "atoms": copied,
                          "open_top": False, "open_bottom": False})
    bbox = (xs[0], ys[0], xs[-1], ys[-1])
    return {"page": page, "last_page": page, "bbox": bbox, "source_bbox": bbox,
            "cells": cells, "trusted": False, "method": "text", "column_starts": tuple(starts)}


def _borderless_grids(source, page, ruled):
    words = [word for word in source.page_words(page) if not any(
        intersection(word["bbox"], grid["bbox"]) > 0 for grid in ruled
    )]
    groups = []
    current = []
    for row in word_lines(words):
        segments = _segments(row)
        if len(segments) < 2:
            if current:
                groups.append(current)
                current = []
            continue
        if current:
            previous = current[-1]
            gap = min(word["bbox"][1] for word in row)-max(word["bbox"][3] for word in previous)
            if gap > 35 or len(_segments(previous)) != len(segments):
                groups.append(current)
                current = []
        current.append(row)
    if current:
        groups.append(current)
    result = []
    for rows in groups:
        if len(rows) < 3:
            continue
        grid = _text_grid(rows, page)
        if grid is not None and assess_text_quality([word for row in rows for word in row], grid["bbox"]).trusted:
            result.append(grid)
    return result


def detect_tables(source, page_ids=None) -> list[TablePart]:
    """Enumerate every independent digital grid, including borderless candidates.

    Text-derived cell boundaries are explicitly untrusted. Scans/damaged layers
    remain discoverable as regions for the converter's cloud layout fallback.
    """
    parts = []
    for page in range(len(source)) if page_ids is None else page_ids:
        grids = extract_grids(source.pdf_page(page))
        grids += _borderless_grids(source, page, grids)
        for grid in sorted(grids, key=lambda item: (item["bbox"][1], item["bbox"][0])):
            parts.append(TablePart(page, grid["bbox"], grid, source.page_size(page),
                                   _caption(source, page, grid["bbox"]), source))
    return parts


def _compatible(first, second):
    if first.page+1 != second.page:
        return False
    if first.table_id and second.table_id and first.table_id != second.table_id:
        return False
    first_title, second_title = _caption_title(first.caption), _caption_title(second.caption)
    if first_title and second_title and first_title != second_title:
        return False
    if min(*first.page_size, *second.page_size) <= 0:
        return False
    if first.bbox[3] < first.page_size[1]*0.65 or second.bbox[1] > second.page_size[1]*0.35:
        return False
    tolerance = max(2, (first.bbox[2]-first.bbox[0])*0.01)
    if any(abs(first.bbox[i]-second.bbox[i]) > tolerance for i in (0, 2)):
        return False
    if first.grid and second.grid:
        first_geometry, second_geometry = grid_positions(first.grid), grid_positions(second.grid)
        if first_geometry is None or second_geometry is None:
            return False
        xs, following_xs = first_geometry[0], second_geometry[0]
        coarse, fine = (xs, following_xs) if len(xs) <= len(following_xs) else (following_xs, xs)
        # An interrupted merged header can expose only a subset of the cuts
        # visible in later data rows; every original cut must still align.
        if any(not any(abs(cut-other) <= tolerance for other in fine) for cut in coarse):
            return False
    return True


def _caption_title(caption):
    caption = TABLE_LABEL.sub('', caption)
    caption = re.sub(r'\(?\b(?:continued|continuation|продолжение)\b\)?', '', caption, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', caption).strip(' :;,.()-').casefold()


def _summary(part):
    text = " ".join(atom.get("text", "") for cell in (part.grid or {}).get("cells", []) for atom in cell["atoms"])
    return f"page={part.page}; caption={part.caption[:200]}; text={text[:600]}"


def _visual_decision(first, second, client):
    if client is None or first.source is None or second.source is None:
        raise RecognitionFailure("Ambiguous table continuation has no visual decision")
    images = [part.source.crop(part.page, part.bbox, dpi=144) for part in (first, second)]
    prompt = ("Decide whether these two source crops continue ONE logical table across a page break. "
              "Use captions, column alignment, headings and row meaning. Identical column counts alone "
              "are insufficient. Return merge=true only with positive continuation evidence.\n"
              "First: " + _summary(first) + "\nSecond: " + _summary(second))
    return client.request(prompt, images, ContinuationDecision)["merge"]


def group_tables(parts, client) -> list[LogicalTable]:
    """Group supported continuations; all uncertain requests use the shared client."""
    tables = []
    for part in sorted(parts, key=lambda item: (item.page, item.bbox[1], item.bbox[0])):
        candidates = [table for table in tables if _compatible(table.parts[-1], part)]
        if len(candidates) != 1:
            warning = ("Multiple plausible table continuations",) if candidates else ()
            for candidate in candidates:
                candidate.warnings += warning
            tables.append(LogicalTable([part], deepcopy(part.grid), warning))
            continue
        table = candidates[0]
        tail = table.parts[-1]
        split = bool(tail.grid and part.grid and tail.grid.get("trusted", True) and
                     part.grid.get("trusted", True) and can_join_grids(tail.grid, part.grid))
        labelled = bool(tail.table_id and tail.table_id == part.table_id)
        warning = ()
        try:
            merge = split or labelled or _visual_decision(tail, part, client)
        except RecognitionFailure as error:
            merge = False
            warning = (error.reason,)
            table.warnings += warning
        if not merge:
            tables.append(LogicalTable([part], deepcopy(part.grid), warning))
            continue
        if table.grid and part.grid:
            table.grid = (join_grids(table.grid, part.grid) if can_join_grids(table.grid, part.grid)
                          else stack_grids(table.grid, part.grid))
        else:
            table.grid = None
        table.parts.append(part)
    return tables


@dataclass
class TableResult(ContentResult):
    grid: dict | None = None


class TableSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["text", "math", "figure"]
    atom_ids: list[str]
    text: str = ""
    latex: str = ""

    @model_validator(mode="after")
    def usable(self):
        if self.kind == "math":
            try:
                latex_to_mathml(self.latex)
            except RecognitionFailure:
                raise ValueError("Required math is unresolved or unsupported") from None
        return self


class TableLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    parts: list[TableSegment]


class TableCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    rowspan: int = Field(ge=1)
    colspan: int = Field(ge=1)
    lines: list[TableLine]


def _table_schema(expected, trusted, row_count, column_count):
    """Semantic checks run inside CloudClient's counted repair attempts."""
    expected_by_id = {cell["id"]: cell for cell in expected}

    class TableResponse(BaseModel):
        model_config = ConfigDict(extra="forbid")
        cells: list[TableCell]

        @model_validator(mode="after")
        def ownership_and_structure(self):
            if len(self.cells) != len(expected_by_id) or {
                cell.id for cell in self.cells
            } != set(expected_by_id):
                raise ValueError("Cell IDs must cover assigned cells once")
            occupied = set()
            for cell in self.cells:
                original = expected_by_id[cell.id]
                if trusted and any(
                    getattr(cell, key) != original[key]
                    for key in ("row", "column", "rowspan", "colspan")
                ):
                    raise ValueError("Trusted physical spans cannot change")
                if (
                    cell.row + cell.rowspan > row_count
                    or cell.column + cell.colspan > column_count
                ):
                    raise ValueError("Cell span exceeds table extent")
                slots = {
                    (r, c)
                    for r in range(cell.row, cell.row + cell.rowspan)
                    for c in range(cell.column, cell.column + cell.colspan)
                }
                if occupied.intersection(slots):
                    raise ValueError("Cell spans overlap")
                occupied.update(slots)
                expected_lines = {line["id"]: line for line in original["lines"]}
                if len(cell.lines) != len(expected_lines) or {
                    line.id for line in cell.lines
                } != set(expected_lines):
                    raise ValueError("Source baseline lines must be covered once")
                for line in cell.lines:
                    source_line = expected_lines[line.id]
                    atoms = {atom["id"]: atom for atom in source_line["atoms"]}
                    consumed = [
                        identity for part in line.parts for identity in part.atom_ids
                    ]
                    if len(consumed) != len(atoms) or set(consumed) != set(atoms):
                        raise ValueError("Line source atoms must be owned exactly once")
                    if source_line["math_required"] and not any(
                        part.kind == "math" for part in line.parts
                    ):
                        raise ValueError("Required complete expression must be math")
                    for part in line.parts:
                        if not part.atom_ids:
                            raise ValueError("Output content has no source ownership")
                        owned = [atoms[identity] for identity in part.atom_ids]
                        if part.kind == "figure" and any(
                            a["kind"] not in ("image", "vector", "unknown")
                            for a in owned
                        ):
                            raise ValueError(
                                "Selectable text cannot disappear into a figure"
                            )
                        if part.kind == "text":
                            # Grid trust describes physical geometry. Raster
                            # ordinary text can need OCR even in ruled cells.
                            if not part.text.strip():
                                raise ValueError("Required text remains unresolved")
            return self

    return TableResponse


def _describe_cells(grid, positions):
    described = []
    originals = {}
    for index, (row, column, rowspan, colspan, cell) in enumerate(
        sorted(positions, key=lambda p: (p[0], p[1]))
    ):
        identity = f"c{index}"
        originals[identity] = cell
        lines = []
        for number, atoms in enumerate(source_lines(cell["atoms"])):
            box = bounds(atoms)
            page = atoms[0].get("source_page", cell.get("source_page", grid["page"]))
            lines.append(
                {
                    "id": f"{identity}l{number}",
                    "page": page,
                    "bbox": box,
                    "layout_y": min(a.get("layout_bbox", a["bbox"])[1] for a in atoms),
                    "math_required": math_required(atoms),
                    "atoms": [
                        {
                            "id": atom_id(atom, page),
                            "kind": atom.get("kind", "text"),
                            "text": atom.get("text", ""),
                            "bbox": atom.get("source_bbox", atom["bbox"]),
                            "page": atom.get("source_page", page),
                        }
                        for atom in atoms
                    ],
                }
            )
        described.append(
            {
                "id": identity,
                "row": row,
                "column": column,
                "rowspan": rowspan,
                "colspan": colspan,
                "lines": lines,
            }
        )
    return described, originals


def _window_prompt(cells, headers, trusted, rows, columns):
    payload = {
        "trusted": trusted,
        "rows": rows,
        "columns": columns,
        "headers": headers,
        "cells": cells,
    }
    return (
        "Read these table source crops and geometry. Return ONLY owned cells and baseline lines in SOURCE. "
        "headers and neighboring crop pixels are repeated context, never extra output. Each source atom ID "
        "must occur exactly once on its assigned line. Preserve baseline IDs/order across all columns; "
        "a merged cell may contain multiple expressions. Retain trusted spans exactly. If trusted=false, "
        "reconstruct row/column spans within the table dimensions without overlap, retaining source cell/line IDs. "
        "Use math with exact LaTeX for complete expressions, figure for diagrams, text for ordinary words. "
        "Preserve phi/varphi, Cyrillic/Latin subscripts, decimal commas and superscripts. "
        "Do not guess unreadable content.\nSOURCE:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _window_images(source, cells, originals, headers, maximum):
    boxes = {}
    for cell in cells:
        if cell["lines"]:
            for line in cell["lines"]:
                boxes.setdefault(line["page"], []).append(line["bbox"])
        else:
            for page, box in originals[cell["id"]].get(
                "source_boxes",
                (
                    (
                        originals[cell["id"]].get("source_page", 0),
                        originals[cell["id"]]["bbox"],
                    ),
                ),
            ):
                boxes.setdefault(page, []).append(box)
    images = []
    for page, source_boxes in sorted(boxes.items()):
        box = bounds([{"bbox": b} for b in source_boxes])
        images.append(source.crop(page, context_box(source, page, box)))
    # A header image is repeated context only when the image budget permits.
    if headers and len(images) < maximum and headers[0].get("page") is not None:
        images.append(
            source.crop(
                headers[0]["page"],
                context_box(source, headers[0]["page"], headers[0]["bbox"]),
            )
        )
    return images


def _table_windows(source, described, originals, client, trusted, rows, columns):
    """Split owned source lines, never truncate cell content or response space."""
    units = []
    for cell in described:
        if cell["lines"]:
            units.extend((cell, line) for line in cell["lines"])
        else:
            units.append((cell, None))
    units.sort(
        key=lambda pair: (
            pair[0]["row"],
            pair[1]["layout_y"] if pair[1] else 0,
            pair[0]["column"],
        )
    )
    headers = [
        {
            "page": line["page"],
            "bbox": line["bbox"],
            "text": " ".join(
                atom["text"] for atom in line["atoms"] if atom["kind"] == "text"
            ),
        }
        for cell in described
        if cell["row"] == 0
        for line in cell["lines"]
    ]
    # Use one compact header descriptor, even for a wide multi-column header.
    if headers:
        first_page = headers[0]["page"]
        same_page = [h for h in headers if h["page"] == first_page]
        headers = [
            {
                "page": first_page,
                "bbox": bounds([{"bbox": h["bbox"]} for h in same_page]),
                "text": " | ".join(h["text"] for h in same_page)[:400],
            }
        ]
    offset = 0
    while offset < len(units):
        selected = []
        best = None
        while offset + len(selected) < len(units):
            candidate = selected + [units[offset + len(selected)]]
            cells_by_id = {}
            for cell, line in candidate:
                entry = cells_by_id.setdefault(
                    cell["id"],
                    {
                        key: cell[key]
                        for key in ("id", "row", "column", "rowspan", "colspan")
                    }
                    | {"lines": []},
                )
                if line is not None:
                    entry["lines"].append(line)
            cells = list(cells_by_id.values())
            # Conservative JSON response reservation: IDs, spans and segment
            # fields cost more than source text alone. Math gets extra room.
            output_estimate = 128 + sum(
                120
                + sum(
                    120
                    + sum(160 + len(a["text"].encode()) for a in line["atoms"])
                    + (240 if line["math_required"] else 0)
                    for line in cell["lines"]
                )
                for cell in cells
            )
            if output_estimate > client.cloud_max_output_tokens:
                break
            context_headers = headers if offset else []
            prompt = _window_prompt(cells, context_headers, trusted, rows, columns)
            schema = _table_schema(cells, trusted, rows, columns)
            pages = {line["page"] for cell in cells for line in cell["lines"]}
            if len(pages) > client.cloud_max_images:
                break
            images = _window_images(
                source, cells, originals, context_headers, client.cloud_max_images
            )
            if len(images) > client.cloud_max_images:
                break
            try:
                fit_request(
                    schema_prompt(prompt, schema),
                    images,
                    client.cloud_context_tokens,
                    client.cloud_max_output_tokens,
                )
            except RecognitionFailure:
                break
            selected = candidate
            best = (cells, prompt, images, schema)
        if best is None:
            raise RecognitionFailure(
                "A required table line cannot fit the bounded request"
            )
        yield best
        offset += len(selected)


def _validate_final_structure(recognized, rows, columns):
    occupied = set()
    for cell in recognized.values():
        slots = {
            (r, c)
            for r in range(cell["row"], cell["row"] + cell["rowspan"])
            for c in range(cell["column"], cell["column"] + cell["colspan"])
        }
        if (
            not slots
            or occupied.intersection(slots)
            or any(r >= rows or c >= columns for r, c in slots)
        ):
            raise RecognitionFailure(
                "Recognized table spans overlap or exceed table extent"
            )
        occupied.update(slots)
    if occupied != {(r, c) for r in range(rows) for c in range(columns)}:
        raise RecognitionFailure("Recognized table does not cover every row and column")


def _render_recognized_table(source, described, originals, recognized, resources, grid):
    html = ["<table><tbody>"]
    source_by_id = {cell["id"]: cell for cell in described}
    cells_by_row = {}
    for cell in recognized.values():
        cells_by_row.setdefault(cell["row"], []).append(cell)
    for row in range(
        max(cell["row"] + cell["rowspan"] for cell in recognized.values())
    ):
        html.append("<tr>")
        for cell in sorted(cells_by_row.get(row, []), key=lambda c: c["column"]):
            attrs = (f' rowspan="{cell["rowspan"]}"' if cell["rowspan"] > 1 else "") + (
                f' colspan="{cell["colspan"]}"' if cell["colspan"] > 1 else ""
            )
            html.append(f"<td{attrs}>")
            original = originals[cell["id"]]
            expected_lines = {
                line["id"]: line for line in source_by_id[cell["id"]]["lines"]
            }
            responses = {line["id"]: line for line in cell["lines"]}
            cell_layout = original.get("layout_bbox", original["bbox"])
            html.append(
                f'<div style="position:relative;height:{cell_layout[3]-cell_layout[1]:.3f}pt">'
            )
            for identity, line in expected_lines.items():
                result = responses[identity]
                # Absolute source baseline placement preserves correspondence
                # even when physical cells have no internal horizontal rules.
                html.append(
                    f'<div data-source-line="{identity}" data-source-page="{line["page"]}" '
                    f'data-source-y="{line["bbox"][1]:.3f}" style="position:absolute;top:{line["layout_y"]-cell_layout[1]:.3f}pt;white-space:nowrap">'
                )
                atom_lookup = {
                    atom_id(a, a.get("source_page", grid["page"])): a
                    for a in original["atoms"]
                }
                for part in result["parts"]:
                    atoms = [atom_lookup[identity] for identity in part["atom_ids"]]
                    if part["kind"] == "math":
                        box = bounds(atoms)
                        page = atoms[0].get("source_page", line["page"])
                        name = resource_name(
                            "formula", page, box, tuple(part["atom_ids"])
                        )
                        resources[name] = source.crop(page, box)
                        html.append(latex_to_mathml(part["latex"]))
                    elif part["kind"] == "figure":
                        box = bounds(atoms)
                        page = atoms[0].get("source_page", line["page"])
                        name = resource_name(
                            "figure", page, box, tuple(part["atom_ids"])
                        )
                        resources[name] = source.resource(page, box)
                        html.append(
                            f'<img src="{name}" alt="Figure" style="width:{box[2]-box[0]:.3f}pt;height:{box[3]-box[1]:.3f}pt">'
                        )
                    else:
                        # Trusted digital text retains its source spelling;
                        # cloud is responsible for damaged/scanned text only.
                        text = (
                            " ".join(a.get("text", "") for a in atoms)
                            if assess_text_quality(atoms, bounds(atoms)).trusted
                            else part["text"]
                        )
                        html.append(escape(text))
                    html.append(" ")
                html.append("</div>")
            html.append("</div></td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)


def recognize_table(source, logical_table, client):
    """Recognize one logical table with bounded owned windows and one HTML grid.

    Task 4 supplies scanned layout as grid cells/atoms in source coordinates,
    trusted=False. No usable geometry returns an auditable incomplete result.
    """
    grid = deepcopy(logical_table.grid)
    resources = {}
    reasons = list(logical_table.warnings)
    recognized = {}
    try:
        if not grid or not grid.get("cells") or grid_positions(grid) is None:
            raise RecognitionFailure("Required table has no valid source cell layout")
        xs, ys, positions = grid_positions(grid)
        rows, columns = len(ys) - 1, len(xs) - 1
        described, originals = _describe_cells(grid, positions)
        trusted = grid.get("trusted", True)
        for cells, prompt, images, schema in _table_windows(
            source, described, originals, client, trusted, rows, columns
        ):
            try:
                result = client.request(prompt, images, schema)
            except RecognitionFailure:
                for cell in cells:
                    for line in cell["lines"]:
                        name = resource_name(
                            "unresolved-table", line["page"], line["bbox"], line["id"]
                        )
                        resources[name] = source.crop(line["page"], line["bbox"])
                raise
            for cell in result["cells"]:
                if cell["id"] in recognized:
                    previous = recognized[cell["id"]]
                    if any(
                        previous[key] != cell[key]
                        for key in ("row", "column", "rowspan", "colspan")
                    ):
                        raise RecognitionFailure(
                            "Split cell structure changed between windows"
                        )
                    previous["lines"].extend(cell["lines"])
                else:
                    recognized[cell["id"]] = cell
        _validate_final_structure(recognized, rows, columns)
        html = _render_recognized_table(
            source, described, originals, recognized, resources, grid
        )
        return TableResult(
            "table", html, html, resources, not reasons, tuple(reasons), grid=grid
        )
    except RecognitionFailure as error:
        reasons.append(error.reason)
        if not resources:
            for part in logical_table.parts:
                name = resource_name("unresolved-table", part.page, part.bbox)
                resources[name] = source.crop(part.page, part.bbox)
        html = "".join(
            f'<p><img src="{name}" alt="Unresolved table source"></p>'
            for name in resources
        )
        return TableResult(
            "table", html, html, resources, False, tuple(reasons), grid=grid
        )
