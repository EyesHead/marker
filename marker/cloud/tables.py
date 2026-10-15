"""Physical table discovery and bounded grouping before content recognition."""

from collections import OrderedDict

import json
import re
from html import escape
from typing import Literal
from copy import deepcopy
from dataclasses import dataclass, field
from statistics import median

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marker.cloud.client import schema_prompt, semantic_error
from marker.cloud.content import (
    MATH_CLASSIFICATION_PROMPT, MATH_TRANSCRIPTION_PROMPT, ContentResult, atom_id, bounds, context_box,
    expression_groups, latex_to_mathml, resource_name, source_lines,
    source_script_hints, validate_script_hints,
)
from marker.cloud.context import RecognitionFailure, fit_request
from marker.cloud.math_evidence import with_math_closeups
from marker.cloud.parallel import ordered_cloud_map
from marker.cloud.regions import assess_text_quality, word_lines
from marker.cloud.source import intersection, object_bbox
from marker.processors.table_geometry import (
    can_join_grids, extract_grids, grid_positions, join_grids, stack_grids,
)


TABLE_LABEL = re.compile(
    r"(?:table|таблиц[аы])\s*(?:№\s*)?((?:[a-zа-яё]{1,3}\s*\.\s*)?\d+(?:\.\d+)*(?:[a-zа-я])?)",
    re.IGNORECASE,
)


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
        return re.sub(r"\s+", "", match[1]).casefold() if match else None

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


def _clear_source_after_table(part):
    """A short final table/header can continue despite a large blank page tail.

    Require readable source evidence for this exception to the usual page-edge
    gate. A missing optional source implementation keeps the old conservative
    behavior; words, drawings or images after the table disprove a clear tail.
    """
    source = part.source
    if source is None or not callable(getattr(source, "page_words", None)):
        return False
    try:
        bottom = part.bbox[3] + 0.75

        def follows(item):
            if isinstance(item, dict):
                box = item.get("bbox") or object_bbox(item)
            else:
                box = item.bbox
            return box[3] >= bottom and box[3] >= box[1]

        if any(str(word.get("text", "")).strip() and follows(word)
               for word in source.page_words(part.page)):
            return False
        if callable(getattr(source, "page_images", None)) and any(
                follows(image) for image in source.page_images(part.page)):
            return False
        if callable(getattr(source, "pdf_page", None)):
            page = source.pdf_page(part.page)
            if any(follows(item) for item in
                   [*getattr(page, "edges", ()), *getattr(page, "curves", ())]):
                return False
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return False
    return True


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
    if second.bbox[1] > second.page_size[1]*0.35:
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
    if first.bbox[3] < first.page_size[1]*0.65 and not _clear_source_after_table(first):
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
              'are insufficient. Return merge=true only with positive continuation evidence. '
              'Return only {"merge":true} or {"merge":false}. Do not reconstruct or transcribe the tables here.\n'
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
                raise semantic_error("math_unsupported") from None
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
                raise semantic_error("table_cell_ownership")
            occupied = set()
            for cell in self.cells:
                original = expected_by_id[cell.id]
                if trusted and any(
                    getattr(cell, key) != original[key]
                    for key in ("row", "column", "rowspan", "colspan")
                ):
                    raise semantic_error("table_trusted_spans")
                if (
                    cell.row + cell.rowspan > row_count
                    or cell.column + cell.colspan > column_count
                ):
                    raise semantic_error("table_span_bounds")
                slots = {
                    (r, c)
                    for r in range(cell.row, cell.row + cell.rowspan)
                    for c in range(cell.column, cell.column + cell.colspan)
                }
                if occupied.intersection(slots):
                    raise semantic_error("table_span_overlap")
                occupied.update(slots)
                expected_lines = {line["id"]: line for line in original["lines"]}
                if len(cell.lines) != len(expected_lines) or {
                    line.id for line in cell.lines
                } != set(expected_lines):
                    raise semantic_error("table_line_ownership")
                for line in cell.lines:
                    source_line = expected_lines[line.id]
                    atoms = {atom["id"]: atom for atom in source_line["atoms"]}
                    consumed = [
                        identity for part in line.parts for identity in part.atom_ids
                    ]
                    if len(consumed) != len(atoms) or set(consumed) != set(atoms):
                        raise semantic_error("table_atom_ownership")
                    for group in source_line["expression_groups"]:
                        # Coverage alone permits text x + isolated math ≥ +
                        # text 0. Every source expression must instead have
                        # exactly one mathematical owner for all its glyphs.
                        owners = [
                            part for part in line.parts
                            if set(group).intersection(part.atom_ids)
                        ]
                        if (
                            len(owners) != 1 or owners[0].kind != "math"
                            or set(owners[0].atom_ids) != set(group)
                        ):
                            raise semantic_error("table_complete_expression")
                    for part in line.parts:
                        if not part.atom_ids:
                            raise semantic_error("table_content_ownership")
                        owned = [atoms[identity] for identity in part.atom_ids]
                        if part.kind == "math":
                            identities = set(part.atom_ids)
                            hints = [hint for hint in source_line.get("script_hints", [])
                                     if hint["base_atom"] in identities
                                     and identities.issuperset(hint["atom_ids"])]
                            try:
                                validate_script_hints(part.latex, hints)
                            except RecognitionFailure:
                                raise semantic_error("math_script_geometry") from None
                        if part.kind == "figure" and any(
                            a["kind"] not in ("image", "vector", "unknown")
                            for a in owned
                        ):
                            raise semantic_error("table_selectable_text_figure")
                        if part.kind == "text":
                            # Grid trust describes physical geometry. Raster
                            # ordinary text can need OCR even in ruled cells.
                            if not part.text.strip():
                                raise semantic_error("table_text_unresolved")
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
            groups = [
                [atom_id(atom, page) for atom in group]
                for group in expression_groups(atoms)
            ]
            script_hints = source_script_hints(atoms)
            lines.append(
                {
                    "id": f"{identity}l{number}",
                    "page": page,
                    "bbox": box,
                    "layout_y": min(a.get("layout_bbox", a["bbox"])[1] for a in atoms),
                    "math_required": bool(groups),
                    "expression_groups": groups,
                    **({"script_hints": script_hints} if script_hints else {}),
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
        + MATH_CLASSIFICATION_PROMPT + MATH_TRANSCRIPTION_PROMPT
        + "Each expression_groups entry must be owned in full by one math part; keep surrounding prose "
        "and independent expressions in separate parts. Never transcribe a required operator alone. "
        "Preserve phi/varphi, Cyrillic/Latin subscripts, decimal commas and superscripts. "
        "Do not guess unreadable content.\nSOURCE:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _window_images(source, cells, originals, headers, maximum, *, crop_cache=None):
    def crop(page, box):
        key = (page, tuple(box))
        if crop_cache is not None and key in crop_cache:
            image = crop_cache.pop(key)
            crop_cache[key] = image
            return image.copy()
        image = source.crop(page, box)
        # Repeated headers and candidate rectangles can share the render, but
        # each returned image still belongs to its own request. Bound retained
        # pixels as well as the number of entries for large source rectangles.
        pixels = image.width * image.height
        if crop_cache is not None and pixels <= 16_000_000:
            while crop_cache and (len(crop_cache) >= 8 or pixels + sum(
                item.width * item.height for item in crop_cache.values()
            ) > 16_000_000):
                crop_cache.popitem(last=False)
            crop_cache[key] = image.copy()
        return image

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
        images.append(crop(page, context_box(source, page, box)))
    # A header image is repeated context only when the image budget permits.
    if headers and len(images) < maximum and headers[0].get("page") is not None:
        images.append(
            crop(
                headers[0]["page"],
                context_box(source, headers[0]["page"], headers[0]["bbox"]),
            )
        )
    return images


def _owned_cells(units):
    """Reassemble disjoint line units without changing IDs, source boxes or spans."""
    cells = {}
    for cell, line in units:
        owned = cells.setdefault(cell["id"], {
            key: cell[key] for key in ("id", "row", "column", "rowspan", "colspan")
        } | {"lines": []})
        if line is not None:
            previous = next((item for item in owned["lines"] if item["id"] == line["id"]), None)
            if previous is None:
                owned["lines"].append(deepcopy(line))
            else:
                previous["atoms"].extend(deepcopy(line["atoms"]))
                previous["bbox"] = bounds(previous["atoms"])
                previous["expression_groups"].extend(deepcopy(line["expression_groups"]))
                previous["math_required"] = bool(previous["expression_groups"])
                if line.get("script_hints"):
                    previous.setdefault("script_hints", []).extend(deepcopy(line["script_hints"]))
    return list(cells.values())


def _line_fragments(line):
    """Split an oversized baseline into source owners, keeping expressions whole."""
    groups = {identity: group for group in line["expression_groups"] for identity in group}
    seen, pieces = set(), []
    for atom in line["atoms"]:
        if atom["id"] in seen:
            continue
        identities = set(groups.get(atom["id"], [atom["id"]]))
        owned = [item for item in line["atoms"] if item["id"] in identities]
        seen.update(identities)
        piece = {**line, "atoms": owned, "bbox": bounds(owned),
                 "expression_groups": [group for group in line["expression_groups"] if identities.issuperset(group)]}
        piece["math_required"] = bool(piece["expression_groups"])
        if "script_hints" in line:
            piece["script_hints"] = [hint for hint in line["script_hints"]
                                     if hint["base_atom"] in identities and identities.issuperset(hint["atom_ids"])]
        pieces.append(piece)
    return pieces


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
    crop_cache = OrderedDict()
    while offset < len(units):
        selected = []
        best = None
        while offset + len(selected) < len(units):
            candidate = selected + [units[offset + len(selected)]]
            cells = _owned_cells(candidate)
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
                source, cells, originals, context_headers, client.cloud_max_images,
                crop_cache=crop_cache,
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
            cell, line = units[offset]
            fragments = _line_fragments(line) if line is not None else []
            if len(fragments) > 1:
                units[offset:offset+1] = [(cell, fragment) for fragment in fragments]
                continue
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
    xs, _, _ = grid_positions(grid)
    width = xs[-1] - xs[0]
    # Absolutely placed source lines contribute no intrinsic table width.
    # Carry the physical column canvas into the HTML itself, independently of
    # the host application's table padding/font/minimum-width defaults.
    html = [
        f'<table style="table-layout:fixed;border-collapse:collapse;width:{width:.3f}pt;'
        f'min-width:{width:.3f}pt;max-width:none"><colgroup>'
    ]
    html.extend(
        f'<col style="width:{right-left:.3f}pt">'
        for left, right in zip(xs, xs[1:])
    )
    html.append("</colgroup><tbody>")
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
            html.append(f'<td{attrs} style="padding:0;vertical-align:top">')
            original = originals[cell["id"]]
            expected_lines = {
                line["id"]: line for line in source_by_id[cell["id"]]["lines"]
            }
            responses = {line["id"]: line for line in cell["lines"]}
            cell_layout = original.get("layout_bbox", original["bbox"])
            cell_width = xs[cell["column"] + cell["colspan"]] - xs[cell["column"]]
            html.append(
                f'<div style="position:relative;min-width:{cell_width:.3f}pt;'
                f'height:{cell_layout[3]-cell_layout[1]:.3f}pt">'
            )
            atom_lookup = {
                atom_id(a, a.get("source_page", grid["page"])): a
                for a in original["atoms"]
            }
            for identity, line in expected_lines.items():
                result = responses[identity]
                line_atoms = [atom_lookup[atom["id"]] for atom in line["atoms"]]
                line_layout = bounds(line_atoms, layout=True)
                default_size = max(6, min(12, line["bbox"][3] - line["bbox"][1]))
                font_size = max(
                    (atom.get("size", default_size) for atom in line_atoms
                     if atom.get("kind", "text") == "text"),
                    default=default_size,
                )
                # Absolute source baseline placement preserves correspondence
                # even when physical cells have no internal horizontal rules.
                html.append(
                    f'<div data-source-line="{identity}" data-source-page="{line["page"]}" '
                    f'data-source-y="{line["bbox"][1]:.3f}" style="position:absolute;'
                    f'left:{line_layout[0]-cell_layout[0]:.3f}pt;'
                    f'top:{line["layout_y"]-cell_layout[1]:.3f}pt;'
                    f'font-size:{font_size:.3f}pt;line-height:1;white-space:nowrap">'
                )
                for part in result["parts"]:
                    atoms = [atom_lookup[identity] for identity in part["atom_ids"]]
                    if part["kind"] == "math":
                        box = bounds(atoms)
                        page = atoms[0].get("source_page", line["page"])
                        name = resource_name(
                            "formula", page, box, tuple(part["atom_ids"])
                        )
                        resources[name] = source.resource(page, box)
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
        def request_window(window, source_call):
            cells, prompt, images, schema = window
            received = []
            # A server may enforce a smaller context than the conservative
            # client estimate. Keep reflowed ownership in the same counted
            # request; only successful disjoint remainders get a new request.
            units = [(cell, line) for cell in cells for line in (cell["lines"] or [None])]
            while units:
                remaining = []

                def reflow():
                    nonlocal units, remaining
                    if len(units) <= 1:
                        return None
                    midpoint = len(units) // 2
                    remaining = units[midpoint:] + remaining
                    units = units[:midpoint]
                    reduced_cells = _owned_cells(units)
                    return (
                        _window_prompt(reduced_cells, [], trusted, rows, columns),
                        source_call(lambda: _window_images(source, reduced_cells, originals, [], client.cloud_max_images)),
                        _table_schema(reduced_cells, trusted, rows, columns),
                    )

                result = client.request(prompt, images, schema, context_reflow=reflow)
                received.extend(result["cells"])
                units = remaining
                if units:
                    pending_cells = _owned_cells(units)
                    prompt = _window_prompt(pending_cells, [], trusted, rows, columns)
                    images = source_call(lambda: _window_images(source, pending_cells, originals, [], client.cloud_max_images))
                    schema = _table_schema(pending_cells, trusted, rows, columns)
            return received

        def prepared_windows():
            for cells, prompt, images, schema in _table_windows(
                source, described, originals, client, trusted, rows, columns
            ):
                images = with_math_closeups(source, cells, images, client, prompt, schema)
                yield cells, prompt, images, schema

        # Finite request caps retain the original sequential reservation order.
        workers = 1 if client.cloud_max_requests else getattr(client, "cloud_max_concurrency", 1)
        for received in ordered_cloud_map(request_window, prepared_windows(), workers):
            for cell in received:
                if cell["id"] in recognized:
                    previous = recognized[cell["id"]]
                    if any(
                        previous[key] != cell[key]
                        for key in ("row", "column", "rowspan", "colspan")
                    ):
                        raise RecognitionFailure(
                            "Split cell structure changed between windows"
                        )
                    for line in cell["lines"]:
                        earlier = next((item for item in previous["lines"] if item["id"] == line["id"]), None)
                        if earlier is None:
                            previous["lines"].append(line)
                        else:
                            earlier["parts"].extend(line["parts"])
                else:
                    recognized[cell["id"]] = cell
        # Fragments keep their original baseline ID; reassemble source order.
        for original in described:
            for line in recognized[original["id"]]["lines"]:
                expected = next(item for item in original["lines"] if item["id"] == line["id"])
                order = {atom["id"]: index for index, atom in enumerate(expected["atoms"])}
                line["parts"].sort(key=lambda part: min(order[identity] for identity in part["atom_ids"]))
                identities = [identity for part in line["parts"] for identity in part["atom_ids"]]
                if len(identities) != len(order) or set(identities) != set(order):
                    raise RecognitionFailure("Reassembled baseline must own every source atom once")
        _validate_final_structure(recognized, rows, columns)
        html = _render_recognized_table(
            source, described, originals, recognized, resources, grid
        )
        return TableResult(
            "table", html, html, resources, not reasons, tuple(reasons), grid=grid
        )
    except RecognitionFailure as error:
        reasons.append(error.reason)
        # Complete physical coverage is required even after some windows passed
        # or formula resources were retained before a later rendering failure.
        for part in logical_table.parts:
            name = resource_name("unresolved-table", part.page, part.bbox)
            resources[name] = source.resource(part.page, part.bbox)
        html = "".join(
            f'<p><img src="{name}" alt="Unresolved table source"></p>'
            for name in resources
        )
        return TableResult(
            "table", html, html, resources, False, tuple(reasons), grid=grid
        )
