"""Physical table discovery and bounded grouping before content recognition."""

import re
from copy import deepcopy
from dataclasses import dataclass, field
from statistics import median

from pydantic import BaseModel

from marker.cloud.context import RecognitionFailure
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
