"""Adaptive source-first PDF conversion with one bounded cloud client."""

import json
import math
import os
from dataclasses import replace
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marker.cloud.client import CloudClient
from marker.cloud.content import (
    ContentResult,
    group_expression_regions,
    recognize_region,
    resource_name,
    atom_id,
    expression_groups,
    source_lines,
)
from marker.cloud.context import RecognitionFailure
from marker.cloud.regions import Region, analyze_page, assess_text_quality
from marker.cloud.source import PdfSource, intersection
from marker.cloud.tables import TablePart, detect_tables, group_tables, recognize_table
from marker.converters import BaseConverter
from marker.processors.table_geometry import grid_positions
from marker.renderers.markdown import MarkdownOutput
from marker.util import assign_config


class LayoutLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox: list[float] = Field(min_length=4, max_length=4)
    kind: Literal["text", "math", "figure"]


class LayoutCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox: list[float] = Field(min_length=4, max_length=4)
    lines: list[LayoutLine]


class LayoutRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["text", "math", "figure", "table", "decoration"]
    bbox: list[float] = Field(min_length=4, max_length=4)
    cells: list[LayoutCell]


def _contains(outer, inner):
    return (
        all(math.isfinite(v) for v in inner)
        and outer[0] <= inner[0] < inner[2] <= outer[2]
        and outer[1] <= inner[1] < inner[3] <= outer[3]
    )


def _layout_schema(box):
    class LayoutResponse(BaseModel):
        model_config = ConfigDict(extra="forbid")
        complete: bool
        regions: list[LayoutRegion]

        @model_validator(mode="after")
        def source_geometry(self):
            if not self.complete or not self.regions:
                raise ValueError("Required visible layout remains unresolved")
            previous = []
            for region in self.regions:
                if not _contains(box, region.bbox):
                    raise ValueError("Layout region exceeds assigned source bounds")
                if any(intersection(region.bbox, other) > 0 for other in previous):
                    raise ValueError("Layout regions overlap source ownership")
                previous.append(region.bbox)
                if region.kind != "table":
                    if region.cells:
                        raise ValueError("Only tables can own cells")
                    continue
                if not region.cells:
                    raise ValueError("Table requires source cells")
                for cell in region.cells:
                    if not _contains(region.bbox, cell.bbox):
                        raise ValueError("Cell exceeds table bounds")
                    for line in cell.lines:
                        if not _contains(cell.bbox, line.bbox):
                            raise ValueError("Baseline exceeds cell bounds")
                    if any(
                        intersection(a.bbox, b.bbox) > 0
                        for i, a in enumerate(cell.lines)
                        for b in cell.lines[i + 1 :]
                    ):
                        raise ValueError("Baseline atoms overlap")
                grid = {"cells": [{"bbox": c.bbox} for c in region.cells]}
                positions = grid_positions(grid)
                if positions is None:
                    raise ValueError("Table source cells must cover a rectangular grid")
                xs, ys, _ = positions
                if any(
                    abs(a - b) > 0.1
                    for a, b in zip((xs[0], ys[0], xs[-1], ys[-1]), region.bbox)
                ):
                    raise ValueError("Table cells must cover the entire table extent")
            return self

    return LayoutResponse


def scan_layout(source, region, client):
    """Discover scan regions and untrusted physical cells/baseline atoms."""
    prompt = (
        "Locate ALL visible content within this source rectangle, including borderless tables, "
        "paragraphs, formulas and figures. Coordinates are original PDF points, top-left origin, "
        "never image pixels or normalized coordinates. Return disjoint regions in reading order. "
        "For each table provide physical cell rectangles tiling the entire table, including merged "
        "cells and blank cells; provide a separate tight line rectangle for EVERY visible baseline "
        "inside each cell. Do not combine multiple baselines into one bitmap. Non-table cells=[]. "
        "Classify formulas as math even when OCR is disabled. Decoration must be explicitly classified. "
        "Set complete=false for any unreadable, missing or uncertain layout.\nSOURCE:\n"
        + json.dumps({"stage": "layout", "page": region.page, "bbox": region.bbox})
    )
    response = client.request(
        prompt, [source.crop(region.page, region.bbox)], _layout_schema(region.bbox)
    )
    regions = []
    for item in response["regions"]:
        box = tuple(item["bbox"])
        if item["kind"] == "decoration":
            continue
        grid = None
        atoms = []
        if item["kind"] == "table":
            cells = []
            for cell in item["cells"]:
                cell_box = tuple(cell["bbox"])
                line_atoms = [
                    {
                        "kind": "unknown",
                        "bbox": tuple(line["bbox"]),
                        "source_bbox": tuple(line["bbox"]),
                        "source_page": region.page,
                        "content_kind": line["kind"],
                    }
                    for line in cell["lines"]
                ]
                atoms.extend(line_atoms)
                cells.append(
                    {
                        "bbox": cell_box,
                        "source_bbox": cell_box,
                        "source_page": region.page,
                        "source_boxes": ((region.page, cell_box),),
                        "atoms": line_atoms,
                        "open_top": False,
                        "open_bottom": False,
                    }
                )
            grid = {
                "page": region.page,
                "last_page": region.page,
                "bbox": box,
                "source_bbox": box,
                "cells": cells,
                "trusted": False,
                "method": "cloud",
            }
        else:
            atoms = [
                {
                    "kind": "unknown",
                    "bbox": box,
                    "source_bbox": box,
                    "source_page": region.page,
                    "content_kind": item["kind"],
                }
            ]
        regions.append(
            Region(
                region.page,
                box,
                item["kind"],
                tuple(atoms),
                assess_text_quality([], box),
                grid=grid,
            )
        )
    return regions


def _table_needs_ocr(parts):
    for part in parts:
        for cell in part.grid["cells"]:
            for line in source_lines(cell["atoms"]):
                formula_ids = {
                    atom_id(a) for group in expression_groups(line) for a in group
                }
                ordinary = [a for a in line if atom_id(a) not in formula_ids]
                words = [a for a in ordinary if a.get("kind") == "text"]
                if ordinary and not assess_text_quality(words, cell["bbox"]).trusted:
                    return True
    return False


class OllamaPdfConverter(BaseConverter):
    """Convert PDF source text/resources and cloud-recognized regions to Markdown."""

    requires_model_dict = False
    cloud_context_tokens: Annotated[int, "Conservative total request token budget."] = (
        8192
    )
    cloud_max_output_tokens: Annotated[int, "Maximum cloud response tokens."] = 2048
    cloud_max_images: Annotated[int, "Maximum images per cloud request."] = 3
    cloud_max_attempts: Annotated[int, "Total attempts including repairs."] = 3
    cloud_max_requests: Annotated[
        int, "Document request cap; zero means unlimited."
    ] = 0
    cloud_crop_dpi: Annotated[int, "Retained vector/composite source crop DPI."] = 600
    cloud_model: Annotated[str, "Ollama Cloud model."] = "gemma4:31b"
    cloud_base_url: Annotated[str, "Ollama Cloud endpoint."] = "https://ollama.com/v1"
    force_ocr: Annotated[
        bool, "Recognize ordinary text even with a trusted source layer."
    ] = False
    disable_ocr: Annotated[
        bool, "Disable ordinary text OCR; required math still runs."
    ] = False
    page_range = None

    def __init__(
        self,
        config=None,
        artifact_dict=None,
        processor_list=None,
        renderer=None,
        llm_service=None,
    ):
        # BaseConverter downloads a rendering font; this source path needs no models/fonts.
        self.config = (
            config.model_dump() if isinstance(config, BaseModel) else dict(config or {})
        )
        assign_config(self, self.config)
        self.force_ocr = self.force_ocr or os.environ.get("FORCE_OCR") == "1"
        self.disable_ocr = self.disable_ocr or os.environ.get("DISABLE_OCR") == "1"
        if self.force_ocr and self.disable_ocr:
            raise ValueError("force_ocr and disable_ocr cannot both be enabled")
        client_config = dict(self.config)
        for key in (
            "cloud_context_tokens",
            "cloud_max_output_tokens",
            "cloud_max_images",
            "cloud_max_attempts",
            "cloud_max_requests",
            "cloud_model",
            "cloud_base_url",
        ):
            client_config[key] = getattr(self, key)
        self.client = (artifact_dict or {}).get("cloud_client") or CloudClient(
            client_config
        )

    def _unresolved(self, source, region, reason):
        name = resource_name("unresolved", region.page, region.bbox)
        try:
            resources = {name: source.resource(region.page, region.bbox)}
        except RecognitionFailure as error:
            return ContentResult(
                region.kind, complete=False, reasons=(reason, error.reason)
            )
        return ContentResult(
            region.kind,
            f"![Unresolved source]({name})",
            resources=resources,
            complete=False,
            reasons=(reason,),
        )

    def _recognize(self, source, region):
        if region.kind == "figure":
            name = resource_name("figure", region.page, region.bbox)
            try:
                image = source.resource(region.page, region.bbox)
            except RecognitionFailure as error:
                return self._unresolved(source, region, error.reason)
            return ContentResult(
                "figure", f"![Figure]({name})", resources={name: image}
            )
        if self.disable_ocr and region.kind == "text" and not region.quality.trusted:
            return self._unresolved(source, region, "Required text OCR is disabled")
        if self.force_ocr and region.kind == "text":
            region = replace(region, quality=replace(region.quality, trusted=False))
        return recognize_region(
            source, region, self.client, allow_text=not self.disable_ocr
        )

    def __call__(self, filepath):
        reasons, images, outputs, regions = [], {}, [], []
        start_requests = self.client.request_count
        with PdfSource(filepath, crop_dpi=self.cloud_crop_dpi) as source:
            pages = list(
                self.page_range if self.page_range is not None else range(len(source))
            )
            native_parts = {}
            for page in pages:
                if page < 0 or page >= len(source):
                    reasons.append(f"Page {page}: requested page is outside document")
                    continue
                try:
                    for part in detect_tables(source, [page]):
                        native_parts[(part.page, part.bbox)] = part
                    page_regions = group_expression_regions(analyze_page(source, page))
                except RecognitionFailure as error:
                    box = (0.0, 0.0, *source.page_size(page))
                    region = Region(
                        page, box, "unknown", (), assess_text_quality([], box)
                    )
                    result = self._unresolved(source, region, error.reason)
                    outputs.append((page, 0.0, 0.0, result))
                    continue
                for region in page_regions:
                    if region.kind != "image":
                        regions.append(region)
                        continue
                    try:
                        regions.extend(scan_layout(source, region, self.client))
                    except RecognitionFailure as error:
                        result = self._unresolved(source, region, error.reason)
                        outputs.append((page, region.bbox[1], region.bbox[0], result))
            parts = [
                native_parts.get((r.page, r.bbox))
                or TablePart(
                    r.page, r.bbox, r.grid, source.page_size(r.page), source=source
                )
                for r in regions
                if r.kind == "table"
            ]
            for table in group_tables(parts, self.client):
                part = table.parts[0]
                table_regions = [
                    r
                    for r in regions
                    if r.kind == "table"
                    and any(r.page == p.page and r.bbox == p.bbox for p in table.parts)
                ]
                needs_ocr = self.force_ocr or _table_needs_ocr(table.parts)
                if self.disable_ocr and needs_ocr:
                    for region in table_regions:
                        result = self._unresolved(
                            source, region, "Required table text OCR is disabled"
                        )
                        outputs.append(
                            (region.page, region.bbox[1], region.bbox[0], result)
                        )
                        candidates = [
                            Region(
                                region.page,
                                tuple(cell["bbox"]),
                                "text",
                                tuple(cell["atoms"]),
                                region.quality,
                            )
                            for cell in region.grid["cells"]
                        ]
                        candidates.extend(
                            Region(
                                region.page,
                                tuple(a["bbox"]),
                                "math",
                                (a,),
                                region.quality,
                            )
                            for a in region.atoms
                            if a.get("content_kind") == "math"
                        )
                        for math_region in group_expression_regions(candidates):
                            if math_region.kind == "math":
                                outputs.append(
                                    (
                                        region.page,
                                        math_region.bbox[1],
                                        math_region.bbox[0],
                                        self._recognize(source, math_region),
                                    )
                                )
                    continue
                if self.force_ocr:
                    # Retain physical geometry while withholding extracted spelling authority.
                    for cell in table.grid["cells"]:
                        formula_ids = {
                            atom_id(a)
                            for line in source_lines(cell["atoms"])
                            for group in expression_groups(line)
                            for a in group
                        }
                        for atom in cell["atoms"]:
                            if (
                                atom.get("kind") == "text"
                                and atom_id(atom) not in formula_ids
                            ):
                                atom["kind"], atom["text"] = "unknown", ""
                result = recognize_table(source, table, self.client)
                outputs.append((part.page, part.bbox[1], part.bbox[0], result))
            for region in group_expression_regions(
                [r for r in regions if r.kind != "table"]
            ):
                outputs.append(
                    (
                        region.page,
                        region.bbox[1],
                        region.bbox[0],
                        self._recognize(source, region),
                    )
                )
        markdown = []
        for page, _, _, result in sorted(outputs, key=lambda item: item[:3]):
            if result.markdown:
                markdown.append(result.markdown)
            images.update(result.resources)
            reasons.extend(f"Page {page}: {reason}" for reason in result.reasons)
            if not result.complete and not result.reasons:
                reasons.append(f"Page {page}: required recognition incomplete")
        return MarkdownOutput(
            markdown="\n\n".join(markdown),
            images=images,
            metadata={
                "cloud_conversion": {
                    "status": "incomplete" if reasons else "complete",
                    "reasons": list(dict.fromkeys(reasons)),
                    "requests": self.client.request_count - start_requests,
                }
            },
        )
