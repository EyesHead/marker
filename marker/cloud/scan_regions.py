"""Compact layout discovery for scans that have no usable native geometry."""

import json
from math import isfinite
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from marker.cloud.context import RecognitionFailure
from marker.cloud.regions import Region, assess_text_quality
from marker.cloud.source import intersection
from marker.cloud.tables import TablePart


_Coordinate = Annotated[float, Field(ge=0, le=1000, allow_inf_nan=False)]


class _ScanRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["text", "math", "figure", "table"]
    bbox: list[_Coordinate] = Field(
        min_length=4,
        max_length=4,
        description="Normalized x0,y0,x1,y1 in 0..1000 relative to the supplied crop.",
    )
    caption: str = ""
    continuation: bool = False

    @model_validator(mode="after")
    def positive_extent(self):
        if self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]:
            raise ValueError("Layout rectangles require positive width and height")
        return self


class _ScanLayout(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    complete: bool
    regions: list[_ScanRegion] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_disjoint_layout(self):
        if not self.complete:
            raise ValueError("Scanned source layout is incomplete")
        for index, region in enumerate(self.regions):
            if any(
                intersection(region.bbox, previous.bbox) > 0
                for previous in self.regions[:index]
            ):
                raise ValueError("Layout rectangles must not overlap")
        return self


_PROMPT = (
    "Locate ALL visible content in this source crop and return one compact layout in reading order. "
    "Coordinates are normalized 0..1000 relative to the entire supplied crop: (0,0) is its top-left, "
    "(1000,1000) its bottom-right. bbox=[x0,y0,x1,y1]. Rectangles must have positive size, stay inside "
    "that range, and be disjoint. Give one rectangle per paragraph, complete display formula, genuine "
    "figure, or whole physical table part. Keep inline mathematics inside its paragraph. Mathematical "
    "glyphs and formula bitmaps are math; actual diagrams are figure. Include each table's headings, "
    "caption and notes in its table rectangle. For table caption copy its identifier/title including "
    "the word Table or its source-language equivalent; mark continuation when this is a continued part. "
    "Return only kind, bbox, caption and continuation for each region. Do not transcribe body text, "
    "LaTeX or HTML, or provide cells, baseline lines or word geometry. Account for all visible content; "
    "set complete=false when any content cannot be located confidently."
)


def _key(page, box):
    return json.dumps(
        {"page": page, "bbox": [float(value) for value in box]},
        sort_keys=True,
        separators=(",", ":"),
    )


def _point(value, start, end):
    if value == 0:
        return start
    if value == 1000:
        return end
    return min(end, max(start, start + value * (end - start) / 1000))


def _cache(store):
    if store is None:
        return {}
    try:
        raw = json.loads(
            (Path(store.root) / "scan-layout.json").read_text(encoding="utf-8")
        )
        if not isinstance(raw, dict):
            return {}
        validated = {}
        for key, value in raw.items():
            parent = json.loads(key)
            if (
                not isinstance(parent, dict)
                or set(parent) != {"page", "bbox"}
                or type(parent["page"]) is not int
                or parent["page"] < 0
                or not isinstance(parent["bbox"], list)
                or len(parent["bbox"]) != 4
                or not all(
                    type(v) in (int, float) and isfinite(v) for v in parent["bbox"]
                )
                or not (
                    0 <= parent["bbox"][0] < parent["bbox"][2]
                    and 0 <= parent["bbox"][1] < parent["bbox"][3]
                )
                or key != _key(parent["page"], parent["bbox"])
            ):
                return {}
            validated[key] = _ScanLayout.model_validate(value).model_dump()
        return validated
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def discover_scan(
    source, region, client, *, store=None
) -> tuple[list[Region], list[TablePart]]:
    """Discover coarse scan regions with one bounded request, or a valid cache hit.

    The caller retains its source draft before discovery. No transcription or
    invented physical cell grid is needed to route later whole-object jobs.
    """
    box = tuple(region.bbox)
    width, height = source.page_size(region.page)
    if (
        len(box) != 4
        or not all(isfinite(value) for value in box)
        or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
    ):
        raise RecognitionFailure("Scanned source rectangle is outside the page")
    key = _key(region.page, box)
    cache = _cache(store)
    response = cache.get(key)
    if response is None:
        image = source.crop(region.page, box, dpi=200)
        try:
            response = client.request(_PROMPT, [image], _ScanLayout)
            response = _ScanLayout.model_validate(response).model_dump()
        except ValidationError:
            raise RecognitionFailure(
                "Scanned layout response failed validation"
            ) from None
        if store is not None:
            cache[key] = response
            store.write_text(
                "scan-layout.json",
                json.dumps(cache, ensure_ascii=False, allow_nan=False),
            )
    layout = _ScanLayout.model_validate(response)
    regions, parts = [], []
    for item in layout.regions:
        left, top, right, bottom = item.bbox
        bounds = (
            _point(left, box[0], box[2]),
            _point(top, box[1], box[3]),
            _point(right, box[0], box[2]),
            _point(bottom, box[1], box[3]),
        )
        atom = {
            "kind": "unknown",
            "bbox": bounds,
            "source_bbox": bounds,
            "source_page": region.page,
            "content_kind": item.kind,
        }
        regions.append(
            Region(
                region.page,
                bounds,
                item.kind,
                (atom,),
                assess_text_quality([], bounds),
                required=region.required,
            )
        )
        if item.kind == "table":
            caption = (
                ("Продолжение " + item.caption).strip()
                if item.continuation
                else item.caption
            )
            parts.append(
                TablePart(region.page, bounds, None, (width, height), caption, source)
            )
    return regions, parts
