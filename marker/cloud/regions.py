"""Source region discovery and local text trust gates; no model factory/OCR."""

import re
import unicodedata
from dataclasses import dataclass, field
from statistics import median

from marker.cloud.source import NativeImage, intersection, object_bbox


@dataclass(frozen=True)
class TextQuality:
    trusted: bool
    reasons: tuple[str, ...]
    character_count: int
    invalid_fraction: float
    image_coverage: float


@dataclass(frozen=True)
class Region:
    """A source rectangle. Quality describes this region, never just its page.

    ``kind`` is text/table/image/unknown/blank. Unknown content must be checked
    by the adaptive converter; a blank region has required=False. Image atoms
    and every text atom retain source_page/source_bbox for expression crops.
    """

    page: int
    bbox: tuple[float, float, float, float]
    kind: str
    atoms: tuple[dict, ...]
    quality: TextQuality
    image: NativeImage | None = None
    grid: dict | None = field(default=None, compare=False)
    required: bool = True


def _coverage(images, box):
    rectangles = []
    for image in images:
        bounds = image.bbox if isinstance(image, NativeImage) else image.get("bbox") or object_bbox(image)
        x0, y0 = max(box[0], bounds[0]), max(box[1], bounds[1])
        x1, y1 = min(box[2], bounds[2]), min(box[3], bounds[3])
        if x1 > x0 and y1 > y0:
            rectangles.append((x0, y0, x1, y1))
    area = 0.0
    xs = sorted({r[i] for r in rectangles for i in (0, 2)})
    for left, right in zip(xs, xs[1:]):
        bands = sorted((r[1], r[3]) for r in rectangles if r[0] < right and r[2] > left)
        end = -float("inf")
        for top, bottom in bands:
            area += (right-left)*max(0, bottom-max(top, end))
            end = max(end, bottom)
    region_area = (box[2]-box[0])*(box[3]-box[1])
    return area/region_area if region_area > 0 else 0.0


def assess_text_quality(words, bbox, images=()) -> TextQuality:
    """Reject missing/damaged layers, while accepting ordinary numeric cells.

    This is a trust gate, not a claim of perfect OCR detection. Inferred table
    structure and content without text remain available as untrusted regions.
    """
    reasons = []
    text = " ".join(str(word.get("text", "")) for word in words).strip()
    count = len(text)
    if not text:
        reasons.append("missing_text")
    bad = [character for character in text if character == "\ufffd" or
           (unicodedata.category(character) in ("Cc", "Cs", "Co", "Cn") and not character.isspace())]
    invalid_fraction = len(bad)/max(1, count)
    if invalid_fraction >= 0.03 or any(unicodedata.category(char) == "Cc" for char in bad):
        reasons.append("invalid_characters")
    if re.search(r"\(cid:\d+\)", text):
        reasons.append("unmapped_glyphs")
    boxes = [word.get("bbox") or object_bbox(word) for word in words]
    if any(box[0] >= box[2] or box[1] >= box[3] for box in boxes):
        reasons.append("invalid_geometry")
    content = re.sub(r"[.·•…_\-]{3,}", "", text)
    nonspace = [char for char in content if not char.isspace()]
    if nonspace and sum(char.isalnum() for char in nonspace)/len(nonspace) < 0.15:
        reasons.append("suspicious_text")
    if any(intersection(box, bbox) <= 0 for box in boxes):
        reasons.append("outside_text")
    return TextQuality(not reasons, tuple(reasons), count, invalid_fraction, _coverage(images, bbox))


def word_lines(words):
    """Group source words by their physical baseline without changing records."""
    rows = []
    for word in sorted(words, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        center = (word["bbox"][1]+word["bbox"][3])/2
        matching = next((row for row in reversed(rows) if abs(
            center-median((item["bbox"][1]+item["bbox"][3])/2 for item in row)) <= 3), None)
        if matching is None:
            rows.append([word])
        else:
            matching.append(word)
    return [sorted(row, key=lambda item: item["bbox"][0]) for row in rows]


def _bounds(atoms):
    boxes = [atom["bbox"] for atom in atoms]
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def analyze_page(source, page) -> list[Region]:
    """Enumerate digital tables, images and text; expose remaining unknown areas.

    Scanned layout detection belongs to the adaptive converter. Here a scan is
    an untrusted image region, not an empty page or an invented table grid.
    """
    from marker.cloud.tables import detect_tables

    width, height = source.page_size(page)
    words = source.page_words(page)
    regions = []
    parts = detect_tables(source, [page])
    for part in parts:
        atoms = tuple(atom for cell in part.grid["cells"] for atom in cell["atoms"])
        regions.append(Region(page, part.bbox, "table", atoms, source.text_quality(page, part.bbox), grid=part.grid))

    def covered(box):
        area = (box[2]-box[0])*(box[3]-box[1])
        return any(intersection(box, region.bbox) >= area*0.8 for region in regions)

    image_groups = []
    for image in source.page_images(page):
        if covered(image.bbox):
            continue
        overlapping = [group for group in image_groups if any(intersection(image.bbox, other.bbox) > 0 for other in group)]
        group = [image]
        for existing in overlapping:
            group.extend(existing)
            image_groups.remove(existing)
        image_groups.append(group)
    table_words = {(atom.get('text'), tuple(atom['bbox'])) for region in regions
                   for atom in region.atoms if atom['kind'] == 'text'}
    owned_words = {index for index, word in enumerate(words)
                   if (word['text'], tuple(word['bbox'])) in table_words}
    image_boxes = [_bounds([{'bbox': image.bbox} for image in group]) for group in image_groups]
    overlays = [[] for _ in image_groups]
    for index, word in enumerate(words):
        if index in owned_words or not image_boxes:
            continue
        owner = max(range(len(image_boxes)), key=lambda group: intersection(word['bbox'], image_boxes[group]))
        if intersection(word['bbox'], image_boxes[owner]) > 0:
            overlays[owner].append(word)
            owned_words.add(index)
    for index, group in enumerate(image_groups):
        atoms = [{"kind": "image", "bbox": image.bbox, "source_bbox": image.bbox,
                  "source_page": page, "image_index": image.index} for image in group]
        atoms.extend(overlays[index])
        # Assigned words can cross the raster edge. The composite crop must
        # include every assigned atom, and ownership must not depend on a
        # coverage threshold or the rectangular gaps in other region boxes.
        box = _bounds(atoms)
        image = group[0] if len(group) == 1 and not overlays[index] and source.isolated_image(group[0]) else None
        regions.append(Region(page, box, "image", tuple(atoms), source.text_quality(page, box), image=image))
    remaining = [word for index, word in enumerate(words) if index not in owned_words]
    paragraphs = []
    for row in word_lines(remaining):
        box = _bounds(row)
        trusted = assess_text_quality(row, box).trusted
        previous = paragraphs[-1] if paragraphs else None
        if (previous and trusted == previous["trusted"] and
                box[1]-previous["bbox"][3] <= max(18, (box[3]-box[1])*1.5) and
                abs(box[0]-previous["bbox"][0]) <= 15):
            previous["atoms"].extend(row)
            previous["bbox"] = _bounds(previous["atoms"])
        else:
            paragraphs.append({"atoms": list(row), "bbox": box, "trusted": trusted})
    for paragraph in paragraphs:
        box = paragraph["bbox"]
        kind = "text"
        if box[0] >= box[2] or box[1] >= box[3]:
            box = (0.0, max(0, box[1]-2), width, min(height, box[3]+2))
            kind = "unknown"
        regions.append(Region(page, box, kind, tuple(paragraph["atoms"]), assess_text_quality(paragraph["atoms"], box)))
    # Curves and ruling lines outside identified content remain visible to the
    # converter. A zero-height line needs a small rectangle for a source crop.
    graphics = []
    for item in [*source.pdf_page(page).edges, *source.pdf_page(page).curves]:
        box = object_bbox(item)
        if any(all(region.bbox[i]-1 <= box[i] for i in (0, 1)) and
               all(box[i] <= region.bbox[i]+1 for i in (2, 3))
               for region in regions if region.kind in ("table", "image")):
            continue
        source_box = box
        box = (max(0, box[0]-0.5), max(0, box[1]-0.5),
               min(width, box[2]+0.5), min(height, box[3]+0.5))
        area = (box[2]-box[0])*(box[3]-box[1])
        if not any(intersection(box, region.bbox) >= area*0.8
                   for region in regions if region.kind in ("table", "image")):
            graphics.append({"kind": "vector", "bbox": source_box, "source_bbox": source_box, "source_page": page, "crop_bbox": box})
    if graphics:
        box = _bounds([dict(atom, bbox=atom["crop_bbox"]) for atom in graphics])
        regions.append(Region(page, box, "unknown", tuple(graphics), source.text_quality(page, box)))
    if not regions:
        box = (0.0, 0.0, width, height)
        regions.append(Region(page, box, "blank", (), assess_text_quality([], box), required=False))
    return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))
