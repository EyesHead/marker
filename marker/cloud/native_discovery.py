"""Reconcile native layout owners with every visible source object.

This stage discovers *objects*, never cell coordinates or transcriptions. All
source atoms keep their original PDF coordinates. A discovery failure leaves
the original native owner tree intact for the caller's incomplete report.
"""

from dataclasses import replace
from math import isfinite
import re
from marker.cloud.content import atom_id, math_required
from marker.cloud.context import RecognitionFailure
from marker.cloud.regions import Region, assess_text_quality
from marker.cloud.scan_regions import discover_scan
from marker.cloud.source import intersection, object_bbox
from marker.cloud.tables import TablePart, _caption
from marker.schema import BlockTypes
from marker.schema.blocks import Equation, Picture, Table, Text
from marker.schema.polygon import PolygonBox


_PICTURES = {BlockTypes.Picture, BlockTypes.Figure, BlockTypes.Diagram}
_TABLES = {BlockTypes.Table, BlockTypes.TableOfContents}
_EXCLUDED = {BlockTypes.PageHeader, BlockTypes.PageFooter}


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _inside(inner, outer, tolerance=0.75):
    return (outer[0] - tolerance <= inner[0] <= inner[2] <= outer[2] + tolerance
            and outer[1] - tolerance <= inner[1] <= inner[3] <= outer[3] + tolerance)


def _box(block, page, source):
    width, height = source.page_size(page.page_id)
    grid = getattr(block, "table_grid", None)
    if block.block_type in _TABLES and grid is not None:
        # TableProcessor inserted these already in pdfplumber's source frame;
        # they must not be scaled a second time through the layout page size.
        box = tuple(grid["bbox"])
        if (all(isfinite(value) for value in box) and _area(box) > 0
                and _inside(box, (0, 0, width, height))):
            return box
        raise RecognitionFailure("Native table grid has invalid source bounds")
    left, top, right, bottom = page.polygon.bbox
    box = block.polygon.bbox
    result = ((box[0] - left) * width / (right - left),
              (box[1] - top) * height / (bottom - top),
              (box[2] - left) * width / (right - left),
              (box[3] - top) * height / (bottom - top))
    if not all(isfinite(value) for value in result) or _area(result) <= 0:
        raise RecognitionFailure("Native source owner has invalid bounds")
    return result


def _polygon(box, page, source):
    width, height = source.page_size(page.page_id)
    left, top, right, bottom = page.polygon.bbox
    return PolygonBox.from_bbox((left + box[0] * (right - left) / width,
                                 top + box[1] * (bottom - top) / height,
                                 left + box[2] * (right - left) / width,
                                 top + box[3] * (bottom - top) / height))


def _tree(document, parent):
    for identity in parent.structure or ():
        block = document.get_block(identity)
        if block is not None and not block.removed:
            yield block
            yield from _tree(document, block)


def _atoms(source, page):
    """Read all source objects, including vectors missed by layout boxes."""
    width, height = source.page_size(page)
    atoms = [dict(word, kind="text", source_page=page,
                  source_bbox=word.get("source_bbox", word["bbox"]))
             for word in source.page_words(page)]
    atoms.extend({"kind": "image", "bbox": tuple(image.bbox),
                  "source_bbox": tuple(image.bbox), "source_page": page,
                  "image_index": image.index}
                 for image in source.page_images(page))
    if hasattr(source, "pdf_page"):
        pdf_page = source.pdf_page(page)
        for item in [*getattr(pdf_page, "edges", ()), *getattr(pdf_page, "curves", ())]:
            original = tuple(item.get("bbox") or object_bbox(item))
            crop = (max(0, original[0] - .25), max(0, original[1] - .25),
                    min(width, original[2] + .25), min(height, original[3] + .25))
            if _area(crop) > 0:
                atoms.append({"kind": "vector", "bbox": original,
                              "source_bbox": original, "source_page": page,
                              "crop_bbox": crop})
    unique = {}
    for atom in atoms:
        unique.setdefault(atom_id(atom, page), atom)
    return list(unique.values())


def _support(atom):
    return atom.get("crop_bbox", atom["bbox"])


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _region(page, box, kind, atoms):
    words = [atom for atom in atoms if atom.get("kind") == "text"]
    return Region(page, tuple(box), kind, tuple(atoms), assess_text_quality(words, box))


def _carrier(atom, box):
    """A bitmap dominating a discovery envelope is partitioned by its layout."""
    return (atom.get("kind") == "image" and _inside(atom["bbox"], box)
            and _area(atom["bbox"]) >= _area(box) * .65)


def _near(first, second):
    vertical = min(first[3], second[3]) - max(first[1], second[1])
    horizontal = min(first[2], second[2]) - max(first[0], second[0])
    xgap = max(0, first[0] - second[2], second[0] - first[2])
    ygap = max(0, first[1] - second[3], second[1] - first[3])
    height = min(first[3] - first[1], second[3] - second[1])
    return (vertical >= max(0, height * .25) and xgap <= 12
            or horizontal > 0 and ygap <= 6
            or intersection(first, second) > 0)


def _components(atoms):
    groups = []
    for atom in atoms:
        touching = [group for group in groups
                    if any(_near(_support(atom), _support(other)) for other in group)]
        joined = [atom]
        for group in touching:
            joined.extend(group)
            groups.remove(group)
        groups.append(joined)
    return groups


def _effective_exclusions(page, source, atoms, owners, exclusions):
    """Recover content mislabelled as a footer inside a closed ruling frame.

    A learned footer box is not evidence for deleting an enclosed table note.
    Only a complete physical frame with native body support can disprove that
    label. Page borders and unframed/actually external headers stay excluded.
    The original source objects and learned polygons remain unchanged.
    """
    strokes = []
    for atom in atoms:
        if atom.get("kind") != "vector":
            continue
        box = atom["bbox"]
        width, height = box[2] - box[0], box[3] - box[1]
        if width >= 20 and height <= 1.5 or height >= 20 and width <= 1.5:
            strokes.append(atom)
    frames = []
    page_width, page_height = source.page_size(page.page_id)
    # PDF rules are often separate filled strips. Their corners can stop a
    # few points short of one another, with tiny connector strips omitted by
    # the long-stroke filter above. Join only near axis-aligned endpoints.
    groups = []
    for stroke in strokes:
        box = stroke["bbox"]
        touching = []
        for group in groups:
            for other in group:
                other_box = other["bbox"]
                xgap = max(0, box[0] - other_box[2], other_box[0] - box[2])
                ygap = max(0, box[1] - other_box[3], other_box[1] - box[3])
                if min(xgap, ygap) <= 2 and max(xgap, ygap) <= 8:
                    touching.append(group)
                    break
        joined = [stroke]
        for group in touching:
            joined.extend(group)
            groups.remove(group)
        groups.append(joined)
    for group in groups:
        if len(group) < 4:
            continue
        frame = _union([atom["bbox"] for atom in group])
        width, height = frame[2] - frame[0], frame[3] - frame[1]
        if min(width, height) < 60 or _area(frame) >= page_width * page_height * .85:
            continue
        horizontal = [a["bbox"] for a in group
                      if a["bbox"][2] - a["bbox"][0] >= width * .8
                      and a["bbox"][3] - a["bbox"][1] <= 1.5]
        vertical = [a["bbox"] for a in group
                    if a["bbox"][3] - a["bbox"][1] >= height * .8
                    and a["bbox"][2] - a["bbox"][0] <= 1.5]
        if (not all(any(abs(line[1] - edge) <= 2 for line in horizontal)
                    for edge in (frame[1], frame[3]))
                or not all(any(abs(line[0] - edge) <= 2 for line in vertical)
                           for edge in (frame[0], frame[2]))):
            continue
        if any(_inside(_support(atom), frame)
               and not any(_inside(_support(atom), excluded) for excluded in exclusions)
               and any(_inside(_support(atom), box) for _, box in owners)
               for atom in atoms if atom.get("kind") != "vector"):
            frames.append(frame)
    effective = []
    for excluded in exclusions:
        support = [_support(atom) for atom in atoms if _inside(_support(atom), excluded)]
        if not support or not any(all(_inside(box, frame) for box in support) for frame in frames):
            effective.append(excluded)
    return effective


def _extended_kind(group, nearby, region_overrides):
    """Keep a complete native equation mathematical when it gains operands.

    The Equation label owns its original glyphs, even plain-font multiletter
    variables. Additional prose cannot inherit that label merely by being
    nearby; only explicit math, numbers, single operands and small glyphs can.
    """
    if not nearby:
        return ("math" if math_required(group) else
                "unknown" if any(a.get("kind") != "text" for a in group) else "text")
    if any(owner.block_type != BlockTypes.Equation
           or owner.id in region_overrides and region_overrides[owner.id].kind != "math"
           for owner, _ in nearby):
        return "text"
    for atom in group:
        box = _support(atom)
        if any(intersection(box, owner_box) / max(_area(box), 1e-12) >= .6
               for _, owner_box in nearby):
            continue
        if atom.get("content_kind") == "math":
            continue
        if atom.get("kind") != "text":
            if max(box[2] - box[0], box[3] - box[1]) <= 24:
                continue
            return "text"
        text = atom.get("text", "").strip()
        if (re.search(r"math|symbol|cmmi|cmsy", atom.get("fontname", ""), re.I)
                or re.fullmatch(r"(?:[+-]?\d+(?:[.,]\d+)*|[A-Za-zА-Яа-я]|"
                                r"[+−*/^_()\[\]{},;:.=<>≤≥≠±∑∏∫√∞∂α-ωΑ-Ωϕϑϱ-]+|\\[A-Za-z]+)", text)):
            continue
        return "text"
    return "math"


def _blank(source, page, atoms):
    if atoms:
        return False
    if not hasattr(source, "crop"):
        image = page.lowres_image
    else:
        width, height = source.page_size(page.page_id)
        image = source.crop(page.page_id, (0, 0, width, height), dpi=72)
    if image is None:
        raise RecognitionFailure("A page with no native owners could not be checked for visible content")
    # A uniform white page is blank. Solid coloured/black backgrounds remain
    # visible content and require discovery rather than an empty success.
    extrema = image.convert("L").getextrema()
    return extrema[0] >= 250


def _new_owner(document, page, source, region):
    cls = {"table": Table, "math": Equation, "text": Text,
           "figure": Picture, "unknown": Picture}[region.kind]
    return cls(page_id=page.page_id, polygon=_polygon(region.bbox, page, source))


def _detach(document, page, removed):
    """Detach only after every proposed replacement has passed validation."""
    containers = [page, *_tree(document, page)]
    for parent in containers:
        if parent.structure is not None:
            parent.structure = [identity for identity in parent.structure if identity not in removed]
    for identity in removed:
        owner = document.get_block(identity)
        owner.ignore_for_output = True
        owner.structure = []
        owner.embedded_images = {}


def _insert(page, owner):
    page.add_full_block(owner)
    structure = list(page.structure or ())
    position = next((index for index, identity in enumerate(structure)
                     if page.get_block(identity).polygon.y_start > owner.polygon.y_start), len(structure))
    structure.insert(position, owner.id)
    page.structure = structure


def _discover(source, page, parent_box, atoms, owners, exclusions, client, store, *, replacing=()):
    """Build a mutation-free proposal, preserving existing verified tables."""
    parent = _region(page.page_id, parent_box, "image", atoms)
    regions, parts = discover_scan(source, parent, client, store=store)
    table_by_box = {(part.page, part.bbox): part for part in parts}
    # Full scan bitmaps carry the source pixels. Their subregions partition
    # the bitmap and therefore must not inherit the same page-sized atom.
    carrier = {atom_id(atom, page.page_id) for atom in atoms
               if _carrier(atom, parent_box)}
    assigned = [[] for _ in regions]
    for atom in atoms:
        if atom_id(atom, page.page_id) in carrier:
            continue
        box = _support(atom)
        if any(_inside(box, excluded) for excluded in exclusions):
            continue
        if not _inside(box, parent_box):
            if intersection(box, parent_box) > 0:
                raise RecognitionFailure("Discovery crop cuts across an original source atom")
            continue
        matches = [index for index, region in enumerate(regions) if _inside(box, region.bbox)]
        if len(matches) != 1:
            raise RecognitionFailure("Scanned discovery omitted or multiply assigned an original source atom")
        assigned[matches[0]].append(atom)
    proposals, removed, preserved = [], set(), set()
    for index, region in enumerate(regions):
        if any(_inside(region.bbox, excluded) for excluded in exclusions):
            continue
        if any(intersection(region.bbox, excluded) > 0 for excluded in exclusions):
            raise RecognitionFailure("Scanned region crosses an intentionally excluded page header or footer")
        overlaps = [(owner, box) for owner, box in owners
                    if owner.id not in replacing and intersection(region.bbox, box) > 0
                    and not _inside(parent_box, box)]
        existing_tables = [(owner, box) for owner, box in overlaps if owner.block_type in _TABLES]
        if existing_tables:
            if (region.kind != "table" or len(existing_tables) != 1
                    or any(abs(a - b) > .75 for a, b in zip(region.bbox, existing_tables[0][1]))):
                raise RecognitionFailure("Discovered table conflicts with an existing native table owner")
            owner, box = existing_tables[0]
            if any(other.id != owner.id and not _inside(other_box, box)
                   for other, other_box in overlaps):
                raise RecognitionFailure("Discovered table also overlaps another native source owner")
            original = table_by_box[(region.page, region.bbox)]
            part = replace(original, bbox=box, grid=owner.table_grid)
            preserved.add(owner.id)
            proposals.append((owner, replace(region, bbox=box, grid=owner.table_grid), part))
            removed.update(other.id for other, _ in overlaps if other.id != owner.id)
            continue
        if any(not _inside(box, region.bbox) for _, box in overlaps):
            raise RecognitionFailure("Scanned region cuts across an existing native source owner")
        removed.update(owner.id for owner, _ in overlaps)
        populated = replace(region, atoms=tuple(assigned[index]) or region.atoms,
                            quality=assess_text_quality(
                                [a for a in assigned[index] if a.get("kind") == "text"], region.bbox))
        proposals.append((None, populated, table_by_box.get((region.page, region.bbox))))
    for owner, box in owners:
        if owner.id in preserved:
            continue
        if owner.id in replacing:
            removed.add(owner.id)
            continue
        if _inside(parent_box, box):
            # A whole-page unknown owner is replaced by the successful layout.
            if owner.block_type not in _PICTURES and not _inside(box, parent_box):
                raise RecognitionFailure("Discovery envelope overlaps a non-picture native owner")
            removed.add(owner.id)
        elif intersection(box, parent_box) > 0 and owner.id not in removed | preserved:
            raise RecognitionFailure("Scanned discovery did not preserve every native content owner")
    if not proposals:
        raise RecognitionFailure("Scanned discovery produced no required content owners")
    return proposals, removed


def _choose_owner(atom, owners):
    """Assign complete source support despite approximate learned rectangles.

    A source atom is never clipped to a layout rectangle. Physical table grids
    take precedence; otherwise a clearly dominant overlap must identify its
    one native owner. Ambiguous near-equal assignments remain errors.
    """
    box = _support(atom)
    tables = [(owner, owner_box) for owner, owner_box in owners
              if owner.block_type in _TABLES and _inside(
                  box, owner_box, tolerance=(3 if getattr(owner, "table_grid", None) is None else 1.25)
                  if atom.get("kind") == "vector" else .75)]
    if len(tables) > 1:
        raise RecognitionFailure("Original source atom belongs to conflicting native tables")
    if tables:
        return tables[0][0]
    scores = sorted(((intersection(box, owner_box) / max(_area(box), 1e-12), owner)
                     for owner, owner_box in owners if intersection(box, owner_box) > 0),
                    key=lambda pair: pair[0], reverse=True)
    if not scores:
        return None
    best, owner = scores[0]
    runner = scores[1][0] if len(scores) > 1 else 0
    if atom.get("kind") == "vector" and best < .6:
        # A long ruling stroke crossing several small text rectangles is a
        # missing whole frame, not an ambiguously owned text glyph.
        return None
    if atom.get("kind") == "image" and box[3] - box[1] <= 72:
        original = next(owner_box for candidate, owner_box in owners if candidate.id == owner.id)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        # Learned text boxes commonly describe only the prose x-height. A
        # taller inline raster formula can protrude above AND below that row.
        # Its center and a uniquely dominant overlap still identify the row.
        if (best >= .25 and best - runner >= .25
                and original[0] <= center[0] <= original[2]
                and original[1] <= center[1] <= original[3]):
            return owner
    if best < .6 or best - runner < .25:
        raise RecognitionFailure("Native layout clips or ambiguously assigns an original source atom")
    return owner


def _owner_boxes(active, page, source, part_overrides):
    return [(owner, part_overrides[owner.id].bbox if owner.id in part_overrides
             else _box(owner, page, source))
            for owner in active if owner.page_id == page.page_id]


def _coarse_table_frames(page, source, atoms, owners, part_overrides):
    """Complete coarse table crops from connected physical ruling support."""
    coarse = [(owner, box) for owner, box in owners
              if owner.block_type in _TABLES and getattr(owner, "table_grid", None) is None]
    if not coarse:
        return
    frames = [_union([_support(atom) for atom in group])
              for group in _components([atom for atom in atoms if atom.get("kind") == "vector"])
              if len(group) >= 4]
    for owner, original in coarse:
        candidates = [frame for frame in frames if _area(frame) >= _area(original) * .75
                      and intersection(frame, original) >= min(_area(frame), _area(original)) * .9]
        if not candidates:
            continue
        if len(candidates) > 1:
            raise RecognitionFailure("Several physical frames compete for one coarse native table")
        frame = candidates[0]
        limit = max(6, min(original[2] - original[0], original[3] - original[1]) * .05)
        if any(abs(a - b) > limit for a, b in zip(frame, original)):
            raise RecognitionFailure("Physical frame disagrees with coarse native table bounds")
        box = _union([frame, original])
        if any(other.id != owner.id and other.block_type in _TABLES
               and intersection(frame, other_box) > 0 for other, other_box in owners):
            raise RecognitionFailure("Physical frame crosses another native table")
        part_overrides[owner.id] = TablePart(
            page.page_id, box, None, source.page_size(page.page_id),
            _caption(source, page.page_id, original), source)


def _discovery_envelope(identity, box, atoms, owners):
    """Include complete uniquely assigned source support before image routing."""
    if identity is None:
        return box
    support = []
    for atom in atoms:
        owner = _choose_owner(atom, owners)
        if owner is not None and owner.id == identity:
            support.append(_support(atom))
    return _union([box, *support])


def _reconcile_support(page, source, atoms, owners, exclusions,
                       part_overrides, region_overrides):
    """Hand exact assigned atoms to LLM jobs, including vector mathematics."""
    assigned = {owner.id: [] for owner, _ in owners}
    changed = set()
    orphan = []
    for atom in atoms:
        box = _support(atom)
        if any(_inside(box, excluded) for excluded in exclusions):
            continue
        owner = _choose_owner(atom, owners)
        if owner is None:
            orphan.append(atom)
            continue
        assigned[owner.id].append(atom)
        overlaps = [(other, other_box) for other, other_box in owners
                    if intersection(box, other_box) > 0]
        if len(overlaps) > 1:
            # Both jobs need their assigned atom lists; rereading every word
            # intersecting their approximate rectangles would duplicate text.
            changed.update(other.id for other, _ in overlaps)
        original = next(owner_box for other, owner_box in owners if other.id == owner.id)
        if not _inside(box, original, tolerance=0) or atom.get("kind") == "vector":
            changed.add(owner.id)
    for owner, original in owners:
        support = assigned[owner.id]
        if not support:
            continue
        previous = region_overrides.get(owner.id)
        box = _union([_support(atom) for atom in support])
        if owner.block_type in _TABLES:
            # Extra stroke ends and actual formula overhangs enlarge only the
            # crop envelope. The physical source grid/cells stay immutable.
            box = _union([original, box])
            if owner.id in part_overrides:
                part_overrides[owner.id] = replace(part_overrides[owner.id], bbox=box)
            else:
                part_overrides[owner.id] = TablePart(
                    page.page_id, box, getattr(owner, "table_grid", None), source.page_size(page.page_id),
                    _caption(source, page.page_id, original), source)
            continue
        if owner.id not in changed and previous is None:
            continue
        if previous is not None:
            # Classified raster owners have visual support beyond selectable
            # atoms; their validated discovery envelope remains authoritative.
            box = _union([previous.bbox, box])
            kind = previous.kind
        else:
            kind = ("math" if owner.block_type == BlockTypes.Equation else
                    "unknown" if owner.block_type in _PICTURES else "text")
        region_overrides[owner.id] = _region(page.page_id, box, kind, support)
    return orphan


def _coalesce_duplicates(document, page, source, atoms, active, region_overrides):
    """Coalesce duplicate learned boxes using shared complete source support.

    Layout can label the same physical line as both Text and Caption/Equation.
    A smaller owner's source atoms must be almost entirely shared, and the
    rectangles must substantially overlap. Nearby disjoint paragraphs never
    qualify. Tables keep their physical ownership and are not coalesced here.
    """
    priority = {BlockTypes.Equation: 9, BlockTypes.TextInlineMath: 8,
                BlockTypes.Footnote: 7, BlockTypes.Caption: 6,
                BlockTypes.SectionHeader: 5, BlockTypes.Text: 1}
    while True:
        owners = [(owner, _box(owner, page, source)) for owner in active
                  if owner.page_id == page.page_id and owner.block_type not in _TABLES | _PICTURES]
        support = {
            owner.id: {atom_id(atom, page.page_id) for atom in atoms
                       if intersection(_support(atom), box) / max(_area(_support(atom)), 1e-12) >= .6}
            for owner, box in owners
        }
        found = None
        for index, (first, first_box) in enumerate(owners):
            for second, second_box in owners[index + 1:]:
                first_atoms, second_atoms = support[first.id], support[second.id]
                if not first_atoms or not second_atoms:
                    continue
                smaller = min(len(first_atoms), len(second_atoms))
                shared = len(first_atoms & second_atoms)
                geometry = intersection(first_box, second_box) / min(_area(first_box), _area(second_box))
                if shared >= smaller * .9 and geometry >= .65:
                    found = first, second, first_box, second_box
                    break
            if found:
                break
        if found is None:
            return active
        first, second, first_box, second_box = found
        canonical = max((first, second), key=lambda owner: (priority.get(owner.block_type, 2),
                                                           len(support[owner.id])))
        box = _union([first_box, second_box])
        structure = list(dict.fromkeys([*(first.structure or ()), *(second.structure or ())]))
        replacement = canonical.model_copy(update={
            "block_id": None, "polygon": _polygon(box, page, source),
            "structure": structure, "source": "processor", "ignore_for_output": False,
        })
        removed = {first.id, second.id}
        _detach(document, page, removed)
        _insert(page, replacement)
        active = [owner for owner in active if owner.id not in removed] + [replacement]
        chosen = support[first.id] | support[second.id]
        complete_atoms = [atom for atom in atoms if atom_id(atom, page.page_id) in chosen]
        kind = ("math" if canonical.block_type == BlockTypes.Equation
                and support[first.id] == support[second.id] else "text")
        region_overrides[replacement.id] = _region(page.page_id, box, kind, complete_atoms)


def _discover_native_owners(document, source, blocks, client, store):
    """Return complete owners plus explicit source TablePart/Region overrides.

    A compact layout response classifies scans before any logical tables are
    grouped. Remaining unowned native atoms become whole composite owners.
    Conflicting rectangles fail closed instead of disappearing from output.
    """
    active = list(blocks)
    part_overrides, region_overrides = {}, {}
    for page in document.pages:
        width, height = source.page_size(page.page_id)
        atoms = _atoms(source, page.page_id)
        exclusions = [_box(block, page, source) for block in _tree(document, page)
                      if block.block_type in _EXCLUDED]
        exclusions = _effective_exclusions(
            page, source, atoms, _owner_boxes(active, page, source, part_overrides), exclusions)
        active = _coalesce_duplicates(document, page, source, atoms, active, region_overrides)
        owners = _owner_boxes(active, page, source, part_overrides)
        _coarse_table_frames(page, source, atoms, owners, part_overrides)
        owners = _owner_boxes(active, page, source, part_overrides)
        large = [(block, box) for block, box in owners if block.block_type in _PICTURES
                 and _area(box) >= width * height * .65]
        carrier = [atom for atom in atoms if atom.get("kind") == "image"
                   and _area(atom["bbox"]) >= width * height * .65]
        candidates = []
        if carrier:
            candidates = [(None, (0.0, 0.0, width, height))]
        elif large:
            if len(large) > 1:
                raise RecognitionFailure("Multiple overlapping whole-page native picture owners")
            candidates = [(large[0][0].id, large[0][1])]
        else:
            candidates = [(block.id, box) for block, box in owners
                          if block.block_type in _PICTURES and _area(box) >= 1500
                          and min(box[2] - box[0], box[3] - box[1]) >= 28]
        if (not candidates and not owners and (not atoms or any(
                not any(_inside(_support(atom), excluded) for excluded in exclusions)
                for atom in atoms)) and not _blank(source, page, atoms)):
            candidates = [(None, (0.0, 0.0, width, height))]
        partitioned = set()
        for identity, candidate in candidates:
            if identity is not None and all(owner.id != identity for owner, _ in owners):
                continue
            candidate = _discovery_envelope(identity, candidate, atoms, owners)
            proposals, removed = _discover(source, page, candidate, atoms, owners,
                                            exclusions, client, store,
                                            replacing=() if identity is None else (identity,))
            new = []
            for owner, region, part in proposals:
                owner = owner or _new_owner(document, page, source, region)
                new.append((owner, region, part))
            _detach(document, page, removed)
            active = [block for block in active if block.id not in removed]
            for owner, region, part in new:
                if owner.block_id is None:
                    _insert(page, owner)
                    active.append(owner)
                region_overrides[owner.id] = region
                if part is not None:
                    part_overrides[owner.id] = part
            owners = _owner_boxes(active, page, source, part_overrides)
            partitioned.update(atom_id(atom, page.page_id) for atom in atoms if _carrier(atom, candidate))
        # Successful complete layouts partition their carrier bitmaps. This
        # applies to subpage raster tables as well as whole scanned pages.
        atoms = [atom for atom in atoms if atom_id(atom, page.page_id) not in partitioned]

        orphan = _reconcile_support(page, source, atoms, owners, exclusions,
                                    part_overrides, region_overrides)
        if not orphan:
            continue
        for group in _components(orphan):
            box = _union([_support(atom) for atom in group])
            nearby = [(owner, owner_box) for owner, owner_box in owners
                      if owner.block_type not in _TABLES and _near(box, owner_box)]
            # Keep a raster operator and all native operands in one paragraph
            # job. A source owner is absorbed whole; never steal just a word.
            if nearby:
                box = _union([box, *[owner_box for _, owner_box in nearby]])
                included = [atom for atom in atoms if _inside(_support(atom), box)]
                group = list({atom_id(atom, page.page_id): atom for atom in [*group, *included]}.values())
            conflicts = [(owner, owner_box) for owner, owner_box in owners
                         if intersection(box, owner_box) > 0 and owner not in [item[0] for item in nearby]]
            if conflicts:
                raise RecognitionFailure("Unowned source component conflicts with an existing native owner")
            if (
                any(a.get("kind") == "image" for a in group)
                and min(box[2] - box[0], box[3] - box[1]) >= 28 and _area(box) >= 1500
                or sum(a.get("kind") == "vector" for a in group) >= 4
            ):
                # A missing raster table or ruled frame needs discovery before
                # joining, even when it occupies only part of a digital page.
                proposals, removed = _discover(source, page, box, group, nearby, exclusions, client, store)
                _detach(document, page, removed)
                active = [item for item in active if item.id not in removed]
                owners = [(item, owner_box) for item, owner_box in owners if item.id not in removed]
                for _, region, part in proposals:
                    owner = _new_owner(document, page, source, region)
                    _insert(page, owner)
                    active.append(owner)
                    owners.append((owner, region.bbox))
                    region_overrides[owner.id] = region
                    if part is not None:
                        part_overrides[owner.id] = part
                continue
            kind = _extended_kind(group, nearby, region_overrides)
            region = _region(page.page_id, box, kind, group)
            owner = _new_owner(document, page, source, region)
            removed = {item[0].id for item in nearby}
            _detach(document, page, removed)
            active = [item for item in active if item.id not in removed]
            owners = [(item, owner_box) for item, owner_box in owners if item.id not in removed]
            _insert(page, owner)
            active.append(owner)
            owners.append((owner, box))
            region_overrides[owner.id] = region
    return active, part_overrides, region_overrides


def discover_native_owners(document, source, blocks, client, store):
    """Discover and audit atom ownership, restoring native trees on failure."""
    page_state = [(page, page.block_id, list(page.children or ()),
                   None if page.structure is None else list(page.structure))
                  for page in document.pages]
    block_state = [(block, block.ignore_for_output,
                    None if block.structure is None else list(block.structure),
                    block.embedded_images)
                   for page in document.pages for block in _tree(document, page)]
    try:
        return _discover_native_owners(document, source, blocks, client, store)
    except BaseException:
        for page, identity, children, structure in page_state:
            page.block_id, page.children, page.structure = identity, children, structure
        for block, ignored, structure, resources in block_state:
            block.ignore_for_output, block.structure = ignored, structure
            block.embedded_images = resources
        raise
