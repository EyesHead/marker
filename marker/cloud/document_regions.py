"""Partition native prose and complete raster candidates before recognition."""

from dataclasses import replace

from marker.cloud.content import atom_id, bounds, group_expression_regions
from marker.cloud.regions import Region, assess_text_quality
from marker.cloud.source import intersection


def _region(page, kind, atoms, *, inline_group=None):
    box = bounds(atoms)
    words = [atom for atom in atoms if atom.get("kind", "text") == "text"]
    return Region(
        page,
        box,
        kind,
        tuple(atoms),
        assess_text_quality(words, box),
        inline_group=inline_group,
    )


def document_regions(regions) -> list[Region]:
    """Retain native owners while separating raster candidates from prose.

    A short bitmap is only a temporary partition seed. Its restored owner
    remains an image candidate unless independent native mathematical evidence
    exists; the recognition stage must decide whether it is math or a figure.
    The height bound chooses partitioning behavior, not a content classification.
    """
    regions = list(regions)
    kinds = {"text", "image", "math"}
    kept = [region for region in regions if region.kind not in kinds]
    kept_ids = {atom_id(atom, region.page) for region in kept for atom in region.atoms}
    originals, parents, pages, proven_math = {}, {}, {}, set()
    for index, region in enumerate(regions):
        if region.kind not in kinds:
            continue
        for atom in region.atoms:
            identity = atom_id(atom, region.page)
            if identity in kept_ids:
                continue
            originals.setdefault(identity, atom)
            parents.setdefault(identity, index)
            pages.setdefault(identity, region.page)
            if region.kind == "math" or atom.get("content_kind") == "math":
                proven_math.add(identity)

    candidates = {
        identity
        for identity, atom in originals.items()
        if atom.get("kind") == "image" and 0 < atom["bbox"][3] - atom["bbox"][1] <= 72
    }
    large = [
        identity
        for identity, atom in originals.items()
        if atom.get("kind") == "image" and atom["bbox"][3] - atom["bbox"][1] > 72
    ]

    # Keep genuine large images and their overlapping labels together. Nearby
    # prose outside those pixels must remain native text with its own owner.
    image_groups = []
    for identity in large:
        overlapping = [
            group
            for group in image_groups
            if pages[group[0]] == pages[identity]
            and any(
                intersection(originals[identity]["bbox"], originals[item]["bbox"]) > 0
                for item in group
            )
        ]
        group = [identity]
        for previous in overlapping:
            group.extend(previous)
            image_groups.remove(previous)
        image_groups.append(group)
    owned = set(large)
    large_images = [list(group) for group in image_groups]
    for identity, atom in originals.items():
        if identity in owned:
            continue
        scores = [
            sum(
                intersection(atom["bbox"], originals[item]["bbox"])
                for item in group
                if pages[item] == pages[identity]
            )
            for group in large_images
        ]
        if scores and max(scores) > 0:
            image_groups[max(range(len(scores)), key=scores.__getitem__)].append(
                identity
            )
            owned.add(identity)
    large_regions = []
    for group in image_groups:
        atoms = [atom for identity, atom in originals.items() if identity in group]
        parent = regions[parents[group[0]]]
        kind = (
            "math"
            if any(
                identity in proven_math and originals[identity].get("kind") == "image"
                for identity in group
            )
            else "image"
        )
        if parent.kind == kind and tuple(atoms) == parent.atoms:
            large_regions.append(parent)
        else:
            large_regions.append(
                _region(pages[group[0]], kind, atoms, inline_group=parent.inline_group)
            )

    temporary = []
    for index, region in enumerate(regions):
        if region.kind not in kinds:
            continue
        atoms = []
        for atom in region.atoms:
            identity = atom_id(atom, region.page)
            if identity in owned or parents.get(identity) != index:
                continue
            copied = dict(atom)
            if identity in candidates:
                copied["content_kind"] = "math"
            atoms.append(copied)
        if atoms:
            temporary.append(replace(region, atoms=tuple(atoms)))

    restored, inherited = [], {}
    for group in group_expression_regions(temporary):
        identities = [atom_id(atom, group.page) for atom in group.atoms]
        atoms = [originals[identity] for identity in identities]
        kind = group.kind
        if any(identity in proven_math for identity in identities):
            kind = "math"
        elif any(identity in candidates for identity in identities):
            # A diagram may contain Greek or math-font labels. Such text does
            # not establish that the unknown bitmap itself is a formula.
            kind = "image"
        inline = {
            regions[parents[identity]].inline_group
            for identity in identities
            if regions[parents[identity]].inline_group is not None
        }
        if len(inline) == 1:
            value = next(iter(inline))
            if group.inline_group is not None:
                inherited[group.inline_group] = value
            else:
                group = replace(group, inline_group=value)
        restored.append(replace(group, kind=kind, atoms=tuple(atoms)))
    restored = [
        replace(
            group, inline_group=inherited.get(group.inline_group, group.inline_group)
        )
        for group in restored
    ]

    # Preserve complete native paragraphs that needed no inline partition.
    owner = {
        atom_id(atom, group.page): group for group in restored for atom in group.atoms
    }
    paragraphs, paragraph_ids = [], set()
    for index, region in enumerate(regions):
        identities = [atom_id(atom, region.page) for atom in region.atoms]
        if (
            region.kind == "text"
            and identities
            and len(identities) == len(set(identities))
            and all(
                parents.get(identity) == index
                and identity in owner
                and owner[identity].kind == "text"
                and owner[identity].inline_group is None
                for identity in identities
            )
        ):
            paragraphs.append(region)
            paragraph_ids.update(identities)
    fragments = []
    for group in restored:
        atoms = [
            atom
            for atom in group.atoms
            if atom_id(atom, group.page) not in paragraph_ids
        ]
        if len(atoms) == len(group.atoms):
            fragments.append(group)
        elif atoms:
            fragments.append(
                _region(group.page, group.kind, atoms, inline_group=group.inline_group)
            )
    return sorted(
        [*kept, *large_regions, *paragraphs, *fragments],
        key=lambda region: (
            region.page,
            region.inline_group[1][1] if region.inline_group else region.bbox[1],
            region.bbox[0],
        ),
    )
