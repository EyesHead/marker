"""Source-owned whole-expression recognition and auditable visible content."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from html import escape
from typing import Literal
from xml.etree import ElementTree as ET

from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator

from marker.cloud.context import RecognitionFailure
from marker.cloud.regions import Region, assess_text_quality

MATH_NAMESPACE = "http://www.w3.org/1998/Math/MathML"
ET.register_namespace("", MATH_NAMESPACE)


@dataclass
class ContentResult:
    kind: str
    markdown: str = ""
    html: str = ""
    resources: dict[str, Image.Image] = field(default_factory=dict)
    complete: bool = True
    reasons: tuple[str, ...] = ()
    latex: str | None = None
    source_atom_ids: tuple[str, ...] = ()

    @property
    def status(self):
        return "complete" if self.complete else "incomplete"


def atom_id(atom, page=None):
    """Stable ownership identity, independent of virtual display coordinates."""
    payload = (
        atom.get("source_page", page),
        tuple(atom.get("source_bbox", atom["bbox"])),
        atom.get("kind", "text"),
        atom.get("text", ""),
        atom.get("image_index"),
    )
    return "a" + hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


def bounds(atoms, *, layout=False):
    boxes = [
        a.get("layout_bbox", a["bbox"]) if layout else a.get("source_bbox", a["bbox"])
        for a in atoms
    ]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def source_lines(atoms):
    """Keep superscripts with their baseline; never mix source pages."""
    lines = []
    for atom in sorted(
        atoms, key=lambda a: (a.get("source_page", 0), a["bbox"][1], a["bbox"][0])
    ):
        box = atom["bbox"]
        candidates = []
        for line in lines:
            if atom.get("source_page", 0) != line[0].get("source_page", 0):
                continue
            other = bounds(line)
            overlap = min(box[3], other[3]) - max(box[1], other[1])
            if overlap >= min(box[3] - box[1], other[3] - other[1]) * 0.3:
                candidates.append(line)
        if candidates:
            # Tall diagrams may overlap several baselines. Assign them once
            # to the closest baseline instead of bridging independent rows.
            line = min(
                candidates,
                key=lambda row: abs(
                    (bounds(row)[1] + bounds(row)[3]) - (box[1] + box[3])
                ),
            )
            line.append(atom)
        else:
            lines.append([atom])
    return [
        sorted(line, key=lambda a: a["bbox"][0])
        for line in sorted(
            lines, key=lambda line: (line[0].get("source_page", 0), bounds(line)[1])
        )
    ]


def math_required(atoms):
    text = " ".join(a.get("text", "") for a in atoms)
    if re.search(r"[=<>≤≥≠±∑∏∫√∞∂]|\\[A-Za-z]+|[α-ωΑ-Ωϕϑϱ]", text):
        return True
    if any(
        re.search(r"math|symbol|cmmi|cmsy", a.get("fontname", ""), re.I) for a in atoms
    ):
        return True
    words = [a for a in atoms if a.get("kind", "text") == "text"]
    if len(words) > 1:
        sizes = [a.get("size", a["bbox"][3] - a["bbox"][1]) for a in words]
        if max(sizes) > min(sizes) * 1.3 and any(
            re.fullmatch(r"[A-Za-zА-Яа-я0-9]+", a.get("text", "")) for a in words
        ):
            return True
    # Small raster operators adjacent to selectable symbols require one
    # complete formula crop. Larger diagrams remain figure candidates.
    return bool(words) and any(
        a.get("kind") == "image"
        and a["bbox"][2] - a["bbox"][0] <= 24
        and a["bbox"][3] - a["bbox"][1] <= 24
        for a in atoms
    )


def expression_groups(atoms):
    """Identify complete mathematical glyph runs within one source baseline.

    Prose, separating punctuation and wide source gaps delimit expressions.
    Digits/ordinary variable glyphs belong to a neighboring math seed even
    though they are not independently sufficient to require transcription.
    """
    baseline_size = max(
        (
            a.get("size", a["bbox"][3] - a["bbox"][1])
            for a in atoms
            if a.get("kind", "text") == "text"
        ),
        default=10,
    )

    def member(atom):
        if atom.get("kind") == "image":
            box = atom["bbox"]
            return box[2] - box[0] <= 24 and box[3] - box[1] <= 24
        if atom.get("kind", "text") != "text":
            return False
        text = atom.get("text", "").strip()
        if not text or text in (",", ";", ":"):
            return False
        if re.search(r"[=<>≤≥≠±∑∏∫√∞∂α-ωΑ-Ωϕϑϱ]|\\[A-Za-z]+", text):
            return True
        if re.fullmatch(r"[+−*/^_()\[\]{}-]+", text):
            return True
        if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)*[,;]?", text):
            return True
        if re.fullmatch(r"[A-Za-zА-Яа-я]", text):
            return True
        if re.fullmatch(r"[A-Za-zА-Яа-я0-9]+", text):
            # PdfSource words expose fontname/size, without the optional
            # table-style italic flag. Use the same explicit math-font signal
            # as math_required so operands cannot be excluded from a formula
            # that their own font makes mandatory.
            if re.search(r"math|symbol|cmmi|cmsy", atom.get("fontname", ""), re.I):
                return True
            size = atom.get("size", atom["bbox"][3] - atom["bbox"][1])
            return size < baseline_size * 0.8 or (atom.get("italic") and len(text) <= 4)
        return False

    runs = []
    current = []
    for atom in sorted(atoms, key=lambda a: a["bbox"][0]):
        if not member(atom):
            if current:
                runs.append(current)
                current = []
            continue
        if current and atom["bbox"][0] - current[-1]["bbox"][2] > 16:
            runs.append(current)
            current = []
        current.append(atom)
    if current:
        runs.append(current)
    return [run for run in runs if math_required(run)]


def group_expression_regions(regions):
    """Partition non-table source atoms into complete expressions once.

    Nearby pixels can be supplied as context later, but this partition alone
    determines ownership. Table/unknown/blank regions keep their identity.
    """
    kept = [r for r in regions if r.kind not in ("text", "image", "math")]
    by_page = {}
    seen = set()
    for region in regions:
        if region.kind not in ("text", "image", "math"):
            continue
        for atom in region.atoms:
            identity = atom_id(atom, region.page)
            if identity in seen:
                continue
            seen.add(identity)
            by_page.setdefault(region.page, []).append(atom)
    for page, atoms in by_page.items():
        for line in source_lines(atoms):
            groups = []
            for atom in line:
                if groups and atom["bbox"][0] - groups[-1][-1]["bbox"][2] <= 16:
                    groups[-1].append(atom)
                else:
                    groups.append([atom])
            for group in groups:
                box = bounds(group)
                words = [a for a in group if a.get("kind", "text") == "text"]
                kind = (
                    "math"
                    if math_required(group)
                    else (
                        "image"
                        if any(a.get("kind") == "image" for a in group)
                        else "text"
                    )
                )
                kept.append(
                    Region(
                        page, box, kind, tuple(group), assess_text_quality(words, box)
                    )
                )
    return sorted(kept, key=lambda r: (r.page, r.bbox[1], r.bbox[0]))


def latex_to_mathml(latex, *, display="inline"):
    """Typeset real MathML, retaining exact recognized LaTeX for inspection."""
    from latex2mathml.converter import convert

    if not latex or re.search(r"[\x00-\x1f]", latex):
        raise RecognitionFailure("Required formula has no usable LaTeX")
    try:
        root = ET.fromstring(convert(latex, display=display))
        if not list(root) or not any((node.text or "").strip() for node in root.iter()):
            raise ValueError("empty math")
        # Unknown commands may be emitted as literal text instead of raising.
        if any("\\" in (node.text or "") for node in root.iter()):
            raise ValueError("unsupported command")
        semantics = ET.Element(f"{{{MATH_NAMESPACE}}}semantics")
        for child in list(root):
            root.remove(child)
            semantics.append(child)
        annotation = ET.SubElement(
            semantics,
            f"{{{MATH_NAMESPACE}}}annotation",
            {"encoding": "application/x-tex"},
        )
        annotation.text = latex
        root.append(semantics)
        return ET.tostring(root, encoding="unicode")
    except Exception:
        raise RecognitionFailure(
            "Required LaTeX could not be rendered as MathML"
        ) from None


def resource_name(kind, page, box, suffix=""):
    digest = hashlib.sha256(repr((page, tuple(box), suffix)).encode()).hexdigest()[:16]
    return f"{kind}-p{page}-{digest}.png"


def context_box(source, page, box, padding=8):
    width, height = source.page_size(page)
    return (
        max(0, box[0] - padding),
        max(0, box[1] - padding),
        min(width, box[2] + padding),
        min(height, box[3] + padding),
    )


class RegionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["math", "figure", "text"]
    resolved: bool
    latex: str = ""
    text: str = ""
    display: Literal["inline", "block"] = "block"

    @model_validator(mode="after")
    def usable_content(self):
        if (
            not self.resolved
            or (self.kind == "math" and not self.latex.strip())
            or (self.kind == "text" and not self.text.strip())
        ):
            raise ValueError("Required content unresolved")
        if self.kind == "math":
            try:
                latex_to_mathml(self.latex)
            except RecognitionFailure:
                raise ValueError("Required math is unresolved or unsupported") from None
        return self


def recognize_region(source, region, client):
    """Return content/resources/status without hiding required recognition loss."""
    identities = tuple(atom_id(a, region.page) for a in region.atoms)
    if not region.required:
        return ContentResult(region.kind, source_atom_ids=identities)
    if (
        region.kind == "text"
        and region.quality.trusted
        and not math_required(region.atoms)
    ):
        text = "\n".join(
            " ".join(a.get("text", "") for a in line)
            for line in source_lines(region.atoms)
        )
        return ContentResult(
            "text", text, "<p>" + escape(text) + "</p>", source_atom_ids=identities
        )
    resources = {}
    try:
        retained = source.resource(region.page, region.bbox)
        name = resource_name(
            "formula" if region.kind == "math" else "source", region.page, region.bbox
        )
        resources[name] = retained
        payload = {
            "page": region.page,
            "bbox": region.bbox,
            "required_kind": region.kind,
            "atoms": [
                {
                    "id": atom_id(a, region.page),
                    "kind": a.get("kind", "text"),
                    "text": a.get("text", ""),
                    "bbox": a.get("source_bbox", a["bbox"]),
                }
                for a in region.atoms
            ],
        }
        prompt = (
            "Recognize only the assigned source atoms in this crop. Nearby pixels are context, never additional output. "
            "Classify a complete mathematical expression as math; transcribe ALL operators, subscripts and superscripts "
            "in one LaTeX string. Preserve phi/varphi, Cyrillic/Latin subscripts and decimal commas exactly. "
            "A diagram is figure. Never guess unreadable content: set resolved=false.\nSOURCE:\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        class RequiredRegionResponse(RegionResponse):
            @model_validator(mode="after")
            def required_kind(self):
                if region.kind == "math" and self.kind != "math":
                    raise ValueError("Required expression must be math")
                return self

        response = client.request(
            prompt,
            [source.crop(region.page, context_box(source, region.page, region.bbox))],
            RequiredRegionResponse,
        )
        if response["kind"] == "math":
            html = latex_to_mathml(response["latex"], display=response["display"])
            delimiter = "$$" if response["display"] == "block" else "$"
            return ContentResult(
                "math",
                delimiter + response["latex"] + delimiter,
                html,
                resources,
                latex=response["latex"],
                source_atom_ids=identities,
            )
        if response["kind"] == "figure":
            return ContentResult(
                "figure",
                f"![Figure]({name})",
                f'<img src="{name}" alt="Figure">',
                resources,
                source_atom_ids=identities,
            )
        return ContentResult(
            "text",
            response["text"],
            "<p>" + escape(response["text"]) + "</p>",
            resources,
            source_atom_ids=identities,
        )
    except RecognitionFailure as error:
        markdown = "\n".join(f"![Unresolved source]({name})" for name in resources)
        return ContentResult(
            region.kind,
            markdown,
            resources=resources,
            complete=False,
            reasons=(error.reason,),
            source_atom_ids=identities,
        )
