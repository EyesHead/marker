"""Local PDF draft followed by independent whole-table and expression jobs."""

from dataclasses import replace
import hashlib
from html import escape
import json
from pathlib import Path
import re
import tempfile
from time import monotonic
from typing import Annotated

from bs4 import BeautifulSoup, NavigableString

from marker.cloud.client import ResponseValidationError
from marker.cloud.content import (
    ContentResult,
    MATH_CLASSIFICATION_PROMPT,
    MATH_BASELINE_PROMPT,
    atom_id,
    latex_to_mathml,
    math_required,
    recognize_region,
    resource_name,
    source_script_hints,
    validate_script_hints,
)
from marker.cloud.context import RecognitionFailure
from marker.cloud.document_regions import document_regions
from marker.cloud.parallel import ordered_cloud_map
from marker.cloud.regions import Region, assess_text_quality
from marker.cloud.resources import retain_png
from marker.cloud.source import PdfSource, intersection
from marker.cloud.stage_store import PIPELINE_VERSION, StageStore
from marker.cloud.table_layout import TableLayoutError, apply_source_table_layout
from marker.cloud.tables import detect_tables, group_tables
from marker.cloud.whole_tables import recognize_whole_table, normalize_table_html
from marker.converters.ollama import OllamaPdfConverter
from marker.logger import get_logger
from marker.renderers.markdown import MarkdownOutput

logger = get_logger()


class OwnerSource:
    """Keep PDFium/pdfplumber work on the caller while HTTP jobs overlap."""

    def __init__(self, source, source_call):
        self.source, self.source_call = source, source_call

    def __getattr__(self, name):
        member = getattr(self.source, name)
        if callable(member):
            return lambda *args, **kwargs: self.source_call(
                lambda: member(*args, **kwargs)
            )
        return member


class BoundaryClient:
    """Cache the few ambiguous continuation decisions before table OCR."""

    def __init__(self, client, store):
        self.client, self.store = client, store
        try:
            self.decisions = json.loads((store.root / "boundaries.json").read_text())
        except (OSError, ValueError):
            self.decisions = {}
        if not isinstance(self.decisions, dict):
            self.decisions = {}

    def request(self, prompt, images, schema):
        digest = hashlib.sha256(prompt.encode())
        for image in images:
            digest.update(repr((image.mode, image.size)).encode())
            digest.update(image.tobytes())
        key = digest.hexdigest()
        if isinstance(self.decisions.get(key), bool):
            return {"merge": self.decisions[key]}
        result = self.client.request(prompt, images, schema)
        self.decisions[key] = result["merge"]
        self.store.write_text("boundaries.json", json.dumps(self.decisions))
        return result


def _used_resources(result):
    """Publish figures actually referenced by the output, not evidence PNGs."""
    visible = result.markdown + "\n" + result.html
    result.resources = {
        name: retain_png(image)
        for name, image in result.resources.items()
        if name in visible
    }
    return result


def _render_entries(entries):
    """Reassemble inline owners by their original baseline, then document order."""
    blocks = []
    inline = {}
    for page, y, x, body, group in entries:
        if not body:
            continue
        if group is None:
            blocks.append((page, y, x, body))
        else:
            inline.setdefault(group, []).append((x, body))
    for (page, box), pieces in inline.items():
        blocks.append(
            (page, box[1], box[0], " ".join(body for _, body in sorted(pieces)))
        )
    return "\n\n".join(item[3] for item in sorted(blocks))


def _discover_scans(source, regions, parts, client, store):
    """Replace covered owners once; retain native table geometry when available."""
    from marker.cloud.scan_regions import discover_scan

    def area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def covered(inner, outer):
        return area(inner) > 0 and intersection(inner, outer) >= area(inner) * 0.999

    candidates = []
    for index, region in enumerate(regions):
        width, height = source.page_size(region.page)
        if (region.kind == "image" and region.bbox[3] - region.bbox[1] > 72) or (
            region.kind == "unknown"
            and (area(region.bbox) >= width * height * 0.65 or len(region.atoms) > 50)
        ):
            candidates.append((index, region))
    suppressed = set()
    adopted = []
    problems = []
    result_parts = list(parts)
    for index, parent in sorted(candidates, key=lambda item: -area(item[1].bbox)):
        if index in suppressed:
            continue
        native = [
            part
            for part in result_parts
            if part.page == parent.page and intersection(part.bbox, parent.bbox) > 0
        ]
        crossing = any(
            region.page == parent.page
            and other != index
            and intersection(region.bbox, parent.bbox) > 0
            and not covered(region.bbox, parent.bbox)
            for other, region in enumerate(regions)
        )
        crossing = crossing or any(
            not covered(part.bbox, parent.bbox) for part in native
        )
        try:
            if crossing:
                raise RecognitionFailure(
                    "Scan crop crosses an existing source owner; explicit layout remains required"
                )
            found, new_parts = discover_scan(source, parent, client, store=store)
            additions = []
            for part in new_parts:
                overlaps = [
                    old for old in native if intersection(old.bbox, part.bbox) > 0
                ]
                if overlaps:
                    if len(overlaps) != 1 or intersection(
                        overlaps[0].bbox, part.bbox
                    ) < 0.8 * max(area(part.bbox), area(overlaps[0].bbox)):
                        raise RecognitionFailure(
                            "Scanned and native table bounds disagree"
                        )
                    # A searchable scan can expose the same table through its
                    # page bitmap and native layer. Keep the original cell grid.
                    continue
                additions.append(part)
            adopted.extend(
                region
                for region in found
                if region.kind == "table"
                or not any(
                    intersection(region.bbox, part.bbox) > 0.8 * area(region.bbox)
                    for part in native
                )
            )
            result_parts.extend(additions)
            suppressed.update(
                other
                for other, region in enumerate(regions)
                if region.page == parent.page and covered(region.bbox, parent.bbox)
            )
        except RecognitionFailure as error:
            problems.append(f"Page {parent.page}: {error.reason}")
            suppressed.add(index)
            adopted.append(replace(parent, kind="scan_unresolved"))
    return (
        [region for index, region in enumerate(regions) if index not in suppressed]
        + adopted,
        result_parts,
        problems,
    )


def _article_validator(
    raw, crop_boxes, *, expected_text=(), require_math=False, script_hints=()
):
    raw = re.sub(r"^```(?:html)?\s*\n|\n```\s*$", "", raw.strip(), flags=re.I)
    soup = BeautifulSoup(raw, "html.parser")
    roots = [
        node
        for node in soup.contents
        if not isinstance(node, NavigableString) or node.strip()
    ]
    if len(roots) != 1 or getattr(roots[0], "name", None) != "article":
        raise ResponseValidationError("Return exactly one complete article element")
    article = roots[0]
    if not raw.rstrip().endswith("</article>"):
        raise ResponseValidationError("The complete article closing tag is required")
    if require_math and not article.find("math"):
        raise ResponseValidationError(
            "Known source formulas must be represented as complete LaTeX math"
        )
    allowed = {
        "article",
        "p",
        "div",
        "span",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "strong",
        "em",
        "b",
        "i",
        "ul",
        "ol",
        "li",
        "br",
        "blockquote",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "caption",
        "colgroup",
        "col",
        "math",
        "semantics",
        "annotation",
        "img",
        "sub",
        "sup",
    }
    for node in article.find_all(True):
        if node.name not in allowed or any(
            str(key).lower().startswith("on") for key in node.attrs
        ):
            raise ResponseValidationError(
                "Executable or unsupported HTML is not allowed"
            )
        if any(
            key
            not in {
                "colspan",
                "rowspan",
                "display",
                "encoding",
                "data-latex",
                "data-crop",
                "data-source-image",
                "alt",
            }
            for key in node.attrs
        ):
            raise ResponseValidationError("Unsupported HTML attributes are not allowed")
    crop_boxes.clear()
    image_names = {}
    for index, node in enumerate(article.find_all("img")):
        value = node.get("data-crop", "")
        try:
            box = tuple(float(v) for v in re.split(r"[,\s]+", value.strip()))
        except ValueError:
            box = ()
        if (
            len(box) != 4
            or not all(0 <= v <= 1000 for v in box)
            or not (box[0] < box[2] and box[1] < box[3])
        ):
            raise ResponseValidationError(
                "Figure crops require four normalized source coordinates"
            )
        if (box[2] - box[0]) * (box[3] - box[1]) > 750000:
            raise ResponseValidationError(
                "A page or table cannot be substituted with a figure crop"
            )
        identity = f"crop{index}"
        filename = f"figure-{identity}.png"
        crop_boxes[identity] = box
        image_names[identity] = filename
        node.attrs = {"data-source-image": identity, "alt": "Figure"}
    for table in list(article.find_all("table")):
        if table.find_parent("table"):
            raise ResponseValidationError("Nested tables are not allowed")
        normalized = normalize_table_html(str(table), source_images=image_names)
        table.replace_with(BeautifulSoup(normalized, "html.parser"))
    if expected_text:
        from collections import Counter
        import unicodedata

        def words(value):
            return [
                word
                for word in re.findall(
                    r"[^\W\d_]+", unicodedata.normalize("NFC", value).casefold()
                )
                if len(word) > 1
            ]

        required_words = Counter(
            word for value in expected_text for word in words(value)
        )
        if required_words - Counter(words(article.get_text(" "))):
            raise ResponseValidationError(
                "Article omitted or changed trusted source text"
            )
    if script_hints:
        formulas = []
        for node in article.find_all("math"):
            annotation = node.find(
                "annotation", attrs={"encoding": "application/x-tex"}
            )
            formulas.append(
                annotation.get_text()
                if annotation
                else node.get("data-latex") or node.get_text()
            )
        try:
            validate_script_hints(r"\quad ".join(formulas), script_hints)
        except RecognitionFailure:
            raise ResponseValidationError(
                "Article indices must agree with source geometry"
            ) from None
    for node in article.find_all("img"):
        identity = node.get("data-source-image")
        if identity in image_names:
            node.attrs = {"src": image_names[identity], "alt": "Figure"}
    for node in list(article.find_all("math")):
        if node.find_parent("table"):
            continue
        annotation = node.find("annotation", attrs={"encoding": "application/x-tex"})
        latex = (
            annotation.get_text()
            if annotation
            else node.get("data-latex") or node.get_text()
        )
        try:
            latex_to_mathml(latex)
        except RecognitionFailure:
            raise ResponseValidationError(
                "Every expression requires usable complete LaTeX"
            ) from None
        node.replace_with(
            BeautifulSoup(
                latex_to_mathml(latex, display=node.get("display", "inline")),
                "html.parser",
            )
        )
    if not article.get_text(strip=True) and not article.find("img"):
        raise ResponseValidationError("The article has no recognized content")
    return "".join(str(child) for child in article.contents)


class StagedOllamaPdfConverter(OllamaPdfConverter):
    """Produce a resumable draft locally, then recognize each logical object."""

    cloud_context_tokens: Annotated[int, "Conservative whole-table context budget."] = (
        245760
    )
    cloud_max_output_tokens: Annotated[int, "Maximum whole-table response tokens."] = (
        16384
    )
    cloud_max_images: Annotated[int, "Maximum page crops of one logical table."] = 8
    cloud_work_dir: Annotated[
        str, "Persistent private intermediate-document directory."
    ] = ""

    def _recognize_object(self, source, region):
        """A single complete expression/image; no geometry response or JSON escapes."""
        if region.kind == "figure":
            name = resource_name("figure", region.page, region.bbox)
            return ContentResult(
                "figure",
                f"![Figure]({name})",
                resources={name: source.resource(region.page, region.bbox)},
            )
        if self.disable_ocr and region.kind == "text" and not region.quality.trusted:
            return ContentResult(
                "text", complete=False, reasons=("Required text OCR is disabled",)
            )
        hints = source_script_hints(region.atoms)
        required = region.kind == "math" or (
            not any(a.get("kind") == "image" for a in region.atoms)
            and math_required(region.atoms)
        )
        crops = {}
        structured = False

        def validate(raw):
            nonlocal structured
            structured = False
            crops.clear()
            raw = re.sub(
                r"^```(?:html|latex|tex)?\s*\n|\n```\s*$", "", raw.strip(), flags=re.I
            )
            if raw.startswith("$$") and raw.endswith("$$"):
                raw = "<math>" + escape(raw[2:-2].strip()) + "</math>"
            elif raw.startswith("$") and raw.endswith("$"):
                raw = "<math>" + escape(raw[1:-1].strip()) + "</math>"
            soup = BeautifulSoup(raw, "html.parser")
            roots = [
                n
                for n in soup.contents
                if not isinstance(n, NavigableString) or n.strip()
            ]
            if len(roots) != 1 or getattr(roots[0], "name", None) not in {
                "math",
                "figure",
                "p",
                "table",
                "article",
            }:
                raise ResponseValidationError(
                    "Return one complete math, figure, p, table or article element"
                )
            node = roots[0]
            if node.name in {"table", "article"} and not required:
                if self.disable_ocr:
                    ordinary = BeautifulSoup(str(node), "html.parser")
                    for expression in ordinary.find_all(["math", "img"]):
                        expression.decompose()
                    if ordinary.get_text(strip=True):
                        raise ResponseValidationError(
                            "Required ordinary text OCR is disabled"
                        )
                html = _article_validator(
                    raw if node.name == "article" else "<article>" + raw + "</article>",
                    crops,
                )
                structured = True
                return ContentResult("text", html, html)
            if node.name == "math":
                if not raw.rstrip().endswith("</math>"):
                    raise ResponseValidationError(
                        "The complete math closing tag is required"
                    )
                annotation = node.find(
                    "annotation", attrs={"encoding": "application/x-tex"}
                )
                if node.find(True) and annotation is None:
                    raise ResponseValidationError(
                        "MathML presentation requires its original LaTeX annotation"
                    )
                latex = (
                    annotation.get_text()
                    if annotation
                    else node.get("data-latex") or node.get_text()
                )
                try:
                    html = latex_to_mathml(latex)
                    validate_script_hints(latex, hints)
                except RecognitionFailure:
                    raise ResponseValidationError(
                        "Complete LaTeX and its indices must agree with the source"
                    ) from None
                delimiter = "$" if region.inline_group else "$$"
                return ContentResult(
                    "math", delimiter + latex + delimiter, html, latex=latex
                )
            if (
                node.name == "figure"
                and not required
                and not node.get_text(strip=True)
                and not node.find(True)
            ):
                return ContentResult("figure")
            if (
                node.name == "p"
                and raw.rstrip().endswith("</p>")
                and not required
                and not self.disable_ocr
                and node.get_text(strip=True)
                and not node.find(True)
            ):
                return ContentResult(
                    "text", node.get_text(), "<p>" + escape(node.get_text()) + "</p>"
                )
            raise ResponseValidationError(
                "A formula must be complete LaTeX and cannot become a figure or ordinary text"
            )

        prompt = (
            "Recognize ONE complete source object. Return raw HTML, never JSON. "
            "For math return exactly <math>COMPLETE LATEX</math>. A single letter/operator/subscript is math. "
            "For an actual diagram/photo/plot return <figure/>. For ordinary prose only return <p>text</p>. "
            "For a ruled or borderless table return one complete <table> with all cells, never a figure. "
            "For multiple objects or a page return <article> with paragraphs, math and HTML tables. "
            'Only genuine diagrams/photos inside an article may use <img data-crop="x0,y0,x1,y1"> '
            "with coordinates normalized0..1000 relative to this crop. "
            "Never turn a formula into a figure, and never summarize or invent unreadable content. "
            + MATH_CLASSIFICATION_PROMPT
            + MATH_BASELINE_PROMPT
            + "Keep Russian/Cyrillic versus Latin, decimal commas and phi/varphi exactly. "
            "The crop contains the assigned object; read its entire expression. "
            "Source script evidence: " + json.dumps(hints, ensure_ascii=False)
        )
        if required:
            prompt += " This source is known math; only a math element is accepted."
        try:
            image = source.resource(region.page, region.bbox)
            if max(image.size) < 600:
                from PIL import Image

                image = image.resize(
                    (image.width * 2, image.height * 2), Image.Resampling.NEAREST
                )
            result = self.client.request_text(prompt, [image], validate)
            if structured:
                return self._finish_article(source, region, result.html, crops)
            if result.kind == "figure":
                name = resource_name("figure", region.page, region.bbox)
                result.markdown = f"![Figure]({name})"
                result.html = f'<img src="{name}" alt="Figure">'
                result.resources = {name: source.resource(region.page, region.bbox)}
            return result
        except RecognitionFailure as error:
            return ContentResult(region.kind, complete=False, reasons=(error.reason,))

    @staticmethod
    def _finish_article(source, region, html, crops):
        soup = BeautifulSoup(html, "html.parser")
        resources = {}
        for identity, normalized in crops.items():
            x0, y0, x1, y1 = region.bbox
            box = (
                x0 + (x1 - x0) * normalized[0] / 1000,
                y0 + (y1 - y0) * normalized[1] / 1000,
                x0 + (x1 - x0) * normalized[2] / 1000,
                y0 + (y1 - y0) * normalized[3] / 1000,
            )
            name = resource_name("figure", region.page, box)
            resources[name] = source.resource(region.page, box)
            for image in soup.find_all("img", src=f"figure-{identity}.png"):
                image["src"] = name
        from marker.renderers.markdown import Markdownify

        markdown = Markdownify(
            False,
            "",
            ("$", "$"),
            ("$$", "$$"),
            True,
            heading_style="ATX",
            bullets="-",
            escape_misc=False,
            escape_underscores=True,
            escape_asterisks=True,
            escape_dollars=True,
            sub_symbol="<sub>",
            sup_symbol="<sup>",
        ).convert(str(soup))
        return ContentResult("text", markdown, str(soup), resources)

    def _recognize_article(self, source, region):
        if self.disable_ocr:
            return ContentResult(
                region.kind,
                complete=False,
                reasons=("Required page text OCR is disabled",),
            )
        crops = {}
        from marker.cloud.content import expression_groups, source_lines

        ordinary = []
        hints = []
        for line in source_lines(region.atoms):
            groups = expression_groups(line)
            math_ids = {atom_id(atom) for group in groups for atom in group}
            ordinary.extend(
                atom.get("text", "")
                for atom in line
                if atom.get("kind", "text") == "text" and atom_id(atom) not in math_ids
            )
            for group in groups:
                hints.extend(source_script_hints(group))
        expected = ordinary if region.quality.trusted else ()
        prompt = (
            "Transcribe this ONE source crop as exactly one <article>...</article>, with no JSON or explanations. "
            "Use ordinary paragraphs/headings for prose, HTML <table> with correct rowspan/colspan for every table. "
            "Read all rows and all source text; do not summarize. A mathematical glyph is math even when stored as pixels. "
            "Write every complete formula as <math>EXACT LATEX</math> (no MathML presentation tree). "
            + MATH_CLASSIFICATION_PROMPT
            + MATH_BASELINE_PROMPT
            + 'For actual drawings/photos only, use <img data-crop="x0,y0,x1,y1"> with coordinates 0..1000 relative '
            "to this crop. Never use an image for a formula, table, or whole page. "
            "Preserve Russian, decimal commas, all indices and phi/varphi. Use the source pixels as authority."
        )
        try:
            html = self.client.request_text(
                prompt,
                [source.crop(region.page, region.bbox, dpi=200)],
                lambda raw: _article_validator(
                    raw,
                    crops,
                    expected_text=expected,
                    require_math=math_required(region.atoms),
                    script_hints=hints,
                ),
            )
            return self._finish_article(source, region, html, crops)
        except RecognitionFailure as error:
            return ContentResult(region.kind, complete=False, reasons=(error.reason,))

    def _recognize_job(self, source, job):
        if "table" in job:
            if self.disable_ocr:
                for part in job["table"].parts:
                    if part.grid is None:
                        return ContentResult(
                            "table",
                            complete=False,
                            reasons=("Required table text OCR is disabled",),
                        )
                    for cell in part.grid["cells"]:
                        words = [
                            a for a in cell["atoms"] if a.get("kind", "text") == "text"
                        ]
                        if (
                            words
                            and not assess_text_quality(words, cell["bbox"]).trusted
                            and not math_required(words)
                        ):
                            return ContentResult(
                                "table",
                                complete=False,
                                reasons=("Required table text OCR is disabled",),
                            )
            return recognize_whole_table(source, job["table"], self.client)
        region = job["region"]
        if job["kind"] == "scan_unresolved":
            return ContentResult(
                region.kind,
                complete=False,
                reasons=("Required scan layout remains unresolved",),
            )
        if job["kind"] == "page" or (
            region.kind == "text"
            and (not region.quality.trusted or math_required(region.atoms))
        ):
            return self._recognize_article(source, region)
        return self._recognize_object(source, region)

    def __call__(self, filepath):
        started = monotonic()
        start_requests = self.client.request_count
        root = (
            Path(self.cloud_work_dir)
            if self.cloud_work_dir
            else Path(tempfile.mkdtemp(prefix="marker-draft-"))
        )
        signature = {
            name: getattr(self, name)
            for name in (
                "cloud_model",
                "cloud_context_tokens",
                "cloud_max_output_tokens",
                "cloud_max_images",
                "cloud_crop_dpi",
                "force_ocr",
                "disable_ocr",
                "page_range",
            )
        }
        for name in (
            "cloud_model",
            "cloud_context_tokens",
            "cloud_max_output_tokens",
            "cloud_max_images",
        ):
            signature[name] = getattr(self.client, name)
        signature["endpoint_hash"] = hashlib.sha256(
            self.client.cloud_base_url.encode()
        ).hexdigest()
        store = StageStore(root, filepath, signature)
        jobs = []
        local = []
        reasons = []
        cached = 0
        results = {}
        table_count = 0
        with PdfSource(filepath, crop_dpi=self.cloud_crop_dpi) as source:
            pages = list(
                self.page_range if self.page_range is not None else range(len(source))
            )
            parts = []
            regions = []
            logger.info(
                "Stage 1: local PDF extraction; pages=%s; draft=%s",
                len(pages),
                store.root / "draft.md",
            )
            for page in pages:
                logger.info(
                    "Local source page %s/%s; cloud requests=%s",
                    page + 1,
                    len(pages),
                    self.client.request_count - start_requests,
                )
                if page < 0 or page >= len(source):
                    reasons.append(f"Page {page}: requested page is outside document")
                    continue
                try:
                    page_parts = detect_tables(source, [page])
                    parts.extend(page_parts)
                    from marker.cloud.regions import analyze_page

                    regions.extend(
                        document_regions(
                            analyze_page(source, page, table_parts=page_parts)
                        )
                    )
                except RecognitionFailure:
                    box = (0.0, 0.0, *source.page_size(page))
                    regions.append(
                        Region(page, box, "image", (), assess_text_quality([], box))
                    )
            # A visible intermediate document exists before any cloud request.
            initial = []
            if not (store.root / "source.md").is_file():
                for region in regions:
                    if (
                        region.kind == "text"
                        and region.quality.trusted
                        and not math_required(region.atoms)
                    ):
                        initial.append(
                            (
                                region.page,
                                region.bbox[1],
                                region.bbox[0],
                                recognize_region(source, region, self.client).markdown,
                                region.inline_group,
                            )
                        )
                    elif region.required:
                        identity = (
                            "source-"
                            + hashlib.sha256(
                                repr((region.page, region.bbox)).encode()
                            ).hexdigest()[:24]
                        )
                        image = source.resource(region.page, region.bbox)
                        name = store.save_source(identity, image)
                        initial.append(
                            (
                                region.page,
                                region.bbox[1],
                                region.bbox[0],
                                f"![Pending {region.kind}]({name})",
                                region.inline_group,
                            )
                        )
                store.write_text("source.md", _render_entries(initial))
            if not (store.root / "draft.md").exists():
                store.write_text("draft.md", (store.root / "source.md").read_text())
            local_elapsed = monotonic() - started
            logger.info(
                "Stage 1 complete in %.1f s; physical table parts=%s; cloud requests=%s",
                local_elapsed,
                len(parts),
                self.client.request_count - start_requests,
            )
            scan_started = monotonic()
            before_scan = self.client.request_count
            regions, parts, scan_problems = _discover_scans(
                source, regions, parts, self.client, store
            )
            reasons.extend(scan_problems)
            scan_elapsed = monotonic() - scan_started
            scan_requests = self.client.request_count - before_scan
            boundary_started = monotonic()
            before_boundary = self.client.request_count
            tables = group_tables(parts, BoundaryClient(self.client, store))
            boundary_requests = self.client.request_count - before_boundary
            for table in tables:
                table_count += 1
                part = table.parts[0]
                identity = (
                    "table-"
                    + hashlib.sha256(
                        repr([(p.page, p.bbox) for p in table.parts]).encode()
                    ).hexdigest()[:24]
                )
                jobs.append(
                    {
                        "id": identity,
                        "kind": "table",
                        "page": part.page,
                        "bbox": part.bbox,
                        "table": table,
                        "inline_group": None,
                    }
                )
            boundary_elapsed = monotonic() - boundary_started
            preparation_started = monotonic()
            for region in regions:
                if region.kind == "table":
                    continue
                if not region.required or (
                    region.kind == "text"
                    and region.quality.trusted
                    and not math_required(region.atoms)
                    and not self.force_ocr
                ):
                    result = recognize_region(source, region, self.client)
                    local.append(
                        (
                            region.page,
                            region.bbox[1],
                            region.bbox[0],
                            result.markdown,
                            region.inline_group,
                        )
                    )
                    continue
                area = (region.bbox[2] - region.bbox[0]) * (
                    region.bbox[3] - region.bbox[1]
                )
                page_area = (
                    source.page_size(region.page)[0] * source.page_size(region.page)[1]
                )
                kind = (
                    "page"
                    if region.kind in ("image", "unknown") and area >= page_area * 0.65
                    else region.kind
                )
                image = source.resource(region.page, region.bbox)
                digest = hashlib.sha256(
                    repr(
                        (kind, image.mode, image.size, bool(region.inline_group))
                    ).encode()
                )
                digest.update(image.tobytes())
                digest.update(
                    repr(
                        [
                            (
                                a.get("kind"),
                                a.get("text", ""),
                                a.get("fontname", ""),
                                a.get("size"),
                                tuple(
                                    round(v - region.bbox[i % 2], 3)
                                    for i, v in enumerate(a["bbox"])
                                ),
                            )
                            for a in region.atoms
                        ]
                    ).encode()
                )
                identity = "region-" + digest.hexdigest()[:32]
                jobs.append(
                    {
                        "id": identity,
                        "kind": kind,
                        "page": region.page,
                        "bbox": region.bbox,
                        "region": region,
                        "inline_group": region.inline_group,
                    }
                )
            jobs.sort(key=lambda job: (job["page"], job["bbox"][1], job["bbox"][0]))
            for job in jobs:
                existing = store.load(job["id"])
                if existing is not None and "table" in job:
                    try:
                        apply_source_table_layout(existing.html, job["table"].grid)
                    except TableLayoutError:
                        existing = None
                if existing is not None:
                    job["sources"] = [
                        path.relative_to(store.root).as_posix()
                        for path in sorted(store._job(job["id"]).glob("source-*.png"))
                    ]
                    results[job["id"]] = existing
                    cached += 1
                elif "table" in job:
                    job["sources"] = [
                        store.save_source(job["id"], source.resource(p.page, p.bbox), i)
                        for i, p in enumerate(job["table"].parts)
                    ]
                else:
                    job["sources"] = [
                        store.save_source(
                            job["id"], source.resource(job["page"], job["bbox"])
                        )
                    ]

            def draft():
                rendered = list(local)
                for job in jobs:
                    result = results.get(job["id"])
                    body = (
                        result.markdown
                        if result and result.complete
                        else "\n\n".join(
                            f"![Pending {job['kind']}]({name})"
                            for name in job["sources"]
                        )
                    )
                    rendered.append(
                        (
                            job["page"],
                            job["bbox"][1],
                            job["bbox"][0],
                            body,
                            job["inline_group"],
                        )
                    )
                return _render_entries(rendered)

            def manifest():
                record = {
                    "pipeline": PIPELINE_VERSION,
                    "input_hash": store.input_hash,
                    "jobs": [
                        {
                            **{
                                key: job[key]
                                for key in ("id", "kind", "page", "bbox", "sources")
                            },
                            "state": "complete"
                            if results.get(job["id"]) and results[job["id"]].complete
                            else "pending",
                        }
                        for job in jobs
                    ],
                }
                store.write_text(
                    "manifest.json", json.dumps(record, ensure_ascii=False, indent=2)
                )
                store.write_text("draft.md", draft())

            manifest()
            preparation_elapsed = monotonic() - preparation_started
            unique = {job["id"]: job for job in jobs if job["id"] not in results}
            pending = list(unique.values())
            recognition_started = monotonic()
            logger.info(
                "Stage 2: %s logical tables, %s distinct pending objects, %s cached; no geometry OCR for formula images",
                table_count,
                len(pending),
                cached,
            )

            def recognize(job, source_call):
                logger.info(
                    "Recognizing %s on page %s; requests=%s",
                    job["kind"],
                    job["page"] + 1,
                    self.client.request_count - start_requests,
                )
                try:
                    if (
                        self.client.cloud_max_requests
                        and self.client.request_count >= self.client.cloud_max_requests
                    ):
                        raise RecognitionFailure(
                            "Document cloud request budget exhausted"
                        )
                    result = self._recognize_job(OwnerSource(source, source_call), job)
                    if result.complete:
                        return job, _used_resources(result)
                    return job, ContentResult(
                        job["kind"], complete=False, reasons=result.reasons
                    )
                except RecognitionFailure as error:
                    return job, ContentResult(
                        job["kind"], complete=False, reasons=(error.reason,)
                    )

            workers = (
                1
                if self.client.cloud_max_requests
                else self.client.cloud_max_concurrency
            )
            for index, (job, result) in enumerate(
                ordered_cloud_map(recognize, pending, workers), 1
            ):
                results[job["id"]] = result
                store.save(job["id"], result)
                manifest()
                logger.info(
                    "Completed object %s/%s: %s; requests=%s; elapsed=%.1f s",
                    index,
                    len(pending),
                    "complete" if result.complete else "pending",
                    self.client.request_count - start_requests,
                    monotonic() - started,
                )
            recognition_elapsed = monotonic() - recognition_started
        layout_started = monotonic()
        final = list(local)
        images = {}
        tables_pending = 0
        formulas_pending = 0
        objects_pending = 0
        for job in jobs:
            result = results[job["id"]]
            body = result.markdown
            if result.complete and "table" in job:
                try:
                    body = apply_source_table_layout(result.html, job["table"].grid)
                except TableLayoutError as error:
                    result = ContentResult(
                        "table", complete=False, reasons=(error.reason,)
                    )
            if not result.complete:
                reasons.extend(
                    f"Page {job['page']}: {reason}"
                    for reason in result.reasons or ("Required recognition incomplete",)
                )
                tables_pending += job["kind"] == "table"
                formulas_pending += job["kind"] in ("math", "image")
                objects_pending += 1
                continue
            final.append(
                (
                    job["page"],
                    job["bbox"][1],
                    job["bbox"][0],
                    body,
                    job["inline_group"],
                )
            )
            images.update(result.resources)
        markdown = _render_entries(final)
        status = {
            "pipeline": PIPELINE_VERSION,
            "status": "incomplete" if reasons else "complete",
            "publishable": not reasons,
            "reasons": list(dict.fromkeys(reasons)),
            "requests": self.client.request_count - start_requests,
            "cached_jobs": cached,
            "tables_expected": table_count,
            "tables_pending": tables_pending,
            "formulas_pending": formulas_pending,
            "objects_pending": objects_pending,
            "tables_rendered": len(
                BeautifulSoup(markdown, "html.parser").find_all("table")
            ),
            "work_dir": str(store.root),
            "elapsed_seconds": round(monotonic() - started, 3),
            "generated_assets": sorted(images),
            "stages_seconds": {
                "local": round(local_elapsed, 3),
                "scan_layout": round(scan_elapsed, 3),
                "boundaries": round(boundary_elapsed, 3),
                "preparation": round(preparation_elapsed, 3),
                "recognition": round(recognition_elapsed, 3),
                "layout": round(monotonic() - layout_started, 3),
            },
            "boundary_requests": boundary_requests,
            "scan_layout_requests": scan_requests,
        }
        store.write_text(
            "report.json", json.dumps(status, ensure_ascii=False, indent=2)
        )
        logger.info(
            "Conversion %s in %.1f s; tables=%s, pending tables=%s, requests=%s",
            status["status"],
            status["elapsed_seconds"],
            table_count,
            tables_pending,
            status["requests"],
        )
        return MarkdownOutput(
            markdown=markdown, images=images, metadata={"cloud_conversion": status}
        )
