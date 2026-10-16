"""Enrich Marker's native document, one complete logical object at a time.

Native layout/text and physical PDF cells remain the source of geometry. The
LLM transcribes complete tables/expressions; it never supplies replacement page
coordinates or draws the figures that will be published.
"""

from copy import copy
import hashlib
import json
import re
import tempfile
from time import monotonic
from typing import Annotated

from pydantic import BaseModel

from marker.cloud.client import ResponseValidationError, validate_response, repair_prompt
from marker.cloud.content import math_required, recognize_region
from marker.cloud.context import RecognitionFailure, fit_request
from marker.cloud.parallel import ordered_cloud_map
from marker.cloud.regions import Region, assess_text_quality
from marker.cloud.source import PdfSource, intersection
from marker.cloud.stage_store import StageStore
from marker.cloud.native_discovery import discover_native_owners
from marker.cloud.table_layout import apply_source_table_layout, TableLayoutError
from marker.cloud.tables import TablePart, _caption, group_tables
from marker.cloud.whole_tables import recognize_whole_table
from marker.converters.ollama_document import WholeObjectRecognizer, OwnerSource, BoundaryClient, _native_punctuation
from marker.logger import get_logger
from marker.processors.llm import BaseLLMProcessor
from marker.schema import BlockTypes


logger = get_logger()
_GROUPS = {BlockTypes.TableGroup, BlockTypes.FigureGroup, BlockTypes.PictureGroup,
           BlockTypes.ListGroup}
_TABLES = {BlockTypes.Table, BlockTypes.TableOfContents}
_PICTURES = {BlockTypes.Picture, BlockTypes.Figure, BlockTypes.Diagram}
_SKIP = {BlockTypes.Line, BlockTypes.Span, BlockTypes.Char, BlockTypes.Reference,
         BlockTypes.TableCell, BlockTypes.PageHeader, BlockTypes.PageFooter}


def content_blocks(document, parent):
    """Visit rendered content owners, never their duplicate lines or spans."""
    for identity in parent.structure or ():
        block = document.get_block(identity)
        if block is None or block.removed or block.ignore_for_output:
            continue
        if block.block_type in _GROUPS:
            yield from content_blocks(document, block)
        elif block.block_type not in _SKIP:
            yield block


def source_box(block, page, source):
    """Map native Marker page points to the visible PDF frame, without rounding."""
    width, height = source.page_size(page.page_id)
    grid = getattr(block, "table_grid", None)
    if grid:
        box = tuple(grid["bbox"])
        if 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height:
            return box
    x0, y0, x1, y1 = page.polygon.bbox
    left, top, right, bottom = block.polygon.bbox
    box = (max(0, (left - x0) * width / (x1 - x0)),
           max(0, (top - y0) * height / (y1 - y0)),
           min(width, (right - x0) * width / (x1 - x0)),
           min(height, (bottom - y0) * height / (y1 - y0)))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise RecognitionFailure("Native block has invalid source bounds")
    return box


def _atoms(source, page, box):
    atoms = source.page_words(page, box)
    atoms.extend({"kind": "image", "bbox": image.bbox, "source_bbox": image.bbox,
                  "source_page": page, "image_index": image.index}
                 for image in source.page_images(page, box))
    return tuple(atoms)


def _identity(kind, boxes):
    return kind + "-" + hashlib.sha256(repr(boxes).encode()).hexdigest()[:32]


def known_math(document, block):
    return block.block_type in {BlockTypes.Equation, BlockTypes.TextInlineMath} or any(
        "math" in (getattr(child, "formats", None) or ())
        for child in block.contained_blocks(document, (BlockTypes.Line, BlockTypes.Span))
    )


class _ServiceText(BaseModel):
    text: str


class ServiceClient:
    """Explicit non-Ollama service overrides keep the whole-object contract."""

    def __init__(self, service, config):
        from marker.cloud.defaults import (DEFAULT_CONTEXT_TOKENS, DEFAULT_MAX_IMAGES,
            DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT)
        self.service = copy(service)
        self.cloud_model = next((getattr(service, name) for name in
            ("openai_model", "gemini_model_name", "ollama_model", "claude_model_name",
             "openrouter_model", "deployment_name") if getattr(service, name, None)),
            type(service).__name__)
        self.cloud_base_url = next((getattr(service, name) for name in
            ("openai_base_url", "ollama_base_url", "openrouter_base_url", "azure_endpoint")
            if getattr(service, name, None)),
            type(service).__module__)
        self.cloud_context_tokens = config.get("cloud_context_tokens", DEFAULT_CONTEXT_TOKENS)
        self.cloud_max_images = config.get("cloud_max_images", DEFAULT_MAX_IMAGES)
        self.cloud_max_output_tokens = config.get("cloud_max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        self.cloud_max_attempts = config.get("cloud_max_attempts", DEFAULT_MAX_ATTEMPTS)
        # A legacy service may have mutable metadata; keep its requests serial.
        self.cloud_max_concurrency = 1
        self.cloud_max_requests = config.get("cloud_max_requests", 0)
        self.timeout = config.get("cloud_timeout", config.get("timeout", DEFAULT_TIMEOUT))
        self.service.max_output_tokens = self.cloud_max_output_tokens
        if hasattr(self.service, "max_claude_tokens"):
            self.service.max_claude_tokens = self.cloud_max_output_tokens
        self.request_count = self.total_tokens = 0

    def start_document(self):
        self.request_count = self.total_tokens = 0

    def _request_once(self, prompt, images, schema):
        if len(images) > self.cloud_max_images:
            raise RecognitionFailure("Required images exceed the cloud request image limit")
        if self.cloud_max_requests and self.request_count >= self.cloud_max_requests:
            raise RecognitionFailure("Document cloud request budget exhausted")
        fitted = fit_request(prompt, images, self.cloud_context_tokens, self.cloud_max_output_tokens,
                             max_image_edge=3072, min_image_edge=2048)
        self.request_count += 1
        response = self.service(prompt, fitted, None, schema, max_retries=0, timeout=self.timeout)
        return validate_response(json.dumps(response), schema)

    def request(self, prompt, images, schema):
        reason = "Required content unresolved"
        original = prompt
        for _ in range(self.cloud_max_attempts):
            try:
                return self._request_once(prompt, images, schema)
            except ResponseValidationError as error:
                reason = str(error)
                prompt = repair_prompt(original, reason)
        raise RecognitionFailure(reason)

    def request_text(self, prompt, images, validator, **kwargs):
        reason = "Required content unresolved"
        original = prompt
        for _ in range(self.cloud_max_attempts):
            try:
                answer = self._request_once(prompt + '\nReturn the complete HTML in JSON field "text".', images, _ServiceText)
                return validator(answer["text"])
            except ResponseValidationError as error:
                reason = str(error)
                prompt = repair_prompt(original, reason)
        raise RecognitionFailure(reason)


class NativeLLMProcessor(WholeObjectRecognizer, BaseLLMProcessor):
    """Default --use_llm processor, after native extraction and table discovery."""

    cloud_crop_dpi: Annotated[int, "Minimum source-resource rendering DPI."] = 600
    cloud_work_dir: Annotated[str, "Resumable private recognition cache directory."] = ""
    disable_ocr: Annotated[bool, "Keep native prose; always recognize required mathematics/tables."] = False
    force_ocr: Annotated[bool, "Recognize all native text blocks from source images."] = False

    def __init__(self, llm_service, config=None):
        super().__init__(llm_service, config)
        config = dict(config or {})
        self.client = getattr(llm_service, "client", None) or ServiceClient(llm_service, config)
        if self.force_ocr and self.disable_ocr:
            raise ValueError("force_ocr and disable_ocr cannot be combined")

    def _table_parts(self, document, source, blocks, overrides=None):
        """Reuse native physical grids; borderless/scanned tables stay untrusted."""
        parts, owners = [], {}
        for block in blocks:
            if block.block_type not in _TABLES:
                continue
            if overrides and block.id in overrides:
                part = overrides[block.id]
                parts.append(part)
                owners[(part.page, part.bbox)] = block
                continue
            page = document.get_page(block.page_id)
            box = source_box(block, page, source)
            grid = block.table_grid
            if grid is not None:
                # A grid is trusted only in exactly the same source coordinate
                # system. Never silently scale/mutate its immutable source boxes.
                if any(abs(a - b) > 0.1 for a, b in zip(grid["bbox"], box)):
                    grid = None
            part = TablePart(block.page_id, box, grid, source.page_size(block.page_id),
                             _caption(source, block.page_id, box), source)
            parts.append(part)
            owners[(part.page, part.bbox)] = block
        return parts, owners

    def _block_region(self, document, source, block):
        page = document.get_page(block.page_id)
        box = source_box(block, page, source)
        atoms = _atoms(source, page.page_id, box)
        quality = assess_text_quality([a for a in atoms if a.get("kind") == "text"], box)
        if block.block_type == BlockTypes.Code and quality.trusted and not self.force_ocr:
            return None
        required_math = known_math(document, block)
        has_images = any(a.get("kind") == "image" for a in atoms)
        kind = "math" if block.block_type == BlockTypes.Equation else "text"
        if block.block_type in _PICTURES:
            # Whole native diagrams are classified once, never per component
            # image/glyph. The original PDF supplies the retained PNG pixels.
            kind = "unknown"
            width, height = source.page_size(page.page_id)
            if (box[2] - box[0]) * (box[3] - box[1]) >= width * height * 0.65:
                kind = "page"
        elif block.block_type != BlockTypes.Equation and (required_math or math_required(atoms) or has_images):
            kind = "text"  # Mixed prose + formulas stays one whole native block.
        elif quality.trusted and not self.force_ocr and not required_math:
            return None
        if self.disable_ocr and kind == "text" and quality.trusted and not has_images and not required_math and not math_required(atoms):
            return None
        region = Region(page.page_id, box, kind, atoms, quality)
        if not self.force_ocr and _native_punctuation(region) is not None:
            return None
        return region

    def __call__(self, document):
        if not self.use_llm:
            return
        if hasattr(self.client, "bind_document"):
            self.client.bind_document(document.filepath)
        if hasattr(self.client, "start_document"):
            self.client.start_document()
        started = monotonic()
        initial_requests = self.client.request_count
        root = self.cloud_work_dir or tempfile.mkdtemp(prefix="marker-native-")
        signature = {"pipeline": "native-marker-v1", "model": self.client.cloud_model,
                     "endpoint": hashlib.sha256(self.client.cloud_base_url.encode()).hexdigest(),
                     "context": self.client.cloud_context_tokens,
                     "output": self.client.cloud_max_output_tokens,
                     "images": self.client.cloud_max_images, "dpi": self.cloud_crop_dpi,
                     "force_ocr": self.force_ocr, "disable_ocr": self.disable_ocr,
                     "pages": [page.page_id for page in document.pages]}
        if hasattr(self.client, "reasoning_effort"):
            signature["reasoning_effort"] = self.client.reasoning_effort
        store = StageStore(root, document.filepath, signature)
        completed, jobs, failures = [], [], []
        with PdfSource(document.filepath, crop_dpi=self.cloud_crop_dpi) as source:
            blocks = [block for page in document.pages for block in content_blocks(document, page)]
            try:
                blocks, part_overrides, region_overrides = discover_native_owners(
                    document, source, blocks, self.client, store)
            except RecognitionFailure as error:
                report = {"status": "incomplete", "pipeline": "native-marker-v1",
                          "source_sha256": store.input_hash,
                          "reasons": [error.reason], "publishable": False,
                          "requests": self.client.request_count - initial_requests,
                          "model": self.client.cloud_model, "work_dir": str(store.root)}
                if hasattr(self.client, "budget_snapshot"):
                    report["budget"] = self.client.budget_snapshot()
                store.write_text("report.json", json.dumps(report, ensure_ascii=False))
                document.cloud_conversion = report
                raise
            parts, owners = self._table_parts(document, source, blocks, part_overrides)
            tables = group_tables(parts, BoundaryClient(self.client, store))
            for table in tables:
                owner_blocks = [owners[(part.page, part.bbox)] for part in table.parts]
                jobs.append({"id": _identity("table", [(p.page, p.bbox) for p in table.parts]),
                             "kind": "table", "table": table, "owners": owner_blocks})
            for block in blocks:
                if block.block_type in _TABLES:
                    continue
                box = source_box(block, document.get_page(block.page_id), source)
                # Geometry cleanup can leave a non-text sibling within a table.
                # The one table job owns ALL of its formula/figure content.
                if any(part.page == block.page_id and intersection(box, part.bbox) >=
                       (box[2] - box[0]) * (box[3] - box[1]) * 0.95 for part in parts):
                    continue
                region = region_overrides.get(block.id) or self._block_region(document, source, block)
                if region is not None:
                    if (region.kind == "text" and region.quality.trusted
                            and not self.force_ocr and not math_required(region.atoms)
                            and not known_math(document, block)
                            and all(atom.get("kind", "text") == "text" for atom in region.atoms)):
                        completed.append(({"owners": [block]}, recognize_region(source, region, self.client)))
                        continue
                    jobs.append({"id": _identity("block", (region.page, region.bbox, region.kind)),
                                 "kind": region.kind, "region": region, "owners": [block],
                                 "require_math": known_math(document, block)})
            logger.info("Native Marker: %d logical tables, %d whole content jobs; cache=%s",
                        len(tables), len(jobs), store.root)
            manifest = []
            for job in jobs:
                result = store.load(job["id"])
                if result is not None and "table" in job:
                    try:
                        apply_source_table_layout(result.html, job["table"].grid)
                    except TableLayoutError:
                        result = None
                if result is not None:
                    completed.append((job, result))
                else:
                    crops = ([(part.page, part.bbox) for part in job["table"].parts]
                             if "table" in job else [(job["region"].page, job["region"].bbox)])
                    for index, (page, box) in enumerate(crops):
                        store.save_source(job["id"], source.resource(page, box), index)
                    manifest.append({"id": job["id"], "kind": job["kind"], "sources": crops})
            store.write_text("manifest.json", json.dumps({"pipeline": "native-marker-v1", "jobs": manifest}))
            pending = [job for job in jobs if all(job is not item[0] for item in completed)]

            def recognize(job, source_call):
                safe_source = OwnerSource(source, source_call)
                if "table" in job:
                    return recognize_whole_table(safe_source, job["table"], self.client)
                region = job["region"]
                # Native text owners give a whole paragraph context. Individual
                # unknown figure owners use the stricter object classifier.
                if region.kind in {"text", "page"}:
                    if self.disable_ocr and region.quality.trusted:
                        # Recognition of inline math is still mandatory. The
                        # ordinary source words are checked by the validator.
                        enabled = WholeObjectRecognizer()
                        enabled.client, enabled.disable_ocr = self.client, False
                        return enabled._recognize_article(safe_source, region, require_math=job.get("require_math", False))
                    return self._recognize_article(safe_source, region, require_math=job.get("require_math", False))
                return self._recognize_object(safe_source, region)

            workers = (1 if self.client.cloud_max_requests else self.client.cloud_max_concurrency)
            for job, result in zip(pending, ordered_cloud_map(recognize, pending, workers=workers)):
                if result.complete and not result.reasons:
                    store.save(job["id"], result)
                    completed.append((job, result))
                else:
                    failures.append({"job": job["id"], "page": job["owners"][0].page_id + 1,
                                     "reasons": list(result.reasons)})
            report = {"status": "incomplete" if failures else "complete",
                      "source_sha256": store.input_hash,
                      "reasons": [f"Page {item['page']}: {reason}" for item in failures for reason in item["reasons"]],
                      "pipeline": "native-marker-v1", "model": self.client.cloud_model,
                      "requests": self.client.request_count - initial_requests,
                      "logical_tables": len(tables), "jobs": len(jobs),
                      "cached": len(jobs) - len(pending), "failures": failures,
                      "elapsed_seconds": monotonic() - started, "work_dir": str(store.root),
                      "generated_assets": sorted({name for _, result in completed for name in result.resources}),
                      "publishable": not failures}
            if hasattr(self.client, "budget_snapshot"):
                report["budget"] = self.client.budget_snapshot()
            store.write_text("report.json", json.dumps(report, ensure_ascii=False))
            document.cloud_conversion = report
            if failures:
                raise RecognitionFailure("Native LLM recognition incomplete; source crops and successful jobs retained at " + str(store.root))
            # Apply validated results only after all required jobs have passed.
            for job, result in completed:
                head, *continuations = job["owners"]
                head.html = (apply_source_table_layout(result.html, job["table"].grid)
                             if "table" in job else result.html)
                head.structure = []
                head.embedded_images = dict(result.resources)
                head.llm_processed = True
                for continuation in continuations:
                    continuation.ignore_for_output = True
                    continuation.structure = []
                    continuation.embedded_images = {}
            # Remove any residual layout sibling already owned by a table.
            table_owner_ids = {b.id for job in jobs if "table" in job for b in job["owners"]}
            for block in blocks:
                if block.id in table_owner_ids:
                    continue
                box = source_box(block, document.get_page(block.page_id), source)
                if any(p.page == block.page_id and intersection(box, p.bbox) >=
                       (box[2]-box[0]) * (box[3]-box[1]) * 0.95 for p in parts):
                    block.ignore_for_output = True
                    block.embedded_images = {}
