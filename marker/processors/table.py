from collections import Counter
import os
from typing import Annotated, List

import pdfplumber
from bs4 import BeautifulSoup
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor, _detect_repeat_loop

from marker.processors import BaseProcessor
from marker.processors.table_recon import (
    reconstruct_table_html,
    table_lines_from_pdftext,
)
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.labels import block_type_to_surya_label
from marker.util import matrix_intersection_area
from marker.logger import get_logger
from marker.processors.table_geometry import attach_images, extract_grids, render_grid, _intersection
from marker.schema.blocks import Table
from marker.schema.polygon import PolygonBox

logger = get_logger()


class TableProcessor(BaseProcessor):
    """Fills in table HTML.

    - Scanned/garbled pages are full-page OCR'd by the OcrBuilder, which already
      sets ``block.html`` on their Table blocks - those are left untouched.
    - Ruled digital tables use PDF lines to define physical cells. Text and
      raster formulas are placed by source coordinates, including unruled
      content bands and headers interrupted by a page boundary.
    - Tables without a reliable vector grid use text-layer heuristics.
    - A digital table the heuristics can't resolve (sparse, vector-drawn, low
      score) falls back to OCRing the table crop with the recognition model.

    There is no dedicated table-structure model.
    """

    block_types = (BlockTypes.Table, BlockTypes.TableOfContents, BlockTypes.Form)
    mode: Annotated[str, "Conversion mode ('balanced' | 'fast')."] = "balanced"
    min_recon_score: Annotated[
        float,
        "Minimum pdftext reconstruction judge score to accept; below this a",
        "digital table falls back to recognition OCR. None = auto by mode:",
        "0.75 in balanced (spend VLM calls on low-confidence tables), 0.5 in",
        "fast (keep VLM fallback rare).",
    ] = None
    ocr_table_token_floor: Annotated[
        int,
        "Minimum token budget when OCRing a table crop to HTML.",
    ] = 2048
    contained_block_types: Annotated[
        List[BlockTypes],
        "Block types to remove if they're contained inside the tables.",
    ] = (BlockTypes.Text, BlockTypes.TextInlineMath)
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False
    disable_ocr: Annotated[bool, "Disable OCR entirely."] = False
    extract_images: Annotated[bool, "Preserve source images inside table cells."] = True
    table_geometry: Annotated[bool, "Build ruled PDF tables from physical cell coordinates."] = True
    table_source_coordinates: Annotated[
        bool, "Use the visible PDF coordinate frame for cloud table source geometry."
    ] = False

    def __init__(
        self,
        recognition_model: RecognitionPredictor,
        config=None,
    ):
        super().__init__(config)

        self.recognition_model = recognition_model
        # Conversion stats, useful for monitoring table quality
        self.table_stats = Counter()

    def __call__(self, document: Document):
        self.table_stats.clear()
        self.prepare_geometry_tables(document)
        tables_by_page = self.collect_tables(document)
        total = sum(len(v) for v in tables_by_page.values())
        if not total:
            return

        ocr_fallback = []  # (page, block) digital tables the heuristics missed
        for page in document.pages:
            for block in tables_by_page.get(page.page_id, []):
                # Scanned/garbled pages: the full-page OCR already produced the
                # table HTML - trust it, don't redo.
                if block.html:
                    self.table_stats["tables_geometry" if getattr(block, "table_grid", None) else "tables_ocr"] += 1
                    continue

                html = self.reconstruct_digital_table(page, block)
                if html:
                    block.structure = []
                    block.html = html
                    block.text_extraction_method = "pdftext"
                    self.table_stats["tables_pdftext"] += 1
                else:
                    ocr_fallback.append((page, block))

        self.run_ocr_fallback(document, ocr_fallback)
        self.cleanup_contained_blocks(document, tables_by_page)

        # Release the cached raw pdftext pages - they hold char-level data
        for page in document.pages:
            page.pdftext_page = None

        self.table_stats["tables_total"] = total
        logger.info(f"Table processing stats: {dict(self.table_stats)}")

    def prepare_geometry_tables(self, document):
        if not self.table_geometry or not os.path.isfile(document.filepath) or not document.filepath.lower().endswith('.pdf'):
            return
        if self.table_source_coordinates:
            from marker.cloud.source import PdfSource
            context = PdfSource(document.filepath)
        else:
            context = pdfplumber.open(document.filepath)
        with context as pdf:
            for page in document.pages:
                source = pdf.pdf_page(page.page_id) if self.table_source_coordinates else pdf.pages[page.page_id]
                grids = extract_grids(source)
                existing = list(page.contained_blocks(document, (BlockTypes.Table,)))
                for grid in grids:
                    matching = [b for b in existing if _intersection(b.polygon.bbox, grid['bbox']) > min(b.polygon.area, (grid['bbox'][2]-grid['bbox'][0])*(grid['bbox'][3]-grid['bbox'][1]))*0.8]
                    if matching:
                        block = max(matching, key=lambda b: _intersection(b.polygon.bbox, grid['bbox']))
                        for other in matching:
                            existing.remove(other)
                            if other is not block:
                                other.ignore_for_output = True
                                other.structure = []
                    else:
                        block = page.add_full_block(Table(page_id=page.page_id, polygon=PolygonBox.from_bbox(grid['bbox'])))
                        structure = list(page.structure or [])
                        position = next((i for i, bid in enumerate(structure) if document.get_block(bid).polygon.y_start > grid['bbox'][1]), len(structure))
                        structure.insert(position, block.id)
                        page.structure = structure
                    block.polygon = PolygonBox.from_bbox(grid['bbox'])
                    block.table_grid = grid
                    if self.extract_images:
                        attach_images(grid, block, page)
                    block.html = render_grid(grid)
                    block.structure = []
                    block.text_extraction_method = 'pdftext'
                if not self.table_source_coordinates:
                    source.close()

    def collect_tables(self, document: Document) -> dict:
        return {
            page.page_id: page.contained_blocks(document, self.block_types)
            for page in document.pages
        }

    def reconstruct_digital_table(self, page, block) -> str | None:
        """Reconstruct a digital table's HTML from the cached pdftext page.
        Returns HTML, or None if there's no text layer or the heuristics can't
        resolve a confident grid."""
        pdftext_page = page.pdftext_page
        if not pdftext_page:
            return None
        lines = table_lines_from_pdftext(pdftext_page, block.polygon.bbox)
        result = reconstruct_table_html(lines)
        if not result:
            return None
        html, score = result
        min_score = self.min_recon_score
        if min_score is None:
            min_score = 0.75 if self.mode == "balanced" else 0.5
        if score < min_score:
            return None
        return html

    def run_ocr_fallback(self, document: Document, fallback: list):
        """OCR the crops of digital tables the pdftext heuristics couldn't
        resolve, with the recognition model (one box per table -> HTML)."""
        if not fallback or self.disable_ocr:
            return

        images, entries = [], []
        for page, block in fallback:
            highres = page.get_image(highres=True)
            image_poly = block.polygon.rescale(page.polygon.size, highres.size)
            images.append(highres.crop(image_poly.bbox))
            entries.append((block, block.layout_token_count or 0))

        layout_results = []
        for image, (block, token_count) in zip(images, entries):
            w, h = image.size
            label = block_type_to_surya_label(block.block_type) or "Table"
            box = LayoutBox(
                polygon=[[0, 0], [w, 0], [w, h], [0, h]],
                label=label,
                raw_label=label,
                position=0,
                count=max(token_count, self.ocr_table_token_floor),
            )
            layout_results.append(LayoutResult(bboxes=[box], image_bbox=[0, 0, w, h]))

        self.recognition_model.disable_tqdm = self.disable_tqdm
        results = self.recognition_model(
            images=images, layout_results=layout_results, full_page=False
        )
        for (block, _tc), page_result in zip(entries, results):
            block_result = page_result.blocks[0] if page_result.blocks else None
            raw = block_result.html if block_result and not block_result.error else ""
            if block.block_type == BlockTypes.Form:
                # Forms rarely OCR to <table> html - accept any cleaned output
                # (matches how full-page OCR sets form html unconditionally).
                html = self.clean_form_html(raw)
            else:
                html = self.clean_table_html(raw)
            if not html:
                self.table_stats["tables_ocr_failed"] += 1
                logger.warning(f"Table OCR failed for block {block.id}")
                continue
            block.structure = []
            block.html = html
            block.text_extraction_method = "surya"
            self.table_stats["tables_ocr"] += 1

    def clean_table_html(self, html: str | None) -> str:
        if not html:
            return ""
        if "<table" not in html:
            return ""
        if _detect_repeat_loop(html):
            return ""

        # Re-serialize to balance tags in case of token-budget truncation
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None or not table.find_all(["td", "th"]):
            return ""
        return str(soup).strip()

    def clean_form_html(self, html: str | None) -> str:
        if not html or _detect_repeat_loop(html):
            return ""
        # Balance any truncated tags; no <table> requirement for forms.
        return str(BeautifulSoup(html, "html.parser")).strip()

    def cleanup_contained_blocks(self, document: Document, tables_by_page: dict):
        # Clean out other blocks inside the table
        # This can happen with stray text blocks inside the table post-merging
        for page in document.pages:
            page_tables = tables_by_page.get(page.page_id, [])
            if not page_tables:
                continue
            child_contained_blocks = page.contained_blocks(
                document, self.contained_block_types
            )
            if not child_contained_blocks:
                continue

            intersections = matrix_intersection_area(
                [c.polygon.bbox for c in child_contained_blocks],
                [block.polygon.bbox for block in page_tables],
            )
            for child_idx, child in enumerate(child_contained_blocks):
                for table_idx in range(len(page_tables)):
                    intersection_pct = intersections[child_idx, table_idx] / max(
                        child.polygon.area, 1
                    )
                    if intersection_pct > 0.95 and child.id in page.structure:
                        page.structure.remove(child.id)
                        break
