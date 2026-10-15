"""Regression checks against a local normative PDF; no cloud or model weights.

Run with MARKER_SAMPLE_PDF pointing to the local СП_16.13330.2017 sample.
The user's PDF is intentionally not included in the repository.
"""
import os
import unittest
from unittest.mock import Mock

import pypdfium2 as pdfium
from bs4 import BeautifulSoup
from pdftext.extraction import dictionary_output

from marker.processors.table import TableProcessor
from marker.processors.llm.llm_table_merge import LLMTableMergeProcessor
from marker.renderers.markdown import MarkdownRenderer
from marker.schema.blocks import Table
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox


SAMPLE = os.environ.get('MARKER_SAMPLE_PDF', '')


@unittest.skipUnless(SAMPLE and os.path.isfile(SAMPLE), 'Set MARKER_SAMPLE_PDF to run source-PDF checks')
class SourcePDFGeometryTests(unittest.TestCase):
    def make_document(self, regions):
        raw = {page['page']: page for page in dictionary_output(SAMPLE, page_range=list(regions),
                                                                keep_chars=True, workers=1)}
        pages = []
        with pdfium.PdfDocument(SAMPLE) as source:
            for page_id, bbox in regions.items():
                pdf_page = source[page_id]
                width, height = pdf_page.get_size()
                image = pdf_page.render(scale=2).to_pil()
                page = PageGroup(page_id=page_id, polygon=PolygonBox.from_bbox([0, 0, width, height]),
                                 children=[], structure=[], pdftext_page=raw[page_id],
                                 lowres_image=image, highres_image=image)
                table = page.add_full_block(Table(page_id=page_id, polygon=PolygonBox.from_bbox(bbox)))
                page.add_structure(table)
                pages.append(page)
                pdf_page.close()
        return Document(filepath=SAMPLE, pages=pages)

    def test_table_6_keeps_coefficient_column_images_and_merged_notes(self):
        document = self.make_document({17: [85.1, 163.58, 793.92, 455.62]})
        with self.assertLogs('marker', level='INFO'):
            TableProcessor(None, {'disable_ocr': True})(document)
        rendered = MarkdownRenderer()(document)
        soup = BeautifulSoup(rendered.markdown, 'html.parser')
        table = soup.find('table')
        self.assertIsNotNone(table)
        self.assertEqual(len(table.find_all('col')), 7, 'Coefficient column must not disappear')
        self.assertTrue(table.find('td', colspan='7'), 'Footnotes must span the complete table')
        self.assertGreaterEqual(len(table.find_all('img')), 10, 'Raster formulas and Greek letters were lost')
        numeric = [cell.get_text(' ', strip=True) for cell in table.find_all('td')]
        self.assertIn('1,70 0,05 0,65', numeric)
        self.assertIn('1,77 0,19 1,0', numeric)
        for image in table.find_all('img'):
            self.assertIn(image['src'], rendered.images)
            self.assertTrue(image['src'].endswith('.png'), 'Formulas require lossless resources')

    def test_table_5_joins_interrupted_header_and_preserves_formula_cells(self):
        document = self.make_document({14: [85.1, 494.02, 786.96, 516.48],
                                       15: [85.1, 57.02, 786.96, 322.15]})
        with self.assertLogs('marker', level='INFO'):
            TableProcessor(None, {'disable_ocr': True})(document)
        merger = LLMTableMergeProcessor(Mock(return_value={'direction': 'bottom', 'merge': 'true'}),
                                        {'use_llm': True, 'disable_tqdm': True})
        merger.rewrite_blocks(document)
        rendered = MarkdownRenderer()(document)
        soup = BeautifulSoup(rendered.markdown, 'html.parser')
        tables = [table for table in soup.find_all('table') if 'болтов классов' in table.get_text(' ', strip=True)]
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertIn('Напряженное', table.get_text())
        self.assertIn('состояние', table.get_text())
        self.assertEqual(len(table.find_all('col')), 8)
        self.assertTrue(table.find(['td', 'th'], colspan='6'), 'Top header must span six columns')
        self.assertTrue(table.find(['td', 'th'], rowspan='3'), 'Header interrupted by page break was not joined')
        self.assertGreaterEqual(len(table.find_all('img')), 15)
        self.assertTrue(table.find(['td', 'th'], colspan='8'), 'Footnote must span the entire table')
        for image in table.find_all('img'):
            self.assertIn(image['src'], rendered.images)


if __name__ == '__main__':
    unittest.main()
