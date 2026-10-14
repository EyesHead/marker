"""Portable geometry/LLM contracts with a hand-built source page."""
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

from bs4 import BeautifulSoup
from PIL import Image

from marker.output import save_output
from marker.processors.table_geometry import extract_grids, attach_images, render_grid, grid_positions
from marker.processors.llm.llm_table import LLMTableProcessor
from marker.processors.llm import BaseLLMProcessor
from marker.processors.llm.llm_page_correction import LLMPageCorrectionProcessor
from marker.renderers.markdown import MarkdownRenderer
from marker.schema.blocks import Table
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox


def source_fixture():
    def edge(x0, y0, x1, y1):
        return {'x0': x0, 'top': y0, 'x1': x1, 'bottom': y1,
                'width': x1-x0, 'height': y1-y0, 'orientation': 'h' if y0==y1 else 'v'}
    edges = [edge(0,0,300,0), edge(50,20,300,20), edge(0,40,300,40), edge(0,100,300,100)]
    edges += [edge(0,0,0,100), edge(50,0,50,100), edge(300,0,300,100), edge(175,20,175,100)]
    def word(text, x, y):
        return {'text': text, 'x0': x, 'x1': x+20, 'top': y, 'bottom': y+8, 'size': 8, 'fontname': 'Arial'}
    words = [word('Коэффициент', 5, 5), word('Значения', 90, 5), word('А',80,25), word('Б',200,25),
             word('1,70',80,45),word('0,05',80,65),word('1,77',200,45),word('0,19',200,65)]
    return SimpleNamespace(page_number=1, edges=edges, extract_words=lambda **_: words,
                           images=[{'x0':15,'top':45,'x1':30,'bottom':55},
                                   {'x0':15,'top':65,'x1':30,'bottom':75}])


def document_fixture():
    page = PageGroup(page_id=0, polygon=PolygonBox.from_bbox([0,0,300,120]),
                     children=[], structure=[], lowres_image=Image.new('RGB',(600,240),'white'),
                     highres_image=Image.new('RGB',(600,240),'white'))
    table = page.add_full_block(Table(page_id=0, polygon=PolygonBox.from_bbox([0,0,300,100])))
    page.add_structure(table)
    table.table_grid = extract_grids(source_fixture())[0]
    attach_images(table.table_grid, table, page)
    table.html = render_grid(table.table_grid)
    return Document(filepath='fixture.pdf', pages=[page]), table


class GeometryContractTests(unittest.TestCase):
    def test_unruled_data_lines_keep_common_baselines_and_merged_header(self):
        document, table = document_fixture()
        soup = BeautifulSoup(table.html, 'html.parser')
        self.assertEqual(len(soup.find_all('col')), 3)
        self.assertIsNotNone(soup.find('td', rowspan='2'))
        self.assertIsNotNone(soup.find('td', colspan='2'))
        data = soup.find_all('tr')[-1]
        self.assertEqual([c.get_text(' ',strip=True) for c in data.find_all('td')], ['', '1,70 0,05', '1,77 0,19'])
        for cell in data.find_all('td'):
            items = cell.find_all(['img','span'])
            self.assertIn('top:5.000pt', items[0]['style'])
            self.assertIn('top:25.000pt', items[1]['style'])

    def test_formula_annotations_cannot_change_cells_or_remove_source_images(self):
        document, table = document_fixture()
        names = list(table.embedded_images)
        response = {'formulas':[{'source_id':names[0], 'latex':r'\alpha_1'},
                                {'source_id':'invented.png', 'latex':'wrong'}]}
        llm = Mock(return_value=response)
        processor = LLMTableProcessor(llm, {'use_llm':True})
        before = [c['bbox'][:] for c in table.table_grid['cells']]
        processor.process_rewriting(document, document.pages[0], table)
        self.assertEqual([c['bbox'] for c in table.table_grid['cells']], before)
        soup = BeautifulSoup(table.html,'html.parser')
        self.assertEqual([im['src'] for im in soup.find_all('img')], names)
        self.assertEqual(soup.find('img')['alt'], r'\alpha_1')
        self.assertEqual(set(table.embedded_images), set(names))

    def test_page_rewrite_cannot_destroy_verified_grid_even_with_all_image_names(self):
        document, table = document_fixture()
        original = table.html
        rewritten = '<table><tr><td>'+''.join(f'<img src="{n}">' for n in table.embedded_images)+'</td></tr></table>'
        for cls in (BaseLLMProcessor, LLMPageCorrectionProcessor):
            with self.subTest(processor=cls.__name__):
                table.html = original
                cls(Mock(), {'use_llm':True}).handle_rewrites([{'id':str(table.id),'html':rewritten}], document)
                self.assertEqual(table.html, original)

    def test_saved_formula_resources_are_actual_png_files(self):
        document, table = document_fixture()
        with tempfile.TemporaryDirectory() as root:
            save_output(MarkdownRenderer()(document), root, 'fixture')
            for name in table.embedded_images:
                self.assertEqual(Path(root,name).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_overlapping_physical_cells_are_rejected(self):
        grid = {'cells':[{'bbox':[0,0,100,20]}, {'bbox':[50,0,150,20]}]}
        self.assertIsNone(grid_positions(grid))


if __name__ == '__main__':
    unittest.main()
