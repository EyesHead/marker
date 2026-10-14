"""Offline HTML table contracts: resources, LaTeX and page continuations."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import openai
from bs4 import BeautifulSoup
from PIL import Image

from marker.output import save_output
from marker.processors.llm.llm_table import LLMTableProcessor, TableSchema
from marker.processors.llm.llm_table_merge import LLMTableMergeProcessor
from marker.renderers.markdown import MarkdownRenderer
from marker.schema.blocks import Table, TableCell
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.services.openai import OpenAIService


def page_with_table(page_id, bbox, html=None):
    image = Image.new('RGB', (1000, 1000), 'white')
    page = PageGroup(page_id=page_id, polygon=PolygonBox.from_bbox([0, 0, 1000, 1000]),
                     children=[], structure=[], lowres_image=image, highres_image=image)
    table = page.add_full_block(Table(page_id=page_id, polygon=PolygonBox.from_bbox(bbox), html=html))
    page.add_structure(table)
    return page, table


class HTMLTableTests(unittest.TestCase):
    def test_model_image_boxes_become_cell_resources_and_latex_survives(self):
        page, table = page_with_table(0, [100, 100, 900, 900], '<table><tr><td>А</td><td>Б</td></tr></table>')
        document = Document(filepath='offline.pdf', pages=[page])
        corrected = r'<table><tr><th rowspan="2">Схема</th><th colspan="2">Расчёт</th></tr><tr><th>А</th><th>Б</th></tr><tr><td><img data-bbox="100,200,400,600"></td><td><math>\varphi = \frac{a_1}{b^2}</math></td><td>0,67</td></tr></table>'
        llm = Mock(return_value={'comparison': 'match', 'corrected_html': corrected, 'analysis': 'match', 'score': 5})
        processor = LLMTableProcessor(llm, {'use_llm': True, 'image_expansion_ratio': 0})
        processor.process_rewriting(document, page, table)
        rendered = MarkdownRenderer()(document)
        soup = BeautifulSoup(rendered.markdown, 'html.parser')
        self.assertEqual(soup.find('th')['rowspan'], '2')
        self.assertEqual(soup.find_all('th')[1]['colspan'], '2')
        img = soup.find('td').find('img')
        self.assertIsNotNone(img)
        name = img['src']
        self.assertIn(name, rendered.images)
        self.assertEqual(rendered.images[name].size, (240, 320))
        self.assertIn(r'$\varphi = \frac{a_1}{b^2}$', rendered.markdown)
        self.assertNotIn('data-bbox', rendered.markdown)
        self.assertNotIn('table_reconstruction', rendered.metadata)
        with tempfile.TemporaryDirectory() as root:
            save_output(rendered, root, 'document')
            self.assertTrue(Path(root, name).is_file())
            self.assertIn(name, Path(root, 'document.md').read_text())
            self.assertFalse(Path(root, 'document_tables.json').exists())

    def test_page_continuation_forms_one_html_table_with_both_resources(self):
        first, table1 = page_with_table(0, [100, 700, 900, 960], '<table><tbody><tr><th>Схема</th><th>Значение</th></tr><tr><td><img src="first.png"></td><td>1</td></tr></tbody></table>')
        second, table2 = page_with_table(1, [100, 50, 900, 300], '<table><tbody><tr><th>Схема</th><th>Значение</th></tr><tr><td><img src="second.png"></td><td>2</td></tr></tbody></table>')
        table1.embedded_images['first.png'] = Image.new('RGB', (10, 10), 'red')
        table2.embedded_images['second.png'] = Image.new('RGB', (20, 20), 'blue')
        document = Document(filepath='offline.pdf', pages=[first, second])
        llm = Mock(return_value={'direction': 'bottom', 'merge': 'true'})
        processor = LLMTableMergeProcessor(llm, {'use_llm': True, 'disable_tqdm': True})
        processor.rewrite_blocks(document)
        rendered = MarkdownRenderer()(document)
        soup = BeautifulSoup(rendered.markdown, 'html.parser')
        self.assertEqual(len(soup.find_all('table')), 1)
        self.assertEqual(len(soup.find_all('tr')), 3)
        self.assertEqual([img['src'] for img in soup.find_all('img')], ['first.png', 'second.png'])
        self.assertEqual(set(rendered.images), {'first.png', 'second.png'})

    def test_cell_table_continuation_keeps_resources_from_second_page(self):
        first, table1 = page_with_table(0, [100, 700, 900, 960])
        second, table2 = page_with_table(1, [100, 50, 900, 300])
        for page, table, name in ((first, table1, 'first.png'), (second, table2, 'second.png')):
            cell = page.add_full_block(TableCell(page_id=page.page_id, polygon=table.polygon,
                                                rowspan=1, colspan=1, row_id=0, col_id=0,
                                                is_header=False, text_lines=[f'<img src="{name}">']))
            table.add_structure(cell)
            table.embedded_images[name] = Image.new('RGB', (10, 10), 'red')
        document = Document(filepath='offline.pdf', pages=[first, second])
        processor = LLMTableMergeProcessor(Mock(return_value={'direction': 'bottom', 'merge': 'true'}),
                                          {'use_llm': True, 'disable_tqdm': True})
        processor.rewrite_blocks(document)
        rendered = MarkdownRenderer()(document)
        soup = BeautifulSoup(rendered.markdown, 'html.parser')
        self.assertEqual(len(soup.find_all('table')), 1)
        self.assertEqual([img['src'] for img in soup.find_all('img')], ['first.png', 'second.png'])
        self.assertEqual(set(rendered.images), {'first.png', 'second.png'})

    def test_ollama_cloud_can_return_html_in_the_small_marker_response(self):
        corrected = '<table><tr><td>А</td><td>Б</td></tr></table>'
        content = {'comparison': 'match', 'corrected_html': corrected, 'analysis': 'match', 'score': 5}
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 0,
                                            'model': 'gemma4:31b',
                                            'choices': [{'index': 0, 'finish_reason': 'stop',
                                                         'message': {'role': 'assistant', 'content': json.dumps(content)}}],
                                            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})
        client = openai.OpenAI(api_key='offline-placeholder', base_url='https://ollama.com/v1',
                               http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        service = OpenAIService({'openai_base_url': 'https://ollama.com/v1', 'openai_model': 'gemma4:31b',
                                 'openai_api_key': 'offline-placeholder', 'openai_image_format': 'png'})
        with client, patch.object(service, 'get_client', return_value=client):
            result = service('Correct the HTML table.', Image.new('RGB', (10, 10)), None, TableSchema)
        self.assertEqual(result['corrected_html'], corrected)
        self.assertEqual(requests[0]['model'], 'gemma4:31b')
        self.assertEqual(requests[0]['temperature'], 0)
        self.assertNotIn('response_format', requests[0])
        self.assertTrue(requests[0]['messages'][0]['content'][0]['image_url']['url'].startswith('data:image/png;'))


if __name__ == '__main__':
    unittest.main()
