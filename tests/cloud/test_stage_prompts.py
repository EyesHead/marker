"""Stage-specific instructions keep geometry requests separate from transcription."""
import unittest
from pathlib import Path
from marker.cloud.client import CloudClient
from marker.cloud.source import PdfSource
from marker.cloud.regions import analyze_page
from marker.cloud.tables import detect_tables,_visual_decision
from marker.converters.ollama import scan_layout
from test_client import OfflineCloud

FIXTURES=Path(__file__).parent/'fixtures'

class StagePromptTests(unittest.TestCase):
    def test_layout_echoes_whole_region_bounds_and_forbids_transcription_fields(self):
        import json
        with PdfSource(FIXTURES/'native_figure.pdf') as source:
            region=next(r for r in analyze_page(source,0) if r.kind=='image')
            answer={'complete':True,'regions':[{'kind':'figure','bbox':region.bbox,'cells':[]}]}
            with OfflineCloud([json.dumps(answer)]) as offline:
                scan_layout(source,region,CloudClient(client=offline.client))
                prompt=offline.requests[0]['messages'][0]['content'][-1]['text']
                self.assertIn('SOURCE.bbox',prompt)
                self.assertIn('only bbox and kind',prompt)
                self.assertIn('do not return text, latex',prompt)

    def test_continuation_expects_only_a_boolean_decision(self):
        with PdfSource(FIXTURES/'three_page_continuation.pdf') as source:
            parts=detect_tables(source)
            with OfflineCloud(['{"merge":true}']) as offline:
                self.assertTrue(_visual_decision(parts[0],parts[1],CloudClient(client=offline.client)))
                prompt=offline.requests[0]['messages'][0]['content'][-1]['text']
                self.assertIn('{"merge":false}',prompt)
                self.assertIn('Do not reconstruct',prompt)
