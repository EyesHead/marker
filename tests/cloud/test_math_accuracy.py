"""Source geometry and rendered MathML must agree about complete indices."""
import json
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from marker.cloud.content import latex_to_mathml
from marker.cloud.context import RecognitionFailure


def atoms(base='0,42R', subscript='bun', raised=False):
    return [
        {'kind':'text','text':base,'bbox':(0,10,35,22),'size':12,'source_page':0},
        {'kind':'text','text':subscript,'bbox':(35,17,49,25) if not raised else (35,5,49,13),'size':8,'source_page':0},
    ]


class MathAccuracyTests(unittest.TestCase):
    def test_annotation_and_visible_base_share_the_entire_index(self):
        root=ET.fromstring(latex_to_mathml(r'0,42R_{\text{bun}}'))
        ns={'m':'http://www.w3.org/1998/Math/MathML'}
        node=root.find('.//m:msub',ns)
        self.assertEqual(''.join(node[0].itertext()),'R')
        self.assertEqual(''.join(node[1].itertext()),'bun')
        self.assertEqual(root.find('.//m:annotation',ns).text,r'0,42R_{\text{bun}}')

    def test_malformed_external_mathml_never_passes_as_a_rendered_formula(self):
        malformed='<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi><msub><mi>bun</mi></msub></mrow></math>'
        with patch('latex2mathml.converter.convert',return_value=malformed):
            with self.assertRaises(RecognitionFailure):
                latex_to_mathml('malformed-for-arity-control')

    def test_source_subscripts_reject_baseline_but_do_not_rewrite_it(self):
        from marker.cloud.content import source_script_hints, validate_script_hints
        hints=source_script_hints(atoms())
        self.assertEqual(hints[0]['text'],'bun')
        self.assertEqual(hints[0]['base'],'R')
        for latex in (r'0,42R_{bun}',r'0,42R_{\text{bun}}'):
            validate_script_hints(latex,hints)
        with self.assertRaises(RecognitionFailure):
            validate_script_hints(r'0,42Rb_{\text{un}}',hints)
        # A source with an actual baseline b is a different formula.
        validate_script_hints(r'0,42Rb_{\text{un}}',source_script_hints(atoms('0,42Rb','un')))

    def test_superscripts_and_nested_scripts_keep_the_source_baseline(self):
        from marker.cloud.content import source_script_hints, validate_script_hints
        hints=source_script_hints(atoms('x','2',raised=True))
        self.assertEqual(hints[0]['position'],'superscript')
        validate_script_hints('x_i^2',hints)
        with self.assertRaises(RecognitionFailure):
            validate_script_hints('x_2',hints)

    def test_nested_indices_are_not_flattened_into_the_outer_index(self):
        from marker.cloud.content import source_script_hints,validate_script_hints
        source=[{'kind':'text','text':'x','bbox':(0,0,8,12),'size':12},
                {'kind':'text','text':'a','bbox':(8,7,13,15),'size':8},
                {'kind':'text','text':'i','bbox':(13,12,16,17),'size':5}]
        hints=source_script_hints(source)
        validate_script_hints('x_{a_i}',hints)
        with self.assertRaises(RecognitionFailure):
            validate_script_hints('x_{ai}',hints)

    def test_styled_and_accented_bases_keep_their_actual_script_owner(self):
        from marker.cloud.content import source_script_hints,validate_script_hints
        hints=source_script_hints(atoms('x','i'))
        for latex in (r'\vec{x}_i',r'\hat{x}_i',r'\overline{x}_i',r'\mathbf{x}_i',r'\mathit{x}_i'):
            with self.subTest(latex=latex):validate_script_hints(latex,hints)
        for letter in ('R','C','N'):
            hints=source_script_hints(atoms(letter,'i'))
            for style in ('mathbb','mathcal','mathfrak'):
                validate_script_hints('\\'+style+'{'+letter+'}_i',hints)

    def test_stacked_fraction_boxes_do_not_invent_hard_script_constraints(self):
        from marker.cloud.content import source_script_hints,validate_script_hints
        source=[{'kind':'text','text':'x','bbox':(0,10,10,22),'size':12},
                {'kind':'text','text':'a','bbox':(12,5,18,13),'size':8},
                {'kind':'text','text':'b','bbox':(12,17,18,25),'size':8}]
        hints=source_script_hints(source)
        self.assertEqual(hints,[])
        validate_script_hints(r'x\frac{a}{b}',hints)

    def test_comparison_keeps_phi_and_varphi_distinct(self):
        from marker.cloud.content import source_script_hints,validate_script_hints
        hints=source_script_hints(atoms('ϕ','i'))
        validate_script_hints(r'\phi_i',hints)
        with self.assertRaises(RecognitionFailure):
            validate_script_hints(r'\varphi_i',hints)

    def test_geometry_does_not_invent_constraints_for_rasters_or_ordinary_text(self):
        from marker.cloud.content import source_script_hints
        self.assertEqual(source_script_hints([{'kind':'image','bbox':(0,0,97,38)}]),[])
        words=atoms('Note','word');words[1]['size']=12
        self.assertEqual(source_script_hints(words),[])

    def test_separately_extracted_script_letters_remain_one_complete_index(self):
        from marker.cloud.content import source_script_hints, validate_script_hints
        words = atoms()[:1] + [
            {'kind':'text','text':letter,'bbox':(35+i*5,17,39+i*5,25),'size':8,'source_page':0}
            for i,letter in enumerate('bun')
        ]
        hints=source_script_hints(words)
        self.assertEqual([h['text'] for h in hints],['bun'])
        validate_script_hints(r'0,42R_{bun}',hints)

    def test_wrong_index_triggers_counted_table_repair(self):
        from marker.cloud.tables import _describe_cells,_table_schema
        from marker.processors.table_geometry import grid_positions
        from test_client import OfflineCloud
        from marker.cloud.client import CloudClient
        source=atoms();grid={'page':0,'cells':[{'bbox':(0,0,60,35),'atoms':source}]}
        xs,ys,positions=grid_positions(grid)
        described,_=_describe_cells(grid,positions)
        ids=[a['id'] for a in described[0]['lines'][0]['atoms']]
        def response(latex):
            return json.dumps({'cells':[{'id':'c0','row':0,'column':0,'rowspan':1,'colspan':1,'lines':[{'id':'c0l0','parts':[{'kind':'math','atom_ids':ids,'latex':latex}]}]}]})
        with OfflineCloud([response(r'0,42Rb_{un}'),response(r'0,42R_{bun}')]) as offline:
            client=CloudClient(client=offline.client)
            result=client.request('Read source.',[],_table_schema(described,True,1,1))
            self.assertEqual(client.request_count,2)
            self.assertEqual(result['cells'][0]['lines'][0]['parts'][0]['latex'],r'0,42R_{bun}')
            repair=offline.requests[1]['messages'][0]['content'][-1]['text']
            self.assertIn('script',repair.lower())


class FormulaEvidenceTests(unittest.TestCase):
    def test_optional_zoom_preserves_existing_evidence_and_native_pixels(self):
        from types import SimpleNamespace
        from PIL import Image, ImageDraw
        from marker.cloud.math_evidence import with_math_closeups
        from test_client import FormulaSchema
        original=Image.new('RGBA',(97,38),'white')
        ImageDraw.Draw(original).text((2,2),'R',fill='black')
        before=original.tobytes()
        class Source:
            def resource(self,page,box):return original
        cells=[{'lines':[{'atoms':[{'id':'formula-source','kind':'image','page':0,'bbox':(0,0,45,18)}]}]}]
        context=Image.new('RGB',(1000,100),'white')
        config=SimpleNamespace(cloud_max_images=3,cloud_context_tokens=32768,cloud_max_output_tokens=2048)
        images=with_math_closeups(Source(),cells,[context],config,'Read.',FormulaSchema)
        self.assertEqual(len(images),2)
        self.assertIs(images[0],context)
        self.assertGreater(images[1].height,original.height)
        self.assertEqual(original.tobytes(),before)
        config.cloud_max_images=1
        self.assertEqual(with_math_closeups(Source(),cells,[context],config,'Read.',FormulaSchema),[context])

    def test_zoom_is_omitted_if_existing_image_would_be_reduced(self):
        from types import SimpleNamespace
        from PIL import Image
        from marker.cloud.math_evidence import with_math_closeups
        from test_client import FormulaSchema
        class Source:
            def resource(self,page,box):return Image.new('RGB',(97,38),'white')
        cells=[{'lines':[{'atoms':[{'id':'a','kind':'image','page':0,'bbox':(0,0,45,18)}]}]}]
        context=Image.new('RGB',(1000,100),'white')
        from marker.cloud.client import schema_prompt
        from marker.cloud.context import estimate_request_tokens
        budget=estimate_request_tokens(schema_prompt('Read.',FormulaSchema),[context])+2048+10
        config=SimpleNamespace(cloud_max_images=3,cloud_context_tokens=budget,cloud_max_output_tokens=2048)
        result=with_math_closeups(Source(),cells,[context],config,'Read.',FormulaSchema)
        self.assertEqual(result,[context])

    def test_packed_closeups_keep_every_previously_fitted_context_pixel(self):
        from types import SimpleNamespace
        from PIL import Image
        from marker.cloud.math_evidence import with_math_closeups
        from marker.cloud.client import schema_prompt
        from marker.cloud.context import estimate_request_tokens,fit_request
        from test_client import FormulaSchema
        class Source:
            def resource(self,page,box):return Image.new('RGB',(97,38),'black')
        cells=[{'lines':[{'atoms':[{'id':'source','kind':'image','page':0,'bbox':(0,0,45,18)}]}]}]
        images=[Image.new('RGB',(400,80),'red'),Image.new('RGB',(400,80),'blue')]
        text=schema_prompt('Read.',FormulaSchema)
        budget=estimate_request_tokens(text,images)+2048+10
        config=SimpleNamespace(cloud_max_images=3,cloud_context_tokens=budget,cloud_max_output_tokens=2048)
        baseline=fit_request(text,images,budget,2048)
        output=with_math_closeups(Source(),cells,images,config,'Read.',FormulaSchema)
        self.assertEqual(len(output),1)
        self.assertTrue(output[0].info.get('source_closeups'))
        transmitted=fit_request(text,output,budget,2048)[0]
        self.assertEqual(transmitted.size,output[0].size)
        top=0
        for original in baseline:
            self.assertEqual(transmitted.crop((0,top,original.width,top+original.height)).convert('RGBA').tobytes(),original.convert('RGBA').tobytes())
            top+=original.height+4


class LongTableBaselineTests(unittest.TestCase):
    def test_long_baseline_splits_without_losing_atoms_or_whole_expressions(self):
        from types import SimpleNamespace
        from PIL import Image,ImageDraw
        from marker.cloud.tables import _describe_cells,_table_windows,_table_schema
        from marker.processors.table_geometry import grid_positions
        from marker.cloud.client import CloudClient
        from copy import deepcopy
        words=[{'kind':'text','text':'paragraph','bbox':(i*25,10,i*25+23,20),'size':10,'source_page':0} for i in range(30)]
        grid={'page':0,'cells':[{'bbox':(0,0,760,40),'atoms':words}]}
        xs,ys,positions=grid_positions(grid)
        described,originals=_describe_cells(grid,positions)
        original=deepcopy(described)
        class Source:
            def page_size(self,page):return (800,100)
            def crop(self,page,box):
                im=Image.new('RGB',(max(1,round((box[2]-box[0])*2)),40),'white')
                ImageDraw.Draw(im).text((1,1),'paragraph',fill='black');return im
        windows=list(_table_windows(Source(),described,originals,CloudClient(),True,1,1))
        ids=[a['id'] for cells,*_ in windows for c in cells for l in c['lines'] for a in l['atoms']]
        expected=[a['id'] for a in described[0]['lines'][0]['atoms']]
        self.assertEqual(ids,expected)
        self.assertGreater(len(windows),1)
        self.assertEqual(described,original)
        self.assertTrue(all(l['id']=='c0l0' for cells,*_ in windows for c in cells for l in c['lines']))

if __name__ == '__main__':
    unittest.main()
