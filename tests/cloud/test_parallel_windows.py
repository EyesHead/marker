import threading
import time
import unittest
from marker.cloud.parallel import ordered_cloud_map


class ParallelWindowTests(unittest.TestCase):
    def test_only_network_overlaps_and_source_and_order_stay_on_owner(self):
        owner=threading.get_ident();barrier=threading.Barrier(2)
        prepared=[];source=[]
        def items():
            for i in range(4):
                self.assertEqual(threading.get_ident(),owner)
                prepared.append(i);yield i
        def request(i,call):
            self.assertNotEqual(threading.get_ident(),owner)
            barrier.wait(timeout=3)
            call(lambda:source.append((i,threading.get_ident())))
            if i%2==0:time.sleep(.01)
            return i
        self.assertEqual(list(ordered_cloud_map(request,items(),2)),[0,1,2,3])
        self.assertEqual(prepared,[0,1,2,3])
        self.assertEqual({identity for _,identity in source},{owner})

    def test_sequential_has_no_worker_threads(self):
        owner=threading.get_ident()
        def request(i,call):
            self.assertEqual(threading.get_ident(),owner)
            return call(lambda:i)
        self.assertEqual(list(ordered_cloud_map(request,range(3),1)),[0,1,2])

    def test_source_exception_reaches_worker_without_a_deadlock(self):
        def failure():raise ValueError('safe fixture failure')
        def request(i,call):return call(failure)
        with self.assertRaisesRegex(ValueError,'safe fixture'):
            list(ordered_cloud_map(request,[0,1],2))

class ParallelTableIntegrationTests(unittest.TestCase):
    def exercise(self,workers,overflow=False):
        import json
        from pathlib import Path
        from unittest.mock import patch
        import httpx,openai
        from test_content import offline_client,table_echo
        from marker.cloud.source import PdfSource
        from marker.cloud.tables import detect_tables,group_tables,recognize_table
        owner=threading.get_ident();lock=threading.Lock();active=0;maximum=0
        overlap=threading.Barrier(2) if workers > 1 else None
        def respond(source,number):
            nonlocal active,maximum
            with lock:active+=1;maximum=max(maximum,active)
            try:
                if overlap is not None and number <= 2:
                    overlap.wait(timeout=5)
                time.sleep(.015)
                if overflow and number==1:
                    raise openai.BadRequestError('context fixture',response=httpx.Response(400,request=httpx.Request('POST','https://offline.invalid')),body={'error':{'code':'context_length_exceeded'}})
                return table_echo(source,number)
            finally:
                with lock:active-=1
        client,seen=offline_client(respond,{'cloud_max_concurrency':workers})
        with PdfSource(Path(__file__).parent/'fixtures/content_lines.pdf') as source:
            table=group_tables(detect_tables(source),None)[0]
            crop,resource=source.crop,source.resource
            def safe_crop(*args,**kwargs):
                self.assertEqual(threading.get_ident(),owner)
                return crop(*args,**kwargs)
            def safe_resource(*args,**kwargs):
                self.assertEqual(threading.get_ident(),owner)
                return resource(*args,**kwargs)
            started=time.perf_counter()
            with patch.object(source,'crop',side_effect=safe_crop),patch.object(source,'resource',side_effect=safe_resource):
                result=recognize_table(source,table,client)
            elapsed=time.perf_counter()-started
        self.assertTrue(result.complete,result.reasons)
        return result,client.request_count,maximum,seen,elapsed

    def test_parallel_table_is_identical_with_two_inflight_http_calls(self):
        first,count,one,seen1,_=self.exercise(1)
        second,count2,two,seen2,_=self.exercise(2)
        self.assertEqual(one,1);self.assertEqual(two,2)
        self.assertEqual(first.html,second.html)
        self.assertEqual(count,count2)
        self.assertEqual(sorted(map(str,seen1)),sorted(map(str,seen2)))
        self.assertEqual(first.resources.keys(),second.resources.keys())
        for key in first.resources:self.assertEqual(first.resources[key].tobytes(),second.resources[key].tobytes())

    def test_parallel_context_reflow_keeps_pdfium_on_owner_thread(self):
        expected,_,_,_,_=self.exercise(1)
        actual,_,maximum,_,_=self.exercise(2,overflow=True)
        self.assertEqual(actual.html,expected.html)
        self.assertEqual(maximum,2)
