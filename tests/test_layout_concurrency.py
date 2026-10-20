import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from app.layout import prepare_layout


class LayoutConcurrencyTests(unittest.TestCase):
    def test_parallel_units_use_isolated_temporary_inputs(self):
        barrier=threading.Barrier(2)
        paths=[]
        def node(command,**kwargs):
            path=Path(command[-1]);paths.append(path)
            source=json.loads(path.read_text(encoding='utf-8'))
            barrier.wait(timeout=5)
            source['layout']={'version':1,'pages':[{'sentence_id':source['marker']}]}
            path.write_text(json.dumps(source),encoding='utf-8')
        with tempfile.TemporaryDirectory() as temporary,patch('app.layout.subprocess.run',side_effect=node),patch('app.layout._page_translations'):
            a,b={'marker':'unit-a'},{'marker':'unit-b'}
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures=[executor.submit(prepare_layout,item,Path(temporary)) for item in (a,b)]
                for future in futures:future.result()
            self.assertEqual(len(set(paths)),2)
            self.assertEqual(a['layout']['pages'][0]['sentence_id'],'unit-a')
            self.assertEqual(b['layout']['pages'][0]['sentence_id'],'unit-b')
            self.assertTrue(all(not path.exists() for path in paths))
