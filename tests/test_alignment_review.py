import json
import io
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['ENGLISH_VIDEO_DATA_DIR']=str(ROOT/'work/test-alignment-profile')
from app.alignment import group_connected_speech
from app import pipeline
from app import providers


class AlignmentReviewTests(unittest.TestCase):
    def test_layout_gate_stops_before_render_and_keeps_previous_video(self):
        from app import renderer
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
            base=Path(temp);cli=base/'node_modules/@remotion/cli/remotion-cli.js';cli.parent.mkdir(parents=True);cli.write_text('test')
            f=base/'episode';f.mkdir();(f/'lesson.mp4').write_bytes(b'previous')
            path=f/'episode.json';path.write_text(json.dumps(self.package()),encoding='utf-8')
            with patch.object(renderer,'ROOT',base),patch.object(renderer.shutil,'which',return_value='node'),patch.object(renderer.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='讲解超出范围')),patch.object(renderer.subprocess,'Popen') as popen:
                with self.assertRaises(pipeline.ReviewRequired):renderer.render_video(path,f,lambda *a:None)
                popen.assert_not_called()
            self.assertEqual((f/'lesson.mp4').read_bytes(),b'previous')
            self.assertTrue(json.loads(path.read_text(encoding='utf-8'))['quality']['needs_review'])

    def test_reopening_studio_reuses_running_local_service(self):
        from app import server
        with patch.object(server,'ThreadingHTTPServer',side_effect=OSError('in use')),patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps({'kind':'universal'}).encode())) as request,patch('webbrowser.open') as opened:
            server.serve(8775,True)
            opened.assert_called_once_with('http://127.0.0.1:8775/')

    def test_truncated_reasoning_response_increases_budget_with_a_cap(self):
        budgets=[]
        def respond(request,**kwargs):
            budgets.append(json.loads(request.data)['max_tokens'])
            value={'choices':[{'message':{'content':'' if len(budgets)==1 else '{"ok":true}'},'finish_reason':'length' if len(budgets)==1 else 'stop'}]}
            return io.BytesIO(json.dumps(value).encode())
        with patch.object(providers,'get_settings',lambda:{'model':'test','base_url':'http://127.0.0.1:11434/v1'}),patch.object(providers,'read_api_key',lambda:'test-only'),patch.object(providers.time,'sleep'),patch.object(providers.urllib.request,'urlopen',side_effect=respond):
            self.assertEqual(providers.ask_deepseek('JSON','test',max_tokens=8192),{'ok':True})
            self.assertEqual(budgets,[8192,16384])

    def test_content_review_can_correct_grammar_points_without_editing_timing(self):
        from app.teaching_review import review_teaching
        p=self.package();p['sentences'][0].update(translation='你好',speaker='')
        p['words'][0]['gloss_zh']='你好'
        p['cards']=[{'id':'c0','type':'grammar','title':'结构','meaning':'释义','structure':'means doing','points':[{'label':'错误','text':'宾语从句'}],'example':'Hello.','translation':'你好','sentence_ids':['s0']}]
        results=[{'ok':True,'corrections':[{'type':'card','id':'c0','field':'points','value':[{'label':'结构','text':'动名词短语'}],'reason':'语法错误'}],'issues':[]},{'ok':True,'corrections':[],'issues':[]}]
        with patch('app.teaching_review.ask_deepseek',side_effect=results):review_teaching(p,lambda *a:None)
        self.assertEqual(p['cards'][0]['points'],[{'label':'结构','text':'动名词短语'}])
        self.assertEqual((p['words'][0]['start_ms'],p['words'][0]['end_ms']),(100,600))

    def test_empty_model_response_retries_without_caching_empty_result(self):
        def response(content):return io.BytesIO(json.dumps({'choices':[{'message':{'content':content},'finish_reason':'stop'}]}).encode())
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp,patch.object(providers,'get_settings',lambda:{'model':'test','base_url':'http://127.0.0.1:11434/v1'}),patch.object(providers,'read_api_key',lambda:'test-only'),patch.object(providers.time,'sleep'),patch.object(providers.urllib.request,'urlopen',side_effect=[response(''),response('{"ok":true}')]) as request:
            self.assertEqual(providers.ask_deepseek('JSON','test',cache_dir=Path(temp)),{'ok':True})
            self.assertEqual(request.call_count,2)
            self.assertEqual(len(list(Path(temp).glob('*.json'))),1)

    def test_inseparable_weak_word_keeps_measured_group(self):
        words=[{'text':'got','start_ms':100,'end_ms':300},{'text':'a','start_ms':300,'end_ms':300},{'text':'ring.','start_ms':300,'end_ms':600}]
        result,records=group_connected_speech(words)
        self.assertEqual([w['text'] for w in result],['got a','ring.'])
        self.assertEqual((result[0]['start_ms'],result[0]['end_ms']),(100,300))
        self.assertEqual(words[0]['text'],'got')
        self.assertEqual(len(records),1)

    def test_content_word_or_truncated_tail_stays_flagged(self):
        for text,following in [('gold',True),('a',False),('a','punctuation')]:
            words=[{'text':'is','start_ms':100,'end_ms':300},{'text':text,'start_ms':300,'end_ms':300}]
            if following:words.append({'text':'-' if following=='punctuation' else 'valuable.','start_ms':300,'end_ms':300 if following=='punctuation' else 600})
            result,records=group_connected_speech(words)
            self.assertEqual(records,[])
            self.assertEqual(result[1]['end_ms'],result[1]['start_ms'])

    def package(self):
        words=[{'id':'w0','text':'Hello.','start_ms':100,'end_ms':600,'probability':1,'sentence_id':'s0'}]
        return {'episode':{'id':'test','duration_s':1},'words':words,'sentences':[{'id':'s0','text':'Hello.','start_ms':100,'end_ms':600,'word_ids':['w0']}],'review':{'recognized_transcript':'Hello.','official_transcript':'Hello.'},'quality':{'flags':[]},'render':{},'processing':{},'cards':[]}

    def test_corrected_source_is_saved_before_api_failure(self):
        p=self.package()
        def realign(package,*args):
            package['words'][0]['text']='Hi.'
            package['sentences'][0]['text']='Hi.'
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
            folder=Path(temp)
            with patch.object(pipeline,'_realign_corrected',realign),patch.object(pipeline,'_annotate',side_effect=RuntimeError('API unavailable')):
                with self.assertRaises(RuntimeError):pipeline.continue_after_review(p,folder,lambda *a:None,'Hi.')
            for name in ('episode.json','episode.checkpoint.json'):
                saved=json.loads((folder/name).read_text(encoding='utf-8'))
                self.assertEqual(saved['words'][0]['text'],'Hi.')
                self.assertEqual(saved['review']['correction_history'][0]['after'],'Hi.')

    def test_confirmation_cannot_bypass_structural_timestamp_error(self):
        for start,end in [(100,100),(-1,600),(100,1500)]:
            p=self.package();p['words'][0].update(start_ms=start,end_ms=end)
            with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp,patch.object(pipeline,'_annotate') as annotate:
                with self.assertRaises(pipeline.ReviewRequired):pipeline.continue_after_review(p,Path(temp),lambda *a:None)
                annotate.assert_not_called()
                self.assertTrue(p['quality']['needs_review'])

if __name__=='__main__':unittest.main()
