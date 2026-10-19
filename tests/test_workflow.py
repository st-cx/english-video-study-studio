import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['ENGLISH_VIDEO_DATA_DIR']=str(ROOT/'work/test-workflow-profile')
from app import providers,server,pipeline,bbc,storage
from app.subtitles import _timestamp,make_captions
from app.transcript import spoken_text,normalize_orthography,official_vocabulary


class WorkflowTests(unittest.TestCase):
    def test_topic_title_uses_episode_metadata_instead_of_site_brand(self):
        from bs4 import BeautifulSoup
        soup=BeautifulSoup('<meta property="og:title" content="BBC Learning English - 6 Minute English / Why do people buy gold?"><h1>Learning English</h1>','html.parser')
        self.assertEqual(bbc.extract_title(soup),'Why do people buy gold?')

    def _cached_episode(self,temp):
        base=Path(temp);audio=base/'source.mp3';audio.write_bytes(b'cached source')
        source={'id':'resume-test','title':'Test','audio_path':str(audio),'source_url':'local import','transcript':'Hello.'}
        folder=base/source['id'];folder.mkdir()
        word={'id':'w0','index':0,'sentence_id':'s0','text':'Hello.','start_ms':100,'end_ms':600,'gloss_zh':'你好','probability':1}
        sentence={'id':'s0','start_ms':100,'end_ms':600,'text':'Hello.','translation':'你好。','speaker':'','word_ids':['w0'],'card_ids':[]}
        package={'version':3,'episode':{'id':source['id'],'title':'Test','title_zh':'测试','source_url':'local import','audio_file':'audio.mp3','fingerprint':'','duration_s':1},'sentences':[sentence],'words':[word],'cards':[],'quality':{},'review':{'recognized_transcript':'Hello.'},'render':{'status':'pending'},'processing':{}}
        (folder/'episode.checkpoint.json').write_text(json.dumps(package),encoding='utf-8')
        return base,source,folder

    def test_checkpoint_resume_uses_cached_duration_without_retranscribing(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
            base,source,folder=self._cached_episode(temp)
            def layout(p,f):p['layout']={'version':1,'pages':[{'id':'p0','sentence_id':'s0','start_ms':100,'end_ms':600,'lines':[['w0']],'translation':'你好。'}]}
            with patch.object(pipeline,'output_root',lambda:base),patch.object(pipeline,'transcribe_words') as transcribe,patch.object(pipeline,'_annotate'),patch('app.teaching_review.review_teaching'),patch('app.layout.prepare_layout',layout):
                package=pipeline._create_episode(source,base,lambda *a:None,False,prepare_only=True)
                self.assertEqual(package['episode']['duration_s'],1)
                self.assertEqual(package['render']['status'],'prepared');transcribe.assert_not_called()

    def test_render_failure_is_not_ready_and_keeps_previous_video(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
            base,source,folder=self._cached_episode(temp)
            (folder/'lesson.mp4').write_bytes(b'previous valid video')
            def layout(p,f):p['layout']={'version':1,'pages':[{'id':'p0','sentence_id':'s0','start_ms':100,'end_ms':600,'lines':[['w0']],'translation':'你好。'}]}
            with patch.object(pipeline,'output_root',lambda:base),patch.object(pipeline,'_annotate'),patch('app.teaching_review.review_teaching'),patch('app.layout.prepare_layout',layout),patch('app.renderer.render_video',side_effect=RuntimeError('render failed')):
                with self.assertRaises(RuntimeError):pipeline._create_episode(source,base,lambda *a:None,False)
            package=json.loads((folder/'episode.json').read_text(encoding='utf-8'))
            self.assertNotEqual(package['render']['status'],'ready')
            self.assertEqual((folder/'lesson.mp4').read_bytes(),b'previous valid video')

    def test_srt_millisecond_precision(self):
        self.assertEqual(_timestamp(1.234),'00:00:01,234')
        self.assertEqual(_timestamp(61.987),'00:01:01,987')
        self.assertEqual(_timestamp(1.234,True),'0:00:01.23')

    def test_pdf_headers_and_speakers_not_treated_as_audio(self):
        raw='BBC LEARNING ENGLISH\n6 Minute English\nTopic\nThis is not a word-for-word transcript.\nNeil\nWe co-opt parts.\nBeth\nYes, 1.3 million.\nNeil\nGoodbye!\nBeth\nBye.\nVOCABULARY\nco-opt\nborrow parts'
        self.assertEqual(spoken_text(raw),'We co-opt parts. Yes, 1.3 million. Goodbye! Bye.')
        self.assertIn('co-opt',official_vocabulary(raw))
        words=[{'text':'co','start_ms':0,'end_ms':100},{'text':'-opt','start_ms':100,'end_ms':300},{'text':'1','start_ms':500,'end_ms':700},{'text':'.3','start_ms':700,'end_ms':900}]
        fixed,changes=normalize_orthography(words,raw)
        self.assertEqual([w['text'] for w in fixed],['co-opt','1.3'])
        self.assertEqual((fixed[0]['start_ms'],fixed[0]['end_ms']),(0,300))
        self.assertEqual(len(changes),2)

    def test_provider_retries_rate_limit_and_caches_valid_result(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
            response=io.BytesIO(json.dumps({'choices':[{'message':{'content':'{"ok":true}'},'finish_reason':'stop'}]}).encode())
            failure=urllib.error.HTTPError('http://127.0.0.1',429,'rate limit',{},io.BytesIO(b'private'))
            with patch.object(providers,'get_settings',lambda:{'model':'test/model','base_url':'http://127.0.0.1:11434/v1'}),patch.object(providers,'read_api_key',lambda:'test-key'),patch.object(providers.time,'sleep'),patch.object(providers.urllib.request,'urlopen',side_effect=[failure,response]) as request:
                self.assertEqual(providers.ask_deepseek('JSON','test',cache_dir=Path(temp)),{'ok':True})
                self.assertEqual(providers.ask_deepseek('JSON','test',cache_dir=Path(temp)),{'ok':True})
                self.assertEqual(request.call_count,2)

    def test_failed_api_validation_preserves_working_settings_and_key(self):
        class Reply:
            def _json(self,status,value):return status,value
        with patch.object(server,'get_settings',lambda:{'model':'old','base_url':'http://127.0.0.1:11434/v1'}),patch.object(server,'save_settings') as save,patch.object(server,'save_api_key') as key,patch.object(server,'ask_deepseek',side_effect=providers.ProviderError('offline')):
            with self.assertRaises(server.APIError) as failure:
                server.Handler._save_settings(Reply(),{'model':'group/new-model','api_key':'replacement','test':True})
            self.assertEqual(failure.exception.status,502);save.assert_not_called();key.assert_not_called()

    def test_duplicate_episode_does_not_start_second_producer(self):
        episode='ep-991231'
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp,patch.object(storage,'DB_PATH',Path(temp)/'jobs.sqlite3'):
            with patch.object(server,'_spawn') as start:
                first=server.create_job('https://www.bbc.co.uk/learningenglish/features/6-minute-english_2099/'+episode)
                second=server.create_job('https://www.bbc.co.uk/learningenglish/features/6-minute-english_2099/'+episode)
                self.assertEqual(first,second);self.assertEqual(start.call_count,1)
            server._set_job(first,status='failed')
            self.assertEqual(next(j['status'] for j in server.list_jobs() if j['id']==first),'failed')

    def test_partial_download_never_overwrites_valid_asset(self):
        class Interrupted:
            url='https://downloads.bbc.co.uk/test.mp3';headers={}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def raise_for_status(self):pass
            def iter_content(self,*args):
                yield b'partial'
                raise ConnectionError('interrupted')
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
            target=Path(temp)/'audio.mp3';target.write_bytes(b'original')
            with patch.object(bbc.requests,'get',return_value=Interrupted()):
                with self.assertRaises(ConnectionError):bbc._download(Interrupted.url,target)
            self.assertEqual(target.read_bytes(),b'original')

    def test_shared_layout_drives_both_subtitle_languages(self):
        source={'episode':{'duration_s':12},'words':[
            {'id':'w0','text':'Hello.','start_ms':8260,'end_ms':9000},
            {'id':'w1','text':'Welcome.','start_ms':9500,'end_ms':10500}],
            'sentences':[{'id':'s0','translation':'你好。欢迎。'}],
            'layout':{'pages':[
                {'id':'p0','sentence_id':'s0','start_ms':8260,'end_ms':9000,'lines':[['w0']],'translation':'你好。'},
                {'id':'p1','sentence_id':'s0','start_ms':9500,'end_ms':10500,'lines':[['w1']],'translation':'欢迎。'}]}}
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
            en,zh,ass=make_captions(source,Path(temp))
            en_entries=en.read_text(encoding='utf-8-sig').strip().split('\n\n')
            zh_entries=zh.read_text(encoding='utf-8-sig').strip().split('\n\n')
            self.assertEqual(len(en_entries),len(source['layout']['pages']))
            self.assertEqual(len(en_entries),len(zh_entries))
            self.assertEqual(en_entries[0].splitlines()[1],zh_entries[0].splitlines()[1])
            self.assertIn('00:00:08,260',en_entries[0])
            self.assertGreater(ass.stat().st_size,1000)

    def test_audio_range_and_head_support(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
            folder=Path(temp);(folder/'sample').mkdir();(folder/'sample/audio.mp3').write_bytes(b'0123456789')
            with patch.object(server,'output_root',lambda:folder):
                http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
                worker=threading.Thread(target=http.serve_forever,daemon=True);worker.start()
                try:
                    url=f'http://127.0.0.1:{http.server_address[1]}/files/sample/audio.mp3'
                    with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=3-5'})) as response:
                        self.assertEqual(response.status,206);self.assertEqual(response.read(),b'345');self.assertEqual(response.headers['Content-Range'],'bytes 3-5/10')
                    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD')) as response:
                        self.assertEqual(response.headers['Content-Length'],'10');self.assertEqual(response.read(),b'')
                    with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=20-30'}))
                    self.assertEqual(error.exception.code,416)
                finally:http.shutdown();http.server_close()


if __name__=='__main__':unittest.main(verbosity=2)
