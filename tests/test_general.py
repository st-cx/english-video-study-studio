import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from app.general import _chapters,_merge_chapter,_chapter_package,_signature,PROCESSING_VERSION,_transcribe_chunks,_alignment_implementation_fingerprint
from app.pipeline import _quality
from app.privacy import public_package
from app.renderer import export_video

def fixture():
    return {'version':4,'episode':{'id':'fixture','title':'Example','duration_s':610},'source':{'kind':'local'},'sentences':[{'id':f's{i}','text':text,'translation':'','start_ms':start,'end_ms':start+1000,'word_ids':[f'w{i}'],'card_ids':[]} for i,(start,text) in enumerate([(0,'Hello.'),(300100,'Again.'),(605000,'Goodbye.')])],'words':[{'id':f'w{i}','sentence_id':f's{i}','index':0,'text':text,'start_ms':start,'end_ms':start+1000,'probability':.9,'gloss_zh':''} for i,(start,text) in enumerate([(0,'Hello.'),(300100,'Again.'),(605000,'Goodbye.')])],'cards':[],'quality':{},'review':{},'render':{},'processing':{}}

class GeneralTests(unittest.TestCase):
    def test_missing_reference_is_not_a_perfect_match(self):
        result=_quality(fixture(),'')
        self.assertIsNone(result['source_similarity']);self.assertFalse(result['reference_available'])

    def test_chapters_follow_measured_sentence_boundaries_and_keep_short_tail(self):
        p=fixture();cs=_chapters(p['sentences'],610)
        self.assertEqual([c['start_ms'] for c in cs],[0,300100]);self.assertEqual(cs[-1]['end_ms'],610000)
        for s in p['sentences']:self.assertEqual(sum(c['start_ms']<=s['start_ms']<c['end_ms'] for c in cs),1)

    def test_embedded_chapter_times_can_be_strings_without_empty_intro(self):
        p=fixture();cs=_chapters(p['sentences'],610,[{'start_time':'0','title':'Introduction'},{'start_time':'300','title':'Practice'}])
        self.assertEqual(len(cs),2);self.assertEqual(cs[0]['title'],'Introduction');self.assertEqual(cs[1]['start_ms'],300100);self.assertEqual(cs[1]['title'],'Practice')

    def test_merge_has_unique_card_ids_and_preserves_other_chapters(self):
        p=fixture();cs=_chapters(p['sentences'],610);part=_chapter_package(p,cs[0]);part['cards']=[{'id':'c0','sentence_ids':['s0'],'start_ms':0,'end_ms':610000}];part['sentences'][0]['card_ids']=['c0'];part['layout']={'version':1,'pages':[{'id':'p0','sentence_id':'s0','start_ms':0}]}
        _merge_chapter(p,part,cs[0]);self.assertEqual(p['cards'][0]['id'],'ch001-c0');self.assertEqual(p['cards'][0]['end_ms'],300100);self.assertEqual(p['sentences'][0]['card_ids'],['ch001-c0']);self.assertEqual(len(p['words']),3)

    def test_export_blocks_only_intersecting_unapproved_chapters(self):
        p=fixture();p['chapters']=[{'id':'one','start_ms':0,'end_ms':300000,'status':'ready'},{'id':'two','start_ms':300000,'end_ms':610000,'status':'review'}]
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);path=folder/'episode.json';path.write_text(json.dumps(p),encoding='utf-8')
            with patch('app.renderer._render_video',return_value=folder/'clip.mp4') as render:
                export_video(path,folder,lambda *a:None,0,1000);render.assert_called_once()
                with self.assertRaisesRegex(ValueError,'待检查'):export_video(path,folder,lambda *a:None,299999,300001)

    def test_cached_speech_resumes_without_loading_model(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d);speech=f/'.speech';speech.mkdir();(speech/'cuts.json').write_text(json.dumps({'duration':2,'cuts':[0,2]}))
            from app.speech_runtime import speech_runtime_plan
            runtime={**{k:v for k,v in speech_runtime_plan().items() if k!='path'},'alignment_implementation_sha256':_alignment_implementation_fingerprint()}
            cached={'key':_signature([PROCESSING_VERSION,'abc',0,2,runtime]),'runtime':runtime,'words':[{'text':'Hello.','start_ms':100,'end_ms':800,'probability':.9}],'backend':'CPU','language':'en','language_probability':1,'suspects':[],'alignment':{},'start_ms':0,'end_ms':2000}
            (speech/'0000.json').write_text(json.dumps(cached))
            with patch('faster_whisper.WhisperModel',side_effect=AssertionError('must not load')):
                words,_=_transcribe_chunks(f/'audio.mp3',2,f,'abc',lambda *a:None)
            self.assertEqual(words[0]['text'],'Hello.')

    def test_portable_package_keeps_learning_data_without_runtime_paths(self):
        p=fixture();p['processing']['internal_path']='private';p['review']['draft_path']='private';p['api_key']='not-real'
        public=public_package(p)
        self.assertNotIn('processing',public);self.assertNotIn('api_key',public);self.assertNotIn('draft_path',public['review']);self.assertEqual(public['words'],p['words']);self.assertIn('processing',p)

if __name__=='__main__':unittest.main()
