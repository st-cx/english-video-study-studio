import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app.pipeline import _realign_corrected

class CorrectionTests(unittest.TestCase):
    def test_invalid_optional_phrase_is_not_taught_after_bounded_repair(self):
        from app.pipeline import _annotate
        p={'episode':{'title':'Example','duration_s':10},'source':{'kind':'local'},'sentences':[{'id':'s0','text':'Develop small habits.','word_ids':['w0','w1','w2'],'start_ms':0,'end_ms':1000,'translation':'','card_ids':[]}],'words':[{'id':f'w{i}','sentence_id':'s0','index':i,'text':text,'start_ms':i*100,'end_ms':i*100+90,'gloss_zh':''} for i,text in enumerate(['Develop','small','habits.'])],'cards':[]}
        result={'title_zh':'示例','translations':[{'id':'s0','zh':'养成小习惯。'}],'glosses':[{'word_id':f'w{i}','zh':zh} for i,zh in enumerate(['养成','小的','习惯'])],'phrases':[{'word_ids':['w0','w2'],'zh':'养成习惯','study':True}],'cards':[]}
        with patch('app.pipeline.ask_deepseek',side_effect=[copy.deepcopy(result),copy.deepcopy(result)]):_annotate(p,'',lambda *a:None)
        self.assertEqual(p['phrases'],[]);self.assertEqual(p['sentences'][0]['translation'],'养成小习惯。')
        self.assertTrue(p['processing']['omitted_phrase_suggestions']);self.assertEqual(p['words'][0]['end_ms'],90)

    def test_edit_realigns_changed_word_without_regressing_unchanged_span(self):
        p={'episode':{'duration_s':1},'words':[{'id':'old0','text':'Hello','start_ms':100,'end_ms':300,'probability':.95},{'id':'old1','text':'Earth.','start_ms':300,'end_ms':700,'probability':.9}],'sentences':[],'cards':[]}
        # The new full alignment regresses the first word to zero length. It
        # must preserve the old, unchanged measured interval while accepting
        # the genuinely new measured interval for the edited word.
        aligned=SimpleNamespace(segments=[SimpleNamespace(start=0,words=[SimpleNamespace(word='Hello',start=0,end=0,probability=None),SimpleNamespace(word='world.',start=.32,end=.72,probability=.9)])])
        model=SimpleNamespace(align=lambda *a,**k:aligned)
        with patch('stable_whisper.load_faster_whisper',return_value=model),patch('app.pipeline._local_whisper_path',return_value='unused'):
            _realign_corrected(p,Path('unused.wav'),'Hello world.',lambda *a:None)
        self.assertEqual([(w['start_ms'],w['end_ms']) for w in p['words']],[(100,300),(320,720)])
        self.assertEqual(p['words'][1]['text'],'world.')
        self.assertNotIn('Earth.',' '.join(w['text'] for w in p['words']))

if __name__=='__main__':unittest.main()
