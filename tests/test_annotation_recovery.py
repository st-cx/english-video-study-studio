import copy
import unittest
from unittest.mock import patch
from app.pipeline import _annotate
from app.providers import ProviderError


def package():
    return {'episode':{'title':'Example','duration_s':2},'source':{'kind':'local'},
      'sentences':[{'id':'s0','text':'Learn slowly.','word_ids':['w0','w1'],'start_ms':100,'end_ms':900,'translation':'','card_ids':[]}],
      'words':[{'id':f'w{i}','sentence_id':'s0','index':i,'text':t,'start_ms':100+i*400,'end_ms':400+i*400,'gloss_zh':''} for i,t in enumerate(['Learn','slowly.'])],'cards':[]}

def answer():
    return {'title_zh':'示例','translations':[{'id':'s0','zh':'慢慢学习。'}],
      'glosses':[{'word_id':'w0','zh':'学习'},{'word_id':'w1','zh':'慢慢地'}],'phrases':[],'cards':[]}


class AnnotationRecoveryTests(unittest.TestCase):
    def test_ungrounded_optional_card_is_rejected_after_bounded_repair(self):
        p=package();result=answer()
        result['cards']=[{'title':'invented','sentence_ids':['s0'],'example':'This was never spoken.'}]
        with patch('app.pipeline.ask_deepseek',side_effect=[copy.deepcopy(result),copy.deepcopy(result)]),patch('app.presentation.translate_excerpts'):
            _annotate(p,'',lambda *a:None)
        self.assertEqual(p['cards'],[])
        self.assertTrue(p['processing']['omitted_card_suggestions'])
        self.assertEqual(p['sentences'][0]['translation'],'慢慢学习。')

    def test_long_and_missing_glosses_are_repaired_with_exact_requested_ids(self):
        p=package();result=answer();result['glosses']=[{'word_id':'w0','zh':'长'*71}]
        responses=[result,{'glosses':[{'word_id':'foreign','zh':'不应写入'}]},
                   {'glosses':[{'word_id':'w0','zh':'学习'},{'word_id':'w1','zh':'慢慢地'}]}]
        with patch('app.pipeline.ask_deepseek',side_effect=responses),patch('app.presentation.translate_excerpts'):
            _annotate(p,'',lambda *a:None)
        self.assertEqual([w['gloss_zh'] for w in p['words']],['学习','慢慢地'])
        self.assertEqual([(w['start_ms'],w['end_ms']) for w in p['words']],[(100,400),(500,800)])

    def test_failed_gloss_repairs_still_block_and_never_invent_meanings(self):
        p=package();result=answer();result['glosses']=[]
        with patch('app.pipeline.ask_deepseek',side_effect=[result,{}, {}, {}]):
            with self.assertRaises(ProviderError):_annotate(p,'',lambda *a:None)
        self.assertTrue(all(w['gloss_zh']=='' for w in p['words']))

    def test_missing_translation_gets_a_bounded_targeted_repair(self):
        p=package();result=answer();result['translations']=[]
        with patch('app.pipeline.ask_deepseek',side_effect=[result,{'translations':[{'id':'s0','zh':'慢慢学习。'}]}]),patch('app.presentation.translate_excerpts'):
            _annotate(p,'',lambda *a:None)
        self.assertEqual(p['sentences'][0]['translation'],'慢慢学习。')
