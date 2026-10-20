from __future__ import annotations
import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.ctc_repair import build_measured_transaction, repair_with_ctc


def words():
    return [{'id':f'w{i}','text':text,'start_ms':i*300,'end_ms':i*300+200,'probability':.5} for i,text in enumerate(['we','can','study','each','word','using','clear','anchors','today'])]


def observations(source,offset=5):
    aligned=[{'index':i,'text':w['text'],'measured_start_ms':w['start_ms']+offset,'measured_end_ms':w['end_ms']+offset,'raw_path_score_not_accuracy':.95} for i,w in enumerate(source)]
    return [{'status':'measured','aligned_context':copy.deepcopy(aligned)},{'status':'measured','aligned_context':copy.deepcopy(aligned)}]


class CtcRepairTests(unittest.TestCase):
    def test_transaction_keeps_both_anchors_and_original_guards(self):
        source=words();obs=observations(source)
        transaction=build_measured_transaction(source,4,obs)
        self.assertEqual(transaction['status'],'measured_transaction')
        self.assertEqual([p['word_index'] for p in transaction['patch']],[3,4,5])
        self.assertEqual(transaction['patch'][1]['original_start_ms'],1200)
        self.assertEqual(source,words())

    def test_low_path_is_rejected_even_when_both_windows_agree(self):
        source=words();obs=observations(source)
        for m in obs:m['aligned_context'][4]['raw_path_score_not_accuracy']=.01
        transaction=build_measured_transaction(source,4,obs)
        self.assertEqual(transaction['status'],'rejected')
        self.assertEqual(transaction['patch'],[])

    def test_unstable_neighborhood_is_not_repaired(self):
        source=words();obs=observations(source)
        obs[1]['aligned_context'][3]['measured_start_ms']+=100
        self.assertEqual(build_measured_transaction(source,4,obs)['status'],'rejected')

    def test_joint_transaction_expands_instead_of_clipping_real_measurement(self):
        source=words();obs=observations(source)
        # Measured left anchor starts before an unchanged neighbor ends.
        for m in obs:
            m['aligned_context'][3]['measured_start_ms']=750
            m['aligned_context'][2]['measured_end_ms']=745
        transaction=build_measured_transaction(source,4,obs)
        self.assertEqual(transaction['status'],'measured_transaction')
        self.assertEqual(transaction['patch'][0]['word_index'],2)
        self.assertEqual(transaction['patch'][1]['start_ms'],750)

    def test_expansion_into_low_score_word_is_rejected(self):
        source=words();obs=observations(source)
        for m in obs:
            m['aligned_context'][3]['measured_start_ms']=750
            m['aligned_context'][2]['raw_path_score_not_accuracy']=.1
        self.assertEqual(build_measured_transaction(source,4,obs)['status'],'rejected')

    def test_repeated_text_requires_independent_asr_consensus(self):
        source=words();source[3]['text']='word';obs=observations(source)
        transaction=build_measured_transaction(source,4,obs)
        self.assertEqual(transaction['status'],'rejected')
        self.assertTrue(any('重复词' in r for r in transaction['reasons']))

    def test_numeric_or_nonunique_context_is_not_assumed_spoken(self):
        source=words();source[4]['text']='3.14';obs=observations(source)
        self.assertEqual(build_measured_transaction(source,4,obs)['status'],'rejected')
        source=words();source+=copy.deepcopy(source);obs=observations(source)
        self.assertEqual(build_measured_transaction(source,4,obs)['status'],'rejected')

    def test_missing_model_preserves_every_word_and_never_downloads(self):
        source=words();source[4]['end_ms']=source[4]['start_ms']
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as cache:
            result,evidence=repair_with_ctc(ROOT/'work/nonexistent-audio.wav',source,[4],lambda _:None,cache_dir=cache)
            self.assertEqual(evidence['status'],'model_not_cached')
            self.assertEqual(result,source)
            self.assertIsNot(result,source)
            self.assertEqual(list(Path(cache).iterdir()),[])

    def test_unknown_dispute_locator_is_rejected_without_guessing(self):
        with self.assertRaises(ValueError):
            repair_with_ctc('missing.wav',words(),[{'word_id':'unknown'}],cache_dir=ROOT/'work/no-cache')

if __name__=='__main__':unittest.main()
