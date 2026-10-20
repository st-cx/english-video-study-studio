"""Acoustic boundary regressions; no speech model is downloaded by these tests."""
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from app.alignment import _compatible_patch, _contextual_matches, _probe_window, refine_word_boundaries,align_words


def word(text,start,end,**extra):return {'text':text,'start_ms':start,'end_ms':end,**extra}


class AlignmentPrecisionTests(unittest.TestCase):
    def test_individual_asr_fallback_cannot_reverse_adjacent_zero_word(self):
        # Actual long-lesson shape: I is zero at 172720; ASR puts think earlier.
        source=[word('programming,',172060,172500),word('I',172720,172720),word('think',172720,172720),word('the',172720,172780)]
        self.assertFalse(_compatible_patch(source,{2:word('think',172510,172670)}))

    def test_context_can_remeasure_connected_cluster_as_one_transaction(self):
        source=[word('programming,',172060,172500),word('I',172720,172720),word('think',172510,172670),word('the',172720,172780)]
        patch={0:word('programming,',172060,172350),1:word('I',172350,172450),2:word('think',172450,172670)}
        self.assertTrue(_compatible_patch(source,patch))
        self.assertEqual(source[1]['end_ms'],172720)

    def test_local_same_spelling_without_context_is_not_a_word_match(self):
        expected=[word('writing',138540,138760),word('this',138760,139080),word('I',139140,139140),word('was',139070,139150),word('trying',139140,139260)]
        observed=[word('I',139000,139100),word('really',139100,139400),word('disagree',139400,139900)]
        self.assertNotIn(2,_contextual_matches(expected,observed))

    def test_same_context_at_distant_occurrence_is_not_accepted(self):
        expected=[word('when',1000,1100),word('you',1100,1100),word('know',1100,1500),word('this',1500,1800)]
        observed=[word(w['text'],w['start_ms']+6000,w['end_ms']+6100) for w in expected]
        self.assertEqual(_contextual_matches(expected,observed),{})

    def test_context_matching_preserves_measured_interval_without_redistribution(self):
        expected=[word('writing',138540,138760),word('this',138760,139080),word('I',139140,139140),word('was',139070,139150),word('trying',139140,139260)]
        observed=[word('writing',138520,138750),word('this',138750,138990),word('I',138990,139030),word('was',139030,139150),word('trying',139150,139290)]
        matches=_contextual_matches(expected,observed)
        self.assertEqual((matches[2]['start_ms'],matches[2]['end_ms']),(138990,139030))

    def test_probe_includes_both_disputed_observations(self):
        expected=[word('like',90040,90140)]
        lo,hi=_probe_window(expected,[0],{0:word('like',84190,84470)})
        self.assertLessEqual(lo,81190)
        self.assertGreaterEqual(hi,93140)

    def test_zero_duration_local_observation_stays_unusable(self):
        expected=[word('when',1000,1100),word('you',1100,1100),word('know',1100,1500),word('this',1500,1800)]
        self.assertNotIn(1,_contextual_matches(expected,expected))

    def test_incomplete_patch_cannot_hide_existing_overlap(self):
        source=[word('sort',2504820,2505620),word('of',2505180,2505200),word('the',2505200,2505440)]
        self.assertFalse(_compatible_patch(source,{1:word('of',2505180,2505260)}))

    def test_realignment_applies_only_observed_spans_and_repairs_neighbour_together(self):
        source=[word('writing',138540,138760),word('this',138760,139080),word('I',139140,139140),word('was',139070,139150),word('trying',139140,139260)]
        observed=[word('writing',138520,138750,probability=.9),word('this',138750,138990,probability=.9),word('I',138990,139030,probability=.9),word('was',139030,139150,probability=.9),word('trying',139150,139290,probability=.9)]
        with patch('app.alignment._acoustic_observation',return_value=observed):
            output,report=refine_word_boundaries(Path('unused.mp3'),source,'unused',lambda *a:None,probe_model=object())
        self.assertEqual((output[2]['start_ms'],output[2]['end_ms']),(138990,139030))
        self.assertEqual((output[1]['start_ms'],output[1]['end_ms']),(138750,138990))
        self.assertEqual((source[2]['start_ms'],source[2]['end_ms']),(139140,139140))
        self.assertTrue(report['alignment_window_audits'][0]['accepted'])

    def test_unsupported_local_token_leaves_dispute_and_old_timestamp(self):
        source=[word('writing',138540,138760),word('this',138760,139080),word('I',139140,139140),word('was',139070,139150),word('trying',139140,139260)]
        dispute={'text':'I','aligned_ms':[139140,139140],'asr_ms':[139000,139040]}
        with patch('app.alignment._acoustic_observation',return_value=[word('I',139000,139040,probability=.99)]):
            output,report=refine_word_boundaries(Path('unused.mp3'),source,'unused',lambda *a:None,disputes=[dispute],probe_model=object())
        self.assertEqual(output,source)
        self.assertEqual(len(report['alignment_disputes']),1)
        self.assertFalse(report['alignment_window_audits'][0]['accepted'])

    def test_wide_disagreement_is_left_for_review_without_truncating_window(self):
        source=[word('like',90040,90140)]
        dispute={'text':'like','aligned_ms':[90040,90140],'asr_ms':[40000,41000]}
        with patch('app.alignment._acoustic_observation') as probe:
            output,report=refine_word_boundaries(Path('unused.mp3'),source,'unused',lambda *a:None,disputes=[dispute],probe_model=object())
        probe.assert_not_called();self.assertEqual(output,source)
        self.assertEqual(len(report['alignment_disputes']),1)
        self.assertEqual(report['alignment_window_audits'][0]['method'],'unresolved_window_exceeds_20_seconds')

    def test_initial_asr_fallback_cannot_commit_invalid_order_before_window_review(self):
        aligned=[word('programming,',172060,172500),word('I',172720,172720),word('think',172720,172720),word('the',172720,172780)]
        recognized=[dict(w,probability=.99) for w in aligned];recognized[2].update(start_ms=172510,end_ms=172670)
        segment=SimpleNamespace(start=171.6,words=[SimpleNamespace(word=w['text'],start=w['start_ms']/1000,end=w['end_ms']/1000) for w in aligned])
        model=SimpleNamespace(align=lambda *a,**kw:SimpleNamespace(segments=[segment]))
        fake_module=SimpleNamespace(load_faster_whisper=lambda *a,**kw:model)
        def keep(words_audio,words,*args,**kwargs):return words,{'alignment_repairs':[],'alignment_disputes':kwargs.get('disputes',[]),'alignment_window_audits':[]}
        with patch.dict('sys.modules',{'stable_whisper':fake_module}),patch('app.alignment.refine_word_boundaries',side_effect=keep) as refine:
            result,_=align_words(Path('unused.mp3'),recognized,'unused',lambda *a:None)
        self.assertEqual((result[2]['start_ms'],result[2]['end_ms']),(172720,172720))
        self.assertEqual(refine.call_args.args[1][1]['start_ms'],172720)

    def test_clip_local_index_is_not_used_as_global_index(self):
        words=[word('like',1000,1200),word('some',2000,2200),word('examples',3000,3200),word('like',90040,90140)]
        dispute={'word_index':0,'text':'like','aligned_ms':[90040,90140],'asr_ms':[84190,84470]}
        with patch('app.alignment._acoustic_observation',return_value=[]):
            result,report=refine_word_boundaries(Path('unused.mp3'),words,'unused',lambda *a:None,disputes=[dispute],probe_model=object())
        self.assertEqual(result,words)
        self.assertEqual(report['alignment_disputes'][0]['word_index'],3)

    def test_failed_short_probe_gets_one_wider_acoustic_observation(self):
        source=[word('writing',138540,138760),word('this',138760,139080),word('I',139140,139140),word('was',139070,139150),word('trying',139140,139260)]
        observed=[word('writing',138520,138750,probability=.9),word('this',138750,138990,probability=.9),word('I',138990,139030,probability=.9),word('was',139030,139150,probability=.9),word('trying',139150,139290,probability=.9)]
        with patch('app.alignment._acoustic_observation',side_effect=[[],observed]) as probe:
            result,report=refine_word_boundaries(Path('unused.mp3'),source,'unused',lambda *a:None,probe_model=object())
        self.assertEqual(probe.call_count,2)
        self.assertLess(probe.call_args_list[1].args[1],probe.call_args_list[0].args[1])
        self.assertEqual((result[2]['start_ms'],result[2]['end_ms']),(138990,139030))
        self.assertEqual(report['alignment_window_audits'][1]['attempt'],2)


    def test_positive_an_is_not_shifted_into_pause_by_high_probability_asr(self):
        source=[word('going',2369420,2369580),word('to',2369580,2369780),word('be',2369780,2370040),
                word('an',2371560,2371660),word('expression.',2371660,2372280)]
        observed=[dict(w,probability=.999) for w in source]
        observed[3].update(start_ms=2370060,end_ms=2370560)
        dispute={'word_index':3,'text':'an','aligned_ms':[2371560,2371660],'asr_ms':[2370040,2370620]}
        with patch('app.alignment._acoustic_observation',return_value=observed):
            result,report=refine_word_boundaries(Path('unused.mp3'),source,'unused',lambda *a:None,disputes=[dispute],probe_model=object())
        self.assertEqual((result[3]['start_ms'],result[3]['end_ms']),(2371560,2371660))
        self.assertEqual(len(report['alignment_disputes']),1)
        self.assertTrue(report['alignment_disputes'][0]['independent_acoustic_confirmation_required'])
        self.assertTrue(all(3 in audit['blocked_weak_function_word_indices'] for audit in report['alignment_window_audits']))
        self.assertEqual(report['alignment_repairs'],[])

    def test_zero_an_has_no_positive_interval_to_protect(self):
        source=[word('going',2369420,2369580),word('to',2369580,2369780),word('be',2369780,2370040),
                word('an',2371560,2371560),word('expression.',2371660,2372280)]
        observed=[dict(w,probability=.999) for w in source]
        observed[3].update(start_ms=2370060,end_ms=2370560)
        with patch('app.alignment._acoustic_observation',return_value=observed):
            result,report=refine_word_boundaries(Path('unused.mp3'),source,'unused',lambda *a:None,probe_model=object())
        self.assertEqual((result[3]['start_ms'],result[3]['end_ms']),(2370060,2370560))
        self.assertTrue(report['alignment_window_audits'][0]['accepted'])

    def test_initial_positive_weak_fallback_also_requires_independent_ctc(self):
        aligned=[word('see',1000,1200),word('an',1400,2900),word('object.',3000,3400)]
        recognized=[dict(w,probability=.999) for w in aligned]
        recognized[1].update(start_ms=2500,end_ms=2700)
        segment=SimpleNamespace(start=1.0,words=[SimpleNamespace(word=w['text'],start=w['start_ms']/1000,end=w['end_ms']/1000) for w in aligned])
        model=SimpleNamespace(align=lambda *a,**kw:SimpleNamespace(segments=[segment]))
        def keep(audio,ws,model_path,progress,raw,disputed,**kwargs):
            return ws,{'alignment_repairs':[],'alignment_disputes':disputed,'alignment_window_audits':[]}
        with patch.dict('sys.modules',{'stable_whisper':SimpleNamespace(load_faster_whisper=lambda *a,**kw:model)}),patch('app.alignment.refine_word_boundaries',side_effect=keep):
            result,report=align_words(Path('unused.mp3'),recognized,'unused',lambda *a:None)
        self.assertEqual((result[1]['start_ms'],result[1]['end_ms']),(1400,2900))
        self.assertTrue(any(d.get('reason')=='weak_function_word_large_asr_shift_requires_independent_ctc' for d in report['alignment_disputes']))

    def test_external_alignment_model_is_reused_for_refinement_without_cpu_reload(self):
        tokens=[word('Hello',100,300),word('new',300,500),word('world.',500,800)]
        recognized=[dict(w,probability=.99) for w in tokens]
        segment=SimpleNamespace(start=.1,words=[SimpleNamespace(word=w['text'],start=w['start_ms']/1000,end=w['end_ms']/1000) for w in tokens])
        shared=SimpleNamespace(align=lambda *a,**kw:SimpleNamespace(segments=[segment]))
        fake_loader=unittest.mock.Mock(side_effect=AssertionError('external model must not reload'))
        fake_module=SimpleNamespace(load_faster_whisper=fake_loader)
        def keep(audio,words,*args,**kwargs):return words,{'alignment_repairs':[],'alignment_disputes':[],'alignment_window_audits':[]}
        with patch.dict('sys.modules',{'stable_whisper':fake_module}),patch('app.alignment.refine_word_boundaries',side_effect=keep) as refine:
            result,_=align_words(Path('unused.mp3'),recognized,'unused',lambda *a:None,alignment_model=shared)
        fake_loader.assert_not_called()
        self.assertIs(refine.call_args.kwargs['probe_model'],shared)
        self.assertEqual([w['text'] for w in result],['Hello','new','world.'])


if __name__=='__main__':unittest.main()
