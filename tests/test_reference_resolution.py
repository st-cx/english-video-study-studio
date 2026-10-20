import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import reference_resolution
from app.acoustic_reference import _audio_digest, _model_fingerprint


class ReferenceResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'work')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.audio = self.folder / 'audio.mp3'; self.audio.write_bytes(b'isolated-audio')
        self.model = self.folder / 'model'; self.model.mkdir()
        (self.model / 'model.bin').write_bytes(b'isolated-weights')
        self.package = {'words': [
            {'text': 'The', 'start_ms': 0, 'end_ms': 200},
            {'text': 'elephants', 'start_ms': 200, 'end_ms': 500},
            {'text': 'have', 'start_ms': 500, 'end_ms': 700},
            {'text': 'fronts.', 'start_ms': 1000, 'end_ms': 1400}],
            'processing': {}, 'quality': {'flags': ['pending']},
            'review': {'approved': False}}
        self.reference = 'The elephants have trunks.'

    def observation(self, text='fronts'):
        return {'observation_id': 'c' * 64, 'audio_sha256': _audio_digest(self.audio),
                'model_fingerprint': _model_fingerprint(str(self.model)),
                'mode': 'independent_unprompted_asr', 'start_ms': 0, 'end_ms': 3000,
                'truncated': False, 'words': [{'text': text, 'start_ms': 1000,
                    'end_ms': 1400, 'probability': .99}]}

    def resolve(self, **kwargs):
        return reference_resolution.resolve_reference_findings(self.package,
            self.reference, self.audio, model_path=str(self.model), **kwargs)

    def test_only_current_positive_audio_evidence_resolves_editorial_substitution(self):
        before = copy.deepcopy(self.package)
        with patch.object(reference_resolution, 'observe_disputes', side_effect=AssertionError('must not run')):
            result = self.resolve(observations=[self.observation()])
        self.assertEqual(len(result['findings']), 1)
        self.assertEqual(result['unresolved'], [])
        self.assertEqual(result['resolutions'][0]['status'], 'supported_recognition')
        self.assertEqual(self.package, before)

    def test_reference_support_provides_candidate_but_keeps_the_gate_closed(self):
        result = self.resolve(observations=[self.observation('trunks')])
        self.assertEqual(len(result['unresolved']), 1)
        self.assertEqual(result['resolutions'][0]['status'], 'reference_supported')
        self.assertEqual(result['resolutions'][0]['candidate_words'][0]['start_ms'], 1000)
        self.assertFalse(self.package['review']['approved'])

    def test_changed_audio_or_model_rejects_old_observations(self):
        observation = self.observation()
        self.audio.write_bytes(b'changed-audio')
        self.assertEqual(len(self.resolve(observations=[observation])['unresolved']), 1)
        observation = self.observation()
        (self.model / 'model.bin').write_bytes(b'changed-model-weights')
        self.assertEqual(len(self.resolve(observations=[observation])['unresolved']), 1)

    def test_saved_verdict_or_approval_without_current_observation_is_not_a_bypass(self):
        self.package['review']['approved'] = True
        self.package['processing']['reference_resolutions'] = [{'status': 'supported_recognition'}]
        result = self.resolve()
        self.assertEqual(len(result['unresolved']), 1)
        self.assertEqual(result['resolutions'][0]['status'], 'unresolved')

    def test_observation_failure_is_sanitized_and_keeps_current_findings(self):
        with patch.object(reference_resolution, 'observe_disputes', side_effect=RuntimeError('C:/private/path key=private-test')):
            result = self.resolve(observe=True)
        self.assertEqual(len(result['unresolved']), 1)
        self.assertTrue(result['observation_error'])
        self.assertNotIn('private-test', json.dumps(result))
        self.assertNotIn('C:/private', json.dumps(result))

    def test_explicit_observation_uses_selected_device_then_reassesses(self):
        with patch.object(reference_resolution, 'observe_disputes', return_value=[self.observation()]) as observe:
            result = self.resolve(observe=True, device='cuda', compute_type='int8_float16')
        self.assertEqual(result['unresolved'], [])
        self.assertEqual(observe.call_args.kwargs['device'], 'cuda')
        self.assertEqual(observe.call_args.kwargs['compute_type'], 'int8_float16')

    def test_pause_from_progress_is_not_swallowed_as_model_failure(self):
        class Paused(RuntimeError): pass
        def progress(*args): raise Paused()
        def observe(*args, **kwargs): args[3]('reference', 74, 'test')
        with patch.object(reference_resolution, 'observe_disputes', side_effect=observe):
            with self.assertRaises(Paused): self.resolve(observe=True, progress=progress)

    def test_chunk_local_alignment_disputes_remain_global_and_unresolved(self):
        self.package['processing']['speech_chunks'] = [{'start_ms': 300000, 'end_ms': 600000,
            'alignment': {'alignment_disputes': [{'text': 'first',
                'aligned_ms': [1000, 1200], 'asr_ms': [1300, 1500], 'resolved': True}]}}]
        before = copy.deepcopy(self.package)
        disputes = reference_resolution.unresolved_alignment(self.package)
        self.assertEqual(len(disputes), 1)
        self.assertEqual((disputes[0]['start_ms'], disputes[0]['end_ms']), (301000, 301500))
        self.assertEqual(self.package, before)


    def test_global_media_chunk_disputes_are_not_offset_twice(self):
        self.package['processing']['speech_chunks'] = [{'start_ms': 300000,
            'clip_start_ms': 298500, 'end_ms': 600000, 'time_basis': 'media',
            'alignment': {'alignment_disputes': [{'text': 'first',
                'aligned_ms': [301000, 301200], 'asr_ms': [301300, 301500]}]}}]
        disputes = reference_resolution.unresolved_alignment(self.package)
        self.assertEqual((disputes[0]['start_ms'], disputes[0]['end_ms']), (301000, 301500))

    def test_local_disputes_use_actual_clip_start_not_nominal_chunk_start(self):
        self.package['processing']['speech_chunks'] = [{'start_ms': 300000,
            'clip_start_ms': 298500, 'end_ms': 600000, 'time_basis': 'local',
            'alignment': {'alignment_disputes': [{'text': 'first',
                'aligned_ms': [2500, 2700], 'asr_ms': [2800, 3000]}]}}]
        disputes = reference_resolution.unresolved_alignment(self.package)
        self.assertEqual((disputes[0]['start_ms'], disputes[0]['end_ms']), (301000, 301500))



    def omission_evidence(self):
        from app.sync_audit import digest
        from app.acoustic_reference import reference_issue_key, reference_words_fingerprint
        self.package['words'] = [{'text': text, 'start_ms': index * 200, 'end_ms': index * 200 + 180}
            for index, text in enumerate('the lecturer you know explains useful things today'.split())]
        self.reference = 'the lecturer explains useful things today'
        issue = reference_resolution.reference_issues(self.package['words'], self.reference)[0]
        observed = self.observation()
        observed['words'] = [{**word, 'probability': .72} for word in self.package['words']]
        windows = []
        for start, end in ((0, 2500), (0, 4500)):
            def candidate(name, text, tokens, nll):
                from app.sync_audit import normalize_ctc
                tokens = len(normalize_ctc(text))
                return {'id': name, 'text': text, 'status': 'measured', 'frame_count': 300,
                    'target_token_count': tokens, 'negative_log_likelihood': nll,
                    'nll_per_frame': nll / 300, 'nll_per_target_token': nll / tokens}
            actual = ' '.join(word['text'] for word in self.package['words'])
            windows.append({'status': 'measured', 'clip_start_ms': start, 'clip_end_ms': end,
                'clip_sha256': 'a' * 64, 'comparison': {'comparable': True, 'edge_wildcards': False,
                'algorithm': 'normalized-logsoftmax-ctc-full-context-nll-v1', 'frame_count': 300,
                'observations': [candidate('actual_recognized_context', actual, 60, 30),
                    candidate('reference_edited_context', self.reference, 50, 80)]}})
        ctc = {'audio_sha256': _audio_digest(self.audio), 'model_sha256': 'b' * 64,
            'input_words_sha256': reference_words_fingerprint(self.package['words']), 'input_words_fingerprint_version': 'text-time-1',
            'source': 'independent_full_context_ctc_likelihood', 'rows': [{**issue,
                'issue_key': reference_issue_key(issue), 'global_word_indices': [2, 3],
                'windows': windows}]}
        return observed, ctc

    def test_dual_ctc_and_unique_continuous_asr_support_editorial_omission(self):
        observed, ctc = self.omission_evidence()
        result = self.resolve(observations=[observed], ctc_observations=ctc)
        self.assertEqual(result['unresolved'], [])
        self.assertEqual(result['resolutions'][0]['mode'], 'independent_asr_and_dual_window_ctc')
        self.assertGreaterEqual(result['resolutions'][0]['confidence_min'], .65)

    def test_ctc_disagreement_stale_words_or_one_window_cannot_resolve(self):
        observed, ctc = self.omission_evidence()
        variants = []
        one = copy.deepcopy(ctc); one['rows'][0]['windows'].pop(); variants.append(one)
        stale = copy.deepcopy(ctc); stale['input_words_sha256'] = 'stale'; variants.append(stale)
        negative = copy.deepcopy(ctc)
        negative['rows'][0]['windows'][0]['comparison']['observations'][0]['nll_per_target_token'] = 10
        variants.append(negative)
        wrong_context = copy.deepcopy(ctc)
        wrong_context['rows'][0]['windows'][0]['comparison']['observations'][0]['text'] = 'a different transcript'
        variants.append(wrong_context)
        for value in variants:
            with self.subTest(value=value):
                self.assertEqual(len(self.resolve(observations=[observed], ctc_observations=value)['unresolved']), 1)

    def test_ctc_without_asr_or_with_low_or_ambiguous_asr_stays_unresolved(self):
        observed, ctc = self.omission_evidence()
        self.assertEqual(len(self.resolve(observations=[], ctc_observations=ctc)['unresolved']), 1)
        observed['words'][2]['probability'] = .64
        self.assertEqual(len(self.resolve(observations=[observed], ctc_observations=ctc)['unresolved']), 1)
        observed, ctc = self.omission_evidence()
        observed['words'] += copy.deepcopy(observed['words'])
        self.assertEqual(len(self.resolve(observations=[observed], ctc_observations=ctc)['unresolved']), 1)



    def test_ctc_binds_text_and_time_but_not_generated_ids_or_glosses(self):
        observed, ctc = self.omission_evidence()
        for word in self.package['words']:
            word.update(id='new-id', gloss='新增词义', sentence_id='s1')
        self.assertEqual(self.resolve(observations=[observed], ctc_observations=ctc)['unresolved'], [])
        self.package['words'][2]['start_ms'] += 1
        self.assertEqual(len(self.resolve(observations=[observed], ctc_observations=ctc)['unresolved']), 1)



    def test_ctc_normalization_cannot_use_an_invented_candidate_length(self):
        observed, ctc = self.omission_evidence()
        for window in ctc['rows'][0]['windows']:
            actual = window['comparison']['observations'][0]
            actual['target_token_count'] += 20
            actual['nll_per_target_token'] = actual['negative_log_likelihood'] / actual['target_token_count']
        self.assertEqual(len(self.resolve(observations=[observed], ctc_observations=ctc)['unresolved']), 1)



    def gap_evidence(self):
        from app.acoustic_reference import reference_issue_key, reference_words_fingerprint
        from app.sync_audit import normalize_ctc
        self.package['words'] = [{'text': text, 'start_ms': index * 200, 'end_ms': index * 200 + 180}
            for index, text in enumerate('the primitive operations subtraction and more'.split())]
        self.reference = 'the primitive operations addition subtraction and more'
        issue = reference_resolution.reference_issues(self.package['words'], self.reference)[0]
        observations=[]
        for end, oid in ((3000, 'a' * 64), (5000, 'b' * 64)):
            observed = self.observation()
            observed.update(observation_id=oid, end_ms=end,
                words=[{**word, 'probability': .94} for word in self.package['words']])
            observations.append(observed)
        windows=[]
        for end in (2500, 4500):
            actual = ' '.join(word['text'] for word in self.package['words'])
            candidates=[]
            for cid,text,nll in [('actual_recognized_context',actual,30),
                    ('reference_edited_context',self.reference,80)]:
                tokens=len(normalize_ctc(text))
                candidates.append({'id':cid,'text':text,'status':'measured','frame_count':300,
                    'target_token_count':tokens,'negative_log_likelihood':nll,
                    'nll_per_frame':nll/300,'nll_per_target_token':nll/tokens})
            windows.append({'status':'measured','clip_start_ms':0,'clip_end_ms':end,
                'clip_sha256':'c'*64,'comparison':{'comparable':True,'edge_wildcards':False,
                    'algorithm':'normalized-logsoftmax-ctc-full-context-nll-v1',
                    'source':'independent_ctc_candidate_likelihood','frame_count':300,'observations':candidates}})
        ctc={'audio_sha256':_audio_digest(self.audio),'model_sha256':'d'*64,
            'input_words_sha256':reference_words_fingerprint(self.package['words']),
            'input_words_fingerprint_version':'text-time-1','rows':[{**issue,
                'issue_key':reference_issue_key(issue),'global_insertion_boundary':3,
                'global_anchor_indices':[2,3],'anchors':copy.deepcopy(self.package['words'][2:4]),'windows':windows}]}
        return observations,ctc

    def test_reference_only_gap_requires_positive_contiguous_anchors_and_both_models(self):
        observations,ctc=self.gap_evidence()
        result=self.resolve(observations=observations,ctc_observations=ctc)
        self.assertEqual(result['unresolved'],[])
        self.assertEqual(result['resolutions'][0]['mode'],'independent_dual_asr_anchors_and_dual_window_ctc')

    def test_reference_only_missing_any_evidence_or_extra_word_does_not_pass(self):
        observations,ctc=self.gap_evidence()
        self.assertEqual(len(self.resolve(observations=observations[:1],ctc_observations=ctc)['unresolved']),1)
        self.assertEqual(len(self.resolve(observations=observations,ctc_observations=None)['unresolved']),1)
        one=copy.deepcopy(ctc);one['rows'][0]['windows'].pop()
        self.assertEqual(len(self.resolve(observations=observations,ctc_observations=one)['unresolved']),1)
        for text in ('and','um','addition'):
            altered=copy.deepcopy(observations)
            altered[1]['words'].insert(3,{'text':text,'start_ms':585,'end_ms':595,'probability':.99})
            self.assertEqual(len(self.resolve(observations=altered,ctc_observations=ctc)['unresolved']),1)
        altered=copy.deepcopy(ctc);altered['rows'][0]['global_anchor_indices']=[1,3]
        self.assertEqual(len(self.resolve(observations=observations,ctc_observations=altered)['unresolved']),1)


if __name__ == '__main__':
    unittest.main()
