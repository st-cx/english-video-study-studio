from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.sync_audit import build_manifest, evaluate_annotations, normalize_ctc, timing_fingerprint, verify_manifest, waveform_envelope, write_review_page, candidate_ctc_text, score_ctc_candidates, is_independent_timed_word


def package():
    words = []
    sentences = []
    for index in range(180):
        text = 'MIT' if index % 13 == 0 else 'two' if index % 11 == 0 else 'context'
        words.append({'id': f'w{index}', 'sentence_id': f's{index}', 'text': text, 'start_ms': index*1000+100, 'end_ms': index*1000+700})
        sentences.append({'id': f's{index}', 'text': text, 'start_ms': index*1000+100, 'end_ms': index*1000+700, 'speaker': ''})
    return {'episode': {'id': 'synthetic-audit-test', 'title': 'Timing test', 'duration_s': 180}, 'words': words, 'sentences': sentences,
            'chapters': [{'id': f'ch{index}', 'start_ms': index*20000, 'end_ms': (index+1)*20000} for index in range(9)]}


def with_evidence(manifest):
    for sample in manifest['samples']:
        sample.update(clip_start_ms=max(0, sample['system_start_ms']-2500), clip_end_ms=sample['system_end_ms']+3000,
                      waveform=[[-.1, .1], [-.2, .3]], clip_file='clips/synthetic.wav', clip_sha256='test-only')
    return manifest


def human_annotations(manifest):
    return {'timing_sha256': manifest['timing_sha256'], 'annotations': [{'sample_id': s['id'], 'observed_start_ms': s['system_start_ms'],
            'observed_end_ms': s['system_end_ms'], 'source': 'human_listening_with_waveform', 'reviewer': 'synthetic-test-fixture', 'listened': True,
            'speaker_change_confirmed': 'yes' if s['stratum'] == 'speaker_change_candidate' else None} for s in manifest['samples']]}


class SyncAuditTests(unittest.TestCase):
    def test_sampling_is_deterministic_unique_and_stratified(self):
        first, second = build_manifest(package()), build_manifest(package())
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first['samples']), 48)
        self.assertEqual(len({s['word_id'] for s in first['samples']}), len(first['samples']))
        self.assertEqual(first['sampling_counts']['time_coverage'], 18)
        self.assertEqual(first['sampling_counts']['chapter_before'], 8)
        self.assertEqual(first['sampling_counts']['chapter_after'], 8)
        self.assertEqual(first['sampling_counts']['number'], 5)
        self.assertEqual(first['sampling_counts']['proper_name'], 5)
        self.assertEqual(first['sampling_counts']['speaker_change_candidate'], 4)
        self.assertEqual({s['temporal_bin'] for s in first['samples']}, set(range(6)))

    def test_measured_phrase_unit_is_not_counted_as_a_single_word(self):
        pkg=package();pkg['words'][0].update(text='NoneType',phrase_unit=True)
        manifest=build_manifest(pkg)
        self.assertFalse(any(item['word_id']=='w0' for item in manifest['samples']))

    def test_multiword_display_units_never_enter_word_acceptance_even_as_challenges(self):
        pkg = package()
        pkg['words'][0].update(text='NoneType', spoken_form='none type')
        pkg['words'][45].update(text='contracted', connected_words=['going', 'to'])
        pkg['words'][90].update(text='combined', spoken_words=[{'text':'computer'},{'text':'science'}])
        pkg['words'][120].update(text='unit', phrase_unit=True)
        pkg['words'][179].update(text='tail', spoken_words='two words')
        for i in [0,45,90,120,179]:
            pkg['words'][i]['end_ms'] = pkg['words'][i]['start_ms']
        manifest = build_manifest(pkg, minimum=50)
        forbidden = {'w0','w45','w90','w120','w179'}
        self.assertFalse(forbidden & {s['word_id'] for s in manifest['samples']})
        self.assertTrue(forbidden <= {s['word_id'] for s in manifest['excluded_units']})
        self.assertGreaterEqual(len(manifest['samples']),50)
        self.assertTrue(all(s['lexical_unit']=='word' for s in manifest['samples']))

    def test_spoken_metadata_change_invalidates_old_word_audit(self):
        pkg = package();manifest = build_manifest(pkg)
        pkg['words'][1]['spoken_form'] = 'two words'
        with self.assertRaises(ValueError):verify_manifest(pkg,manifest)
        self.assertFalse(is_independent_timed_word(pkg['words'][1]))
        self.assertTrue(is_independent_timed_word({'text':'Ana','spoken_form':'Anna'}))
        self.assertFalse(is_independent_timed_word({'text':'word','connected_words':[{'text':'two words'}]}))

    def test_silence_does_not_become_a_confirmed_speaker_change(self):
        manifest = build_manifest(package())
        for sample in manifest['samples']:
            if sample['stratum'] == 'speaker_change_candidate':
                self.assertIsNone(sample['speaker_change_confirmed'])
                self.assertFalse(sample['turn_evidence']['acoustic_speaker_change_measured'])
        self.assertTrue(any('不能据此声称' in warning for warning in manifest['warnings']))

    def test_structural_failures_are_added_not_filtered(self):
        pkg = package()
        pkg['words'][1]['end_ms'] = pkg['words'][1]['start_ms']
        manifest = build_manifest(pkg)
        self.assertTrue(any(s['word_id'] == 'w1' and s.get('structural_challenge') for s in manifest['samples']))

    def test_timing_or_spelling_change_invalidates_old_audit(self):
        pkg = package()
        manifest = build_manifest(pkg)
        pkg['words'][0]['start_ms'] += 1
        with self.assertRaises(ValueError):
            verify_manifest(pkg, manifest)
        before = timing_fingerprint(pkg)
        pkg['words'][0]['text'] = 'changed'
        self.assertNotEqual(before, timing_fingerprint(pkg))

    def test_model_agreement_cannot_pass_human_acceptance(self):
        manifest = with_evidence(build_manifest(package()))
        labels = human_annotations(manifest)
        for row in labels['annotations']:
            row['source'] = 'independent_ctc_forced_alignment'
        result = evaluate_annotations(manifest, labels)
        self.assertEqual(result['human_acceptance']['status'], 'not_completed')
        self.assertEqual(result['human_acceptance']['word_count'], 0)
        self.assertTrue(all(row['human_status'] == 'not_human_source' for row in result['rows']))

    def test_no_annotations_produces_null_errors_and_pending(self):
        result = evaluate_annotations(with_evidence(build_manifest(package())))
        self.assertEqual(result['human_acceptance']['status'], 'not_completed')
        self.assertIsNone(result['human_acceptance']['within_fraction'])
        self.assertTrue(all(row['human_error_ms'] is None for row in result['rows']))

    def test_precision_does_not_claim_unconfirmed_speaker_coverage(self):
        manifest = with_evidence(build_manifest(package()))
        labels = human_annotations(manifest)
        for row in labels['annotations']:
            row['speaker_change_confirmed'] = None
        self.assertEqual(evaluate_annotations(manifest, labels)['human_acceptance']['status'], 'precision_passed_speaker_coverage_unconfirmed')

    def test_both_word_boundaries_must_meet_threshold(self):
        manifest = with_evidence(build_manifest(package()))
        labels = human_annotations(manifest)
        # These are explicitly synthetic tests, not actual listening annotations.
        for row in labels['annotations'][:3]:
            row['observed_end_ms'] += 250
        result = evaluate_annotations(manifest, labels)
        self.assertEqual(result['human_acceptance']['status'], 'failed')
        self.assertEqual(result['rows'][0]['human_error_ms']['end_signed'], -250)
        labels['annotations'][2]['observed_end_ms'] -= 250
        self.assertEqual(evaluate_annotations(manifest, labels)['human_acceptance']['status'], 'passed')

    def test_uncertain_missing_or_unattested_labels_never_count_as_passes(self):
        manifest = with_evidence(build_manifest(package()))
        for field, value in [('uncertain', True), ('listened', False), ('reviewer', ''), ('observed_end_ms', -1)]:
            labels = human_annotations(manifest)
            labels['annotations'][0][field] = value
            result = evaluate_annotations(manifest, labels)
            self.assertEqual(result['human_acceptance']['status'], 'not_completed')

    def test_annotations_from_another_timeline_are_rejected(self):
        manifest = with_evidence(build_manifest(package()))
        labels = human_annotations(manifest)
        labels['timing_sha256'] = 'stale'
        with self.assertRaises(ValueError):
            evaluate_annotations(manifest, labels)

    def test_numeric_pronunciation_is_not_invented_for_ctc(self):
        self.assertIsNone(normalize_ctc('6.0001'))
        self.assertEqual(normalize_ctc("Python's"), "PYTHON'S")
        self.assertEqual(normalize_ctc('six point zero'), 'SIX POINT ZERO')

    def test_candidate_requires_provenance_and_explicit_numeric_spoken_form(self):
        for candidate in [dict(text='This is Python'), dict(text='This is 6.0001', source='test_model'), dict(text='Print x + y', source='test_model')]:
            with self.assertRaises(ValueError):
                candidate_ctc_text(candidate)
        candidate = dict(text='This is 6.0001', source='test_model', spoken_form='This is six point zero zero zero one')
        self.assertEqual(candidate_ctc_text(candidate), 'THIS IS SIX POINT ZERO ZERO ZERO ONE')
        self.assertEqual(candidate['text'], 'This is 6.0001')

    def test_candidate_ctc_uses_normalized_probabilities_and_preserves_sources(self):
        import torch
        labels = ('-', '|', 'A', 'B')
        logits = torch.tensor([[[0., -4., 5., -4.], [4., -4., 0., -4.], [0., -4., 5., -4.], [4., -4., 0., -4.]]])
        candidates = [dict(id='a', text='AA', source='synthetic_a'), dict(id='b', text='BB', source='synthetic_b')]
        result = score_ctc_candidates(logits, labels, candidates)
        self.assertFalse(result['human_ground_truth'])
        self.assertFalse(result['edge_wildcards'])
        self.assertEqual(result['ranking_per_frame'][0], 'a')
        self.assertEqual(result['ranking_per_target_token'][0], 'a')
        expected = torch.nn.functional.ctc_loss(torch.log_softmax(logits, -1).transpose(0, 1), torch.tensor([2, 2]), torch.tensor([4]), torch.tensor([2]), reduction='sum')
        self.assertAlmostEqual(result['observations'][0]['negative_log_likelihood'], float(expected), places=5)
        self.assertEqual(result['observations'][0]['source'], 'synthetic_a')
        self.assertEqual(candidates[0]['text'], 'AA')

    def test_candidate_failure_is_not_a_zero_probability_success(self):
        import torch
        labels = ('-', '|', 'A', 'B')
        logits = torch.zeros((1, 4, 4))
        candidates = [dict(id='too_long', text='AAAAAAAA', source='synthetic'), dict(id='normal', text='AB', source='synthetic')]
        result = score_ctc_candidates(logits, labels, candidates)
        self.assertFalse(result['comparable'])
        self.assertEqual(result['observations'][0]['status'], 'not_measurable')
        self.assertNotIn('negative_log_likelihood', result['observations'][0])

    def test_candidate_probability_comparison_rejects_extra_wildcard(self):
        import torch
        candidates = [dict(id='a', text='A', source='synthetic'), dict(id='b', text='B', source='synthetic')]
        with self.assertRaises(ValueError):
            score_ctc_candidates(torch.zeros((1, 4, 5)), ('-', '|', 'A', 'B'), candidates)
        candidates[1]['id'] = 'a'
        with self.assertRaises(ValueError):
            score_ctc_candidates(torch.zeros((1, 4, 4)), ('-', '|', 'A', 'B'), candidates)

    def test_waveform_is_derived_from_actual_pcm(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            path = Path(tmp)/'sound.wav'
            with wave.open(str(path), 'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
                stream.writeframes(b'\x00\x40\x00\xc0'*8000)
            values, duration = waveform_envelope(path, 50)
            self.assertEqual(duration, 1000)
            self.assertEqual(len(values), 50)
            self.assertEqual(values[0], [-.5, .5])

    def test_review_page_does_not_prefill_human_boundaries(self):
        manifest = with_evidence(build_manifest(package()))
        manifest['title'] = '<script>alert(1)</script>'
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            write_review_page(manifest, Path(tmp))
            text = (Path(tmp)/'review.html').read_text(encoding='utf-8')
            self.assertNotIn('<h2><script>', text)
            self.assertIn('let state={reviewer:', text)
            self.assertIn('observed_start_ms:null', text)
            self.assertIn('本页没有替你完成听辨', text)


if __name__ == '__main__':
    unittest.main()
