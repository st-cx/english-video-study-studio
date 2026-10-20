import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import acoustic_reference


def timed_word(text, start=.5, end=.9, probability=.95):
    return SimpleNamespace(word=text, start=start, end=end, probability=probability)


class AcousticObservationTests(unittest.TestCase):
    def observation(self, words):
        return {'observation_id': 'c' * 64, 'audio_sha256': 'a' * 64,
                'model_fingerprint': 'b' * 64, 'mode': 'independent_unprompted_asr',
                'start_ms': 7500, 'end_ms': 13000, 'truncated': False, 'words': words}

    def assess(self, issue, observation):
        return acoustic_reference.assess_observations([issue], [observation],
                audio_sha256='a' * 64, model_fingerprint='b' * 64)[0]

    def test_unprompted_observation_merges_windows_and_caches_measured_words(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as temp:
            base = Path(temp); audio = base / 'audio.mp3'; audio.write_bytes(b'isolated-audio')
            issues = [{'start_ms': 10000, 'end_ms': 10400, 'recognized': 'Bulls', 'reference': 'Bools'},
                      {'start_ms': 10800, 'end_ms': 11000, 'recognized': 'right', 'reference': ''}]
            original = copy.deepcopy(issues)
            model = Mock(); model.transcribe.return_value = ([SimpleNamespace(words=[timed_word('Bools', 2.5, 2.9)])], None)
            with patch.object(acoustic_reference.subprocess, 'run') as extract:
                observed = acoustic_reference.observe_disputes(audio, issues, 'isolated-model', lambda *a: None, model=model, cache_dir=base / 'cache')
                cached = acoustic_reference.observe_disputes(audio, issues, 'isolated-model', lambda *a: None, model=model, cache_dir=base / 'cache')
            self.assertEqual(issues, original)
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]['words'][0]['start_ms'], 10000)
            self.assertEqual(observed[0]['words'][0]['end_ms'], 10400)
            self.assertEqual(observed[0]['mode'], 'independent_unprompted_asr')
            self.assertNotIn('initial_prompt', model.transcribe.call_args.kwargs)
            self.assertEqual(model.transcribe.call_count, 1)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(observed, cached)
            self.assertNotIn('model_path', json.dumps(observed))

    def test_audio_change_invalidates_observation_cache(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as temp:
            base = Path(temp); audio = base / 'audio.mp3'; audio.write_bytes(b'first')
            issues = [{'start_ms': 1000, 'end_ms': 1200, 'recognized': 'task', 'reference': 'test'}]
            model = Mock(); model.transcribe.return_value = ([SimpleNamespace(words=[timed_word('task')])], None)
            with patch.object(acoustic_reference.subprocess, 'run'):
                first = acoustic_reference.observe_disputes(audio, issues, 'isolated-model', lambda *a: None, model=model, cache_dir=base / 'cache')
                audio.write_bytes(b'second')
                second = acoustic_reference.observe_disputes(audio, issues, 'isolated-model', lambda *a: None, model=model, cache_dir=base / 'cache')
            self.assertEqual(model.transcribe.call_count, 2)
            self.assertNotEqual(first[0]['audio_sha256'], second[0]['audio_sha256'])

    def test_degenerate_asr_timing_is_preserved_as_unresolved_and_not_cached(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as temp:
            base = Path(temp); audio = base / 'audio.mp3'; audio.write_bytes(b'isolated')
            issues = [{'start_ms': 1000, 'end_ms': 1200, 'recognized': '', 'reference': 'write'}]
            model = Mock(); model.transcribe.return_value = ([SimpleNamespace(words=[timed_word('write', .5, .5)])], None)
            with patch.object(acoustic_reference.subprocess, 'run'):
                observed = acoustic_reference.observe_disputes(audio, issues, 'isolated-model', lambda *a: None, model=model, cache_dir=base / 'cache')
            self.assertEqual(observed[0]['status'], 'needs_check')
            self.assertTrue(observed[0]['warnings'])
            self.assertEqual(observed[0]['words'][0]['start_ms'], observed[0]['words'][0]['end_ms'])
            self.assertFalse(list((base / 'cache').glob('*.json')))

    def test_reconcile_does_not_hint_the_expected_reference_to_asr(self):
        source = [{'text': 'Bulls.', 'start_ms': 1000, 'end_ms': 1400, 'probability': .9}]
        model = Mock(); model.transcribe.return_value = ([SimpleNamespace(words=[timed_word('Bools', 1, 1.4)])], None)
        module = SimpleNamespace(WhisperModel=Mock(return_value=model))
        with patch.dict(sys.modules, {'faster_whisper': module}), patch.object(acoustic_reference.subprocess, 'run'):
            updated, changes = acoustic_reference.reconcile(Path('isolated-audio.mp3'), source, 'Bools.', 'isolated-model', lambda *a: None)
        self.assertNotIn('initial_prompt', model.transcribe.call_args.kwargs)
        self.assertEqual(updated[0]['text'], 'Bools.')
        self.assertEqual(source[0]['text'], 'Bulls.')
        self.assertEqual(changes[0]['measured_ms'], [1000, 1400])

    def test_reference_alone_cannot_overwrite_an_independently_recognized_word(self):
        source = [{'text': 'task.', 'start_ms': 1000, 'end_ms': 1400, 'probability': .9}]
        model = Mock(); model.transcribe.return_value = ([SimpleNamespace(words=[timed_word('task', 1, 1.4)])], None)
        module = SimpleNamespace(WhisperModel=Mock(return_value=model))
        with patch.dict(sys.modules, {'faster_whisper': module}), patch.object(acoustic_reference.subprocess, 'run'):
            updated, changes = acoustic_reference.reconcile(Path('isolated-audio.mp3'), source, 'test.', 'isolated-model', lambda *a: None)
        self.assertEqual(updated, source)
        self.assertEqual(changes, [])

    def test_high_confidence_actual_extra_is_supported_with_current_evidence_only(self):
        issue = {'start_ms': 10000, 'end_ms': 10400, 'recognized': 'right so', 'reference': ''}
        observation = self.observation([
            {'text': 'right', 'start_ms': 10000, 'end_ms': 10200, 'probability': .99},
            {'text': 'so', 'start_ms': 10200, 'end_ms': 10400, 'probability': .98}])
        decision = self.assess(issue, observation)
        self.assertEqual(decision['status'], 'supported_recognition')
        self.assertEqual(decision['kind'], 'reference_omission')
        self.assertEqual(decision['issue_key'], acoustic_reference.reference_issue_key(issue))
        self.assertTrue(acoustic_reference.reference_resolution_is_current(issue, decision, audio_sha256='a' * 64, model_fingerprint='b' * 64))
        self.assertFalse(acoustic_reference.reference_resolution_is_current({**issue, 'end_ms': 10500}, decision, audio_sha256='a' * 64, model_fingerprint='b' * 64))
        self.assertEqual(self.assess(issue, {**observation, 'audio_sha256': 'd' * 64})['status'], 'unresolved')

    def test_absence_in_asr_does_not_prove_a_reference_only_word_was_not_spoken(self):
        issue = {'start_ms': 10000, 'end_ms': 10400, 'recognized': '', 'reference': 'write'}
        decision = self.assess(issue, self.observation([
            {'text': 'code', 'start_ms': 10400, 'end_ms': 10600, 'probability': .99}]))
        self.assertEqual(decision['status'], 'unresolved')

    def test_reference_supported_candidate_keeps_measured_timing_and_does_not_approve(self):
        issue = {'start_ms': 10000, 'end_ms': 10400, 'recognized': 'fronts', 'reference': 'trunks'}
        decision = self.assess(issue, self.observation([
            {'text': 'trunks', 'start_ms': 10020, 'end_ms': 10420, 'probability': .99}]))
        self.assertEqual(decision['status'], 'reference_supported')
        self.assertEqual(decision['candidate_words'][0]['start_ms'], 10020)
        self.assertFalse(acoustic_reference.reference_resolution_is_current(issue, decision, audio_sha256='a' * 64, model_fingerprint='b' * 64))

    def test_both_alternatives_low_confidence_or_invalid_timing_stay_unresolved(self):
        issue = {'start_ms': 10000, 'end_ms': 10400, 'recognized': 'correct', 'reference': 'incorrect'}
        for words in ([{'text': 'correct', 'start_ms': 10000, 'end_ms': 10200, 'probability': .99},
                       {'text': 'incorrect', 'start_ms': 10200, 'end_ms': 10400, 'probability': .99}],
                      [{'text': 'correct', 'start_ms': 10000, 'end_ms': 10400, 'probability': .5}],
                      [{'text': 'correct', 'start_ms': 10000, 'end_ms': 10000, 'probability': .99}]):
            self.assertEqual(self.assess(issue, self.observation(words))['status'], 'unresolved')

    def test_ambiguous_or_malformed_observation_cannot_be_ignored_to_find_a_pass(self):
        issue = {'start_ms': 10000, 'end_ms': 10400, 'recognized': 'correct', 'reference': 'incorrect'}
        definite = self.observation([{'text': 'correct', 'start_ms': 10000, 'end_ms': 10200, 'probability': .99}])
        ambiguous = self.observation(definite['words'] + [{'text': 'incorrect', 'start_ms': 10200, 'end_ms': 10400, 'probability': .99}])
        decisions = acoustic_reference.assess_observations([issue], [definite, ambiguous], audio_sha256='a' * 64, model_fingerprint='b' * 64)
        self.assertEqual(decisions[0]['status'], 'unresolved')
        malformed = self.observation([{'text': 'unknown', 'start_ms': 10000, 'end_ms': None, 'probability': .99}] + definite['words'])
        self.assertEqual(self.assess(issue, malformed)['status'], 'unresolved')


if __name__ == '__main__':
    unittest.main()
