import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.acoustic_reference import reference_issue_key, reference_words_fingerprint
from app.review_reconcile import (
    _apply_duplicate, _boundary_support, _cross_boundary_evidence, _duplicate_repair,
    _orthographic_repair, _substitution_retention, inventory, merge_ctc_sources,
    rebind_ctc_sources, reconcile_reviews,
)
from app.sync_audit import normalize_ctc


def duplicate_fixture():
    texts = ['we', 'divide', 'i', 'by', 'j.', 'j.', 'And', 'raise']
    words = [{'id': str(i), 'sentence_id': 's' if i != 5 else 'extra', 'text': text,
              'start_ms': 1000 + i * 500, 'end_ms': 1200 + i * 500} for i, text in enumerate(texts)]
    words[5].update(start_ms=4200, end_ms=4400)
    words[6].update(start_ms=5000, end_ms=5200)
    words[7].update(start_ms=5500, end_ms=5700)
    issue = {'recognized': 'j', 'reference': '', 'start_ms': 4200, 'end_ms': 4400, 'reason': 'reference'}
    package = {'words': words, 'sentences': [
        {'id': 's', 'text': 'we divide i by j. And raise', 'word_ids': [str(i) for i in range(8) if i != 5], 'start_ms': 1000, 'end_ms': 5700},
        {'id': 'extra', 'text': 'j.', 'word_ids': ['5'], 'start_ms': 4200, 'end_ms': 4400}],
        'cards': [{'word_ids': ['5'], 'sentence_ids': ['extra']}],
        'phrases': [{'word_ids': ['5'], 'sentence_id': 'extra'}],
        'processing': {}, 'review': {'official_transcript': 'we divide i by j. And raise'}}
    observations = []
    for name in ('b', 'c'):
        obs_words = [{**w, 'probability': .99} for w in words if w['id'] != '5']
        observations.append({'observation_id': name * 64, 'model_fingerprint': name * 64,
            'audio_sha256': 'a' * 64, 'mode': 'independent_unprompted_asr',
            'start_ms': 0, 'end_ms': 10000, 'truncated': False, 'words': obs_words})
    windows = []
    for begin in (0, 100):
        actual = ' '.join(w['text'] for w in words)
        alternative = ' '.join(w['text'] for w in words if w['id'] != '5')
        candidates = []
        for cid, text, nll in [('actual_recognized_context', actual, 100), ('reference_edited_context', alternative, 40)]:
            tokens = len(normalize_ctc(text))
            candidates.append({'id': cid, 'text': text, 'status': 'measured', 'frame_count': 500,
                'target_token_count': tokens, 'negative_log_likelihood': nll,
                'nll_per_frame': nll / 500, 'nll_per_target_token': nll / tokens})
        windows.append({'clip_start_ms': begin, 'clip_end_ms': begin + 10000, 'clip_sha256': 'd' * 64,
            'status': 'measured', 'comparison': {'algorithm': 'normalized-logsoftmax-ctc-full-context-nll-v1',
                'source': 'independent_ctc_candidate_likelihood', 'edge_wildcards': False,
                'comparable': True, 'frame_count': 500, 'observations': candidates}})
    source = {'audio_sha256': 'a' * 64, 'model_sha256': 'e' * 64,
        'input_words_sha256': reference_words_fingerprint(words),
        'rows': [{**issue, 'issue_key': reference_issue_key(issue), 'global_word_indices': [5], 'windows': windows}]}
    return package, issue, observations, source


class ReviewReconcileTests(unittest.TestCase):
    def test_resume_batches_join_only_same_audio_model_and_complete_word_clock(self):
        package, issue, observations, source = duplicate_fixture()
        source.update(source='independent_full_context_ctc_likelihood',
                      input_words_fingerprint_version='text-time-1')
        left, right = copy.deepcopy(source), copy.deepcopy(source)
        left['rows'][0]['windows'] = source['rows'][0]['windows'][:1]
        left['rows'][0]['untrusted_saved_approval'] = True
        right['rows'][0]['windows'] = source['rows'][0]['windows'][1:]
        result = merge_ctc_sources([left, right, left], package['words'], 'a' * 64)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]['rows'][0]['windows']), 2)
        self.assertEqual(len(result[0]['merged_from_sha256']), 2)
        self.assertNotIn('untrusted_saved_approval', result[0]['rows'][0])
        stale = copy.deepcopy(right)
        stale['input_words_sha256'] = 'f' * 64
        wrong_audio = copy.deepcopy(right)
        wrong_audio['audio_sha256'] = 'f' * 64
        other_model = copy.deepcopy(right)
        other_model['model_sha256'] = 'c' * 64
        result = merge_ctc_sources([left, stale, wrong_audio, other_model], package['words'], 'a' * 64)
        self.assertEqual(sorted(len(x['rows'][0]['windows']) for x in result), [1, 1])

    def test_resume_batches_reject_conflicting_copies_of_the_same_crop(self):
        package, issue, observations, source = duplicate_fixture()
        source.update(source='independent_full_context_ctc_likelihood',
                      input_words_fingerprint_version='text-time-1')
        altered = copy.deepcopy(source)
        altered['rows'][0]['windows'][0]['comparison']['observations'][0]['text'] = 'tampered'
        result = merge_ctc_sources([source, altered, source], package['words'], 'a' * 64)
        self.assertEqual(len(result[0]['rows'][0]['windows']), 1)
        self.assertEqual(result[0]['rows'][0]['windows'][0]['clip_start_ms'], 100)

    def test_duplicate_requires_distinct_models_and_real_raw_comparisons(self):
        package, issue, observations, source = duplicate_fixture()
        self.assertIsNotNone(_duplicate_repair(package, issue, observations, [source], 'a' * 64))
        self.assertIsNone(_duplicate_repair(package, issue, observations[:1] * 2, [source], 'a' * 64))
        source['rows'][0]['windows'] = source['rows'][0]['windows'][:1]
        self.assertIsNone(_duplicate_repair(package, issue, observations, [source], 'a' * 64))

    def test_sequence_matcher_can_mark_first_duplicate_but_earlier_id_is_retained(self):
        package, issue, observations, source = duplicate_fixture()
        issue.update(start_ms=package['words'][4]['start_ms'], end_ms=package['words'][4]['end_ms'])
        source['rows'][0].update(**issue, issue_key=reference_issue_key(issue), global_word_indices=[4])
        repair = _duplicate_repair(package, issue, observations, [source], 'a' * 64)
        self.assertIsNotNone(repair)
        self.assertEqual((repair['removed_word_id'], repair['retained_word_id']), ('5', '4'))

    def test_current_boundary_follows_exact_retained_id_revision_not_nearest_word(self):
        words = [{'id': '0', 'text': 'we', 'start_ms': 1500, 'end_ms': 1700},
                 {'id': '1', 'text': 'read', 'start_ms': 2000, 'end_ms': 2300},
                 {'id': '2', 'text': 'books', 'start_ms': 2500, 'end_ms': 2900}]
        row = {'text': 'read', 'details': {'aligned_ms': [8000, 8300]}}
        survey = {'audio_sha256': 'a' * 64, 'model_sha256': 'b' * 64, 'windows': []}
        for begin in (0, 100):
            survey['windows'].append({'start_ms': begin, 'end_ms': begin + 10000,
                'aligned': [{**w, 'path_score_not_accuracy': .99} for w in words],
                'greedy_words': copy.deepcopy(words)})
        package = {'words': words, 'processing': {'timing_revision_history': [
            {'patches': [{'word_id': '1', 'text': 'read', 'before_ms': [8000, 8300], 'after_ms': [2000, 2300]}]}]}}
        self.assertIsNotNone(_boundary_support(package, row, [survey], 'a' * 64))
        package['processing'] = {}
        self.assertIsNone(_boundary_support(package, row, [survey], 'a' * 64))

    def test_cross_model_relocation_requires_raw_old_new_scores_and_two_models(self):
        texts = ['to', 'the', 'console.', 'So', "that's"]
        spans = [(1500, 1700), (1800, 1950), (9000, 9500), (11000, 11200), (11300, 11700)]
        words = [dict(id=str(i), sentence_id='s', text=t, start_ms=a, end_ms=b)
                 for i, (t, (a, b)) in enumerate(zip(texts, spans))]
        issue = {'text': 'console.', 'start_ms': 4000, 'end_ms': 9500,
                 'details': {'aligned_ms': [9000, 9500], 'asr_ms': [4000, 4500]}}
        observations = []
        for name in ('b', 'c'):
            measured = [{**w, 'probability': .99} for w in words]
            measured[2].update(start_ms=4000, end_ms=4500)
            observations.append({'observation_id': name * 64, 'model_fingerprint': name * 64,
                'audio_sha256': 'a' * 64, 'mode': 'independent_unprompted_asr',
                'start_ms': 0, 'end_ms': 14000, 'words': measured})
        survey = {'audio_sha256': 'a' * 64, 'model_sha256': 'd' * 64, 'windows': []}
        for begin in (0, 100):
            aligned = [{**w, 'path_score_not_accuracy': .2} for w in words]
            aligned[2].update(start_ms=4000, end_ms=4500)
            intervals = {}
            for key, bounds, nll in [('before', [9000, 9500], 1000), ('after', [4000, 4500], 100)]:
                intervals[key] = {'interval_ms': bounds, 'negative_log_likelihood': nll,
                    'frame_count': 35, 'target_token_count': 7, 'nll_per_frame': nll / 35,
                    'nll_per_target_token': nll / 7}
            survey['windows'].append({'start_ms': begin, 'end_ms': begin + 14000, 'aligned': aligned,
                'emissions_sha256': 'e' * 64, 'interval_comparisons': [{
                    'algorithm': 'same-target-two-interval-ctc-nll-v1', 'word_id': '2', 'text': 'console.',
                    'before_ms': [9000, 9500], 'after_ms': [4000, 4500], 'padding_ms': 100,
                    'emissions_sha256': 'e' * 64, 'intervals': intervals}]})
        result = _cross_boundary_evidence({'words': words}, issue, observations, [survey], 'a' * 64)
        self.assertEqual(result['after_ms'], [4000, 4500])
        self.assertIsNone(_cross_boundary_evidence({'words': words}, issue, observations[:1] * 2, [survey], 'a' * 64))
        for window in survey['windows']:
            window['interval_comparisons'][0]['intervals']['after']['nll_per_frame'] = .01
        self.assertIsNone(_cross_boundary_evidence({'words': words}, issue, observations, [survey], 'a' * 64))

    def test_substitution_retention_requires_two_models_and_context_not_nearby_pronoun(self):
        texts = ['says.', 'So', 'out', 'just', 'tells', 'you']
        words = [dict(id=str(i), sentence_id='s', text=t, start_ms=1000 + i * 400,
                      end_ms=1200 + i * 400) for i, t in enumerate(texts)]
        issue = {'recognized': 'So out', 'reference': 'it', 'start_ms': 1400, 'end_ms': 2000}
        observations = []
        for model in ('b', 'c'):
            observations.append({'observation_id': model * 64, 'model_fingerprint': model * 64,
                'audio_sha256': 'a' * 64, 'mode': 'independent_unprompted_asr',
                'words': [{'text': 'it', 'start_ms': 800, 'end_ms': 900, 'probability': .99}] +
                         [{**w, 'probability': .75 if w['text'] == 'So' else .99} for w in words]})
        candidates = []
        actual = ' '.join(texts)
        edited = 'says. it just tells you'
        for cid, text, nll in [('actual_recognized_context', actual, 40), ('reference_edited_context', edited, 90)]:
            tokens = len(normalize_ctc(text))
            candidates.append({'id': cid, 'text': text, 'status': 'measured', 'frame_count': 400,
                'target_token_count': tokens, 'negative_log_likelihood': nll,
                'nll_per_frame': nll / 400, 'nll_per_target_token': nll / tokens})
        windows = [{'clip_start_ms': begin, 'clip_end_ms': 8000 + begin, 'clip_sha256': 'd' * 64,
            'status': 'measured', 'comparison': {'algorithm': 'normalized-logsoftmax-ctc-full-context-nll-v1',
                'source': 'independent_ctc_candidate_likelihood', 'edge_wildcards': False,
                'comparable': True, 'frame_count': 400, 'observations': copy.deepcopy(candidates)}} for begin in (0, 100)]
        source = {'audio_sha256': 'a' * 64, 'model_sha256': 'e' * 64,
            'input_words_sha256': reference_words_fingerprint(words),
            'rows': [{**issue, 'issue_key': reference_issue_key(issue), 'windows': windows}]}
        self.assertIsNotNone(_substitution_retention({'words': words}, issue, observations, [source], 'a' * 64))
        self.assertIsNone(_substitution_retention({'words': words}, issue, observations[:1] * 2, [source], 'a' * 64))
        alternate = copy.deepcopy(observations[0])
        alternate['words'] = [{**w, 'probability': .99} for w in words[:1]] + [
            {'text': 'it', 'start_ms': 1400, 'end_ms': 2000, 'probability': .99}] + [
            {**w, 'probability': .99} for w in words[3:]]
        self.assertIsNone(_substitution_retention({'words': words}, issue, observations + [alternate], [source], 'a' * 64))
        for a, b in [('13', '3'), ('not valid', 'valid'), ('can not', 'can')]:
            self.assertIsNone(_substitution_retention({'words': words}, {**issue, 'recognized': a, 'reference': b}, observations, [source], 'a' * 64))

    def test_weak_named_token_or_changed_acoustic_audio_cannot_delete(self):
        package, issue, observations, source = duplicate_fixture()
        observations[1]['words'][4]['probability'] = .84
        self.assertIsNone(_duplicate_repair(package, issue, observations, [source], 'a' * 64))
        observations[1]['words'][4]['probability'] = .99
        self.assertIsNone(_duplicate_repair(package, issue, observations, [source], 'f' * 64))

    def test_fake_winner_flags_and_tampered_normalized_scores_do_not_pass(self):
        package, issue, observations, source = duplicate_fixture()
        source['rows'][0]['two_window_ctc_supports_actual'] = True
        for window in source['rows'][0]['windows']:
            window['comparison']['observations'][1]['nll_per_frame'] = .001
        self.assertIsNone(_duplicate_repair(package, issue, observations, [source], 'a' * 64))

    def test_continuous_full_context_prevents_latching_to_a_nearby_repeat(self):
        package, issue, observations, source = duplicate_fixture()
        # Both recognizers really say the token twice: no reduced-context match.
        for obs in observations:
            obs['words'].insert(5, {**package['words'][5], 'probability': .99})
        self.assertIsNone(_duplicate_repair(package, issue, observations, [source], 'a' * 64))

    def test_deletion_retains_aliases_and_all_card_phrase_references(self):
        package, issue, observations, source = duplicate_fixture()
        repair = _duplicate_repair(package, issue, observations, [source], 'a' * 64)
        _apply_duplicate(package, repair)
        self.assertEqual(package['processing']['identity_aliases'], {'words': {'5': '4'}, 'sentences': {'extra': 's'}})
        self.assertEqual(package['cards'][0], {'word_ids': ['4'], 'sentence_ids': ['s']})
        self.assertEqual(package['phrases'][0], {'word_ids': ['4'], 'sentence_id': 's'})
        self.assertEqual(len(package['sentences']), 1)

    def test_changed_full_candidate_context_rejects_stale_ctc(self):
        package, issue, observations, source = duplicate_fixture()
        changed = copy.deepcopy(package)
        changed['words'][0]['text'] = 'they'
        rebound = rebind_ctc_sources(package, changed, [source], 'a' * 64)
        self.assertEqual(rebound[0]['rows'], [])
        self.assertEqual(rebound[0]['rejected_rebindings_count'], 1)

    def test_apostrophe_normalization_preserves_ids_and_exact_timing(self):
        package = {'words': [{'id': 'w', 'sentence_id': 's', 'text': 'responders', 'start_ms': 10, 'end_ms': 20}],
            'sentences': [{'id': 's', 'text': 'the responders sheet'}]}
        original = copy.deepcopy(package)
        issue = {'recognized': 'responders', 'reference': "responder's", 'start_ms': 10, 'end_ms': 20}
        repair = _orthographic_repair(package, issue)
        self.assertEqual(package['words'][0]['text'], "responder's")
        self.assertEqual(repair['before_ms'], repair['after_ms'])
        self.assertEqual(package['words'][0]['id'], original['words'][0]['id'])
        for actual, expected in [('its', "it's"), ('thats', "that's"), ('Wookiee', 'Wookie')]:
            self.assertIsNone(_orthographic_repair(package, {**issue, 'recognized': actual, 'reference': expected}))

    def test_current_boundary_requires_two_windows_positive_frames_and_full_context(self):
        words = [{'id': '0', 'text': 'we', 'start_ms': 1500, 'end_ms': 1700},
            {'id': '1', 'text': 'read', 'start_ms': 2000, 'end_ms': 2300},
            {'id': '2', 'text': 'books', 'start_ms': 2500, 'end_ms': 2900}]
        row = {'text': 'read', 'details': {'aligned_ms': [2000, 2300]}}
        survey = {'audio_sha256': 'a' * 64, 'model_sha256': 'b' * 64, 'windows': []}
        for begin in (0, 100):
            aligned = [{**w, 'path_score_not_accuracy': .99} for w in words]
            survey['windows'].append({'start_ms': begin, 'end_ms': begin + 10000,
                'aligned': aligned, 'greedy_words': copy.deepcopy(words)})
        self.assertIsNotNone(_boundary_support({'words': words}, row, [survey], 'a' * 64))
        changed = copy.deepcopy(words)
        changed[0]['text'] = 'they'
        self.assertIsNone(_boundary_support({'words': changed}, row, [survey], 'a' * 64))
        survey['windows'][0]['aligned'][1]['path_score_not_accuracy'] = 1.1
        self.assertIsNone(_boundary_support({'words': words}, row, [survey], 'a' * 64))

    def test_non_source_structural_and_teaching_flags_survive_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'audio'; audio.write_bytes(b'test original audio')
            package = {'words': [{'id': 'w', 'sentence_id': 's', 'text': 'Hello.', 'start_ms': 1000, 'end_ms': 2000}],
                'sentences': [{'id': 's', 'text': 'Hello.', 'word_ids': ['w']}],
                'review': {'official_transcript': 'Hello.'}, 'processing': {'reference_review': {'unresolved': []}},
                'chapters': [{'id': 'c', 'title': 'unit', 'start_ms': 0, 'end_ms': 3000,
                    'reference_issues': [{'reason': 'old reference'}], 'alignment_issues': [],
                    'flags': ['old reference', 'real structural issue'],
                    'quality': {'flags': ['old reference', 'real structural issue'], 'structural_flags': ['real structural issue']},
                    'teaching_review': {'status': 'needs_review', 'issues': [{'reason': 'wrong negation'}]}}],
                'quality': {'flags': ['unit：old reference', 'unit：real structural issue', 'global problem'],
                            'structural_flags': ['global structure']}}
            before = copy.deepcopy(package)
            with patch('app.review_reconcile._assess', return_value=[]):
                after = reconcile_reviews(package, audio, model_paths=[], observations=[])['episode']
            self.assertEqual(package, before)
            flags = after['chapters'][0]['flags']
            self.assertNotIn('old reference', flags)
            self.assertIn('real structural issue', flags)
            self.assertIn('教学内容：wrong negation', flags)
            self.assertIn('global problem', after['quality']['flags'])
            self.assertEqual(after['quality']['structural_flags'], ['global structure'])

    def test_overlapping_chapter_reference_rows_have_one_inventory_record(self):
        issue = {'recognized': 'word', 'reference': '', 'start_ms': 10, 'end_ms': 20}
        package = {'chapters': [{'reference_issues': [issue]}, {'reference_issues': [copy.deepcopy(issue)]}]}
        self.assertEqual(len(inventory(package)), 1)


if __name__ == '__main__':
    unittest.main()
