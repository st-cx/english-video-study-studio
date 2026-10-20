import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.providers import ProviderError
from app.study_sections import ensure_study_sections, resolve_sections, source_fingerprint


def fixture():
    return {'episode': {'title': 'A programming lesson', 'duration_s': 102},
            'sentences': [{'id': 'source-' + str(i), 'text': f'Sentence {i}.',
                           'start_ms': i * 10000 + 250, 'end_ms': i * 10000 + 9000} for i in range(10)],
            'chapters': [{'id': 'processing-1', 'start_ms': 0, 'end_ms': 102000, 'status': 'review'}],
            'processing': {}}


def result():
    return {'sections': [{'start_sentence_id': 'source-0', 'title': '课程介绍',
                          'summary': '首先说明本节课的目标和学习方法。', 'evidence_sentence_ids': ['source-0', 'source-2']},
                         {'start_sentence_id': 'source-4', 'title': '变量与赋值',
                          'summary': '随后介绍变量如何绑定表达式的结果。', 'evidence_sentence_ids': ['source-4', 'source-9']}]}


class StudySectionTests(unittest.TestCase):
    def test_boundaries_only_come_from_measured_sentences_and_cover_the_course(self):
        package = fixture(); before = copy.deepcopy(package); proposed = result()
        proposed['sections'][1].update(start_ms=1, end_ms=2)
        sections = resolve_sections(package, proposed)
        self.assertEqual([(s['start_ms'], s['end_ms']) for s in sections], [(0, 40250), (40250, 102000)])
        self.assertEqual([sid for s in sections for sid in s['sentence_ids']], [s['id'] for s in package['sentences']])
        self.assertEqual(package, before)

    def test_unknown_duplicate_and_out_of_order_anchors_are_rejected(self):
        for anchors in [('missing', 'source-4'), ('source-0', 'source-0'), ('source-4', 'source-0')]:
            with self.subTest(anchors=anchors):
                proposed = result()
                for section, anchor in zip(proposed['sections'], anchors):section['start_sentence_id'] = anchor
                with self.assertRaises(ValueError):resolve_sections(fixture(), proposed)

    def test_evidence_cannot_leak_in_from_another_topic_or_be_malformed(self):
        for evidence in [['source-4'], [{}], [], None]:
            with self.subTest(evidence=evidence):
                proposed = result(); proposed['sections'][0]['evidence_sentence_ids'] = evidence
                with self.assertRaises(ValueError):resolve_sections(fixture(), proposed)
        proposed = result(); proposed['sections'][0]['start_sentence_id'] = {}
        with self.assertRaises(ValueError):resolve_sections(fixture(), proposed)

    def test_request_uses_short_aliases_and_cache_preserves_review_and_timing(self):
        package = fixture(); original = copy.deepcopy(package)
        proposed = result()
        for section in proposed['sections']:
            section['start_sentence_id'] = 'S' + f"{int(section['start_sentence_id'].split('-')[-1]) + 1:04d}"
            section['evidence_sentence_ids'] = ['S' + f"{int(s.split('-')[-1]) + 1:04d}" for s in section['evidence_sentence_ids']]
        with tempfile.TemporaryDirectory() as temporary, patch('app.study_sections.get_settings', return_value={'model': 'test', 'base_url': 'https://example.invalid'}), patch('app.study_sections.ask_deepseek', return_value=proposed) as ask:
            folder = Path(temporary)
            self.assertTrue(ensure_study_sections(package, folder))
            self.assertIn('S0001', ask.call_args.args[1]); self.assertNotIn('source-0', ask.call_args.args[1])
            self.assertEqual(package['sentences'], original['sentences']); self.assertEqual(package['chapters'], original['chapters'])
            package['study_sections']['items'][1]['start_ms'] = 7
            self.assertTrue(ensure_study_sections(package, folder));self.assertEqual(package['study_sections']['items'][1]['start_ms'], 40250)
            other = fixture();self.assertTrue(ensure_study_sections(other, folder));self.assertEqual(ask.call_count, 1)

    def test_provider_failure_keeps_the_original_lesson_and_records_unavailability(self):
        package = fixture(); original = copy.deepcopy(package)
        with tempfile.TemporaryDirectory() as temporary, patch('app.study_sections.ask_deepseek', side_effect=ProviderError('not available')):
            self.assertFalse(ensure_study_sections(package, Path(temporary)))
        self.assertEqual(package['sentences'], original['sentences']); self.assertEqual(package['chapters'], original['chapters'])
        self.assertEqual(package['processing']['study_section_generation']['status'], 'unavailable')

    def test_source_text_or_measured_time_changes_invalidate_the_outline(self):
        package = fixture(); original = source_fingerprint(package)
        package['sentences'][0]['text'] = 'Corrected source.';self.assertNotEqual(source_fingerprint(package), original)
        package = fixture();package['sentences'][0]['start_ms'] += 40;self.assertNotEqual(source_fingerprint(package), original)


if __name__ == '__main__':unittest.main()
