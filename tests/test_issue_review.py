import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import issue_review


def lesson():
    return {'episode': {'id': 'private-lesson', 'title': 'Audio lesson', 'duration_s': 60, 'audio_file': 'audio.mp3'},
            'words': [{'id': 'w1', 'text': 'Hello', 'start_ms': 1000, 'end_ms': 1400},
                      {'id': 'w2', 'text': 'world', 'start_ms': 1400, 'end_ms': 1900}],
            'chapters': [{'reference_issues': [{'start_ms': 1000, 'end_ms': 1400, 'recognized': 'Hello', 'reference': 'Well', 'reason': 'different'}],
                          'alignment_issues': [{'start_ms': 1100, 'end_ms': 1600, 'text': 'Hello', 'reason': 'boundary'}]}]}


class IssueReviewTests(unittest.TestCase):
    def test_overlapping_records_share_one_replay_without_mutating_input(self):
        package = lesson(); before = copy.deepcopy(package)
        manifest = issue_review.build_manifest(package)
        self.assertEqual(manifest['issue_count'], 2)
        self.assertEqual(manifest['group_count'], 1)
        self.assertEqual(manifest['groups'][0]['context'], 'Hello world')
        self.assertEqual(package, before)
        self.assertEqual(manifest, issue_review.build_manifest(package))

    def test_feedback_is_revision_bound_and_not_word_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(issue_review, 'data_root', return_value=Path(tmp)):
            package = lesson(); manifest = issue_review.build_manifest(package)
            issue = manifest['groups'][0]['issues'][0]['id']
            payload = {'revision': manifest['revision'], 'decisions': {issue: {'choice': 'current', 'note': 'heard'}}}
            response = issue_review.save_feedback(package, payload)
            self.assertFalse(response['word_boundary_acceptance'])
            self.assertTrue(issue_review.feedback(package)['decisions'][issue]['human_listening_self_report'])
            package['words'][0]['end_ms'] += 30
            with self.assertRaisesRegex(ValueError, '已修订'):
                issue_review.save_feedback(package, payload)
            self.assertEqual(issue_review.feedback(package)['decisions'], {})

    def test_html_preserves_media_and_escapes_source_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder / 'audio.mp3').write_bytes(b'local-test')
            package = lesson(); package['words'][0]['text'] = '</script><script>alert(1)</script>'
            result = issue_review.write_issue_review(package, folder)
            page = (folder / result['page']).read_text(encoding='utf-8')
            self.assertIn('"media": "../audio.mp3"', page)
            self.assertNotIn('</script><script>alert(1)</script>', page)
            self.assertIn('已听清：当前稿', page)
            package['episode']['audio_file'] = '../outside.mp3'
            with self.assertRaisesRegex(ValueError, '当前课程'):
                issue_review.write_issue_review(package, folder)

    def test_invalid_and_unknown_findings_do_not_save(self):
        package = lesson(); manifest = issue_review.build_manifest(package)
        with tempfile.TemporaryDirectory() as tmp, patch.object(issue_review, 'data_root', return_value=Path(tmp)):
            with self.assertRaises(ValueError):
                issue_review.save_feedback(package, {'revision': manifest['revision'], 'decisions': {'unknown': {'choice': 'current'}}})
        package['chapters'][0]['reference_issues'][0]['start_ms'] = float('nan')
        with self.assertRaises(ValueError):
            issue_review.build_manifest(package)

    def test_replacement_audio_and_fingerprint_invalidate_feedback(self):
        package = lesson(); before = issue_review.revision(package)
        package['episode']['fingerprint'] = 'new-material'
        self.assertNotEqual(before, issue_review.revision(package))
        before = issue_review.revision(package)
        package['episode']['audio_file'] = 'replacement.mp3'
        self.assertNotEqual(before, issue_review.revision(package))
