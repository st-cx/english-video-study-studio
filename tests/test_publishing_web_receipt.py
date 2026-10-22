"""Explicit official-web receipts, with real local revision/manifest checks."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import publishing
from app.render_identity import create_manifest, manifest_path


class WebPublicationReceiptTests(unittest.TestCase):
    def setUp(self):
        work = Path(__file__).resolve().parents[1] / 'work'
        work.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='web-receipt-', dir=work)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile, self.folder = self.root / 'profile', self.root / 'lesson'
        self.profile.mkdir()
        self.folder.mkdir()
        for name, content in (('audio.mp3', b'fixture audio'), ('source.mp4', b'fixture source'),
                              ('lesson.mp4', b'fixture rendered video')):
            (self.folder / name).write_bytes(content)
        self.package = {
            'episode': {'id': 'web-receipt-lesson', 'title': 'An original fixture', 'duration_s': 2,
                        'audio_file': 'audio.mp3'},
            'source': {'name': 'Original fixture', 'url': 'https://example.org/source'},
            'media': {'audio_file': 'audio.mp3', 'video_file': 'source.mp4'},
            'words': [{'id': 'w1', 'sentence_id': 's1', 'text': 'Hello.', 'start_ms': 100, 'end_ms': 600}],
            'sentences': [{'id': 's1', 'text': 'Hello.', 'translation': '你好。', 'start_ms': 100, 'end_ms': 600}],
            'cards': [], 'phrases': [], 'chapters': [{'status': 'ready'}],
            'quality': {'needs_review': False, 'flags': [], 'structural_flags': []},
            'artifacts': {'video': 'ready'}, 'render': {'status': 'ready', 'video_file': 'lesson.mp4'},
        }
        self._save_package()
        manifest = create_manifest(self.package, self.folder, self.folder / 'lesson.mp4', 'lesson.mp4', 0, 2000)
        self.manifest_file = manifest_path(self.folder, 'lesson.mp4')
        self.manifest_file.write_text(json.dumps(manifest), encoding='utf-8')
        for item in (patch.object(publishing, 'data_root', return_value=self.profile),
                     patch.object(publishing, '_credentials', side_effect=AssertionError('web receipt must not read credentials')),
                     patch.object(publishing, '_transport', side_effect=AssertionError('web receipt must not access network'))):
            item.start()
            self.addCleanup(item.stop)
        self.draft = self._draft()

    def _save_package(self):
        (self.folder / 'episode.json').write_text(json.dumps(self.package), encoding='utf-8')

    def _draft(self, **options):
        return publishing.prepare_publication(self.package, self.folder,
            {'rights_confirmed': True, 'auto_publish': True, **options})

    def record(self, bvid='BV1234567890', **options):
        return publishing.record_web_receipt(self.draft['id'], bvid,
            confirmed_in_creator_center=True, **options)

    def test_confirmed_receipt_records_web_channel_without_sdk_login(self):
        result = self.record()
        self.assertEqual(result['status'], 'published')
        self.assertEqual(result['bvid'], 'BV1234567890')
        self.assertEqual(result['url'], 'https://www.bilibili.com/video/BV1234567890')
        self.assertFalse(result['auto_publish'])
        self.assertEqual(result['submission_method'], 'official_web')
        self.assertEqual(result['receipt_verified_via'], 'user_confirmation')
        self.assertEqual(result['platform_status'], 'submitted')
        self.assertTrue(result['submitted_at'])
        self.assertEqual(result['submitted_at'], result['attempted_at'])
        self.assertFalse(result.get('sdk_verified', False))
        self.assertFalse(publishing.public_state()['live_submission_verified'])
        self.assertNotIn('aid', result)
        self.assertFalse((self.profile / 'bilibili.key.dpapi').exists())
        self.assertFalse((self.profile / 'settings.json').exists())
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertNotIn('_folder', result)

    def test_confirmation_must_be_explicit_boolean_true(self):
        for confirmed in (False, None, 'true', 1):
            with self.subTest(confirmed=confirmed), self.assertRaises(ValueError):
                publishing.record_web_receipt(self.draft['id'], 'BV1234567890',
                    confirmed_in_creator_center=confirmed)
        self.assertEqual(publishing.list_publications()[0]['status'], 'draft')

    def test_invalid_video_identifiers_are_rejected(self):
        for bvid in ('', 'bv1234567890', 'BV123', 'BV12345678900', 'https://www.bilibili.com/video/BV1234567890'):
            with self.subTest(bvid=bvid), self.assertRaises(ValueError):
                self.record(bvid)

    def test_repeating_same_receipt_is_idempotent_and_does_not_rewrite_time(self):
        first = self.record()
        raw = (self.profile / 'publications.json').read_bytes()
        with patch.object(publishing, '_now', return_value='2099-01-01T00:00:00Z'):
            second = self.record()
        self.assertEqual(first, second)
        self.assertEqual((self.profile / 'publications.json').read_bytes(), raw)

    def test_same_platform_video_cannot_be_claimed_by_a_different_draft(self):
        self.record()
        other = self._draft(title='A separate draft')
        with self.assertRaises(ValueError):
            publishing.record_web_receipt(other['id'], 'BV1234567890', confirmed_in_creator_center=True)
        self.assertEqual(next(job for job in publishing.list_publications() if job['id'] == other['id'])['status'], 'draft')

    def test_active_started_attempted_or_uncertain_jobs_cannot_be_overwritten(self):
        for status, attempted in (('submitting', False), ('published', False), ('uncertain', False),
                                  ('draft', True), ('blocked', True)):
            job = next(item for item in publishing._read_jobs() if item['id'] == self.draft['id'])
            original = copy.deepcopy(job)
            fields = {'status': status}
            if attempted:
                fields['attempted_at'] = '2026-01-01T00:00:00Z'
            publishing._finish(self.draft['id'], **fields)
            with self.subTest(status=status, attempted=attempted), self.assertRaises(ValueError):
                self.record()
            publishing._mutate(lambda jobs: jobs.__setitem__(0, original))
        publishing._ACTIVE.add(self.draft['id'])
        try:
            with self.assertRaises(ValueError):
                self.record()
        finally:
            publishing._ACTIVE.discard(self.draft['id'])

    def test_different_bvid_cannot_overwrite_an_existing_web_receipt(self):
        first = self.record()
        with self.assertRaises(ValueError):
            self.record('BVabcdefghij')
        self.assertEqual(publishing.list_publications()[0], first)

    def test_package_revision_change_rejects_receipt(self):
        self.package['words'][0]['end_ms'] = 650
        self._save_package()
        with self.assertRaises(ValueError):
            self.record()
        self.assertEqual(publishing.list_publications()[0]['status'], 'draft')

    def test_missing_or_changed_render_manifest_rejects_receipt(self):
        original = self.manifest_file.read_bytes()
        self.manifest_file.unlink()
        with self.assertRaises(ValueError):
            self.record()
        self.manifest_file.write_bytes(original)
        (self.folder / 'lesson.mp4').write_bytes(b'different rendered bytes')
        with self.assertRaises(ValueError):
            self.record()
        self.assertNotIn('attempted_at', publishing.list_publications()[0])

    def test_quality_blockers_cannot_be_overridden_by_web_confirmation(self):
        self.package['quality']['needs_review'] = True
        self._save_package()
        self.draft = self._draft()
        self.assertEqual(self.draft['status'], 'blocked')
        with self.assertRaises(ValueError):
            self.record()
        self.assertEqual(publishing.list_publications()[0]['status'], 'blocked')

    def test_safe_blocked_draft_without_attempt_can_register_explicit_receipt(self):
        publishing._finish(self.draft['id'], status='blocked', error='earlier upload preparation stopped')
        result = self.record(package_provider=lambda episode_id: (self.package, self.folder))
        self.assertEqual(result['status'], 'published')
        self.assertEqual(result['error'], '')


if __name__ == '__main__':
    unittest.main()
