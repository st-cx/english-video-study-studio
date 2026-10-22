from __future__ import annotations

import base64
import copy
import importlib.util
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app import publishing as pub


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.error = None

    async def create_qr(self):
        return object(), 'data:image/png;base64,' + base64.b64encode(b'fake-qr').decode()

    async def poll_qr(self, login):
        return 'done', {'sessdata': 'private-session-example', 'bili_jct': 'private-csrf-example'}

    async def submit(self, credentials, video, cover, metadata):
        self.calls.append((credentials, video, cover, metadata))
        if self.error:
            raise self.error
        return {'bvid': 'BV1234567890', 'aid': 123}


class PublishingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile = self.root / 'profile'
        self.profile.mkdir()
        self.folder = self.root / 'lesson'
        self.folder.mkdir()
        self.package = {'episode': {'id': 'lesson-1', 'title': 'A local lesson', 'duration_s': 5},
                        'source': {'url': 'https://example.org/source'},
                        'media': {'video_file': 'source.mp4'},
                        'sentences': [{'id': 's1', 'text': 'Hello world.'}],
                        'words': [{'id': 'w1', 'start_ms': 10, 'end_ms': 100}],
                        'chapters': [{'id': 'c1', 'status': 'ready'}],
                        'quality': {'needs_review': False, 'flags': [], 'structural_flags': []},
                        'artifacts': {'video': 'ready'}, 'render': {'status': 'ready', 'video_file': 'lesson.mp4'}}
        self.write_package()
        (self.folder / 'lesson.mp4').write_bytes(b'fake-rendered-video')
        (self.folder / 'source.mp4').write_bytes(b'fake-original-video')
        self.transport = FakeTransport()
        self.patches = [patch.object(pub, 'data_root', return_value=self.profile),
                        patch.object(pub, '_TRANSPORT', self.transport),
                        patch.object(pub.security, '_protect', side_effect=lambda value, decrypt=False: value[4:] if decrypt else b'enc:' + value)]
        for item in self.patches:
            item.start()
        pub._SESSIONS.clear()
        pub._ACTIVE.clear()

    def tearDown(self):
        deadline = time.monotonic() + 5
        while pub._ACTIVE and time.monotonic() < deadline:
            time.sleep(.01)
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def write_package(self):
        (self.folder / 'episode.json').write_text(json.dumps(self.package), encoding='utf-8')

    def draft(self, **options):
        return pub.prepare_publication(self.package, self.folder, {'rights_confirmed': True, **options})

    def login(self):
        start = pub.start_login()
        return pub.poll_login(start['login_id'])

    def wait_finished(self):
        deadline = time.monotonic() + 5
        while pub._ACTIVE and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(pub._ACTIVE)
        return pub.list_publications()[0]

    def test_qr_credentials_encrypted_and_state_never_discloses_them(self):
        self.assertFalse(pub.public_state()['configured'])
        start = pub.start_login()
        self.assertTrue(start['qr_image'].startswith('data:image/png;base64,'))
        self.assertEqual(pub.poll_login(start['login_id'])['status'], 'done')
        encrypted = (self.profile / 'bilibili.key.dpapi').read_bytes()
        self.assertTrue(encrypted.startswith(b'enc:'))
        public = json.dumps(pub.public_state())
        self.assertNotIn('private-session', public)
        self.assertNotIn('csrf', public)
        self.assertFalse((self.profile / 'settings.json').exists())
        self.assertFalse((self.profile / 'deepseek.key.dpapi').exists())
        self.assertFalse(pub.disconnect()['configured'])
        self.assertEqual(pub.poll_login(start['login_id'])['status'], 'expired')

    def test_qr_expiry_does_not_save_credentials(self):
        start = pub.start_login()
        pub._SESSIONS[start['login_id']]['created'] -= 200
        self.assertEqual(pub.poll_login(start['login_id'])['status'], 'expired')
        self.assertFalse((self.profile / 'bilibili.key.dpapi').exists())

    def test_prepare_is_persistent_idempotent_private_and_no_network(self):
        one = self.draft()
        two = self.draft()
        self.assertEqual(one['id'], two['id'])
        self.assertEqual(one['status'], 'draft')
        self.assertEqual(len(pub.list_publications()), 1)
        self.assertNotIn('_folder', json.dumps(one))
        self.assertNotIn(str(self.root), json.dumps(one))
        self.assertFalse(self.transport.calls)
        self.assertEqual(json.loads((self.profile / 'publications.json').read_text())['version'], 1)

    def test_rights_explicit_and_auto_publish_cannot_enable_implicitly(self):
        draft = pub.prepare_publication(self.package, self.folder, {})
        self.assertEqual(draft['status'], 'blocked')
        self.assertFalse(draft['rights_confirmed'])
        with self.assertRaises(ValueError):
            pub.prepare_publication(self.package, self.folder, {'auto_publish': True})
        draft = self.draft(auto_publish='true')
        self.assertFalse(draft['auto_publish'])

    def test_metadata_validation_repost_requires_source(self):
        no_source = copy.deepcopy(self.package)
        no_source['source'] = {}
        with self.assertRaisesRegex(ValueError, '来源'):
            pub.prepare_publication(no_source, self.folder, {'rights_confirmed': True})
        own = pub.prepare_publication(no_source, self.folder, {'copyright': 1, 'rights_confirmed': True})
        self.assertEqual(own['copyright'], 1)
        for options in ({'copyright': True}, {'tid': True}, {'tags': []}, {'tags': ['x'] * 11}, {'title': 'x' * 81}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.draft(**options)

    def test_quality_chapters_and_sync_cannot_be_bypassed(self):
        original = copy.deepcopy(self.package)
        for field in ('quality', 'chapters', 'sync_review'):
            self.package = copy.deepcopy(original)
            if field == 'quality':
                self.package['quality']['needs_review'] = True
            elif field == 'chapters':
                self.package['chapters'][0]['status'] = 'review'
            else:
                self.package['sync_review'] = {'human_status': 'incomplete', 'human_annotated_count': 0}
            with self.subTest(field=field):
                self.write_package()
                draft = self.draft()
                self.assertEqual(draft['status'], 'blocked')
                with self.assertRaises(ValueError):
                    pub.submit_publication(draft['id'])
                self.assertEqual(self.package[field], json.loads((self.folder / 'episode.json').read_text())[field])
        self.assertFalse(self.transport.calls)

    def test_unrendered_media_cannot_be_posted(self):
        for name in ('source.mp4', '../outside.mp4', str(self.folder / 'lesson.mp4'), 'lesson.rendering.mp4', 'audio.mp3'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.draft(video_file=name)
        (self.folder / 'lesson.mp4').unlink()
        draft = self.draft()
        self.assertEqual(draft['status'], 'blocked')
        self.assertTrue(any('导出' in text for text in draft['blockers']))

    def test_stale_timeline_render_and_cover_rejected(self):
        draft = self.draft()
        self.package['words'][0]['end_ms'] += 10
        self.write_package()
        with self.assertRaisesRegex(ValueError, '修订'):
            pub.submit_publication(draft['id'])
        draft = self.draft()
        (self.folder / 'lesson.mp4').write_bytes(b'changed-video')
        with self.assertRaisesRegex(ValueError, '视频已改变'):
            pub.submit_publication(draft['id'])

    def test_submit_uses_actual_transport_shape_and_prevents_duplicates(self):
        self.login()
        draft = self.draft()
        self.assertEqual(pub.submit_publication(draft['id'])['status'], 'submitting')
        final = self.wait_finished()
        self.assertEqual(final['status'], 'published')
        self.assertEqual(final['bvid'], 'BV1234567890')
        self.assertEqual(len(self.transport.calls), 1)
        credentials, video, cover, metadata = self.transport.calls[0]
        self.assertEqual(video, self.folder / 'lesson.mp4')
        self.assertTrue(cover.is_file())
        self.assertEqual(metadata['tag'], '英语学习,精听')
        self.assertEqual(metadata['copyright'], 2)
        self.assertEqual(metadata['source'], 'https://example.org/source')
        self.assertEqual(pub.submit_publication(draft['id'])['status'], 'published')
        self.assertEqual(len(self.transport.calls), 1)
        self.assertNotIn('private-session', json.dumps(pub.list_publications()))

    def test_timeout_is_uncertain_never_resubmitted(self):
        self.login()
        self.transport.error = TimeoutError('secret-cookie-must-not-leak')
        draft = self.draft(auto_publish=True)
        pub.submit_publication(draft['id'])
        final = self.wait_finished()
        self.assertEqual(final['status'], 'uncertain')
        self.assertNotIn('secret-cookie', json.dumps(final))
        self.assertEqual(pub.process_ready_publications(), [])
        self.assertEqual(pub.submit_publication(draft['id'])['status'], 'uncertain')
        self.assertEqual(len(self.transport.calls), 1)

    def test_restart_submitting_becomes_uncertain_no_blind_retry(self):
        draft = self.draft(auto_publish=True)
        pub._finish(draft['id'], status='submitting')
        self.assertEqual(pub.recover_publications()[0]['status'], 'uncertain')
        self.assertEqual(pub.process_ready_publications(), [])
        self.assertFalse(self.transport.calls)

    def test_new_title_does_not_bypass_uncertain_duplicate_guard(self):
        self.login()
        draft = self.draft()
        pub._finish(draft['id'], status='uncertain', attempted_at='2026-01-01T00:00:00Z')
        alternative = self.draft(title='An edited title')
        self.assertNotEqual(draft['id'], alternative['id'])
        with self.assertRaisesRegex(ValueError, '禁止重复'):
            pub.submit_publication(alternative['id'])
        self.assertFalse(self.transport.calls)

    @unittest.skipUnless(importlib.util.find_spec('bilibili_api'), 'optional Bilibili adapter is not installed')
    def test_installed_adapter_construction_contract_without_network(self):
        draft = self.draft()
        job = pub._read_jobs()[0]
        cover = pub._cover(job, self.folder)
        with patch('bilibili_api.video_uploader.VideoUploader') as uploader:
            uploader.return_value.start = AsyncMock(return_value={'bvid': 'BV1234567890', 'aid': 10})
            result = pub._runner().run(pub._BilibiliTransport().submit(
                {'sessdata': 'local-test', 'bili_jct': 'local-test'}, self.folder / 'lesson.mp4', cover, pub._metadata(job)))
            self.assertEqual(result['aid'], 10)
            args, kwargs = uploader.call_args
            self.assertEqual(args[1]['title'], draft['title'])
            self.assertEqual(kwargs['cover'], str(cover))
            uploader.return_value.start.assert_awaited_once()

    def test_auto_publish_only_explicit_ready_opt_in(self):
        self.login()
        manual = self.draft()
        self.assertEqual(pub.process_ready_publications(), [])
        automatic = self.draft(auto_publish=True)
        result = pub.process_ready_publications(lambda episode_id: (self.package, self.folder))
        self.assertEqual(len(result), 1)
        self.wait_finished()
        manual_after = next(job for job in pub.list_publications() if job['id'] == manual['id'])
        automatic_after = next(job for job in pub.list_publications() if job['id'] == automatic['id'])
        self.assertEqual(manual_after['status'], 'draft')
        self.assertEqual(automatic_after['status'], 'published')
        self.assertEqual(len(self.transport.calls), 1)

    def test_bundle_is_self_contained_and_has_no_authentication(self):
        self.login()
        draft = self.draft()
        result = pub.export_publication_bundle(draft['id'])
        with zipfile.ZipFile(self.folder / result['bundle_file']) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), {'lesson.mp4', 'cover.png', 'metadata.json', '投稿说明.txt'})
            metadata = json.loads(archive.read('metadata.json'))
            self.assertEqual(metadata['title'], 'A local lesson')
            self.assertEqual(archive.read('lesson.mp4'), b'fake-rendered-video')
            content = b''.join(archive.read(name) for name in archive.namelist())
            self.assertNotIn(b'private-session', content)
            self.assertNotIn(b'csrf', content)
        self.assertFalse(self.transport.calls)

    def test_corrupt_persistence_is_not_overwritten(self):
        path = self.profile / 'publications.json'
        path.write_text('{broken')
        with self.assertRaisesRegex(ValueError, '损坏'):
            self.draft()
        self.assertEqual(path.read_text(), '{broken')


if __name__ == '__main__':
    unittest.main()
