from __future__ import annotations

import base64
import asyncio
import copy
import importlib.util
import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import urlsplit

from app import publishing as pub


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.error = None
        self.error_before_submit = False
        self.valid = True
        self.archives = []

    async def check_account(self, credentials):
        return {'valid': self.valid, 'name': 'Local test account' if self.valid else ''}

    async def recent_uploads(self, credentials):
        return self.archives

    async def create_qr(self):
        return object(), 'data:image/png;base64,' + base64.b64encode(b'fake-qr').decode()

    async def poll_qr(self, login):
        return 'done', {'sessdata': 'private-session-example', 'bili_jct': 'private-csrf-example'}

    async def submit(self, credentials, video, cover, metadata, progress=None):
        self.calls.append((credentials, video, cover, metadata))
        if progress:
            progress({'stage': 'uploading', 'progress': 50, 'stage_message': 'Uploading fixture'})
        if self.error and self.error_before_submit:
            raise self.error
        if progress:
            progress({'stage': 'submitting', 'progress': 97, 'stage_message': 'Submitting fixture', 'cid': 9876})
        if self.error:
            raise self.error
        if progress:
            progress({'stage': 'accepted', 'progress': 100, 'bvid': 'BV1234567890', 'aid': 123})
        return {'bvid': 'BV1234567890', 'aid': 123}


class PublishingTests(unittest.TestCase):
    def setUp(self):
        work = Path(__file__).resolve().parents[1] / 'work'
        work.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='publishing-test-', dir=work)
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
                        patch.object(pub, '_verify_render', return_value={'status': 'verified'}),
                        patch.object(pub.security, '_protect', side_effect=lambda value, decrypt=False: value[4:] if decrypt else b'enc:' + value)]
        for item in self.patches:
            item.start()
        pub._SESSIONS.clear()
        pub._ACTIVE.clear()
        pub._ACCOUNT_CHECK.clear()
        pub._LAST_LOGIN_DIAGNOSTIC.clear()
        pub._LOGIN_DIAGNOSTIC_HISTORY.clear()

    def tearDown(self):
        sessions = list(pub._SESSIONS.values())
        with pub._LOCK:
            pub._invalidate_login_sessions()
        for session in sessions:
            if session.get('worker'):
                session['worker'].join(timeout=5)
                self.assertFalse(session['worker'].is_alive())
        deadline = time.monotonic() + 5
        while pub._ACTIVE and time.monotonic() < deadline:
            time.sleep(.01)
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def write_package(self):
        (self.folder / 'episode.json').write_text(json.dumps(self.package), encoding='utf-8')

    def draft(self, **options):
        return pub.prepare_publication(self.package, self.folder, {'rights_confirmed': True, 'copyright': 2, **options})

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
        self.assertFalse(pub.public_state()['verified'])
        self.assertTrue(pub.check_account()['verified'])
        self.assertEqual(pub.public_state()['account_name'], 'Local test account')
        self.assertFalse(pub.disconnect()['configured'])
        self.assertEqual(pub.poll_login(start['login_id'])['status'], 'expired')

    def test_qr_expiry_does_not_save_credentials(self):
        start = pub.start_login()
        pub._SESSIONS[start['login_id']]['created'] -= 200
        self.assertEqual(pub.poll_login(start['login_id'])['status'], 'expired')
        self.assertFalse((self.profile / 'bilibili.key.dpapi').exists())

    def test_completed_login_stays_done_after_qr_lifetime_but_disconnect_invalidates_it(self):
        started = pub.start_login()
        self.assertEqual(pub.poll_login(started['login_id'])['status'], 'done')
        path = self.profile / 'bilibili.key.dpapi'
        saved = path.read_bytes()
        pub._SESSIONS[started['login_id']]['created'] -= 181
        with patch.object(self.transport, 'poll_qr', AsyncMock(side_effect=AssertionError('completed login must not repoll'))) as query:
            result = pub.poll_login(started['login_id'])
        self.assertEqual(result['status'], 'done')
        self.assertTrue(result['configured'])
        self.assertEqual(pub.public_state()['login_diagnostic']['status'], 'done')
        self.assertEqual(path.read_bytes(), saved)
        query.assert_not_awaited()
        pub.disconnect()
        self.assertEqual(pub.poll_login(started['login_id'])['status'], 'expired')
        self.assertFalse(path.exists())

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
            pub.prepare_publication(no_source, self.folder, {'copyright': 2, 'rights_confirmed': True})
        own = pub.prepare_publication(no_source, self.folder, {'copyright': 1, 'rights_confirmed': True})
        self.assertEqual(own['copyright'], 1)
        for options in ({'copyright': True}, {'tid': True}, {'tags': []}, {'tags': ['x'] * 11}, {'title': 'x' * 81}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.draft(**options)

    def test_new_draft_defaults_to_original_and_preserves_source_in_platform_description(self):
        description = '作品作者、许可和修改说明完整保留。'
        draft = pub.prepare_publication(self.package, self.folder,
            {'rights_confirmed': True, 'description': description})
        self.assertEqual(draft['copyright'], 1)
        self.assertEqual(draft['source'], 'https://example.org/source')
        metadata = pub._metadata(draft)
        self.assertNotIn('source', metadata)
        self.assertEqual(metadata['desc'], description + '\n\n原始来源：https://example.org/source')
        self.assertEqual(draft['description'], description)

    def test_explicit_repost_still_overrides_the_new_default(self):
        own = pub.prepare_publication(self.package, self.folder, {'rights_confirmed': True})
        repost = pub.prepare_publication(self.package, self.folder, {'rights_confirmed': True, 'copyright': 2})
        self.assertEqual(own['copyright'], 1)
        self.assertEqual(repost['copyright'], 2)
        self.assertNotEqual(own['id'], repost['id'])
        self.assertEqual(pub._metadata(repost)['source'], 'https://example.org/source')
        self.assertEqual(pub._metadata(repost)['desc'], '')

    def test_new_default_never_mutates_saved_or_submitted_repost_records(self):
        self.draft(title='Saved repost')
        submitted = self.draft(title='Previously submitted repost')
        pub._finish(submitted['id'], status='published', bvid='BV1234567890', aid=123,
                    attempted_at='2026-01-01T00:00:00Z')
        previous = copy.deepcopy(pub._read_jobs())
        new = pub.prepare_publication(self.package, self.folder, {'rights_confirmed': True})
        self.assertEqual(new['copyright'], 1)
        current = {job['id']: job for job in pub._read_jobs()}
        for old in previous:
            with self.subTest(status=old['status']):
                self.assertEqual(current[old['id']], old)
                self.assertEqual(current[old['id']]['copyright'], 2)
        self.assertFalse(self.transport.calls)

    def test_original_metadata_does_not_duplicate_an_existing_source_or_remove_attribution(self):
        description = '原作者与授权链接保留。\n原始来源：https://example.org/source\n节选与字幕修改说明。'
        draft = self.draft(copyright=1, description=description)
        metadata = pub._metadata(draft)
        self.assertEqual(metadata['desc'], description)
        self.assertEqual(metadata['desc'].count('https://example.org/source'), 1)
        self.assertNotIn('source', metadata)

    def test_description_limit_includes_added_source_before_preparing_upload(self):
        with self.assertRaisesRegex(ValueError, '简介'):
            self.draft(copyright=1, description='x' * 2000)
        self.assertFalse(self.transport.calls)
        self.assertEqual(pub.list_publications(), [])
        repost = self.draft(copyright=2, description='x' * 2000)
        self.assertEqual(len(pub._metadata(repost)['desc']), 2000)

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
        self.assertEqual(metadata['no_reprint'], 0)
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

    def test_render_manifest_is_checked_before_preparation_not_only_after_draft(self):
        with patch.object(pub, '_verify_render', side_effect=ValueError('成片对应旧内容，请重新导出。')):
            draft = self.draft()
            self.assertEqual(draft['status'], 'blocked')
            self.assertIn('成片对应旧内容', ' '.join(draft['blockers']))
        self.assertFalse(self.transport.calls)

    def test_expired_login_stops_before_upload_and_remains_private(self):
        self.login()
        self.transport.valid = False
        self.assertEqual(pub.check_account()['status'], 'expired')
        draft = self.draft()
        pub.submit_publication(draft['id'])
        final = self.wait_finished()
        self.assertEqual(final['status'], 'blocked')
        self.assertNotIn('attempted_at', final)
        self.assertFalse(self.transport.calls)
        self.assertNotIn('private-session', json.dumps(final))

    def test_upload_failure_is_retryable_but_not_automatically_replayed(self):
        self.login()
        self.transport.error = TimeoutError('private-upload-url')
        self.transport.error_before_submit = True
        draft = self.draft(auto_publish=True)
        pub.submit_publication(draft['id'])
        final = self.wait_finished()
        self.assertEqual(final['status'], 'blocked')
        self.assertEqual(final['stage'], 'uploading')
        self.assertNotIn('attempted_at', final)
        self.assertEqual(pub.process_ready_publications(), [])
        self.transport.error = None
        pub.submit_publication(draft['id'])
        self.assertEqual(self.wait_finished()['status'], 'published')
        self.assertEqual(len(self.transport.calls), 2)

    def test_recovery_distinguishes_upload_from_final_archive_submission(self):
        draft = self.draft()
        pub._finish(draft['id'], status='submitting', stage='uploading')
        self.assertEqual(pub.recover_publications()[0]['status'], 'blocked')
        pub._finish(draft['id'], status='submitting', stage='submitting', attempted_at='2026-01-01T00:00:00Z')
        self.assertEqual(pub.recover_publications()[0]['status'], 'uncertain')
        self.assertFalse(self.transport.calls)

    def test_uncertain_lookup_only_returns_safe_matching_candidates_never_retries(self):
        self.login()
        draft = self.draft()
        pub._finish(draft['id'], status='uncertain', attempted_at='2026-01-01T00:00:00Z', cid=42)
        self.transport.archives = [
            {'Archive': {'bvid': 'BV1234567890', 'aid': 123, 'title': draft['title'], 'private': 'secret'}},
            {'archive': {'bvid': 'BVabcdefghij', 'aid': 124, 'title': 'Other title', 'ctime': 100}, 'videos': [{'cid': 42}]},
            {'Archive': {'bvid': 'BV0123456789', 'aid': 125, 'title': 'Unrelated'}},
            {'Archive': {'bvid': 'javascript:bad', 'aid': 123, 'title': draft['title']}},
            {'Archive': {'bvid': 'BV1234567890', 'aid': 123, 'title': draft['title']}},
        ]
        result = pub.reconcile_publication(draft['id'])
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(len(result['candidates']), 2)
        self.assertEqual(result['candidates'][0]['matched_by'], 'uploaded_part')
        self.assertNotIn('secret', json.dumps(result))
        self.assertNotIn('private-session', json.dumps(result))
        pub.submit_publication(draft['id'])
        self.assertFalse(self.transport.calls)

    def test_empty_remote_list_never_proves_submission_failed(self):
        self.login()
        draft = self.draft()
        pub._finish(draft['id'], status='uncertain', attempted_at='2026-01-01T00:00:00Z')
        result = pub.reconcile_publication(draft['id'])
        self.assertEqual(result['reconciliation_status'], 'not_found')
        self.assertEqual(result['status'], 'uncertain')
        self.assertFalse(self.transport.calls)

    def test_valid_archive_response_survives_later_sdk_failure(self):
        self.login()
        async def accepted_then_fail(credentials, video, cover, metadata, progress=None):
            progress({'stage': 'submitting', 'progress': 97})
            progress({'stage': 'accepted', 'progress': 100, 'bvid': 'BV1234567890', 'aid': 123})
            raise RuntimeError('private-sdk-tail-failure')
        self.transport.submit = accepted_then_fail
        draft = self.draft()
        pub.submit_publication(draft['id'])
        result = self.wait_finished()
        self.assertEqual(result['status'], 'published')
        self.assertEqual(result['bvid'], 'BV1234567890')
        self.assertNotIn('private-sdk', json.dumps(result))

    def test_disconnect_during_upload_stops_before_final_remote_submit(self):
        self.login()
        submitted = []
        async def disconnect_then_submit(credentials, video, cover, metadata, progress=None):
            pub.disconnect()
            progress({'stage': 'submitting', 'progress': 97})
            submitted.append(True)
        self.transport.submit = disconnect_then_submit
        draft = self.draft()
        pub.submit_publication(draft['id'])
        result = self.wait_finished()
        self.assertEqual(result['status'], 'blocked')
        self.assertNotIn('attempted_at', result)
        self.assertEqual(submitted, [])

    def test_qr_encryption_failure_exposes_safe_stage_not_private_exception(self):
        start = pub.start_login()
        with patch.object(pub.security, '_protect', side_effect=OSError('private-credential-and-path')):
            with self.assertRaisesRegex(ValueError, r'login\.protect\.system') as error:
                pub.poll_login(start['login_id'])
        self.assertNotIn('private-credential', str(error.exception))
        self.assertFalse((self.profile / 'bilibili.key.dpapi').exists())

    def test_qr_save_failure_keeps_existing_credentials_and_has_safe_stage(self):
        path = self.profile / 'bilibili.key.dpapi'
        previous = b'enc:{"sessdata":"old-session","bili_jct":"old-csrf"}'
        path.write_bytes(previous)
        start = pub.start_login()
        with patch.object(Path, 'replace', side_effect=PermissionError('private-credential-and-path')):
            with self.assertRaisesRegex(ValueError, r'login\.save\.permission') as error:
                pub.poll_login(start['login_id'])
        self.assertEqual(path.read_bytes(), previous)
        self.assertNotIn('private-credential', str(error.exception))
        self.assertEqual(list(self.profile.glob('*.tmp')), [])

    def test_explicit_monitor_saves_phone_confirmation_without_client_polling(self):
        started = pub.start_login(monitor=True)
        session = pub._SESSIONS[started['login_id']]
        session['worker'].join(timeout=3)
        self.assertFalse(session['worker'].is_alive())
        self.assertTrue((self.profile / 'bilibili.key.dpapi').is_file())
        state = pub.public_state()
        self.assertEqual(state['login_diagnostic']['status'], 'done')
        self.assertEqual(state['last_login_error'], '')
        self.assertNotIn('private-session', json.dumps(state))

    def test_disconnect_invalidates_inflight_monitor_before_saving_credentials(self):
        requested, release = threading.Event(), threading.Event()
        async def delayed_confirmation(login):
            requested.set()
            while not release.is_set():
                await asyncio.sleep(.005)
            return 'done', {'sessdata': 'private-old-session', 'bili_jct': 'private-csrf'}
        self.transport.poll_qr = delayed_confirmation
        started = pub.start_login(monitor=True)
        session = pub._SESSIONS[started['login_id']]
        try:
            self.assertTrue(requested.wait(timeout=2))
            pub.disconnect()
        finally:
            release.set()
            session['worker'].join(timeout=3)
        self.assertFalse(session['worker'].is_alive())
        self.assertFalse((self.profile / 'bilibili.key.dpapi').exists())
        self.assertFalse(pub.public_state()['configured'])

    def test_timeout_keeps_previous_meaningful_login_error(self):
        self.transport.poll_qr = AsyncMock(side_effect=[TimeoutError('private-token-in-error'), ('timeout', None)])
        started = pub.start_login()
        with self.assertRaisesRegex(ValueError, r'login\.poll\.timeout'):
            pub.poll_login(started['login_id'])
        pub._SESSIONS[started['login_id']]['last_poll'] -= 2
        final = pub.poll_login(started['login_id'])
        diagnostic = pub.public_state()['login_diagnostic']
        self.assertEqual(final['status'], 'timeout')
        self.assertEqual(diagnostic['status'], 'timeout')
        self.assertEqual(diagnostic['error_category'], 'timeout')
        self.assertIn('login.poll.timeout', final['last_error'])
        self.assertIn('login.poll.timeout', pub.public_state()['last_login_error'])
        self.assertNotIn('private-token', json.dumps(diagnostic))
        self.assertNotIn('login_id', diagnostic)

    def test_reuse_keeps_same_qr_and_reports_remaining_lifetime_without_reset(self):
        self.transport.create_qr = AsyncMock(wraps=self.transport.create_qr)
        first = pub.start_login()
        session = pub._SESSIONS[first['login_id']]
        session['created'] -= 35
        created = session['created']
        second = pub.start_login(reuse=True)
        self.assertEqual(second['login_id'], first['login_id'])
        self.assertEqual(second['qr_image'], first['qr_image'])
        self.assertTrue(140 <= second['expires_in'] <= 145)
        self.assertEqual(session['created'], created)
        self.transport.create_qr.assert_awaited_once()
        self.assertFalse(session['stop'].is_set())

    def test_default_new_qr_still_invalidates_previous_session(self):
        first = pub.start_login()
        session = pub._SESSIONS[first['login_id']]
        second = pub.start_login()
        self.assertNotEqual(first['login_id'], second['login_id'])
        self.assertTrue(session['stop'].is_set())
        self.assertNotIn(first['login_id'], pub._SESSIONS)

    def test_done_timeout_and_locally_expired_qr_are_not_reused(self):
        for status, elapsed in (('done', 0), ('timeout', 0), ('scan', 181)):
            first = pub.start_login()
            session = pub._SESSIONS[first['login_id']]
            session['status'] = status
            session['created'] -= elapsed
            second = pub.start_login(reuse=True)
            with self.subTest(status=status, elapsed=elapsed):
                self.assertNotEqual(first['login_id'], second['login_id'])
                self.assertTrue(session['stop'].is_set())

    def test_reusing_unmonitored_session_can_enable_background_confirmation(self):
        first = pub.start_login()
        second = pub.start_login(monitor=True, reuse=True)
        self.assertEqual(first['login_id'], second['login_id'])
        session = pub._SESSIONS[first['login_id']]
        session['worker'].join(timeout=3)
        self.assertFalse(session['worker'].is_alive())
        self.assertTrue((self.profile / 'bilibili.key.dpapi').exists())

    def test_new_qr_invalidates_inflight_old_monitor_before_writing(self):
        requested, release = threading.Event(), threading.Event()
        async def delayed_confirmation(login):
            requested.set()
            while not release.is_set():
                await asyncio.sleep(.005)
            return 'done', {'sessdata': 'private-old-session', 'bili_jct': 'private-csrf'}
        self.transport.poll_qr = delayed_confirmation
        old = pub.start_login(monitor=True)
        session = pub._SESSIONS[old['login_id']]
        try:
            self.assertTrue(requested.wait(timeout=2))
            current = pub.start_login()
            self.assertNotEqual(old['login_id'], current['login_id'])
        finally:
            release.set()
            session['worker'].join(timeout=3)
        self.assertFalse(session['worker'].is_alive())
        self.assertFalse((self.profile / 'bilibili.key.dpapi').exists())

    def test_automatic_draft_waits_for_login_without_blocking_then_dispatches(self):
        draft = self.draft(auto_publish=True)
        with patch.object(pub, 'submit_publication') as submit, patch.object(pub, 'start_login') as generate_qr:
            self.assertEqual(pub.process_ready_publications(), [])
            submit.assert_not_called()
            generate_qr.assert_not_called()
        waiting = pub.list_publications()[0]
        self.assertEqual(waiting['status'], 'draft')
        self.assertEqual(waiting['error'], '')
        self.assertNotIn('attempted_at', waiting)
        self.login()
        results = pub.process_ready_publications()
        self.assertEqual(results[0]['id'], draft['id'])
        self.assertEqual(self.wait_finished()['status'], 'published')
        self.assertEqual(len(self.transport.calls), 1)

    def test_waiting_login_does_not_bypass_quality_blockers(self):
        self.package['quality']['needs_review'] = True
        self.write_package()
        draft = self.draft(auto_publish=True)
        self.assertEqual(draft['status'], 'blocked')
        self.assertEqual(pub.process_ready_publications(), [])
        self.login()
        self.assertEqual(pub.process_ready_publications(), [])
        self.assertEqual(pub.list_publications()[0]['status'], 'blocked')
        self.assertFalse(self.transport.calls)

    def test_pause_automatic_draft_prevents_dispatch_without_deleting_draft(self):
        self.login()
        draft = self.draft(auto_publish=True)
        paused = pub.pause_auto_publication(draft['id'])
        self.assertFalse(paused['auto_publish'])
        self.assertEqual(paused['status'], 'draft')
        self.assertEqual(len(pub.list_publications()), 1)
        self.assertEqual(pub.process_ready_publications(), [])
        self.assertFalse(self.transport.calls)

    def test_pausing_after_automatic_scanner_snapshot_still_prevents_claim(self):
        self.login()
        draft = self.draft(auto_publish=True)
        stale = pub.list_publications()
        pub.pause_auto_publication(draft['id'])
        with patch.object(pub, 'list_publications', return_value=stale):
            self.assertEqual(pub.process_ready_publications(), [])
        self.assertFalse(self.transport.calls)
        self.assertFalse(pub._ACTIVE)

    def test_pause_never_claims_to_recall_started_or_uncertain_publication(self):
        draft = self.draft(auto_publish=True)
        for status, attempted in (('submitting', False), ('published', False), ('uncertain', False),
                                  ('draft', True), ('blocked', True)):
            fields = {'status': status}
            if attempted:
                fields['attempted_at'] = '2026-01-01T00:00:00Z'
            pub._finish(draft['id'], **fields)
            with self.subTest(status=status, attempted=attempted), self.assertRaisesRegex(ValueError, '不能.*撤回'):
                pub.pause_auto_publication(draft['id'])
        self.assertTrue(pub.list_publications()[0]['auto_publish'])

    def test_safe_login_history_survives_new_qr_without_tokens_or_session_identifiers(self):
        first = pub.start_login()
        session = pub._SESSIONS[first['login_id']]
        error = pub._QRFlowError('credentials', 'unexpected_address',
            urlsplit('https://account.bilibili.com/crossDomain?SESSDATA=private-query#private-fragment'))
        pub._record_login_diagnostic(first['login_id'], session, 'error', error)
        pub._record_login_diagnostic(first['login_id'], session, 'error', error)
        self.assertEqual(len(pub.public_state()['login_diagnostic_history']), 2)
        second = pub.start_login()
        current = pub._SESSIONS[second['login_id']]
        pub._record_login_diagnostic(second['login_id'], current, 'scan')
        history = pub.public_state()['login_diagnostic_history']
        self.assertEqual(len(history), 3)
        self.assertEqual(history[1]['callback_hostname'], 'account.bilibili.com')
        self.assertEqual(history[1]['callback_path'], '/crossDomain')
        serialized = json.dumps(history)
        for private in ('private-query', 'private-fragment', first['login_id'], second['login_id'], 'qr_image', 'qrcode_key'):
            self.assertNotIn(private, serialized)

    def test_login_history_is_bounded_and_disconnect_clears_it(self):
        for _ in range(15):
            pub.start_login()
        self.assertEqual(len(pub.public_state()['login_diagnostic_history']), 12)
        pub.disconnect()
        self.assertEqual(pub.public_state()['login_diagnostic_history'], [])
        self.assertEqual(pub.public_state()['login_diagnostic'], {})


if __name__ == '__main__':
    unittest.main()
