"""Installed SDK contracts, without account access, downloads or publication."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app import publishing


class _QRApiStub:
    responses = []
    calls = []

    def __init__(self, **definition):
        self.record = {'definition': definition}
        self.calls.append(self.record)

    def update_params(self, **params):
        self.record['params'] = params
        return self

    def update_headers(self, **headers):
        self.record['headers'] = headers
        return self

    @property
    def result(self):
        async def deliver():
            result = self.responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return deliver()


@unittest.skipUnless(importlib.util.find_spec('bilibili_api'), 'optional Bilibili SDK unavailable')
class InstalledUploadContractTests(unittest.TestCase):
    def setUp(self):
        _QRApiStub.responses = []
        _QRApiStub.calls = []

    def _poll(self, response):
        _QRApiStub.responses.append(response)
        with patch('bilibili_api.utils.network.Api', _QRApiStub):
            return asyncio.run(publishing._BilibiliTransport().poll_qr({'qrcode_key': 'private-test-key'}))

    def test_qr_generation_is_in_memory_and_each_request_busts_caches(self):
        _QRApiStub.responses = [{'url': 'https://passport.bilibili.com/h5-app/passport/login/scan?key=test',
                                'qrcode_key': 'private-test-key'}] * 2
        with patch('bilibili_api.utils.network.Api', _QRApiStub):
            first = asyncio.run(publishing._BilibiliTransport().create_qr())
            second = asyncio.run(publishing._BilibiliTransport().create_qr())
        self.assertEqual(first[0], {'qrcode_key': 'private-test-key'})
        self.assertTrue(base64.b64decode(second[1].split(',', 1)[1]).startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertNotEqual(_QRApiStub.calls[0]['params']['_nonce'], _QRApiStub.calls[1]['params']['_nonce'])
        self.assertTrue(all('no-cache' in call['headers']['Cache-Control'] for call in _QRApiStub.calls))
        self.assertTrue(all(any(key.lower() == 'user-agent' for key in call['headers']) for call in _QRApiStub.calls))

    def test_current_official_account_scan_host_is_allowed_but_other_hosts_are_rejected(self):
        for host, accepted in (('account.bilibili.com', True), ('passport.bilibili.com', True),
                               ('other.bilibili.com', False), ('account.bilibili.com.evil.example', False)):
            _QRApiStub.responses = [{'url': f'https://{host}/h5/account-h5/auth/scan-web?key=test',
                                    'qrcode_key': 'private-test-key'}]
            with self.subTest(host=host), patch('bilibili_api.utils.network.Api', _QRApiStub):
                if accepted:
                    result = asyncio.run(publishing._BilibiliTransport().create_qr())
                    self.assertTrue(result[1].startswith('data:image/png;base64,'))
                else:
                    with self.assertRaisesRegex(ValueError, r'login\.generate\.unexpected_address'):
                        asyncio.run(publishing._BilibiliTransport().create_qr())

    def test_poll_success_without_refresh_token_preserves_encoded_cookie_once(self):
        from bilibili_api import Credential
        status, result = self._poll({'code': 0, 'url':
            'https://passport.biligame.com/crossDomain?SESSDATA=a%2Cb%26c%3Dd&bili_jct=csrf&DedeUserID=42&gourl=x'})
        self.assertEqual(status, 'done')
        self.assertEqual(result, {'sessdata': 'a%2Cb%26c%3Dd', 'bili_jct': 'csrf', 'dedeuserid': '42'})
        self.assertEqual(Credential(**result).sessdata, 'a%2Cb%26c%3Dd')
        _, nested = self._poll({'code': 0, 'url':
            'https://passport.bilibili.com/crossDomain?SESSDATA=a%2526b&bili_jct=csrf', 'refresh_token': 'refresh'})
        self.assertEqual(nested['sessdata'], 'a%2526b')
        self.assertEqual(nested['ac_time_value'], 'refresh')

    def test_poll_requests_remain_fresh_across_scan_confirm_and_timeout(self):
        for code, status in ((86101, 'scan'), (86090, 'confirm'), (86038, 'timeout')):
            with self.subTest(code=code):
                self.assertEqual(self._poll({'code': code}), (status, None))
        nonces = [call['params']['_nonce'] for call in _QRApiStub.calls]
        self.assertEqual(len(set(nonces)), 3)
        self.assertTrue(all('no-cache' in call['headers']['Cache-Control'] for call in _QRApiStub.calls))

    def test_unknown_or_missing_status_never_defaults_to_login_success(self):
        for response in ({'code': 12345}, {'code': False}, {'code': '0'}, {}):
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, r'login\.poll\.'):
                self._poll(response)

    def test_missing_or_ambiguous_cookie_fields_have_safe_stage_errors(self):
        for address in (
                'https://passport.bilibili.com/crossDomain?SESSDATA=private-secret',
                'https://passport.bilibili.com/crossDomain?SESSDATA=&bili_jct=csrf',
                'https://passport.bilibili.com/crossDomain?SESSDATA=x&sessdata=y&bili_jct=csrf',
                'https://evil.example/crossDomain?SESSDATA=private-secret&bili_jct=csrf',
                ''):
            with self.subTest(address=address), self.assertRaisesRegex(ValueError, r'login\.credentials\.') as error:
                self._poll({'code': 0, 'url': address})
            self.assertNotIn('private-secret', str(error.exception))
            if address:
                self.assertNotIn(address, str(error.exception))

    def test_sdk_request_failure_never_exposes_sensitive_exception_text(self):
        with self.assertRaisesRegex(ValueError, r'login\.poll\.timeout') as error:
            self._poll(TimeoutError('private-cookie-url'))
        self.assertNotIn('private-cookie', str(error.exception))

    def test_synchronous_sdk_listener_marks_submission_before_remote_call(self):
        from bilibili_api.utils.AsyncEvent import AsyncEvent
        progress, order = [], []

        class SDKUploader(AsyncEvent):
            def __init__(self, *args, **kwargs):
                super().__init__()

            async def start(self):
                self.dispatch('PREUPLOAD', {'auth': 'never-export-this'})
                for number in (1, 0, 1):
                    self.dispatch('AFTER_CHUNK', {'chunk_number': number, 'total_chunk_count': 2,
                                                  'page': object(), 'private_url': 'never-export-this'})
                self.dispatch('PRE_SUBMIT', {'videos': [{'cid': 42, 'filename': 'private-object-name'}],
                                             'csrf': 'never-export-this'})
                order.append('remote-call')
                result = {'bvid': 'BV1234567890', 'aid': 123, 'private': 'never-export-this'}
                self.dispatch('AFTER_SUBMIT', result)
                return result

        def record(update):
            progress.append(update)
            if update['stage'] == 'submitting':
                order.append('durable-submit-boundary')

        work = Path(__file__).resolve().parents[1] / 'work'
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='upload-contract-', dir=work) as temporary, patch('bilibili_api.video_uploader.VideoUploader', SDKUploader):
            result = asyncio.run(publishing._BilibiliTransport().submit(
                {'sessdata': 'fake', 'bili_jct': 'fake'}, Path(temporary) / 'lesson.mp4',
                Path(temporary) / 'cover.png', {'title': 'Original test'}, progress=record))
        self.assertEqual(result['aid'], 123)
        self.assertEqual(order, ['durable-submit-boundary', 'remote-call'])
        self.assertEqual(progress[-1]['stage'], 'accepted')
        self.assertEqual(next(row for row in progress if row['stage'] == 'submitting')['cid'], 42)
        self.assertNotIn('never-export-this', str(progress))
        uploads = [row['progress'] for row in progress if row['stage'] == 'uploading']
        self.assertEqual(uploads, [44, 83, 83])  # duplicated completion does not double count

    def test_real_login_check_has_safe_return_and_no_invalid_profile_query(self):
        from bilibili_api import Credential
        adapter = publishing._BilibiliTransport()
        with patch.object(Credential, 'check_valid', AsyncMock(return_value=False)), \
                patch('bilibili_api.user.get_self_info', AsyncMock()) as profile:
            result = asyncio.run(adapter.check_account({'sessdata': 'fake', 'bili_jct': 'fake'}))
        self.assertEqual(result, {'valid': False, 'name': ''})
        profile.assert_not_awaited()
        with patch.object(Credential, 'check_valid', AsyncMock(return_value=True)), \
                patch('bilibili_api.user.get_self_info', AsyncMock(return_value={'name': 'Test', 'mid': 999,
                                                                               'cookie': 'never-export-this'})):
            result = asyncio.run(adapter.check_account({'sessdata': 'fake', 'bili_jct': 'fake'}))
        self.assertEqual(result, {'valid': True, 'name': 'Test'})

    def test_recent_uploads_uses_date_sort_and_only_returns_archives(self):
        from bilibili_api import creative_center
        with patch.object(creative_center, 'get_video_upload_manager_info', AsyncMock(return_value={
                'arc_audits': [{'Archive': {'aid': 1}}], 'private': 'never-export-this'})) as recent:
            result = asyncio.run(publishing._BilibiliTransport().recent_uploads(
                {'sessdata': 'fake', 'bili_jct': 'fake'}))
        self.assertEqual(result, [{'Archive': {'aid': 1}}])
        self.assertEqual(recent.call_args.kwargs['order'], creative_center.UploadManagerOrder.SENDDATE)
        self.assertEqual(recent.call_args.kwargs['ps'], 30)


if __name__ == '__main__':
    unittest.main()
