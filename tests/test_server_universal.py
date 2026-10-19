import array
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['ENGLISH_VIDEO_DATA_DIR'] = str(ROOT / 'work/test-universal-profile')
from app import pipeline, server, storage


class UniversalServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'work')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.outputs = self.folder / 'outputs'
        self.outputs.mkdir()
        self.uploads = self.folder / 'uploads'
        self.uploads.mkdir()
        self.profile = self.folder / 'profile'
        self.profile.mkdir()
        for patcher in (
            patch.dict(os.environ, {'ENGLISH_VIDEO_DATA_DIR': str(self.profile)}),
            patch.object(storage, 'DB_PATH', self.profile / 'studio.sqlite3'),
            patch.object(storage, 'CONFIG_PATH', self.profile / 'settings.json'),
            patch.object(server, 'WORK_ROOT', self.folder / 'work'),
            patch.object(server, 'UPLOAD_ROOT', self.uploads),
            patch.object(server, 'output_root', return_value=self.outputs),
            patch.object(server, 'has_api_key', return_value=False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.spawn = patch.object(server, '_spawn')
        self.spawned = self.spawn.start()
        self.addCleanup(self.spawn.stop)
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        worker = threading.Thread(target=self.http.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.port = self.http.server_address[1]

    def request(self, method, path, body=None, headers=None):
        headers = dict(headers or {})
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            headers.setdefault('Content-Type', 'application/json')
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            value = json.loads(raw) if response.getheader('Content-Type', '').startswith('application/json') and raw else raw
            return response.status, value, dict(response.headers)
        finally:
            connection.close()

    def upload(self, payload, name='lesson.mp4', role='media', complete=True, options=None):
        digest = hashlib.sha256(payload).hexdigest()
        status, meta, _ = self.request('POST', '/api/uploads', {'name': name, 'size': len(payload), 'role': role, 'sha256': digest})
        self.assertEqual(status, 201)
        for offset in range(0, len(payload), server.CHUNK_SIZE):
            status, _, _ = self.request('PUT', '/api/uploads/' + meta['upload_id'], payload[offset:offset + server.CHUNK_SIZE],
                                        {'Content-Type': 'application/octet-stream', 'X-Chunk-Offset': str(offset)})
            self.assertEqual(status, 200)
        if complete:
            status, result, _ = self.request('POST', '/api/uploads/' + meta['upload_id'] + '/complete', options or {})
            self.assertEqual(status, 200 if role == 'subtitle' else 202, result)
            return result
        return meta

    def job(self, status='queued', episode_id=''):
        job_id = server._enqueue({'url': 'https://example.org/lesson.mp4', 'render_mode': 'page'}, start_immediately=False)
        server._set_job(job_id, status=status, episode_id=episode_id)
        return job_id

    def package(self, review=False):
        folder = self.outputs / 'episode-test'
        folder.mkdir()
        package = {'version': 4, 'episode': {'id': 'episode-test', 'title': 'Test', 'duration_s': 20},
                   'artifacts': {'study': 'ready', 'video': 'not_requested'}, 'render': {'status': 'page_ready'},
                   'chapters': [{'id': 'ch0', 'title': 'First', 'start_ms': 0, 'end_ms': 10000, 'status': 'ready'},
                                {'id': 'ch1', 'title': 'Second', 'start_ms': 10000, 'end_ms': 20000, 'status': 'review' if review else 'ready'}],
                   'processing': {'media_path': r'C:\private\source.mp4', 'api_key': 'should-not-leak'},
                   'sentences': [], 'words': [], 'cards': []}
        (folder / 'episode.json').write_text(json.dumps(package), encoding='utf-8')
        (folder / 'study.html').write_text('<h1>Offline lesson</h1>', encoding='utf-8')
        return package

    def test_raw_multichunk_upload_hash_and_subtitle_association(self):
        subtitle = self.upload(b'1\n00:00:00,100 --> 00:00:01,000\nHello.\n', 'lesson.srt', 'subtitle')
        self.assertNotIn('job_ids', subtitle)
        payload = b'v' * (server.CHUNK_SIZE + 37)
        uploaded = self.upload(payload, options={'subtitle_upload_id': subtitle['upload_id'], 'render_mode': 'page', 'start_ms': 100, 'end_ms': 2000})
        self.assertEqual(uploaded['received_bytes'], len(payload))
        self.assertEqual(uploaded['sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(uploaded['chunk_size'], 8388608)
        row = server._get_job(uploaded['job_ids'][0])
        spec = json.loads(row['input_json'])['spec']
        self.assertTrue(Path(spec['media_path']).is_file())
        self.assertTrue(Path(spec['subtitle_path']).is_file())
        self.assertEqual((spec['start_ms'], spec['end_ms']), (100, 2000))

    def test_resume_offset_idempotent_duplicate_and_conflicting_chunk(self):
        _, upload, _ = self.request('POST', '/api/uploads', {'name': 'test.mp3', 'size': 6})
        path = '/api/uploads/' + upload['upload_id']
        self.assertEqual(self.request('PUT', path, b'abc', {'X-Chunk-Offset': '0'})[0], 200)
        self.assertEqual(self.request('GET', path)[1]['received_bytes'], 3)
        self.assertEqual(self.request('PUT', path, b'abc', {'X-Chunk-Offset': '0'})[1]['received_bytes'], 3)
        self.assertEqual(self.request('PUT', path, b'xxx', {'X-Chunk-Offset': '0'})[0], 409)
        self.assertEqual(self.request('PUT', path, b'f', {'X-Chunk-Offset': '5'})[0], 409)
        self.assertEqual(self.request('PUT', path, b'def', {'X-Chunk-Offset': '3'})[0], 200)
        _, result, _ = self.request('POST', path + '/complete', {})
        self.assertEqual(result['sha256'], hashlib.sha256(b'abcdef').hexdigest())
        self.assertEqual(self.request('POST', path + '/complete', {})[1]['job_ids'], result['job_ids'])

    def test_integrity_failures_do_not_complete_or_create_job(self):
        meta = self.upload(b'data', complete=False)
        status, _, _ = self.request('POST', '/api/uploads/' + meta['upload_id'] + '/complete', {'sha256': '0' * 64})
        self.assertEqual(status, 422)
        self.assertFalse(self.request('GET', '/api/uploads/' + meta['upload_id'])[1]['complete'])
        self.assertEqual(server.list_jobs(), [])

    def test_chunk_hash_mismatch_keeps_original_resume_offset(self):
        _, meta, _ = self.request('POST', '/api/uploads', {'name': 'x.mp3', 'size': 3})
        path = '/api/uploads/' + meta['upload_id']
        self.assertEqual(self.request('PUT', path, b'abc', {'X-Chunk-Offset': '0', 'X-Chunk-SHA256': '0' * 64})[0], 422)
        self.assertEqual(self.request('GET', path)[1]['received_bytes'], 0)

    def test_upload_size_limits_and_incomplete_finish(self):
        self.assertEqual(self.request('POST', '/api/uploads', {'name': 'x', 'size': server.MAX_MEDIA_SIZE + 1})[0], 413)
        _, meta, _ = self.request('POST', '/api/uploads', {'name': 'x.mp3', 'size': 4})
        self.assertEqual(self.request('POST', '/api/uploads/' + meta['upload_id'] + '/complete', {})[0], 409)

    def test_cross_origin_host_and_json_csrf_are_rejected(self):
        self.assertEqual(self.request('POST', '/api/jobs', {'urls': ['https://example.org/x.mp4']}, {'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.request('GET', '/api/status', headers={'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.request('POST', '/api/jobs', b'{"urls":["https://example.org/x.mp4"]}', {'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.request('POST', '/api/jobs', {'urls': ['https://example.org/x.mp4']}, {'Origin': f'http://127.0.0.1:{self.port}'})[0], 202)

    def test_client_cannot_choose_local_paths_or_private_urls(self):
        self.assertEqual(self.request('POST', '/api/jobs', {'specs': [{'media_path': r'C:\secret\file.mp4'}]})[0], 400)
        for url in ['file:///secret', 'http://127.0.0.1/x.mp4', 'http://192.168.0.1/x.mp4', 'https://user:secret@example.org/x']:
            self.assertEqual(self.request('POST', '/api/jobs', {'urls': [url]})[0], 400)

    def test_queue_batch_deduplicates_and_pause_resume_is_conditional(self):
        _, response, _ = self.request('POST', '/api/jobs', {'urls': ['https://example.org/a.mp4', 'https://example.org/a.mp4']})
        self.assertEqual(len(response['job_ids']), 1)
        job_id = response['job_ids'][0]
        self.assertEqual(self.request('POST', f'/api/jobs/{job_id}/pause', {})[0], 202)
        self.assertEqual(server._get_job(job_id)['status'], 'paused')
        self.assertEqual(self.request('POST', f'/api/jobs/{job_id}/resume', {})[0], 202)
        self.assertEqual(self.request('POST', f'/api/jobs/{job_id}/resume', {})[0], 409)

    def test_running_pause_is_observed_by_progress(self):
        job_id = self.job('running')
        self.assertEqual(self.request('POST', f'/api/jobs/{job_id}/pause', {})[0], 202)
        self.assertEqual(server._get_job(job_id)['status'], 'pause_requested')
        with self.assertRaises(server.JobPaused):
            server._progress(job_id)('align', 40, 'Checkpoint')
        self.assertEqual(server._get_job(job_id)['status'], 'pause_requested')

    def test_worker_pauses_without_marking_failed(self):
        job_id = self.job()
        with patch.object(pipeline, 'create_media_episode', side_effect=server.JobPaused(), create=True):
            server._run_job(job_id)
        self.assertEqual(server._get_job(job_id)['status'], 'paused')

    def test_pause_while_worker_claims_queued_job_is_not_overwritten(self):
        job_id = self.job()
        original_get = server._get_job
        def pause_during_read(identifier):
            row = original_get(identifier)
            server._set_job(identifier, status='paused')
            return row
        with patch.object(server, '_get_job', side_effect=pause_during_read), patch.object(pipeline, 'create_media_episode', create=True) as produce:
            server._run_job(job_id)
        produce.assert_not_called()
        self.assertEqual(original_get(job_id)['status'], 'paused')

    def test_worker_failure_does_not_leak_paths_or_secrets(self):
        job_id = self.job()
        with patch.object(pipeline, 'create_media_episode', side_effect=RuntimeError(r'C:\private\file token=private'), create=True):
            server._run_job(job_id)
        self.assertEqual(server._get_job(job_id)['status'], 'failed')
        public = self.request('GET', '/api/jobs')[1]
        self.assertNotIn('private', json.dumps(public))
        self.assertNotIn('input_json', public[0])
        self.assertEqual(self.request('POST', f'/api/jobs/{job_id}/retry', {})[0], 202)

    def test_ready_study_page_is_available_without_mp4_even_during_review(self):
        self.package(review=True)
        job_id = self.job('review', 'episode-test')
        _, jobs, _ = self.request('GET', '/api/jobs')
        job = next(item for item in jobs if item['id'] == job_id)
        self.assertTrue(job['study_url'].endswith('study.html'))
        self.assertEqual(job['video_url'], '')
        self.assertEqual(len(self.request('GET', '/api/episodes')[1]), 1)
        for endpoint in ['/api/episode?id=episode-test', '/files/episode-test/episode.json']:
            package = self.request('GET', endpoint)[1]
            self.assertNotIn('private', json.dumps(package))
            self.assertNotIn('should-not-leak', json.dumps(package))

    def test_chapter_continue_persists_correction_once(self):
        self.package(review=True)
        job_id = self.job('review', 'episode-test')
        path = f'/api/jobs/{job_id}/continue'
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: self.request('POST', path, {'chapter_id': 'ch1', 'corrected_transcript': 'Saved correction.'})[0], range(2)))
        self.assertEqual(sorted(results), [202, 409])
        inputs = json.loads(server._get_job(job_id)['input_json'])
        self.assertEqual((inputs['chapter_id'], inputs['corrected_transcript']), ('ch1', 'Saved correction.'))

    def test_export_rejects_review_chapter_but_accepts_approved_segment(self):
        self.package(review=True)
        job_id = self.job('review', 'episode-test')
        path = f'/api/jobs/{job_id}/export'
        self.assertEqual(self.request('POST', path, {})[0], 409)
        self.assertEqual(self.request('POST', path, {'start_ms': 0, 'end_ms': 10000})[0], 202)
        self.assertEqual(self.request('POST', path, {'start_ms': 10000, 'end_ms': 30000})[0], 400)

    def test_changed_full_transcript_requires_a_chapter_before_claiming(self):
        self.package(review=True)
        job_id = self.job('review', 'episode-test')
        status, _, _ = self.request('POST', f'/api/jobs/{job_id}/continue', {'corrected_transcript': 'Changed full transcript.'})
        self.assertEqual(status, 400)
        self.assertEqual(server._get_job(job_id)['status'], 'review')

    def test_unchanged_transcript_is_confirmation_without_realigning_all_chapters(self):
        package = self.package(review=True)
        package['review'] = {'recognized_transcript': 'Unchanged source words.'}
        (self.outputs / 'episode-test/episode.json').write_text(json.dumps(package), encoding='utf-8')
        job_id = self.job('review', 'episode-test')
        status, _, _ = self.request('POST', f'/api/jobs/{job_id}/continue', {'corrected_transcript': 'Unchanged source\nwords.'})
        self.assertEqual(status, 202)
        self.assertEqual(json.loads(server._get_job(job_id)['input_json'])['corrected_transcript'], '')

    def test_recovery_preserves_pause_requests_and_continuation_inputs(self):
        job_id = self.job('running')
        server._set_job(job_id, input_json=json.dumps({'operation': 'continue', 'chapter_id': 'ch1', 'corrected_transcript': 'Keep me.'}))
        server.recover_saved_jobs()
        self.assertEqual(json.loads(server._get_job(job_id)['input_json'])['corrected_transcript'], 'Keep me.')
        server._set_job(job_id, status='pause_requested')
        self.spawned.reset_mock()
        server.recover_saved_jobs()
        self.assertEqual(server._get_job(job_id)['status'], 'paused')
        self.spawned.assert_not_called()

    def test_files_reject_private_checkpoints_and_traversal(self):
        self.package()
        self.assertEqual(self.request('GET', '/files/episode-test/episode.checkpoint.json')[0], 403)
        self.assertEqual(self.request('GET', '/files/episode-test/.teaching-cache/test.json')[0], 403)
        self.assertEqual(self.request('GET', '/files/episode-test/%2e%2e/%2e%2e/secret.txt')[0], 403)

    def test_browser_asset_mime_types_do_not_depend_on_windows_registry(self):
        dist = self.folder / 'dist'
        dist.mkdir()
        for extension in ('.js', '.mjs', '.css', '.html'):
            (dist / ('asset' + extension)).write_text('test', encoding='utf-8')
        with patch.object(server, 'ROOT', self.folder), patch.object(server.mimetypes, 'guess_type', return_value=('text/plain', None)):
            for extension, expected in {'.js': 'application/javascript', '.mjs': 'application/javascript', '.css': 'text/css', '.html': 'text/html'}.items():
                status, _, headers = self.request('GET', '/asset' + extension)
                self.assertEqual(status, 200)
                self.assertEqual(headers['Content-Type'], expected)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required for local track inspection')
    def test_uploaded_audio_track_inspection_and_actual_selected_preview(self):
        from app.media import probe_media
        source = self.folder / 'dual.mka'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=frequency=220:duration=15',
                        '-f', 'lavfi', '-i', 'sine=frequency=880:duration=15', '-map', '0:a', '-map', '1:a', '-c:a', 'aac',
                        '-metadata:s:a:0', 'language=zho', '-metadata:s:a:1', 'language=eng',
                        '-disposition:a:0', 'default', '-disposition:a:1', '0', str(source)],
                       capture_output=True, check=True, timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        uploaded = self.upload(source.read_bytes(), name='dual.mka', complete=False)
        path = '/api/uploads/' + uploaded['upload_id']
        status, info, _ = self.request('POST', path + '/inspect', {})
        self.assertEqual(status, 200)
        self.assertEqual(info['suggested_audio_stream_index'], 1)
        self.assertEqual([stream['language'] for stream in info['audio_streams']], ['zho', 'eng'])
        self.assertTrue(info['audio_streams'][0]['default'])
        self.assertNotIn('path', json.dumps(info))
        self.spawned.assert_not_called()
        status, content, headers = self.request('GET', path + '/preview?stream=1')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'audio/mpeg')
        preview = self.uploads / uploaded['upload_id'] / 'preview-audio-1.mp3'
        self.assertEqual(preview.read_bytes(), content)
        self.assertLess(float(probe_media(preview)['format']['duration']), 12.2)
        pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(preview), '-t', '0.5', '-f', 'f32le', '-ac', '1', '-ar', '8000', '-'],
                             capture_output=True, check=True, timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
        samples = array.array('f')
        samples.frombytes(pcm)
        crossings = sum((left < 0) != (right < 0) for left, right in zip(samples, samples[1:]))
        frequency = crossings / (2 * len(samples) / 8000)
        self.assertTrue(800 < frequency < 950, frequency)
        with patch.object(server.subprocess, 'run', side_effect=AssertionError('Preview must be cached')):
            self.assertEqual(self.request('GET', path + '/preview?stream=1', headers={'Range': 'bytes=0-10'})[:2], (206, content[:11]))
            head_status, head_body, _ = self.request('HEAD', path + '/preview?stream=1')
            self.assertEqual((head_status, head_body), (200, b''))
        self.assertEqual(self.request('GET', path + '/preview?stream=999')[0], 400)

    def test_upload_inspection_requires_all_blocks_and_does_not_guess_unknown_multitrack(self):
        _, uploaded, _ = self.request('POST', '/api/uploads', {'name': 'two.mkv', 'size': 4})
        path = '/api/uploads/' + uploaded['upload_id']
        self.assertEqual(self.request('POST', path + '/inspect', {})[0], 409)
        self.request('PUT', path, b'data', {'X-Chunk-Offset': '0'})
        metadata = {'format': {'duration': '20'}, 'streams': [
            {'index': 0, 'codec_type': 'audio', 'channels': 2, 'tags': {'language': 'und'}, 'disposition': {'default': 1}},
            {'index': 1, 'codec_type': 'audio', 'channels': 2, 'tags': {'language': 'fra'}}]}
        with patch('app.media.probe_media', return_value=metadata):
            status, info, _ = self.request('POST', path + '/inspect', {})
        self.assertEqual(status, 200)
        self.assertIsNone(info['suggested_audio_stream_index'])
        self.assertEqual(self.request('GET', path + '/preview')[0], 400)
        self.spawned.assert_not_called()

    def test_failed_gateway_test_preserves_existing_configuration(self):
        storage.save_settings({'model': 'old', 'base_url': 'http://localhost:11434/v1'})
        with patch.object(server, 'ask_deepseek', side_effect=RuntimeError('private network body')), patch.object(server, 'save_api_key') as key:
            status, body, _ = self.request('POST', '/api/settings', {'model': 'group/new', 'api_key': 'not-a-real-key', 'test': True})
        self.assertEqual(status, 502)
        self.assertNotIn('private network', json.dumps(body))
        key.assert_not_called()
        self.assertEqual(storage.get_settings()['model'], 'old')


if __name__ == '__main__':
    unittest.main(verbosity=2)
