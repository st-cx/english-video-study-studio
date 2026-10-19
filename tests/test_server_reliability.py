import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['ENGLISH_VIDEO_DATA_DIR'] = str(ROOT / 'work/test-reliability-profile')
from app import server, storage


class Reply:
    def _json(self, status, value):
        return status, value


class ServerReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'work')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        for patcher in (
            patch.dict(os.environ, {'ENGLISH_VIDEO_DATA_DIR': str(self.folder)}),
            patch.object(storage, 'DB_PATH', self.folder / 'studio.sqlite3'),
            patch.object(storage, 'CONFIG_PATH', self.folder / 'settings.json'),
            patch.object(server, 'has_api_key', return_value=False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_explicit_profile_is_independent_from_bbc_or_store_profiles(self):
        old = self.folder / 'Packages/OpenAI.Codex_test/LocalCache/Local/BBCEnglishStudio'
        old.mkdir(parents=True)
        (old / 'deepseek.key.dpapi').write_bytes(b'opaque-test-data')
        with patch.dict(os.environ, {'LOCALAPPDATA': str(self.folder), 'BBC_ENGLISH_DATA_DIR': str(old)}):
            self.assertEqual(storage.data_root(), self.folder)
        with patch.dict(os.environ, {'ENGLISH_VIDEO_DATA_DIR': ''}):
            self.assertEqual(storage.data_root(), ROOT / 'work/profile')

    def test_settings_merge_across_independent_processes(self):
        storage.save_settings({'model': 'initial', 'base_url': 'initial'})
        script = ('import sys,time;from app.storage import save_settings;'
                  '[(save_settings({sys.argv[1]:sys.argv[2]+str(i)}),time.sleep(.01)) for i in range(8)]')
        processes = [subprocess.Popen([sys.executable, '-B', '-c', script, field, prefix], cwd=ROOT,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                     for field, prefix in [('model', 'test/model-'), ('base_url', 'http://localhost:')]]
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stdout + stderr)
        settings = storage.get_settings()
        self.assertEqual(settings['model'], 'test/model-7')
        self.assertEqual(settings['base_url'], 'http://localhost:7')
        self.assertEqual(list(self.folder.glob('*.tmp')), [])

    def test_settings_validation_does_not_restore_an_untouched_stale_field(self):
        storage.save_settings({'model': 'old', 'base_url': 'http://localhost:1'})
        def validate(*args, **kwargs):
            storage.save_settings({'base_url': 'http://localhost:2'})
            return {'ok': True}
        with patch.object(server, 'ask_deepseek', side_effect=validate):
            status, _ = server.Handler._save_settings(Reply(), {'model': 'test/new', 'test': True})
        self.assertEqual(status, 200)
        self.assertEqual(storage.get_settings()['model'], 'test/new')
        self.assertEqual(storage.get_settings()['base_url'], 'http://localhost:2')

    def test_simultaneous_enqueue_claims_same_content_only_once(self):
        with patch.object(server, '_spawn') as spawn, ThreadPoolExecutor(max_workers=2) as executor:
            jobs = list(executor.map(lambda _: server._enqueue({'media_sha256': 'same-hash', 'media_path': 'private-path', 'render_mode': 'page'}), range(2)))
        self.assertEqual(jobs[0], jobs[1])
        self.assertEqual(spawn.call_count, 1)

    def test_production_lock_timeout_is_a_persisted_failure(self):
        job_id = server._enqueue({'url': 'https://example.org/clip.mp4'}, start_immediately=False)
        with patch.object(server, 'production_lock', side_effect=TimeoutError('test timeout')):
            server._run_job(job_id)
        self.assertEqual(server._get_job(job_id)['status'], 'failed')

    def test_completed_worker_is_not_run_twice(self):
        job_id = server._enqueue({'url': 'https://example.org/clip.mp4'}, start_immediately=False)
        server._set_job(job_id, status='complete')
        with patch('app.pipeline.create_media_episode', create=True) as produce:
            server._run_job(job_id)
        produce.assert_not_called()

    def test_failed_atomic_episode_replace_preserves_previous_json(self):
        old = {'version': 4, 'title': 'original'}
        path = storage.save_episode(old, self.folder)
        with patch.object(storage.os, 'replace', side_effect=OSError('replace failed')):
            with self.assertRaises(OSError):
                storage.save_episode({'title': 'replacement'}, self.folder)
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), old)
        self.assertEqual(list(self.folder.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
