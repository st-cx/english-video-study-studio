import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import speech_runtime


@unittest.skipUnless(os.name == 'nt', 'Windows DLL lookup')
class SpeechRuntimeTests(unittest.TestCase):
    def test_project_dll_search_is_process_local_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            expected = []
            for name in ('cublas', 'cudnn'):
                directory = base / 'nvidia' / name / 'bin'
                directory.mkdir(parents=True)
                (directory / 'library.dll').write_bytes(b'fixture')
                expected.append(str(directory.resolve()))
            with patch.dict(speech_runtime._CUDA_HANDLES, {}, clear=True), \
                 patch.dict(os.environ, {'PATH': 'existing-runtime'}), \
                 patch.object(os, 'add_dll_directory', return_value=object()) as add:
                self.assertEqual(speech_runtime.configure_cuda_runtime(base), 2)
                self.assertEqual(speech_runtime.configure_cuda_runtime(base), 2)
                self.assertEqual(add.call_count, 2)
                self.assertEqual(os.environ['PATH'].split(os.pathsep), expected + ['existing-runtime'])

    def test_missing_optional_libraries_leave_cpu_environment_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'PATH': 'unchanged'}):
            self.assertEqual(speech_runtime.configure_cuda_runtime(Path(temporary)), 0)
            self.assertEqual(os.environ['PATH'], 'unchanged')
