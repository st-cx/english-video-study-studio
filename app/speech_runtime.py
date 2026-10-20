"""Find optional CUDA DLLs in this Python environment without system changes."""
from __future__ import annotations

import os
import sysconfig
from pathlib import Path

_CUDA_HANDLES: dict[str, object] = {}


def configure_cuda_runtime(python_site: Path | None = None) -> int:
    if os.name != 'nt':
        return 0
    base = python_site or Path(sysconfig.get_paths()['purelib'])
    directories = [base / 'nvidia' / name / 'bin' for name in ('cublas', 'cudnn', 'cuda_nvrtc')]
    existing = [str(path.resolve()) for path in directories if path.is_dir() and any(path.glob('*.dll'))]
    for directory in existing:
        if directory not in _CUDA_HANDLES:
            _CUDA_HANDLES[directory] = os.add_dll_directory(directory)
    # CTranslate2's own LoadLibrary calls also consult the process PATH. This
    # affects only this process and children, never the Windows user/system PATH.
    previous = os.environ.get('PATH', '').split(os.pathsep)
    if existing:
        os.environ['PATH'] = os.pathsep.join(existing + [entry for entry in previous if entry not in existing])
    return len(existing)


def speech_runtime_plan() -> dict:
    """Prefer a complete cached large model on CUDA; CPU stays conservative."""
    configure_cuda_runtime()
    try:
        import ctranslate2
        gpu = ctranslate2.get_cuda_device_count() > 0
    except (ImportError, RuntimeError):
        gpu = False
    hub = Path.home() / '.cache' / 'huggingface' / 'hub'
    names = ['large-v3', 'medium'] if gpu else ['medium']
    for name in names:
        snapshots = hub / ('models--Systran--faster-whisper-' + name) / 'snapshots'
        for candidate in sorted(snapshots.glob('*'), reverse=True):
            if all((candidate / item).is_file() for item in ('model.bin', 'config.json', 'tokenizer.json')):
                return {'path': str(candidate), 'model': name, 'snapshot': candidate.name,
                        'device': 'cuda' if gpu else 'cpu',
                        'compute_type': 'int8_float16' if gpu else 'int8'}
    return {'path': 'medium.en', 'model': 'medium.en', 'snapshot': '',
            'device': 'cuda' if gpu else 'cpu',
            'compute_type': 'int8_float16' if gpu else 'int8'}
