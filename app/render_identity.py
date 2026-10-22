"""Bind an exported video to the exact lesson and local media it rendered.

This is local provenance, not a signature or a claim that the source is accurate.
Operational review, progress and artifact state deliberately do not change the
identity of otherwise identical pictures and sound.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

VERSION = 1
WIDTH, HEIGHT, FPS = 1920, 1080, 30
ROOT = Path(__file__).resolve().parent.parent
_RENDER_FILES = ('src/remotion-entry.tsx', 'src/video.tsx', 'src/LessonScene.tsx',
                 'src/shared.ts', 'src/fontMetrics.ts', 'src/lesson.css')
_RENDER_DEPENDENCIES = ('react', 'react-dom', 'remotion', '@remotion/bundler',
                        '@remotion/cli', '@remotion/renderer')


def _digest(value) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (ValueError, TypeError) as exc:
        raise ValueError('课程画面数据格式无效，无法核验成片版本。') from exc
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    except OSError as exc:
        raise ValueError('成片或关联媒体无法读取，请重新导出。') from exc
    return digest.hexdigest()


def _inside(folder: Path, relative: str) -> Path:
    candidate = Path(str(relative))
    if not relative or candidate.is_absolute() or candidate.drive or '\\' in str(relative):
        raise ValueError('成片和媒体必须使用学习包内的相对路径。')
    base = Path(folder).resolve()
    path = (base / candidate).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError('成片或媒体路径超出学习包。') from exc
    return path


def manifest_path(folder: Path, video_relative: str) -> Path:
    video = _inside(folder, video_relative)
    if video.suffix.lower() != '.mp4':
        raise ValueError('成片必须为 MP4。')
    return video.with_name(video.name + '.manifest.json')


def _pick(value: dict, keys: tuple[str, ...]) -> dict:
    return {key: value[key] for key in keys if key in value}


def _source_name(package: dict) -> str:
    source = package.get('source', {})
    if source.get('name'):
        return source['name']
    if source.get('kind') == 'local':
        return '本地素材'
    address = source.get('url') or package.get('episode', {}).get('source_url', '')
    try:
        return (urlparse(address).hostname or '学习素材').removeprefix('www.')
    except (TypeError, ValueError) as exc:
        raise ValueError('课程来源格式无效，无法核验成片版本。') from exc


def content_fingerprint(package: dict) -> str:
    """Project only fields consumed by LessonScene, shared state and audio.

    Keep this projection in step with the rendering components. In particular,
    source names, caption page membership, glosses and card ordering are visible.
    Quality decisions, human annotations, caches and artifact status are not.
    """
    try:
        content = {
            'episode': _pick(package['episode'], ('title', 'title_zh', 'duration_s',
                'source_date', 'cover', 'audio_file', 'video_file')),
            'source_name': _source_name(package),
            'media': _pick(package.get('media', {}), ('audio_file', 'video_file')),
            'sentences': [_pick(item, ('id', 'start_ms', 'end_ms', 'text', 'translation', 'speaker'))
                          for item in package.get('sentences', [])],
            'words': [_pick(item, ('id', 'sentence_id', 'index', 'text', 'start_ms', 'end_ms', 'gloss_zh'))
                      for item in package.get('words', [])],
            'phrases': [_pick(item, ('id', 'sentence_id', 'text', 'gloss_zh', 'word_ids'))
                        for item in package.get('phrases', [])],
            'cards': [_pick(item, ('id', 'type', 'title', 'meaning', 'structure', 'example',
                'translation', 'start_ms', 'end_ms', 'display', 'show_meaning', 'points'))
                      for item in package.get('cards', [])],
            'layout': {'version': package['layout'].get('version'), 'pages': [
                _pick(page, ('id', 'sentence_id', 'start_ms', 'end_ms', 'lines', 'translation'))
                for page in package['layout'].get('pages', [])]} if package.get('layout') else None,
        }
        return _digest(content)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('课程画面数据不完整，无法核验成片版本。') from exc


def renderer_fingerprint() -> str:
    """Bind pictures to rendering code/styles, without unrelated app versions.

    Application version bumps, creator UI changes and audit-only metadata do not
    force a new video. Caption layout code, scene CSS and the actual rendering
    dependency resolutions do. Keep this direct dependency list in step with
    remotion-entry's imports when new visual components or styles are added.
    """
    try:
        files = {name: file_sha256(ROOT / name) for name in _RENDER_FILES}
        for name in ('remotion.config.ts', 'remotion.config.js', 'remotion.config.mjs'):
            if (ROOT / name).is_file():
                files[name] = file_sha256(ROOT / name)
        locked = json.loads((ROOT / 'package-lock.json').read_text(encoding='utf-8'))['packages']
        dependencies = {}
        for name in _RENDER_DEPENDENCIES:
            value = locked['node_modules/' + name]
            dependencies[name] = _pick(value, ('version', 'integrity'))
            if not dependencies[name].get('version'):
                raise ValueError()
        return _digest({'files': files, 'dependencies': dependencies})
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError('视频制作组件不完整，无法核验成片版本；请修复安装后重新导出。') from exc


def input_identity(package: dict, folder: Path) -> dict:
    episode = package.get('episode', {})
    media = package.get('media', {})
    names = {'audio': media.get('audio_file') or episode.get('audio_file') or 'audio.mp3',
             'video': media.get('video_file') or episode.get('video_file') or '',
             'cover': episode.get('cover') or ''}
    assets = {}
    for role, name in names.items():
        if not name:
            continue
        if not isinstance(name, str):
            raise ValueError('课程媒体路径格式无效。')
        path = _inside(folder, name)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('课程关联媒体缺失或为空，无法核验成片版本。')
        assets[role] = {'file': path.relative_to(Path(folder).resolve()).as_posix(),
                        'sha256': file_sha256(path)}
    return {'content_sha256': content_fingerprint(package), 'media': assets,
            'renderer_sha256': renderer_fingerprint()}


def _range(package: dict, start_ms, end_ms) -> dict:
    try:
        duration_ms = round(float(package['episode']['duration_s']) * 1000)
        if (type(start_ms) is not int or type(end_ms) is not int or
                not math.isfinite(duration_ms) or not 0 <= start_ms < end_ms <= duration_ms):
            raise ValueError()
    except (TypeError, ValueError, OverflowError, KeyError) as exc:
        raise ValueError('成片时间范围无效，请重新导出。') from exc
    return {'start_ms': start_ms, 'end_ms': end_ms}


def create_manifest(package: dict, folder: Path, video_path: Path, video_relative: str,
                    start_ms: int, end_ms: int, *, expected_inputs: dict | None = None) -> dict:
    """Create metadata after probing a staged MP4, before atomic publication."""
    manifest_path(folder, video_relative)
    inputs = input_identity(package, folder)
    if expected_inputs is not None and inputs != expected_inputs:
        raise ValueError('渲染期间课程或媒体已改变，已保留原成片，请重新导出。')
    if not video_path.is_file() or not video_path.stat().st_size:
        raise ValueError('没有可核验的精听视频。')
    return {'version': VERSION, 'episode_id': package.get('episode', {}).get('id'),
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'video_file': video_relative, 'video_sha256': file_sha256(video_path),
            'range': _range(package, start_ms, end_ms),
            'width': WIDTH, 'height': HEIGHT, 'fps': FPS, **inputs}


def verify_render(package: dict, folder: Path, video_relative: str) -> dict:
    """Reject stale/unbound exports; return the verified local manifest."""
    try:
        return _verify_render(package, folder, video_relative)
    except OSError as exc:
        raise ValueError('无法读取成片版本凭据或关联媒体，请重新导出。') from exc


def _verify_render(package: dict, folder: Path, video_relative: str) -> dict:
    path = manifest_path(folder, video_relative)
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('成片缺少有效版本凭据，请重新导出精听视频。') from exc
    if not isinstance(manifest, dict) or type(manifest.get('version')) is not int or manifest.get('version') != VERSION:
        raise ValueError('成片版本凭据不受支持，请重新导出。')
    if manifest.get('episode_id') != package.get('episode', {}).get('id') or manifest.get('video_file') != video_relative:
        raise ValueError('成片不属于当前课程或文件，请重新导出。')
    if any(type(manifest.get(key)) is not int or manifest[key] != value
           for key, value in (('width', WIDTH), ('height', HEIGHT), ('fps', FPS))):
        raise ValueError('成片画面参数与制作标准不一致，请重新导出。')
    timing = manifest.get('range')
    if not isinstance(timing, dict):
        raise ValueError('成片缺少时间范围，请重新导出。')
    timing = _range(package, timing.get('start_ms'), timing.get('end_ms'))
    if video_relative == 'lesson.mp4' and timing != _range(package, 0, round(float(package['episode']['duration_s']) * 1000)):
        raise ValueError('整课成片的时间范围不完整，请重新导出。')
    inputs = input_identity(package, folder)
    if manifest.get('renderer_sha256') != inputs['renderer_sha256']:
        raise ValueError('字幕布局、画面样式或制作组件已更新，禁止投稿旧成片，请重新导出。')
    if any(manifest.get(key) != inputs[key] for key in ('content_sha256', 'media')):
        raise ValueError('课程内容、时间轴或媒体已修订，禁止投稿旧成片，请重新导出。')
    video = _inside(folder, video_relative)
    if not video.is_file() or not video.stat().st_size or manifest.get('video_sha256') != file_sha256(video):
        raise ValueError('成片文件与版本凭据不一致，请重新导出。')
    return manifest
