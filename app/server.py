from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .pipeline import ReviewRequired
from .providers import ask_deepseek
from .security import delete_api_key, has_api_key, save_api_key
from .storage import connect, data_root, get_settings, output_root, production_lock, save_settings

ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = data_root() / 'work'
UPLOAD_ROOT = data_root() / 'uploads'
WORK_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 8 * 1024 * 1024
MAX_MEDIA_SIZE = 20 * 1024**3
MAX_SUBTITLE_SIZE = 20 * 1024**2
MAX_JSON_SIZE = 2 * 1024**2
LOG = logging.getLogger('english_video_studio')
logging.basicConfig(filename=str(data_root() / 'studio.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
_upload_locks: dict[str, threading.Lock] = {}
_upload_locks_guard = threading.Lock()


class JobPaused(RuntimeError):
    """Raised by progress at a checkpoint after a persisted pause request."""


class APIError(ValueError):
    def __init__(self, status: int, message: str, **details):
        super().__init__(message)
        self.status, self.details = status, details


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.writing-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _identifier(value):
    if not re.fullmatch(r'[a-f0-9]{32}', value or ''):
        raise APIError(404, '无效的编号。')
    return value


def _public_url(value):
    parsed = urlparse(str(value).strip())
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise APIError(400, '请提供公开的 HTTP 或 HTTPS 媒体链接。')
    host = parsed.hostname.lower()
    if host in {'localhost', 'localhost.localdomain'} or host.endswith(('.localhost', '.local')):
        raise APIError(400, '媒体链接不能指向本机或内网。')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise APIError(400, '媒体链接不能指向本机或内网。')
    return parsed.geturl()


def _private_value(value):
    if isinstance(value, dict):
        return {key: _private_value(item) for key, item in value.items()
                if not key.endswith('_path') and key not in {'processing', 'input_json', 'api_key', 'api_key_override', 'authorization', 'token', 'password', 'cache_dir', 'job_folder'}}
    if isinstance(value, list):
        return [_private_value(item) for item in value]
    if isinstance(value, str) and (re.match(r'^[A-Za-z]:[\\/]', value) or value.startswith(('\\\\', '/Users/', '/home/', '/tmp/', '/var/'))):
        return ''
    return value


def _get_job(job_id):
    db = connect()
    try:
        row = db.execute('SELECT * FROM jobs WHERE id=?', (_identifier(job_id),)).fetchone()
        return dict(row) if row else None
    finally:
        db.close()


def _set_job(job_id, **values):
    allowed = {'episode_id', 'title', 'status', 'stage', 'progress', 'message', 'error', 'package_path', 'input_json'}
    fields = {key: value for key, value in values.items() if key in allowed}
    if not fields:
        return
    db = connect()
    try:
        assignment = ','.join(f'{key}=?' for key in fields)
        db.execute(f'UPDATE jobs SET {assignment},updated_at=? WHERE id=?', (*fields.values(), _now(), job_id))
        db.commit()
    finally:
        db.close()


def _package_file(episode_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', episode_id or ''):
        raise APIError(404, '无效的学习包编号。')
    return output_root() / episode_id / 'episode.json'


def _read_package(episode_id):
    try:
        return json.loads(_package_file(episode_id).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise APIError(404, '学习包尚未生成。') from exc


def _study_ready(package, folder):
    artifacts = package.get('artifacts', {})
    ready = artifacts.get('study') == 'ready' or not artifacts and package.get('render', {}).get('status') == 'ready'
    return ready and (folder / 'study.html').is_file()


def _job_row(row):
    data = dict(row)
    result = {key: data.get(key) for key in ('id', 'episode_id', 'title', 'status', 'stage', 'progress', 'message', 'error', 'created_at', 'updated_at')}
    result.update(study_url='', video_url='', package_path='', chapters=[])
    episode_id = data.get('episode_id')
    if episode_id:
        try:
            package = _read_package(episode_id)
            folder = _package_file(episode_id).parent
            prefix = f'/files/{quote(episode_id)}/'
            result['package_path'] = prefix + 'episode.json'
            result['chapters'] = _private_value(package.get('chapters', []))
            if _study_ready(package, folder):
                result['study_url'] = prefix + 'study.html'
            inputs = json.loads(data.get('input_json') or '{}')
            video = inputs.get('export_file') or package.get('render', {}).get('video_file', 'lesson.mp4')
            video_path = (folder / video).resolve()
            video_path.relative_to(folder.resolve())
            if video_path.is_file() and (inputs.get('export_file') or package.get('artifacts', {}).get('video') == 'ready' or package.get('render', {}).get('status') == 'ready'):
                result['video_url'] = prefix + '/'.join(quote(part) for part in Path(video).parts)
        except (APIError, OSError, ValueError, TypeError):
            pass
    return result


def list_jobs():
    db = connect()
    try:
        return [_job_row(row) for row in db.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 200').fetchall()]
    finally:
        db.close()


def all_episodes():
    result = []
    for path in output_root().glob('*/episode.json'):
        try:
            package = json.loads(path.read_text(encoding='utf-8'))
            if _study_ready(package, path.parent):
                result.append(_private_value({**package['episode'], 'quality': package.get('quality', {}),
                                              'source': package.get('source', {}), 'media': package.get('media', {}),
                                              'artifacts': package.get('artifacts', {}), 'chapters': package.get('chapters', [])}))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(result, key=lambda item: item.get('created_at', ''), reverse=True)


def _upload_lock(upload_id):
    with _upload_locks_guard:
        return _upload_locks.setdefault(_identifier(upload_id), threading.Lock())


def _upload_meta(upload_id):
    folder = UPLOAD_ROOT / _identifier(upload_id)
    try:
        meta = json.loads((folder / 'upload.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise APIError(404, '上传记录不存在。') from exc
    return meta, folder


def _upload_public(meta, folder):
    target = folder / meta['file']
    return {key: meta[key] for key in ('upload_id', 'name', 'size', 'role', 'complete', 'sha256') if key in meta} | {
        'chunk_size': CHUNK_SIZE, 'received_bytes': target.stat().st_size if target.is_file() else 0}


def _upload_asset(upload_id, role):
    meta, folder = _upload_meta(upload_id)
    target = folder / meta['file']
    if meta.get('role') != role or not meta.get('complete') or not target.is_file() or target.stat().st_size != meta['size']:
        raise APIError(409, '请先完成相应素材的上传和完整性校验。')
    return meta, target


def _inspect_uploaded_media(meta, folder):
    from .media import media_duration, probe_media
    target = folder / meta['file']
    if meta['role'] != 'media':
        raise APIError(400, '只有音视频素材可以检查音轨。')
    if not target.is_file() or target.stat().st_size != meta['size']:
        raise APIError(409, '请先传完所有媒体块，再检查音轨。')
    state = target.stat()
    signature = {'size': state.st_size, 'mtime_ns': state.st_mtime_ns}
    try:
        cached = json.loads((folder / 'inspection.json').read_text(encoding='utf-8'))
        if cached.get('signature') == signature:
            return cached['inspection']
    except (OSError, ValueError):
        pass
    try:
        info = probe_media(target)
        duration = media_duration(info)
    except (ValueError, RuntimeError) as exc:
        raise APIError(422, '媒体尚不能读取，请检查是否为完整音视频文件。') from exc
    audio = []
    video = None
    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'audio':
            tags = stream.get('tags', {})
            audio.append({'index': int(stream['index']), 'language': str(tags.get('language', 'und'))[:32],
                          'title': _private_value(str(tags.get('title', '')))[:200], 'channels': int(stream.get('channels') or 0),
                          'default': bool(stream.get('disposition', {}).get('default'))})
        elif stream.get('codec_type') == 'video' and not stream.get('disposition', {}).get('attached_pic') and video is None:
            video = {'width': int(stream.get('width') or 0), 'height': int(stream.get('height') or 0)}
    english = [stream for stream in audio if stream['language'].lower() in {'en', 'eng', 'english'} or stream['language'].lower().startswith('en-')]
    defaults = [stream for stream in english if stream['default']]
    suggested = (defaults or english)[0]['index'] if english else audio[0]['index'] if len(audio) == 1 else None
    result = {'duration_s': duration, 'audio_streams': audio, 'video': video, 'suggested_audio_stream_index': suggested}
    _atomic_json(folder / 'inspection.json', {'signature': signature, 'inspection': result})
    return result


def _normalize_spec(raw):
    if not isinstance(raw, dict):
        raise APIError(400, '任务参数必须是对象。')
    if any(key.endswith('_path') or key in {'api_key', 'package_path', 'export_file'} for key in raw):
        raise APIError(400, '不能从网页指定本机文件路径，请先上传素材。')
    spec = {}
    upload_id = raw.get('upload_id') or raw.get('media_upload_id')
    if upload_id:
        meta, target = _upload_asset(str(upload_id), 'media')
        spec.update(media_path=str(target), media_sha256=meta['sha256'], title=meta['name'].rsplit('.', 1)[0], source_kind='local')
    elif raw.get('url'):
        spec['url'] = _public_url(raw['url'])
    else:
        raise APIError(400, '请上传音视频，或填写媒体链接。')
    if raw.get('subtitle_upload_id'):
        meta, target = _upload_asset(str(raw['subtitle_upload_id']), 'subtitle')
        spec.update(subtitle_path=str(target), subtitle_sha256=meta['sha256'])
    for key in ('title', 'transcript'):
        if key in raw:
            if not isinstance(raw[key], str) or len(raw[key]) > (1_000_000 if key == 'transcript' else 200):
                raise APIError(400, '标题或文稿超过限制。')
            spec[key] = raw[key].strip()
    if raw.get('source_url'):
        spec['source_url'] = _public_url(raw['source_url'])
    for key in ('audio_stream_index', 'start_ms', 'end_ms'):
        if raw.get(key) is not None:
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise APIError(400, '音轨编号和裁剪时间必须是非负整数。')
            spec[key] = value
    if spec.get('audio_stream_index', 0) > 99:
        raise APIError(400, '音轨编号超过范围。')
    if 'end_ms' in spec and spec['end_ms'] <= spec.get('start_ms', 0):
        raise APIError(400, '结束时间必须晚于开始时间。')
    spec['render_mode'] = raw.get('render_mode', 'auto')
    if spec['render_mode'] not in {'auto', 'page', 'video'}:
        raise APIError(400, '请选择自动、学习页或视频模式。')
    return spec


def _enqueue(spec, operation='create', extra=None, start_immediately=True):
    identity = {key: value for key, value in spec.items() if key not in {'media_path', 'subtitle_path', 'title'}}
    signature = hashlib.sha256(json.dumps([operation, identity, extra or {}], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    db = connect()
    try:
        db.execute('BEGIN IMMEDIATE')
        for row in db.execute("SELECT id,input_json FROM jobs WHERE status IN ('queued','running','pause_requested','paused','review','complete')").fetchall():
            if json.loads(row['input_json'] or '{}').get('dedup_key') == signature:
                db.commit()
                return row['id']
        job_id = uuid.uuid4().hex
        inputs = {'spec': spec, 'operation': operation, 'dedup_key': signature, **(extra or {})}
        episode_id = (extra or {}).get('episode_id', '')
        db.execute('INSERT INTO jobs (id,episode_id,url,title,status,stage,progress,message,input_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   (job_id, episode_id, spec.get('url', ''), spec.get('title', ''), 'queued', 'queued', 0, '等待处理', json.dumps(inputs, ensure_ascii=False), _now(), _now()))
        db.commit()
    finally:
        db.close()
    if start_immediately:
        _spawn(job_id)
    return job_id


def create_job(url, title='', imported=False, imported_data=None, start_immediately=True):
    """Compatibility entry for trusted local callers; HTTP uses upload IDs."""
    spec = dict(imported_data or {}) if imported else {'url': _public_url(url)}
    if title:
        spec['title'] = title
    return _enqueue(spec, start_immediately=start_immediately)


def _spawn(job_id):
    threading.Thread(target=_run_job, args=(job_id,), name=f'english-job-{job_id[:8]}', daemon=True).start()


def _progress(job_id):
    def update(stage, progress, message):
        db = connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['status'] in {'pause_requested', 'paused'}:
                raise JobPaused('已在检查点暂停。')
            db.execute("UPDATE jobs SET status='running',stage=?,progress=?,message=?,error='',updated_at=? WHERE id=?",
                       (str(stage), max(0, min(99, int(progress))), str(message)[:500], _now(), job_id))
            db.commit()
        finally:
            db.close()
    return update


def _run_job(job_id):
    try:
        with production_lock():
            row = _get_job(job_id)
            if not row or row['status'] != 'queued':
                return
            db = connect()
            try:
                claimed = db.execute("UPDATE jobs SET status='running',stage='prepare',message=?,error='',updated_at=? WHERE id=? AND status='queued'",
                                     ('正在准备学习素材…', _now(), job_id)).rowcount
                db.commit()
            finally:
                db.close()
            if not claimed:
                return
            inputs = json.loads(row['input_json'])
            operation = inputs.get('operation', 'create')
            if operation == 'continue':
                from .pipeline import continue_after_review
                path = _package_file(row['episode_id'])
                package = json.loads(path.read_text(encoding='utf-8'))
                package = continue_after_review(package, path.parent, _progress(job_id), inputs.get('corrected_transcript', ''), chapter_id=inputs.get('chapter_id', ''))
            elif operation == 'export':
                from .renderer import export_video
                path = _package_file(row['episode_id'])
                package = json.loads(path.read_text(encoding='utf-8'))
                _validate_export(package, inputs.get('start_ms', 0), inputs.get('end_ms'))
                result = export_video(path, path.parent, _progress(job_id), start_ms=inputs.get('start_ms', 0), end_ms=inputs.get('end_ms'))
                inputs['export_file'] = str(Path(result).resolve().relative_to(path.parent.resolve()))
                _set_job(job_id, input_json=json.dumps(inputs, ensure_ascii=False))
                package = json.loads(path.read_text(encoding='utf-8'))
            else:
                from .pipeline import create_media_episode
                package = create_media_episode(inputs['spec'], WORK_ROOT / job_id, _progress(job_id))
            episode_id = package['episode']['id']
            _set_job(job_id, episode_id=episode_id, title=package['episode']['title'], package_path=str(_package_file(episode_id)),
                     status='complete', stage='complete', progress=100, message='视频导出完成' if operation == 'export' else '学习页已生成', error='')
    except JobPaused:
        _set_job(job_id, status='paused', message='已暂停；继续时将复用已完成的步骤。')
    except ReviewRequired as exc:
        episode_id = exc.package['episode']['id']
        _set_job(job_id, episode_id=episode_id, title=exc.package['episode']['title'], status='review', stage='review',
                 message='部分章节需要检查；已通过的章节仍可学习。', error='请查看待检查章节和问题标记。', package_path=str(_package_file(episode_id)))
    except Exception as exc:
        LOG.error('job %s failed: %s', job_id, type(exc).__name__)
        _set_job(job_id, status='failed', stage='failed', message='处理失败，已保留可恢复的步骤。', error=f'{type(exc).__name__}：请检查素材或模型连接后重试。')


def _validate_export(package, start_ms=0, end_ms=None):
    duration = round(float(package['episode']['duration_s']) * 1000)
    end_ms = duration if end_ms is None else end_ms
    if isinstance(start_ms, bool) or isinstance(end_ms, bool) or not isinstance(start_ms, int) or not isinstance(end_ms, int) or not 0 <= start_ms < end_ms <= duration:
        raise APIError(400, '导出区间必须位于素材时长内，单位为整数毫秒。')
    if package.get('quality', {}).get('structural_flags'):
        raise APIError(409, '时间轴结构尚未通过检查，暂时不能导出。')
    if any(chapter.get('status') != 'ready' and chapter['start_ms'] < end_ms and chapter['end_ms'] > start_ms for chapter in package.get('chapters', [])):
        raise APIError(409, '所选区间包含待检查或未完成章节，请先确认或缩小导出范围。')


def recover_saved_jobs():
    db = connect()
    try:
        rows = db.execute("SELECT id,status FROM jobs WHERE status IN ('queued','running','pause_requested') ORDER BY created_at").fetchall()
        for row in rows:
            if row['status'] == 'pause_requested':
                db.execute("UPDATE jobs SET status='paused',message=?,updated_at=? WHERE id=?", ('已按上次请求暂停。', _now(), row['id']))
            else:
                db.execute("UPDATE jobs SET status='queued',stage='queued',message=?,updated_at=? WHERE id=?", ('正在恢复上次中断的步骤…', _now(), row['id']))
        db.commit()
    finally:
        db.close()
    for row in rows:
        if row['status'] != 'pause_requested':
            _spawn(row['id'])


class Handler(BaseHTTPRequestHandler):
    server_version = 'EnglishVideoStudio/0.5.2'

    def log_message(self, fmt, *args):
        LOG.info('%s %s', self.command, urlparse(self.path).path)

    def _guard(self, mutation=False):
        port = self.server.server_address[1]
        host = self.headers.get('Host', '')
        if host not in {f'127.0.0.1:{port}', f'localhost:{port}'}:
            raise APIError(403, '访问来源无效。')
        if mutation:
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + host or self.headers.get('Sec-Fetch-Site') == 'cross-site':
                raise APIError(403, '只接受当前本地页面发起的修改。')

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _body(self):
        if self.headers.get('Transfer-Encoding'):
            raise APIError(411, '请提供请求内容的长度。')
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError as exc:
            raise APIError(400, '请求长度无效。') from exc
        if not 0 <= length <= MAX_JSON_SIZE:
            raise APIError(413, '请求超过大小限制，请使用分块上传。')
        if not length:
            return {}
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            raise APIError(415, '此接口只接受 JSON。媒体请使用分块上传。')
        self.connection.settimeout(30)
        try:
            result = json.loads(self.rfile.read(length).decode('utf-8'))
        except (ValueError, UnicodeError) as exc:
            raise APIError(400, 'JSON 格式无效。') from exc
        if not isinstance(result, dict):
            raise APIError(400, '请求必须是 JSON 对象。')
        return result

    def _dispatch_error(self, exc):
        if isinstance(exc, APIError):
            return self._json(exc.status, {'error': str(exc), **exc.details})
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return self._json(408, {'error': '请求传输超时，请重试。'})
        LOG.error('HTTP request failed: %s', type(exc).__name__)
        return self._json(500, {'error': '本地操作未完成，请检查输入或稍后重试。'})

    def do_GET(self):
        try:
            self._guard()
            route = urlparse(self.path)
            if route.path == '/api/status':
                settings = get_settings()
                return self._json(200, {'name': '英语视频学习工作室', 'kind': 'universal', 'version': '0.5.2',
                                        'api_key_configured': has_api_key(), 'model': settings['model'], 'base_url': settings['base_url'],
                                        'daily_check_enabled': False, 'chunk_size': CHUNK_SIZE, 'max_media_size': MAX_MEDIA_SIZE})
            if route.path == '/api/jobs':
                return self._json(200, list_jobs())
            if route.path == '/api/episodes':
                return self._json(200, all_episodes())
            if route.path == '/api/episode':
                return self._json(200, _private_value(_read_package(parse_qs(route.query).get('id', [''])[0])))
            preview = re.fullmatch(r'/api/uploads/([a-f0-9]{32})/preview', route.path)
            if preview:
                self._guard(True)
                return self._upload_preview(preview.group(1), parse_qs(route.query).get('stream', [''])[0])
            if route.path.startswith('/api/uploads/'):
                meta, folder = _upload_meta(route.path.split('/')[-1])
                return self._json(200, _upload_public(meta, folder))
            if route.path.startswith('/files/'):
                return self._serve_output(route.path)
            if route.path.startswith('/api/'):
                raise APIError(404, '找不到这个接口。')
            return self._serve_app(route.path)
        except Exception as exc:
            return self._dispatch_error(exc)

    def do_POST(self):
        try:
            self._guard(True)
            path, body = urlparse(self.path).path, self._body()
            if path == '/api/settings':
                return self._save_settings(body)
            if path == '/api/sources/inspect':
                from .sources import inspect_source
                try:
                    inspected = inspect_source(_public_url(body.get('url', '')))
                except APIError:
                    raise
                except ValueError as exc:
                    raise APIError(400, '该来源暂不可公开获取，请检查链接、登录限制或改用本地导入。') from exc
                return self._json(200, _private_value(inspected))
            if path in {'/api/jobs', '/api/episodes'}:
                return self._add_jobs(body)
            if path == '/api/import':
                if 'audio_base64' in body or 'video_base64' in body:
                    raise APIError(400, '请先通过分块上传接口传入媒体，再提供 upload_id。')
                return self._add_jobs({'specs': [body]})
            if path == '/api/uploads':
                return self._new_upload(body)
            inspection = re.fullmatch(r'/api/uploads/([a-f0-9]{32})/inspect', path)
            if inspection:
                with _upload_lock(inspection.group(1)):
                    meta, folder = _upload_meta(inspection.group(1))
                    return self._json(200, _inspect_uploaded_media(meta, folder))
            match = re.fullmatch(r'/api/uploads/([a-f0-9]{32})/complete', path)
            if match:
                return self._complete_upload(match.group(1), body)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/(pause|resume|retry|continue|export)', path)
            if match:
                return self._job_action(*match.groups(), body)
            raise APIError(404, '找不到这个接口。')
        except Exception as exc:
            return self._dispatch_error(exc)

    def do_PUT(self):
        try:
            self._guard(True)
            match = re.fullmatch(r'/api/uploads/([a-f0-9]{32})', urlparse(self.path).path)
            if not match:
                raise APIError(404, '找不到上传记录。')
            return self._upload_chunk(match.group(1))
        except Exception as exc:
            return self._dispatch_error(exc)

    def _save_settings(self, body):
        settings, updates = get_settings(), {}
        if 'model' in body:
            model = str(body['model']).strip()
            if len(model) > 160 or not re.fullmatch(r'[A-Za-z0-9_.:-]+(?:/[A-Za-z0-9_.:-]+)*', model):
                raise APIError(400, '模型名称格式无效。')
            updates['model'] = model
        if 'base_url' in body:
            base = str(body['base_url']).strip().rstrip('/')
            parsed = urlparse(base)
            allowed = parsed.scheme == 'https' and parsed.hostname in {'api.deepseek.com', 'api.deepseek.com.cn'} or parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost'}
            if not allowed or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise APIError(400, '服务地址仅接受 DeepSeek 官方 HTTPS 地址或本机兼容接口。')
            updates['base_url'] = base
        settings.update(updates)
        if body.get('test'):
            try:
                answer = ask_deepseek('只返回 JSON object。', '请返回 {"ok":true}。', max_tokens=512, settings_override=settings,
                                      api_key_override=str(body['api_key']) if body.get('api_key') else None)
                if answer.get('ok') is not True:
                    raise ValueError('Unexpected model response')
            except Exception as exc:
                raise APIError(502, '模型连接验证失败；原有配置与密钥已保留。') from exc
        if body.get('api_key'):
            save_api_key(str(body['api_key']))
        if body.get('clear_api_key'):
            delete_api_key()
        save_settings(updates)
        return self._json(200, {'saved': True, 'api_key_configured': has_api_key(), 'model': settings['model'], 'daily_check_enabled': False})

    def _add_jobs(self, body):
        urls = body.get('urls', [])
        if isinstance(urls, str):
            urls = [line.strip() for line in urls.splitlines() if line.strip()]
        specs = body.get('specs', [])
        if not isinstance(urls, list) or not isinstance(specs, list) or not 1 <= len(urls) + len(specs) <= 25:
            raise APIError(400, '每次可添加 1 至 25 项素材。')
        normalized = [_normalize_spec({'url': url}) for url in urls] + [_normalize_spec(spec) for spec in specs]
        return self._json(202, {'job_ids': list(dict.fromkeys(_enqueue(spec) for spec in normalized))})

    def _new_upload(self, body):
        size, role = body.get('size'), body.get('role', 'media')
        if role not in {'media', 'subtitle'} or isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size > (MAX_SUBTITLE_SIZE if role == 'subtitle' else MAX_MEDIA_SIZE):
            raise APIError(413, '上传大小或素材类型不符合限制。')
        name = re.sub(r'[^\w.\- ]', '_', str(body.get('name', 'media')))[:180].strip(' .') or 'media'
        suffix = Path(name).suffix.lower()
        if role == 'subtitle' and suffix not in {'.srt', '.vtt', '.ass', '.ssa', '.txt', '.pdf'}:
            raise APIError(400, '字幕支持 SRT、VTT、ASS、SSA、TXT 或 PDF。')
        expected = body.get('sha256', '')
        if expected and not re.fullmatch(r'[a-fA-F0-9]{64}', str(expected)):
            raise APIError(400, 'SHA-256 格式无效。')
        upload_id = uuid.uuid4().hex
        folder = UPLOAD_ROOT / upload_id
        folder.mkdir(parents=True)
        meta = {'upload_id': upload_id, 'name': name, 'size': size, 'role': role, 'file': 'source' + suffix,
                'complete': False, 'expected_sha256': str(expected).lower(), 'created_at': _now()}
        _atomic_json(folder / 'upload.json', meta)
        return self._json(201, _upload_public(meta, folder))

    def _upload_chunk(self, upload_id):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            offset = int(self.headers.get('X-Chunk-Offset', '-1'))
        except ValueError as exc:
            raise APIError(400, '上传块长度或偏移无效。') from exc
        if self.headers.get('Transfer-Encoding') or not 0 < length <= CHUNK_SIZE or offset < 0:
            raise APIError(413, '每块最多 8 MiB，且必须提供块偏移和长度。')
        with _upload_lock(upload_id):
            meta, folder = _upload_meta(upload_id)
            target = folder / meta['file']
            received = target.stat().st_size if target.is_file() else 0
            if meta.get('complete') or offset > received or offset + length > meta['size'] or offset < received < offset + length:
                raise APIError(409, '上传偏移不匹配，请查询断点后继续。', received_bytes=received)
            self.connection.settimeout(60)
            temporary = None
            try:
                digest = hashlib.sha256()
                with tempfile.NamedTemporaryFile(dir=folder, prefix='.chunk-', delete=False) as chunk:
                    temporary = Path(chunk.name)
                    remaining = length
                    while remaining:
                        data = self.rfile.read(min(256 * 1024, remaining))
                        if not data:
                            raise APIError(400, '上传块传输不完整，可从原断点重试。')
                        chunk.write(data)
                        digest.update(data)
                        remaining -= len(data)
                expected = self.headers.get('X-Chunk-SHA256', '').lower()
                if expected and expected != digest.hexdigest():
                    raise APIError(422, '上传块校验失败，可重新发送这一块。')
                if offset < received:
                    previous = hashlib.sha256()
                    with target.open('rb') as stream:
                        stream.seek(offset)
                        remaining = length
                        while remaining:
                            data = stream.read(min(256 * 1024, remaining))
                            if not data:
                                raise APIError(409, '已保存的上传块不完整，请重新上传。')
                            previous.update(data)
                            remaining -= len(data)
                    if previous.digest() != digest.digest():
                        raise APIError(409, '重复上传块与已保存的内容不一致。', received_bytes=received)
                else:
                    with target.open('ab') as stream, temporary.open('rb') as chunk:
                        while data := chunk.read(256 * 1024):
                            stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                return self._json(200, _upload_public(meta, folder))
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def _complete_upload(self, upload_id, body):
        with _upload_lock(upload_id):
            meta, folder = _upload_meta(upload_id)
            target = folder / meta['file']
            if not target.is_file() or target.stat().st_size != meta['size']:
                raise APIError(409, '素材尚未完整上传。', received_bytes=target.stat().st_size if target.is_file() else 0)
            digest = hashlib.sha256()
            with target.open('rb') as stream:
                while data := stream.read(1024 * 1024):
                    digest.update(data)
            expected = body.get('sha256') or meta.get('expected_sha256')
            if expected and str(expected).lower() != digest.hexdigest():
                raise APIError(422, '整文件 SHA-256 校验不一致，请重新上传。')
            meta.update(complete=True, sha256=digest.hexdigest())
            _atomic_json(folder / 'upload.json', meta)
        result = _upload_public(meta, folder)
        if meta['role'] == 'media':
            raw = {key: value for key, value in body.items() if key != 'sha256'} | {'upload_id': upload_id}
            result['job_ids'] = [_enqueue(_normalize_spec(raw))]
        return self._json(200 if meta['role'] == 'subtitle' else 202, result)

    def _upload_preview(self, upload_id, selected):
        with _upload_lock(upload_id):
            meta, folder = _upload_meta(upload_id)
            info = _inspect_uploaded_media(meta, folder)
            if selected == '':
                index = info['suggested_audio_stream_index']
                if index is None:
                    raise APIError(400, '多音轨未发现明确英语轨道，请先选择一个音轨试听。')
            elif re.fullmatch(r'\d{1,4}', selected):
                index = int(selected)
            else:
                raise APIError(400, '音轨编号无效。')
            if index not in {stream['index'] for stream in info['audio_streams']}:
                raise APIError(400, '请选择实际存在的音频轨道。')
            preview = folder / f'preview-audio-{index}.mp3'
            if not preview.is_file():
                ffmpeg = shutil.which('ffmpeg')
                if not ffmpeg:
                    raise APIError(503, '未找到 FFmpeg，请先安装媒体依赖。')
                temporary = folder / f'preview-audio-{index}.part.mp3'
                try:
                    result = subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-y', '-protocol_whitelist', 'file,pipe',
                                             '-i', str(folder / meta['file']), '-t', '12', '-map', f'0:{index}', '-vn', '-sn', '-dn',
                                             '-c:a', 'libmp3lame', '-ac', '2', '-ar', '44100', '-b:a', '128k', str(temporary)],
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                            timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    if result.returncode or not temporary.is_file() or temporary.stat().st_size < 128:
                        raise APIError(422, '该音轨暂时无法生成试听，请检查素材。')
                    os.replace(temporary, preview)
                except subprocess.TimeoutExpired as exc:
                    raise APIError(504, '生成试听超时，请稍后重试。') from exc
                finally:
                    temporary.unlink(missing_ok=True)
        return self._serve_path(preview)

    def _job_action(self, job_id, action, body):
        row = _get_job(job_id)
        if not row:
            raise APIError(404, '任务不存在。')
        if action == 'pause':
            db = connect()
            try:
                changed = db.execute("UPDATE jobs SET status=CASE WHEN status='queued' THEN 'paused' ELSE 'pause_requested' END,message=?,updated_at=? WHERE id=? AND status IN ('queued','running')",
                                     ('正在等待安全的暂停检查点…', _now(), job_id)).rowcount
                db.commit()
            finally:
                db.close()
            if not changed:
                raise APIError(409, '该任务当前不能暂停。')
            return self._json(202, {'accepted': True})
        if action == 'export':
            if not row.get('episode_id'):
                raise APIError(409, '请先生成学习包。')
            package = _read_package(row['episode_id'])
            start, end = body.get('start_ms', 0), body.get('end_ms')
            _validate_export(package, start, end)
            revision = hashlib.sha256(json.dumps({key: package.get(key) for key in ('sentences', 'words', 'cards', 'phrases', 'layout')},
                                                   sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            export_id = _enqueue({'title': package['episode']['title']}, operation='export',
                                 extra={'episode_id': row['episode_id'], 'start_ms': start, 'end_ms': end, 'package_revision': revision})
            return self._json(202, {'accepted': True, 'job_ids': [export_id]})
        inputs = json.loads(row['input_json'] or '{}')
        expected = {'resume': 'paused', 'retry': 'failed', 'continue': 'review'}[action]
        if row['status'] != expected:
            raise APIError(409, '任务状态已经变化，请刷新后操作。')
        if action == 'continue':
            corrected = body.get('corrected_transcript', '')
            if not isinstance(corrected, str) or len(corrected) > 1_000_000:
                raise APIError(400, '校正稿超过大小限制。')
            chapter_id = str(body.get('chapter_id', ''))
            package = _read_package(row['episode_id'])
            chapters = package.get('chapters', [])
            chapter = next((item for item in chapters if item['id'] == chapter_id and item['status'] == 'review'), None)
            if chapter_id and chapter is None:
                raise APIError(409, '该章节当前不需要确认。')
            original = ' '.join(sentence['text'] for sentence in package.get('sentences', [])
                                if not chapter or chapter['start_ms'] <= sentence['start_ms'] < chapter['end_ms'])
            if not chapter_id:
                original = package.get('review', {}).get('recognized_transcript') or original
            if ' '.join(corrected.split()) == ' '.join(original.split()):
                corrected = ''
            if corrected.strip() and not chapter_id and len(chapters) > 1:
                raise APIError(400, '修改文字时请先选择一个待检查章节，避免覆盖其他章节。')
            inputs.update(operation='continue', corrected_transcript=corrected, chapter_id=chapter_id)
        db = connect()
        try:
            claimed = db.execute("UPDATE jobs SET status='queued',stage='queued',message=?,error='',input_json=?,updated_at=? WHERE id=? AND status=?",
                                 ('等待继续处理…', json.dumps(inputs, ensure_ascii=False), _now(), job_id, expected)).rowcount
            db.commit()
        finally:
            db.close()
        if not claimed:
            raise APIError(409, '这项任务已被其他操作接收。')
        _spawn(job_id)
        return self._json(202, {'accepted': True, 'job_ids': [job_id]})

    def _serve_app(self, raw_path):
        base = ROOT / 'dist'
        candidate = (base / unquote(raw_path).lstrip('/')).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError as exc:
            raise APIError(403, '访问路径无效。') from exc
        if not candidate.is_file():
            candidate = base / 'index.html'
        if not candidate.is_file():
            raise APIError(503, '请先构建并启动工作室。')
        return self._serve_path(candidate)

    def _serve_output(self, raw_path):
        parts = [unquote(part) for part in raw_path.split('/') if part][1:]
        if len(parts) < 2 or any(part.startswith('.') for part in parts):
            raise APIError(403, '访问路径无效。')
        base = output_root().resolve()
        candidate = base.joinpath(*parts).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise APIError(403, '访问路径无效。') from exc
        if candidate.name == 'episode.json':
            return self._json(200, _private_value(_read_package(parts[0])))
        if candidate.suffix.lower() == '.json' or candidate.name.endswith(('.log', '.sqlite3', '.dpapi', '.tmp')):
            raise APIError(403, '此文件不是公开学习资料。')
        if not candidate.is_file():
            raise APIError(404, '学习资料尚未生成。')
        return self._serve_path(candidate)

    def _serve_path(self, path):
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        requested = self.headers.get('Range', '')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
            if match and any(match.groups()):
                left, right = match.groups()
                if left:
                    start, end = int(left), min(int(right) if right else size - 1, size - 1)
                else:
                    start, end = max(0, size - int(right)), size - 1
            else:
                start = size
            if start >= size or start > end:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header('Accept-Ranges', 'bytes')
        if requested:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        kind = {'.js': 'application/javascript', '.mjs': 'application/javascript', '.mp3': 'audio/mpeg',
                '.css': 'text/css', '.html': 'text/html'}.get(path.suffix.lower()) or mimetypes.guess_type(str(path))[0]
        self.send_header('Content-Type', kind or 'application/octet-stream')
        self.send_header('Cache-Control', 'no-cache' if path.suffix in {'.html', '.js', '.css'} else 'private, max-age=3600')
        self.send_header('X-Content-Type-Options', 'nosniff')
        remaining = end - start + 1 if size else 0
        self.send_header('Content-Length', str(remaining))
        self.end_headers()
        if self.command == 'HEAD':
            return
        try:
            with path.open('rb') as stream:
                stream.seek(start)
                while remaining:
                    data = stream.read(min(256 * 1024, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_HEAD(self):
        self.do_GET()


def serve(port=8775, open_browser=False):
    try:
        http = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    except OSError as exc:
        if open_browser:
            from urllib.request import urlopen
            try:
                with urlopen(f'http://127.0.0.1:{port}/api/status', timeout=3) as response:
                    existing = json.load(response)
                if existing.get('kind') == 'universal':
                    import webbrowser
                    webbrowser.open(f'http://127.0.0.1:{port}/')
                    return
            except (OSError, ValueError):
                pass
        raise RuntimeError(f'无法使用本地端口 {port}，请关闭已有窗口或选择其他端口。') from exc
    http.daemon_threads = True
    recover_saved_jobs()
    if open_browser:
        import webbrowser
        threading.Timer(1, lambda: webbrowser.open(f'http://127.0.0.1:{port}/')).start()
    print(f'英语视频学习工作室已启动：http://127.0.0.1:{port}', flush=True)
    try:
        http.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        pass
    finally:
        http.server_close()
