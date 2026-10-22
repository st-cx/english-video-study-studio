"""Local Bilibili drafts, explicit QR authorisation and guarded publication.

The optional adapter uses bilibili-api-python's installed interface. It is not
an official Bilibili SDK; successful account submission requires a real account
and is deliberately never exercised by the test suite.
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import importlib.metadata
import io
import json
import re
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlsplit

from . import security
from .storage import _file_lock, _write_json_atomic, data_root

CREATOR_URL = 'https://member.bilibili.com/platform/upload/video/frame'
_LOCK = threading.RLock()
_LOGIN_LOCK = threading.Lock()
_SESSIONS: dict = {}
_ACTIVE: set[str] = set()
_RUNNER = None
_TRANSPORT = None
_SESSION_SECONDS = 180
_ACCOUNT_CHECK: dict = {}
_LOGIN_GENERATION = 0
_LOGIN_POLL_SECONDS = 2
_LAST_LOGIN_DIAGNOSTIC: dict = {}
_LOGIN_DIAGNOSTIC_HISTORY: list[dict] = []


class _QRFlowError(ValueError):
    """A stage/category is diagnostic; upstream exception text never is."""
    def __init__(self, stage, category, address=None):
        self.stage, self.category = stage, category
        self.safe_address = {}
        if address is not None:
            # Only public route diagnostics, never query/fragment or a URL.
            path = address.path
            self.safe_address = {'callback_scheme': address.scheme if address.scheme in {'http', 'https'} else 'other',
                                 'callback_hostname': address.hostname or '',
                                 'callback_path': path if path in {'', '/', '/crossDomain'}
                                    or re.fullmatch(r'/(?:x/passport-login|h5|h5-app)/[A-Za-z_/-]{1,120}', path) else '[omitted]'}
        labels = {'generate': '二维码获取未完成', 'poll': '扫码状态查询未完成',
                  'credentials': '手机确认已返回，但电脑未取得完整登录信息',
                  'protect': '电脑未能加密登录信息', 'save': '电脑未能保存登录信息'}
        super().__init__(f'{labels[stage]}（login.{stage}.{category}）。请稍后重试；原有配置已保留。')


def _invalidate_login_sessions():
    """Caller holds _LOCK; stop watchers before discarding their authority."""
    for session in _SESSIONS.values():
        if session.get('stop'):
            session['stop'].set()
    _SESSIONS.clear()


def _publish_login_diagnostic(record, *, force=False):
    """Caller holds _LOCK; retain only bounded, explicitly safe state records."""
    signature = ('status', 'error_stage', 'error_category', 'callback_scheme', 'callback_hostname', 'callback_path')
    previous = _LOGIN_DIAGNOSTIC_HISTORY[-1] if _LOGIN_DIAGNOSTIC_HISTORY else {}
    if force or not _LOGIN_DIAGNOSTIC_HISTORY or any(previous.get(key) != record.get(key) for key in signature):
        _LOGIN_DIAGNOSTIC_HISTORY.append(dict(record))
        del _LOGIN_DIAGNOSTIC_HISTORY[:-12]
    _LAST_LOGIN_DIAGNOSTIC.clear()
    _LAST_LOGIN_DIAGNOSTIC.update(record)


def _record_login_diagnostic(login_id, session, status, error=None, *, new_session=False):
    with _LOCK:
        if _SESSIONS.get(login_id) is not session:
            return
        if status == 'done':
            session.pop('last_error', None)
        elif error is not None:
            # A later expiry must not erase the actual parsing, network or
            # encryption failure that prevented receipt of the phone approval.
            if (not session.get('last_error')
                    or error.category not in {'local_expired', 'platform_timeout'}):
                session['last_error'] = {'error_stage': error.stage, 'error_category': error.category,
                                         'message': str(error), **error.safe_address}
        _publish_login_diagnostic({'status': status,
            'elapsed_s': round(max(0, time.monotonic()-session['created']), 1),
            'error_stage': '', 'error_category': '', 'message': '', **session.get('last_error', {})}, force=new_session)


def _login_reply(session, status=None):
    return {'status': status or session['status'], 'configured': public_state()['configured'],
            'last_error': session.get('last_error', {}).get('message', '')}


def _monitor_login(login_id, session):
    while not session['stop'].is_set():
        with _LOCK:
            if _SESSIONS.get(login_id) is not session:
                return
        try:
            result = poll_login(login_id)
            if result['status'] in {'done', 'expired', 'timeout'}:
                return
        except ValueError:
            pass  # Only the safe diagnostic is retained; no upstream logging.
        session['stop'].wait(_LOGIN_POLL_SECONDS)


def _qr_failure_category(exc):
    if isinstance(exc, TimeoutError):
        return 'timeout'
    if type(exc).__name__ in {'NetworkException', 'ClientError', 'ConnectionError', 'ConnectError'}:
        return 'network'
    if type(exc).__name__ == 'ResponseCodeException':
        code = getattr(exc, 'code', None)
        return 'platform_' + str(code) if type(code) is int else 'platform_response'
    if isinstance(exc, (KeyError, IndexError, TypeError, ValueError, UnicodeError)):
        return 'response_format'
    if isinstance(exc, PermissionError):
        return 'permission'
    if isinstance(exc, OSError):
        return 'system'
    return 'unavailable'


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _paths():
    root = data_root()
    return root / 'publications.json', root / 'publications.lock', root / 'bilibili.key.dpapi'


def _read_jobs():
    path = _paths()[0]
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('version') != 1 or not isinstance(data.get('jobs'), list):
            raise ValueError()
        return data['jobs']
    except (ValueError, OSError, AttributeError) as exc:
        raise ValueError('投稿记录损坏，已停止写入；请从备份恢复。') from exc


def _mutate(callback):
    path, lock_path, _ = _paths()
    with _LOCK, _file_lock(lock_path, 30, '投稿记录正在使用，请稍后重试。', interval=.02):
        jobs = _read_jobs()
        result = callback(jobs)
        _write_json_atomic(path, {'version': 1, 'jobs': jobs})
        return result


def _safe(job):
    allowed = {'id', 'job_id', 'episode_id', 'status', 'title', 'tid', 'tags', 'description',
               'copyright', 'source', 'video_file', 'cover_file', 'package_revision',
               'rights_confirmed', 'auto_publish', 'error', 'created_at', 'updated_at',
               'bvid', 'aid', 'url', 'creator_url', 'bundle_file', 'blockers', 'attempted_at',
               'stage', 'progress', 'stage_message', 'last_event', 'cid', 'error_code',
               'candidates', 'reconciliation_status', 'reconciled_at'}
    return {key: job[key] for key in allowed if key in job}


def _get(jobs, publication_id):
    if not re.fullmatch(r'[a-f0-9]{32}', str(publication_id)):
        raise ValueError('投稿编号无效。')
    found = next((job for job in jobs if job['id'] == publication_id), None)
    if found is None:
        raise ValueError('找不到该投稿记录。')
    return found


def _credentials():
    path = _paths()[2]
    if not path.is_file():
        return None
    try:
        result = json.loads(security._protect(path.read_bytes(), decrypt=True).decode('utf-8'))
        if not isinstance(result, dict) or not result.get('sessdata') or not result.get('bili_jct'):
            raise ValueError()
        return result
    except Exception as exc:
        raise ValueError('B站登录凭据无法解密，请重新扫码登录；原学习配置不受影响。') from exc


def public_state() -> dict:
    try:
        installed = importlib.metadata.version('bilibili-api-python')
    except importlib.metadata.PackageNotFoundError:
        installed = ''
    path = _paths()[2]
    configured = path.is_file() and path.stat().st_size > 0
    verified = bool(configured and _ACCOUNT_CHECK.get('credential_hash') == _file_hash(path)
                    and time.monotonic() - _ACCOUNT_CHECK.get('checked_at', 0) < 300)
    state = {'configured': configured, 'status': 'configured' if configured else 'disconnected',
            'adapter_available': bool(installed), 'adapter_version': installed,
            'creator_url': CREATOR_URL, 'live_submission_verified': False,
            'message': '扫码后可尝试直接投稿；接口不兼容时可下载投稿包，转到官方创作中心。'}
    if verified:
        state.update(status='verified' if _ACCOUNT_CHECK['valid'] else 'expired',
                     verified=_ACCOUNT_CHECK['valid'], account_name=_ACCOUNT_CHECK.get('name', ''))
    else:
        state['verified'] = False
    with _LOCK:
        state['login_diagnostic'] = dict(_LAST_LOGIN_DIAGNOSTIC)
        state['login_diagnostic_history'] = [dict(record) for record in _LOGIN_DIAGNOSTIC_HISTORY]
        state['last_login_error'] = str(_LAST_LOGIN_DIAGNOSTIC.get('message') or '')
    return state


def check_account() -> dict:
    """Explicit, read-only remote check; file existence is never a valid login."""
    credentials = _credentials()
    if not credentials:
        return public_state()
    credential_hash = _file_hash(_paths()[2])
    try:
        result = _runner().run(_transport().check_account(credentials))
        if not isinstance(result, dict) or type(result.get('valid')) is not bool:
            raise ValueError()
    except Exception:
        raise ValueError('B站登录验证暂未完成，请检查网络后重试；原登录信息已保留。') from None
    with _LOCK:
        # A QR refresh or disconnect while checking cannot certify another file.
        path = _paths()[2]
        if path.is_file() and _file_hash(path) == credential_hash:
            _ACCOUNT_CHECK.clear()
            _ACCOUNT_CHECK.update(credential_hash=credential_hash, checked_at=time.monotonic(),
                                  valid=result['valid'], name=str(result.get('name') or '')[:80])
    return public_state()


class _AsyncRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name='bilibili-network')
        self.thread.start()

    def run(self, coroutine, timeout=45):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError('B站接口响应超时。') from None


def _runner():
    global _RUNNER
    with _LOCK:
        if _RUNNER is None:
            _RUNNER = _AsyncRunner()
        return _RUNNER


class _BilibiliTransport:
    async def check_account(self, credentials):
        from bilibili_api import Credential, user
        credential = Credential(**credentials)
        valid = await credential.check_valid()
        if not valid:
            return {'valid': False, 'name': ''}
        # Only the display name crosses the transport boundary, never cookies,
        # numeric account IDs, complete profile data or request metadata.
        info = await user.get_self_info(credential)
        return {'valid': True, 'name': str(info.get('name') or '')[:80]}

    async def create_qr(self):
        from bilibili_api import login_v2
        from bilibili_api.utils.network import Api, HEADERS
        import qrcode
        try:
            # Keep the SDK's configured network client and API definitions, but
            # never reuse a cacheable generation URL or a shared qrcode.png.
            data = await (Api(**login_v2.API['qrcode']['web']['get_qrcode_and_token'])
                          .update_params(source='main-fe-header', _nonce=uuid.uuid4().hex,
                                         _ts=int(time.time() * 1000))
                          .update_headers(**{**HEADERS, 'Cache-Control': 'no-cache, no-store', 'Pragma': 'no-cache',
                                             'Referer': 'https://www.bilibili.com/'}).result)
            if (not isinstance(data, dict) or not isinstance(data.get('qrcode_key'), str)
                    or not data['qrcode_key'] or not isinstance(data.get('url'), str)):
                raise _QRFlowError('generate', 'missing_fields')
            address = urlsplit(data['url'])
            if (address.scheme != 'https' or address.hostname not in {'passport.bilibili.com', 'account.bilibili.com'}
                    or address.username or address.password):
                raise _QRFlowError('generate', 'unexpected_address', address)
            image = io.BytesIO()
            qrcode.make(data['url']).save(image, format='PNG')
            return {'qrcode_key': data['qrcode_key']}, 'data:image/png;base64,' + base64.b64encode(image.getvalue()).decode('ascii')
        except _QRFlowError:
            raise
        except Exception as exc:
            raise _QRFlowError('generate', _qr_failure_category(exc)) from None

    async def poll_qr(self, login):
        from bilibili_api import login_v2
        from bilibili_api.utils.network import Api, HEADERS
        try:
            if not isinstance(login, dict) or not isinstance(login.get('qrcode_key'), str) or not login['qrcode_key']:
                raise _QRFlowError('poll', 'invalid_session')
            data = await (Api(**login_v2.API['qrcode']['web']['get_events'])
                          .update_params(qrcode_key=login['qrcode_key'], source='main-fe-header',
                                         _nonce=uuid.uuid4().hex, _ts=int(time.time() * 1000))
                          .update_headers(**{**HEADERS, 'Cache-Control': 'no-cache, no-store', 'Pragma': 'no-cache',
                                             'Referer': 'https://www.bilibili.com/'}).result)
            if not isinstance(data, dict) or type(data.get('code')) is not int:
                raise _QRFlowError('poll', 'missing_status')
            code = data['code']
            states = {86101: 'scan', 86090: 'confirm', 86038: 'timeout'}
            if code in states:
                return states[code], None
            if code != 0:
                raise _QRFlowError('poll', 'unknown_status_' + str(code))
            # A successful scan can omit refresh_token. It is useful for future
            # credential refresh, but not required for the current login.
            address = urlsplit(data.get('url') or '')
            if (address.scheme != 'https' or address.hostname not in {'passport.bilibili.com', 'passport.biligame.com'}
                    or address.username or address.password):
                raise _QRFlowError('credentials', 'unexpected_address', address)
            credentials = {}
            for key, value in parse_qsl(address.query, keep_blank_values=True):
                field = {'sessdata': 'sessdata', 'bili_jct': 'bili_jct', 'dedeuserid': 'dedeuserid'}.get(key.lower())
                if field is None:
                    continue
                if field in credentials:
                    raise _QRFlowError('credentials', 'ambiguous_fields')
                # parse_qsl decodes once. Credential expects URL-encoded
                # SESSDATA and only quotes values without a percent sign, so
                # re-encode it exactly once, including literal encoded '&'.
                credentials[field] = quote(value, safe='') if field == 'sessdata' else value
            if not credentials.get('sessdata') or not credentials.get('bili_jct'):
                raise _QRFlowError('credentials', 'missing_fields')
            if isinstance(data.get('refresh_token'), str) and data['refresh_token']:
                credentials['ac_time_value'] = data['refresh_token']
            return 'done', credentials
        except _QRFlowError:
            raise
        except Exception as exc:
            raise _QRFlowError('poll', _qr_failure_category(exc)) from None

    async def submit(self, credentials, video, cover, metadata, progress=None):
        from bilibili_api import Credential, video_uploader
        credential = Credential(**credentials)
        page = video_uploader.VideoUploaderPage(str(video), metadata['title'])
        uploader = video_uploader.VideoUploader([page], metadata, credential, cover=str(cover))
        completed_chunks = set()
        stages = {'PREUPLOAD': ('preupload', 3, '正在准备上传线路'),
                  'PRE_PAGE': ('uploading', 5, '正在上传精听视频'),
                  'PRE_PAGE_SUBMIT': ('assembling', 85, '正在合并已上传的视频分块'),
                  'AFTER_PAGE_SUBMIT': ('uploaded', 88, '视频已上传，正在准备封面'),
                  'PRE_COVER': ('cover', 90, '正在上传封面'),
                  'AFTER_COVER': ('cover_ready', 94, '封面已上传'),
                  'PRE_SUBMIT': ('submitting', 97, '正在向B站提交稿件'),
                  'AFTER_SUBMIT': ('response_received', 99, '投稿响应已返回，正在核对稿件编号')}

        def receive(event):
            name = str(event.get('name', ''))
            raw = event.get('data') or []
            data = raw[0] if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict) else {}
            update = None
            if name in stages:
                stage, percent, message = stages[name]
                update = {'stage': stage, 'progress': percent, 'stage_message': message, 'last_event': name}
                if name == 'PRE_SUBMIT':
                    videos = data.get('videos', [])
                    if videos and isinstance(videos[0], dict) and type(videos[0].get('cid')) is int:
                        update['cid'] = videos[0]['cid']
                if name == 'AFTER_SUBMIT':
                    if re.fullmatch(r'BV[A-Za-z0-9]{10}', str(data.get('bvid', ''))) and type(data.get('aid')) is int:
                        update.update(stage='accepted', progress=100, stage_message='B站已返回稿件编号',
                                      bvid=data['bvid'], aid=data['aid'])
            elif name == 'AFTER_CHUNK':
                number, total = data.get('chunk_number'), data.get('total_chunk_count')
                if type(number) is int and type(total) is int and 0 <= number < total:
                    completed_chunks.add(number)
                    update = {'stage': 'uploading', 'progress': 5 + round(78 * len(completed_chunks) / total),
                              'stage_message': f'正在上传视频分块：{len(completed_chunks)} / {total}', 'last_event': name}
            elif name == 'CHUNK_FAILED':
                update = {'stage': 'uploading', 'stage_message': '部分视频分块暂未上传成功，正在重试', 'last_event': name}
            # The SDK dispatches ordinary handlers synchronously. In particular,
            # PRE_SUBMIT must finish durable bookkeeping BEFORE its API call.
            # Do not turn this callback into an async listener.
            if update is not None and progress is not None:
                progress(update)

        uploader.add_event_listener('__ALL__', receive)
        return await uploader.start()

    async def recent_uploads(self, credentials):
        from bilibili_api import Credential, creative_center
        result = await creative_center.get_video_upload_manager_info(
            Credential(**credentials), pn=1, ps=30, order=creative_center.UploadManagerOrder.SENDDATE)
        return result.get('arc_audits', []) if isinstance(result, dict) else []


def _transport():
    global _TRANSPORT
    if _TRANSPORT is None:
        _TRANSPORT = _BilibiliTransport()
    return _TRANSPORT


def start_login(monitor=False, reuse=False) -> dict:
    # One current, in-memory QR session; no shared image file or browser cookies.
    global _LOGIN_GENERATION
    with _LOGIN_LOCK:
        with _LOCK:
            if reuse is True:
                for existing_id, existing in _SESSIONS.items():
                    remaining = _SESSION_SECONDS - (time.monotonic()-existing['created'])
                    if (existing['status'] in {'scan', 'confirm'} and remaining > 0
                            and existing.get('qr_image') and not existing['stop'].is_set()):
                        worker = existing.get('worker')
                        if monitor is True and (worker is None or not worker.is_alive()):
                            worker = threading.Thread(target=_monitor_login, args=(existing_id, existing),
                                                      daemon=True, name='bilibili-login-monitor')
                            existing['worker'] = worker
                            worker.start()
                        return {'login_id': existing_id, 'qr_image': existing['qr_image'],
                                'status': existing['status'], 'expires_in': max(1, int(remaining))}
            _LOGIN_GENERATION += 1
            generation = _LOGIN_GENERATION
            _invalidate_login_sessions()
            _LAST_LOGIN_DIAGNOSTIC.clear()
        try:
            login, image = _runner().run(_transport().create_qr())
        except _QRFlowError as exc:
            with _LOCK:
                if generation == _LOGIN_GENERATION:
                    _publish_login_diagnostic({'status': 'error', 'elapsed_s': 0,
                        'error_stage': exc.stage, 'error_category': exc.category, 'message': str(exc), **exc.safe_address}, force=True)
            raise ValueError(str(exc)) from None
        except Exception as exc:
            error = _QRFlowError('generate', _qr_failure_category(exc))
            with _LOCK:
                if generation == _LOGIN_GENERATION:
                    _publish_login_diagnostic({'status': 'error', 'elapsed_s': 0,
                        'error_stage': error.stage, 'error_category': error.category, 'message': str(error)}, force=True)
            raise error from None
        login_id = uuid.uuid4().hex
        with _LOCK:
            if generation != _LOGIN_GENERATION:
                raise _QRFlowError('generate', 'cancelled')
            session = {'login': login, 'qr_image': image, 'created': time.monotonic(), 'status': 'scan',
                       'last_poll': 0, 'poll_lock': threading.Lock(), 'stop': threading.Event()}
            _SESSIONS[login_id] = session
            _record_login_diagnostic(login_id, session, 'scan', new_session=True)
            if monitor is True:
                worker = threading.Thread(target=_monitor_login, args=(login_id, session),
                                          daemon=True, name='bilibili-login-monitor')
                session['worker'] = worker
                worker.start()
        return {'login_id': login_id, 'qr_image': image, 'status': 'scan', 'expires_in': _SESSION_SECONDS}


def poll_login(login_id: str) -> dict:
    with _LOCK:
        session = _SESSIONS.get(login_id)
    if not session:
        return {'status': 'expired', 'configured': public_state()['configured']}
    with session['poll_lock']:
        # QR expiry only limits receipt of a new phone confirmation. Once the
        # credentials were saved, a sleeping tab must still observe success.
        # Explicit disconnect/new QR invalidates the session map above.
        if session['status'] == 'done':
            return _login_reply(session)
        if time.monotonic() - session['created'] > _SESSION_SECONDS:
            with _LOCK:
                _record_login_diagnostic(login_id, session, 'expired', _QRFlowError('poll', 'local_expired'))
                if _SESSIONS.get(login_id) is session:
                    _SESSIONS.pop(login_id, None)
                session['stop'].set()
            return _login_reply(session, 'expired')
        if session['status'] == 'timeout':
            return _login_reply(session)
        if time.monotonic() - session['last_poll'] < 1.5:
            return _login_reply(session)
        session['last_poll'] = time.monotonic()
        stage = 'poll'
        try:
            status, credentials = _runner().run(_transport().poll_qr(session['login']))
            if status == 'done':
                stage = 'credentials'
                if not credentials or not credentials.get('sessdata') or not credentials.get('bili_jct'):
                    raise _QRFlowError(stage, 'missing_fields')
                stage = 'protect'
                encrypted = security._protect(json.dumps(credentials).encode('utf-8'))
                stage = 'save'
                with _LOCK:
                    # A concurrent disconnect invalidates the outstanding QR.
                    if _SESSIONS.get(login_id) is not session:
                        return {'status': 'expired', 'configured': False}
                    path = _paths()[2]
                    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
                    try:
                        temporary.write_bytes(encrypted)
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
            session['status'] = status
            _record_login_diagnostic(login_id, session, status,
                                     _QRFlowError('poll', 'platform_timeout') if status == 'timeout' else None)
            return _login_reply(session)
        except _QRFlowError as exc:
            _record_login_diagnostic(login_id, session, 'error', exc)
            raise
        except Exception as exc:
            error = _QRFlowError(stage, _qr_failure_category(exc))
            _record_login_diagnostic(login_id, session, 'error', error)
            raise error from None


def disconnect() -> dict:
    global _LOGIN_GENERATION
    with _LOCK:
        _LOGIN_GENERATION += 1
        _invalidate_login_sessions()
        _LAST_LOGIN_DIAGNOSTIC.clear()
        _LOGIN_DIAGNOSTIC_HISTORY.clear()
        _ACCOUNT_CHECK.clear()
        _paths()[2].unlink(missing_ok=True)
    return public_state()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _inside(folder, name):
    folder = Path(folder).resolve()
    candidate = Path(str(name))
    if not name or candidate.is_absolute() or candidate.drive:
        raise ValueError('投稿文件必须使用学习包内的相对路径。')
    candidate = (folder / candidate).resolve()
    try:
        candidate.relative_to(folder)
    except ValueError as exc:
        raise ValueError('投稿文件不能超出当前学习包。') from exc
    return candidate


def _video_file(package, folder, requested=''):
    name = requested or package.get('render', {}).get('video_file', '') or 'lesson.mp4'
    path = _inside(folder, name)
    relative = path.relative_to(Path(folder).resolve()).as_posix()
    media_names = {str(package.get('media', {}).get('video_file', '')), 'source.mp4', 'source.webm', 'audio.mp3'}
    allowed = relative == 'lesson.mp4' or (len(Path(relative).parts) > 1 and Path(relative).parts[0] in {'exports', 'render'})
    if not allowed or relative in media_names or path.suffix.lower() != '.mp4' or path.name.endswith('.rendering.mp4'):
        raise ValueError('只能投稿已导出的精听 MP4，不能直接投稿原始媒体。')
    return path, relative


def _verify_render(package, folder, relative):
    from .render_identity import verify_render
    return verify_render(package, folder, relative)


def _blockers(package, folder, video_name, rights):
    reasons = []
    quality = package.get('quality', {})
    if quality.get('needs_review') or quality.get('flags') or quality.get('structural_flags'):
        reasons.append('课程仍有待检查问题。')
    if any(chapter.get('status') != 'ready' for chapter in package.get('chapters', [])):
        reasons.append('存在未完成或待检查的制作单元。')
    sync = package.get('sync_review')
    if sync:
        status = sync.get('human_status') or sync.get('human_acceptance', {}).get('status')
        if status != 'passed':
            reasons.append('逐词人工同步验收尚未通过。')
    if not rights:
        reasons.append('请确认拥有该素材的发布权利。')
    path, _ = _video_file(package, folder, video_name)
    if not path.is_file() or path.stat().st_size == 0:
        reasons.append('精听视频尚未导出。')
    elif path.name == 'lesson.mp4' and not (package.get('artifacts', {}).get('video') == 'ready' or package.get('render', {}).get('status') == 'ready'):
        reasons.append('精听视频尚未通过导出检查。')
    else:
        try:
            _verify_render(package, folder, video_name)
        except (ValueError, OSError) as exc:
            reasons.append(str(exc))
    return reasons


def _text(value, name, maximum, required=False):
    if not isinstance(value, str):
        raise ValueError(f'{name}格式不正确。')
    result = value.strip()
    if len(result) > maximum or required and not result:
        raise ValueError(f'{name}不能为空且最多 {maximum} 个字符。' if required else f'{name}最多 {maximum} 个字符。')
    return result


def prepare_publication(package: dict, folder: Path, options: dict) -> dict:
    if not isinstance(package, dict) or not isinstance(options, dict):
        raise ValueError('投稿参数格式不正确。')
    episode = package.get('episode', {})
    episode_id = str(episode.get('id', ''))
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', episode_id):
        raise ValueError('课程编号无效。')
    title = _text(options.get('title') or episode.get('title_zh') or episode.get('title', ''), '标题', 80, True)
    tid = options.get('tid', 208)
    if isinstance(tid, bool) or not isinstance(tid, int) or tid < 1:
        raise ValueError('请选择有效的B站分区编号。')
    try:
        from bilibili_api.video_zone import get_zone_info_by_tid
    except ImportError:
        # The optional SDK is not required for preparation of an offline bundle.
        # Validate the actual platform table again before a live submission.
        pass
    else:
        parent, child = get_zone_info_by_tid(tid)
        if not parent or not child:
            raise ValueError('该B站投稿子分区不存在，请选择有效分区。')
    copyright_value = options.get('copyright', 2)
    if type(copyright_value) is not int or copyright_value not in {1, 2}:
        raise ValueError('投稿类型须为自制或转载。')
    source = _text(options.get('source') or package.get('source', {}).get('url') or episode.get('source_url', ''), '来源', 200)
    if copyright_value == 2 and not source:
        raise ValueError('转载投稿必须填写原始来源。')
    tags = options.get('tags', ['英语学习', '精听'])
    if isinstance(tags, str):
        tags = re.split('[,，]', tags)
    if not isinstance(tags, list) or not 1 <= len(tags) <= 10:
        raise ValueError('请填写 1 至 10 个标签。')
    tags = [_text(tag, '标签', 20, True) for tag in tags]
    if any(',' in tag or '\n' in tag for tag in tags):
        raise ValueError('单个标签不能包含逗号或换行。')
    description = _text(options.get('description', ''), '简介', 2000)
    rights = options.get('rights_confirmed') is True
    auto = options.get('auto_publish') is True
    if auto and not rights:
        raise ValueError('开启自动发布前必须确认素材发布权利。')
    video, relative = _video_file(package, folder, options.get('video_file', ''))
    cover_name = options.get('cover_file') or episode.get('cover', '')
    if cover_name:
        cover_path = _inside(folder, cover_name)
        if cover_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'} or not cover_path.is_file():
            raise ValueError('封面必须是学习包内已有的图片。')
        cover_name = cover_path.relative_to(Path(folder).resolve()).as_posix()
    blockers = _blockers(package, folder, relative, rights)
    identity = uuid.uuid4().hex
    job = {'id': identity, 'job_id': identity, 'episode_id': episode_id,
           'status': 'blocked' if blockers else 'draft', 'title': title, 'tid': tid,
           'tags': tags, 'description': description, 'copyright': copyright_value, 'source': source,
           'video_file': relative, 'cover_file': cover_name, 'package_revision': _digest(package),
           'rights_confirmed': rights, 'auto_publish': auto, 'blockers': blockers, 'error': '',
           'creator_url': CREATOR_URL, 'created_at': _now(), 'updated_at': _now(),
           '_folder': str(Path(folder).resolve()), '_video_sha256': _file_hash(video) if video.is_file() else '',
           '_cover_sha256': _file_hash(cover_path) if cover_name else ''}
    def save(jobs):
        # Repeated clicks keep the same draft, never create a duplicate submit.
        existing = next((old for old in jobs if old['episode_id'] == episode_id and
                         old.get('package_revision') == job['package_revision'] and
                         all(old.get(key) == job.get(key) for key in ('title', 'tid', 'tags', 'description', 'copyright', 'source', 'video_file', 'cover_file', '_cover_sha256', '_video_sha256', 'rights_confirmed', 'auto_publish'))), None)
        if existing:
            return _safe(existing)
        jobs.append(job)
        return _safe(job)
    return _mutate(save)


def list_publications() -> list:
    with _LOCK, _file_lock(_paths()[1], 30, '读取投稿记录超时。', interval=.02):
        return [_safe(job) for job in reversed(_read_jobs())]


def _current(job, provider=None):
    if provider:
        result = provider(job['episode_id'])
        if isinstance(result, tuple) and len(result) == 2:
            package, folder = result
        else:
            raise ValueError('课程读取器必须返回课程数据和学习包目录。')
    else:
        folder = Path(job['_folder'])
        try:
            package = json.loads((folder / 'episode.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('无法读取原学习包，请重新准备投稿。') from exc
    folder = Path(folder).resolve()
    if folder != Path(job['_folder']).resolve() or package.get('episode', {}).get('id') != job['episode_id']:
        raise ValueError('学习包目录或课程编号已改变，请重新准备投稿。')
    if _digest(package) != job['package_revision']:
        raise ValueError('课程内容或时间轴已修订，请重新准备投稿，禁止发布旧成片。')
    blockers = _blockers(package, folder, job['video_file'], job['rights_confirmed'])
    if blockers:
        raise ValueError(' '.join(blockers))
    video, _ = _video_file(package, folder, job['video_file'])
    if not job.get('_video_sha256') or _file_hash(video) != job['_video_sha256']:
        raise ValueError('导出视频已改变，请重新准备投稿。')
    if job.get('cover_file') and _file_hash(_inside(folder, job['cover_file'])) != job.get('_cover_sha256'):
        raise ValueError('封面已改变，请重新准备投稿。')
    return package, folder, video


def _metadata(job):
    value = {'title': job['title'], 'copyright': job['copyright'], 'tid': job['tid'],
             'tag': ','.join(job['tags']), 'desc_format_id': 9999, 'desc': job['description'],
             'recreate': -1, 'dynamic': '', 'interactive': 0, 'act_reserve_create': 0,
             'no_disturbance': 0, 'no_reprint': 0 if job['copyright'] == 2 else 1, 'subtitle': {'open': 0, 'lan': ''},
             'dolby': 0, 'lossless_music': 0, 'web_os': 1}
    if job['copyright'] == 2:
        value['source'] = job['source']
    return value


def _cover(job, folder):
    if job.get('cover_file'):
        return _inside(folder, job['cover_file'])
    # A local, original title card keeps uploads usable without a source cover.
    from PIL import Image, ImageDraw
    directory = folder / 'publications'
    directory.mkdir(exist_ok=True)
    path = directory / (job['id'] + '-cover.png')
    image = Image.new('RGB', (1146, 717), '#17345b')
    draw = ImageDraw.Draw(image)
    draw.text((72, 245), 'ENGLISH STUDY', fill='#ffd064', font_size=70)
    draw.text((76, 350), 'Listen - Replay - Understand', fill='white', font_size=36)
    image.save(path)
    return path


def export_publication_bundle(publication_id: str, package_provider=None) -> dict:
    with _LOCK, _file_lock(_paths()[1], 30, '读取投稿记录超时。', interval=.02):
        job = dict(_get(_read_jobs(), publication_id))
    package, folder, video = _current(job, package_provider)
    cover = _cover(job, folder)
    directory = folder / 'publications'
    directory.mkdir(exist_ok=True)
    final = directory / (job['id'] + '.zip')
    temporary = final.with_name(final.name + '.' + uuid.uuid4().hex + '.tmp')
    metadata = _metadata(job)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.write(video, 'lesson.mp4')
            archive.write(cover, 'cover' + cover.suffix.lower())
            archive.writestr('metadata.json', json.dumps(metadata, ensure_ascii=False, indent=2))
            archive.writestr('投稿说明.txt', '打开B站官方创作中心：' + CREATOR_URL + '\n上传 lesson.mp4 和封面；按 metadata.json 填写标题、分区、标签、简介和来源。\n本包不包含账号、密钥或浏览器数据。\n')
        temporary.replace(final)
    finally:
        temporary.unlink(missing_ok=True)
    def update(jobs):
        current = _get(jobs, publication_id)
        current.update(bundle_file=final.relative_to(folder).as_posix(), updated_at=_now())
        return _safe(current)
    return _mutate(update)


def _finish(publication_id, **fields):
    def update(jobs):
        job = _get(jobs, publication_id)
        job.update(fields, updated_at=_now())
        return _safe(job)
    return _mutate(update)


def _submit_worker(job, package_provider):
    submit_started = False
    accepted_result = None
    def progress(fields):
        nonlocal submit_started, accepted_result
        safe = {key: fields[key] for key in ('stage', 'progress', 'stage_message', 'last_event', 'cid') if key in fields}
        if fields.get('stage') == 'submitting':
            if _credentials() != credentials:
                raise ValueError('登录信息在上传期间已改变，请重新检查后显式重试。')
            safe['attempted_at'] = _now()
        # A valid archive response remains evidence even if a later SDK event
        # raises. Do not turn a returned bvid into an ambiguous retry.
        if fields.get('stage') == 'accepted' and re.fullmatch(r'BV[A-Za-z0-9]{10}', str(fields.get('bvid', ''))) and type(fields.get('aid')) is int:
            accepted_result = {'bvid': fields['bvid'], 'aid': fields['aid']}
        _finish(job['id'], **safe)
        if fields.get('stage') == 'submitting':
            submit_started = True
    try:
        _, folder, video = _current(job, package_provider)
        credentials = _credentials()
        if not credentials:
            raise ValueError('请先扫码登录B站。')
        progress({'stage': 'verifying', 'progress': 1, 'stage_message': '正在验证B站登录状态'})
        if not check_account().get('verified'):
            raise ValueError('B站登录已失效，请重新扫码。')
        cover = _cover(job, folder)
        transport = _transport()
        # Uploading media alone does not create a public archive. The synchronous
        # PRE_SUBMIT callback marks the irreversible boundary durably first.
        result = _runner().run(transport.submit(credentials, video, cover, _metadata(job), progress=progress), timeout=6 * 3600)
        bvid = str((result or {}).get('bvid', ''))
        aid = (result or {}).get('aid')
        if not re.fullmatch(r'BV[A-Za-z0-9]{10}', bvid) or type(aid) is not int:
            raise ValueError('投稿接口没有返回完整稿件编号。')
        _finish(job['id'], status='published', bvid=bvid, aid=aid,
                url='https://www.bilibili.com/video/' + bvid, error='', blockers=[], stage='accepted', progress=100,
                stage_message='B站已返回稿件编号')
    except Exception as exc:
        # Upstream exceptions can contain response URLs/cookies; never expose
        # them through status, logs, a traceback string or persisted records.
        if accepted_result:
            bvid = accepted_result['bvid']
            _finish(job['id'], status='published', **accepted_result,
                    url='https://www.bilibili.com/video/' + bvid, error='', blockers=[],
                    stage='accepted', progress=100, stage_message='B站已返回稿件编号')
        elif submit_started:
            _finish(job['id'], status='uncertain', error='提交结果尚未确认，请先核对稿件，禁止自动重投。',
                    error_code=exc.code if type(getattr(exc, 'code', None)) is int else None)
        else:
            _finish(job['id'], status='blocked', error='稿件尚未提交；请检查登录、上传阶段或网络后显式重试。',
                    error_code=exc.code if type(getattr(exc, 'code', None)) is int else None)
    finally:
        with _LOCK:
            _ACTIVE.discard(job['id'])


def submit_publication(publication_id: str, package_provider=None, *, automatic=False) -> dict:
    with _LOCK, _file_lock(_paths()[1], 30, '投稿记录正在使用。', interval=.02):
        jobs = _read_jobs()
        job = _get(jobs, publication_id)
        if automatic and not (job.get('auto_publish') is True and job.get('rights_confirmed') is True
                              and job['status'] == 'draft' and not job.get('attempted_at')):
            return _safe(job)
        if job['status'] in {'published', 'submitting', 'uncertain'} or job.get('attempted_at'):
            return _safe(job)
        other = next((item for item in jobs if item['id'] != job['id'] and
                      item['episode_id'] == job['episode_id'] and item.get('_video_sha256') and
                      item['_video_sha256'] == job.get('_video_sha256') and
                      (item.get('attempted_at') or item.get('status') in {'submitting', 'published', 'uncertain'})), None)
        if other:
            raise ValueError('同一成片已有投稿或待确认记录，请先到创作中心核对，禁止重复提交。')
        _current(job, package_provider)
        if not _credentials():
            raise ValueError('请先扫码登录B站，或下载投稿包后到官方创作中心上传。')
        job.update(status='submitting', error='', blockers=[], stage='checking', progress=0,
                   stage_message='正在检查本地成片与登录状态', error_code=None, updated_at=_now())
        _write_json_atomic(_paths()[0], {'version': 1, 'jobs': jobs})
        _ACTIVE.add(job['id'])
        worker = threading.Thread(target=_submit_worker, args=(dict(job), package_provider),
                                  daemon=True, name='bilibili-submit-' + job['id'][:8])
        try:
            worker.start()
        except Exception:
            _ACTIVE.discard(job['id'])
            job.update(status='draft', error='投稿线程未启动，请重试。')
            _write_json_atomic(_paths()[0], {'version': 1, 'jobs': jobs})
            raise
        return _safe(job)


def recover_publications() -> list:
    def recover(jobs):
        for job in jobs:
            if job.get('status') == 'submitting' and job['id'] not in _ACTIVE:
                # Older versions did not persist phases; remain conservative.
                safe_upload = job.get('stage') in {'checking', 'verifying', 'preupload', 'uploading', 'assembling', 'uploaded', 'cover', 'cover_ready'} and not job.get('attempted_at')
                job.update(status='blocked' if safe_upload else 'uncertain',
                           error='上传被中断，稿件尚未提交；可检查网络后显式重试。' if safe_upload else '上次投稿被中断；请先核对稿件，系统不会自动重投。', updated_at=_now())
        return [_safe(job) for job in jobs]
    return _mutate(recover)


def pause_auto_publication(publication_id: str) -> dict:
    """Withdraw only an unclaimed automatic draft, never promise a recall."""
    def pause(jobs):
        job = _get(jobs, publication_id)
        if job['status'] not in {'draft', 'blocked'} or job.get('attempted_at'):
            raise ValueError('稿件已开始提交或需要核对结果，不能通过暂停自动投稿撤回；请到创作中心处理。')
        job.update(auto_publish=False, updated_at=_now())
        return _safe(job)
    return _mutate(pause)


def reconcile_publication(publication_id: str) -> dict:
    """Read recent account archives; candidates never authorise a retry.

    Titles can be reused and archive lists can lag. Even one apparent match is
    therefore presented as a candidate for opening, not silently certified.
    """
    with _LOCK, _file_lock(_paths()[1], 30, '读取投稿记录超时。', interval=.02):
        job = dict(_get(_read_jobs(), publication_id))
    if job.get('status') not in {'uncertain', 'published'}:
        raise ValueError('只有提交结果待核对或已提交的记录需要查询稿件。')
    credentials = _credentials()
    if not credentials:
        raise ValueError('请先扫码登录B站，再查询本账号的稿件。')
    if not check_account().get('verified'):
        raise ValueError('B站登录已失效，请重新扫码。')
    try:
        rows = _runner().run(_transport().recent_uploads(credentials))
    except Exception:
        raise ValueError('暂时无法查询稿件；请打开官方创作中心核对，系统不会自动重投。') from None
    if not isinstance(rows, list):
        raise ValueError('稿件列表格式暂不兼容；请到官方创作中心核对，系统不会自动重投。')
    candidates, seen = [], set()
    for row in rows[:100]:
        if not isinstance(row, dict):
            continue
        archive = row.get('Archive') or row.get('archive') or row
        if not isinstance(archive, dict):
            continue
        bvid, aid = str(archive.get('bvid', '')), archive.get('aid')
        if not re.fullmatch(r'BV[A-Za-z0-9]{10}', bvid) or type(aid) is not int or bvid in seen:
            continue
        videos = row.get('videos') or archive.get('videos') or []
        cid_match = (type(job.get('cid')) is int and isinstance(videos, list)
                     and any(isinstance(part, dict) and part.get('cid') == job['cid'] for part in videos))
        known = bvid == job.get('bvid')
        if not (cid_match or known or archive.get('title') == job['title']):
            continue
        seen.add(bvid)
        candidates.append({'bvid': bvid, 'aid': aid, 'title': str(archive.get('title') or '')[:80],
                           'url': 'https://www.bilibili.com/video/' + bvid,
                           'matched_by': 'archive_id' if known else 'uploaded_part' if cid_match else 'title_only',
                           'created_at': archive.get('ctime') if type(archive.get('ctime')) is int else None,
                           'state': archive.get('state') if type(archive.get('state')) is int else None})
    candidates.sort(key=lambda candidate: candidate['matched_by'] == 'title_only')
    return _finish(publication_id, candidates=candidates[:10], reconciled_at=_now(),
                   reconciliation_status='candidates' if candidates else 'not_found',
                   error='' if job['status'] == 'published' else '找到候选稿件，请打开核对；系统仍禁止重投。' if candidates else '暂未找到匹配稿件；列表可能延迟，请到创作中心核对，系统仍禁止重投。')


def process_ready_publications(package_provider=None) -> list:
    # An authorised draft can legitimately be waiting for the user's scan.
    # Absence of a local login is not a failed publication and must not make
    # that draft permanently blocked. Never generate a QR from this scanner.
    if not public_state()['configured']:
        return []
    results = []
    for job in list_publications():
        # Only a per-draft, explicit opt-in can reach the submit method.
        if job.get('auto_publish') and job.get('rights_confirmed') and job['status'] == 'draft' and not job.get('attempted_at'):
            try:
                result = submit_publication(job['id'], package_provider, automatic=True)
                if result.get('auto_publish') is True:
                    results.append(result)
            except (ValueError, OSError):
                results.append(_finish(job['id'], status='blocked', error='自动投稿前检查未通过，请重新准备草稿。'))
    return results
