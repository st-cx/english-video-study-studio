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
import json
import re
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

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
               'bvid', 'aid', 'url', 'creator_url', 'bundle_file', 'blockers', 'attempted_at'}
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
    return {'configured': configured, 'status': 'configured' if configured else 'disconnected',
            'adapter_available': bool(installed), 'adapter_version': installed,
            'creator_url': CREATOR_URL, 'live_submission_verified': False,
            'message': '扫码后可尝试直接投稿；接口不兼容时可下载投稿包，转到官方创作中心。'}


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
    async def create_qr(self):
        from bilibili_api import login_v2
        login = login_v2.QrCodeLogin()
        await login.generate_qrcode()
        picture = login.get_qrcode_picture()
        return login, 'data:image/png;base64,' + base64.b64encode(picture.content).decode('ascii')

    async def poll_qr(self, login):
        event = await login.check_state()
        status = str(event.value)
        if status != 'done':
            return status, None
        credential = login.get_credential()
        # A strict allowlist avoids proxy settings and internal client state.
        keys = ('sessdata', 'bili_jct', 'buvid3', 'buvid4', 'dedeuserid', 'ac_time_value')
        return status, {key: getattr(credential, key, None) for key in keys if getattr(credential, key, None)}

    async def submit(self, credentials, video, cover, metadata):
        from bilibili_api import Credential, video_uploader
        credential = Credential(**credentials)
        page = video_uploader.VideoUploaderPage(str(video), metadata['title'])
        uploader = video_uploader.VideoUploader([page], metadata, credential, cover=str(cover))
        return await uploader.start()


def _transport():
    global _TRANSPORT
    if _TRANSPORT is None:
        _TRANSPORT = _BilibiliTransport()
    return _TRANSPORT


def start_login() -> dict:
    # The upstream QR helper uses a shared temporary filename.
    with _LOGIN_LOCK:
        with _LOCK:
            _SESSIONS.clear()
        try:
            login, image = _runner().run(_transport().create_qr())
        except Exception:
            raise ValueError('暂时无法生成B站登录二维码；可使用官方创作中心和本地投稿包。') from None
        login_id = uuid.uuid4().hex
        with _LOCK:
            _SESSIONS[login_id] = {'login': login, 'created': time.monotonic(), 'status': 'scan',
                                   'last_poll': 0, 'poll_lock': threading.Lock()}
        return {'login_id': login_id, 'qr_image': image, 'status': 'scan', 'expires_in': _SESSION_SECONDS}


def poll_login(login_id: str) -> dict:
    with _LOCK:
        session = _SESSIONS.get(login_id)
    if not session:
        return {'status': 'expired', 'configured': public_state()['configured']}
    with session['poll_lock']:
        if time.monotonic() - session['created'] > _SESSION_SECONDS:
            with _LOCK:
                _SESSIONS.pop(login_id, None)
            return {'status': 'expired', 'configured': public_state()['configured']}
        if session['status'] in {'done', 'timeout'}:
            return {'status': session['status'], 'configured': public_state()['configured']}
        if time.monotonic() - session['last_poll'] < 1.5:
            return {'status': session['status'], 'configured': public_state()['configured']}
        session['last_poll'] = time.monotonic()
        try:
            status, credentials = _runner().run(_transport().poll_qr(session['login']))
            if status == 'done':
                if not credentials or not credentials.get('sessdata') or not credentials.get('bili_jct'):
                    raise ValueError()
                encrypted = security._protect(json.dumps(credentials).encode('utf-8'))
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
            return {'status': status, 'configured': public_state()['configured']}
        except Exception:
            raise ValueError('扫码状态查询失败，请稍后重试；未记录任何登录信息到日志。') from None


def disconnect() -> dict:
    with _LOCK:
        _SESSIONS.clear()
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
             'no_disturbance': 0, 'no_reprint': 1, 'subtitle': {'open': 0, 'lan': ''},
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
    network_started = False
    try:
        _, folder, video = _current(job, package_provider)
        credentials = _credentials()
        if not credentials:
            raise ValueError('请先扫码登录B站。')
        cover = _cover(job, folder)
        transport = _transport()
        # Persist before the remote call. A crash can only become uncertain,
        # never an automatic replay of a potentially accepted publication.
        _finish(job['id'], attempted_at=_now())
        network_started = True
        result = _runner().run(transport.submit(credentials, video, cover, _metadata(job)), timeout=6 * 3600)
        bvid = str((result or {}).get('bvid', ''))
        aid = (result or {}).get('aid')
        if not re.fullmatch(r'BV[A-Za-z0-9]{10}', bvid) or not isinstance(aid, int):
            raise ValueError('投稿接口没有返回完整稿件编号。')
        _finish(job['id'], status='published', bvid=bvid, aid=aid,
                url='https://www.bilibili.com/video/' + bvid, error='', blockers=[])
    except Exception:
        # Upstream exceptions can contain response URLs/cookies; never expose
        # them through status, logs, a traceback string or persisted records.
        if network_started:
            _finish(job['id'], status='uncertain', error='提交结果尚未确认，请先到B站创作中心核对，禁止自动重投。')
        else:
            _finish(job['id'], status='blocked', error='投稿前检查未通过，请刷新草稿、检查登录与本地文件。')
    finally:
        with _LOCK:
            _ACTIVE.discard(job['id'])


def submit_publication(publication_id: str, package_provider=None) -> dict:
    with _LOCK, _file_lock(_paths()[1], 30, '投稿记录正在使用。', interval=.02):
        jobs = _read_jobs()
        job = _get(jobs, publication_id)
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
        job.update(status='submitting', error='', blockers=[], updated_at=_now())
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
                job.update(status='uncertain', error='上次投稿被中断；请先到官方创作中心核对，系统不会自动重投。', updated_at=_now())
        return [_safe(job) for job in jobs]
    return _mutate(recover)


def process_ready_publications(package_provider=None) -> list:
    results = []
    for job in list_publications():
        # Only a per-draft, explicit opt-in can reach the submit method.
        if job.get('auto_publish') and job.get('rights_confirmed') and job['status'] == 'draft' and not job.get('attempted_at'):
            try:
                results.append(submit_publication(job['id'], package_provider))
            except (ValueError, OSError):
                results.append(_finish(job['id'], status='blocked', error='自动投稿前检查未通过，请重新准备草稿。'))
    return results
