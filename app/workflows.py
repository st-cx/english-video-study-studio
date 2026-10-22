"""Opt-in source subscriptions, with durable discovery and publication handoff.

Existing production jobs remain the authority for media processing. This module
only discovers public links and remembers their job IDs; it never edits model
settings, user learning state or a lesson's quality decisions.
"""
from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import re
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlsplit

from .storage import data_root, _file_lock, _write_json_atomic
from .sources import checked_url, _read_url, inspect_source


def _path():
    return data_root() / 'resource-workflows.json'


def _read():
    try:
        value = json.loads(_path().read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {'version': 1, 'sources': []}
    if value.get('version') != 1 or not isinstance(value.get('sources'), list):
        raise ValueError('资源订阅记录格式异常，原文件已保留。')
    return value


def _lock():
    return _file_lock(_path().with_suffix('.lock'), 20, '资源订阅正在保存，请稍后重试。', interval=.03)


def list_sources():
    with _lock():
        return copy.deepcopy(_read()['sources'])


def save_source(raw):
    if not isinstance(raw, dict):
        raise ValueError('资源订阅参数必须是对象。')
    mode = raw.get('mode', 'rss')
    if mode not in {'rss', 'playlist', 'single'}:
        raise ValueError('请选择 RSS、YouTube 播放列表或单个视频。')
    url = checked_url(str(raw.get('url', '')), resolve=False)
    host = urlsplit(url).hostname.lower().rstrip('.')
    if host in {'localhost', 'localhost.localdomain'} or host.endswith(('.localhost', '.local', '.internal')):
        raise ValueError('订阅不能指向本机或内网。')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global:
            raise ValueError('订阅不能指向本机、内网或保留网络。')
    interval = raw.get('interval_minutes', 60)
    if isinstance(interval, bool) or not isinstance(interval, int) or not 15 <= interval <= 10080:
        raise ValueError('检查间隔需在 15 分钟至 7 天之间。')
    options = raw.get('publication', {})
    if not isinstance(options, dict):
        raise ValueError('投稿选项格式错误。')
    automatic = raw.get('auto_publish') is True
    if automatic and (raw.get('auto_generate') is not True or options.get('rights_confirmed') is not True):
        raise ValueError('自动投稿需要先启用自动制作，并确认素材的发布权限。')
    # Publication fields are revalidated by the publishing module before use.
    selected = {k: options[k] for k in ('tid', 'tags', 'description', 'copyright', 'rights_confirmed') if k in options}
    with _lock():
        state = _read()
        identifier = raw.get('id') or uuid.uuid4().hex
        if not isinstance(identifier, str) or not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('资源订阅编号无效。')
        previous = next((s for s in state['sources'] if s['id'] == identifier), None)
        if previous and (previous['url'] != url or previous['mode'] != mode):
            raise ValueError('修改来源地址请新建订阅，以保留旧来源的去重记录。')
        record = {**(previous or {}), 'id': identifier, 'name': str(raw.get('name') or '英语资源')[:120],
                  'url': url, 'mode': mode, 'enabled': raw.get('enabled', True) is True,
                  'interval_minutes': interval, 'auto_generate': raw.get('auto_generate') is True,
                  'auto_publish': automatic, 'publication': selected,
                  'revision': uuid.uuid4().hex, 'seen': (previous or {}).get('seen', []),
                  'items': (previous or {}).get('items', []), 'last_checked': (previous or {}).get('last_checked', 0)}
        state['sources'] = [s for s in state['sources'] if s['id'] != identifier] + [record]
        _write_json_atomic(_path(), state)
    return copy.deepcopy(record)


def delete_source(identifier):
    with _lock():
        state = _read()
        state['sources'] = [s for s in state['sources'] if s['id'] != identifier]
        _write_json_atomic(_path(), state)
    return {'deleted': True}


def enqueue_item(identifier, item_id, enqueue):
    with _lock():
        state = _read()
        source = next((s for s in state['sources'] if s['id'] == identifier), None)
        item = next((r for r in (source or {}).get('items', []) if r['id'] == item_id), None)
        if not item:
            raise ValueError('该资源项不存在，请刷新订阅。')
        if not item.get('job_id'):
            item['job_id'] = enqueue({'url': item['url'], 'source_url': item['url'], 'title': item['title'],
                                      'render_mode': 'video' if source['auto_publish'] else 'auto'})
            item['status'] = 'queued'
            _write_json_atomic(_path(), state)
        return copy.deepcopy(item)


def discover(source):
    if source['mode'] == 'single':
        info = inspect_source(source['url'])
        return [{'url': source['url'], 'title': info.get('title') or source['name']}]
    if source['mode'] == 'playlist':
        info = inspect_source(source['url'])
        entries = info.get('entries', [])
        return [{'url': checked_url(e.get('source_url') or e.get('webpage_url') or e.get('url', ''), resolve=False),
                 'title': str(e.get('title') or '英语视频')[:200]} for e in entries[:50]]
    content, final, _ = _read_url(source['url'], limit=2 * 1024 * 1024)
    if b'<!doctype' in content.lower() or b'<!entity' in content.lower():
        raise ValueError('请使用不含外部实体的 RSS／Atom 订阅地址。')
    root = ET.fromstring(content)
    rows = []
    for element in root.iter():
        if element.tag.rsplit('}', 1)[-1] not in {'item', 'entry'}:
            continue
        title, link, enclosure = '', '', ''
        for child in element:
            tag = child.tag.rsplit('}', 1)[-1]
            if tag == 'title':
                title = ''.join(child.itertext()).strip()
            if tag == 'enclosure' and child.get('url'):
                enclosure = child.get('url')
            if tag == 'link':
                if child.get('rel') == 'enclosure':
                    enclosure = child.get('href', '')
                elif child.get('rel', 'alternate') == 'alternate':
                    link = child.get('href') or (child.text or '').strip()
        target = enclosure or link
        if target:
            rows.append({'title': title[:200] or '英语资源', 'url': checked_url(urljoin(final, target), resolve=False)})
        if len(rows) >= 50:
            break
    if not rows:
        raise ValueError('订阅没有提供可处理的视频或音频链接。')
    return list({r['url']: r for r in rows}.values())


def scan_source(identifier, enqueue, *, force=True):
    # This OS lock prevents two application instances from racing discovery.
    with _file_lock(data_root() / 'resource-scan.lock', 1, '资源检查正在进行，请稍后查看。', interval=.05):
        source = next((s for s in list_sources() if s['id'] == identifier), None)
        if not source:
            raise ValueError('资源订阅不存在。')
        if not force and (not source['enabled'] or time.time() - source['last_checked'] < source['interval_minutes'] * 60):
            return source
        try:
            entries = discover(source)
            with _lock():
                state = _read()
                current = next((s for s in state['sources'] if s['id'] == identifier), None)
                if not current or current['revision'] != source['revision']:
                    raise ValueError('检查期间订阅已修改，请重新检查。')
                unseen = [e for e in entries if e['url'] not in current['seen']]
                # First enable processes the first listed item only, avoiding a historical flood.
                selected = unseen[:1 if not current.get('initialized') else 5]
                for entry in selected:
                    row = {**entry, 'id': hashlib.sha256(entry['url'].encode()).hexdigest()[:24],
                           'status': 'discovered', 'discovered_at': time.time()}
                    if current['auto_generate']:
                        row['job_id'] = enqueue({'url': row['url'], 'source_url': row['url'], 'title': row['title'],
                                                 'render_mode': 'video' if current['auto_publish'] else 'auto'})
                        row['status'] = 'queued'
                    current['items'].insert(0, row)
                # The first scan intentionally skips historical items. Later
                # scans may cap processing, but must retain an unseen backlog.
                marked = entries if not current.get('initialized') else selected
                current['seen'] = list(dict.fromkeys(current['seen'] + [e['url'] for e in marked]))
                current.update(initialized=True, last_checked=time.time(), error='')
                _write_json_atomic(_path(), state)
                return copy.deepcopy(current)
        except Exception as exc:
            with _lock():
                state = _read()
                current = next((s for s in state['sources'] if s['id'] == identifier), None)
                if current:
                    current.update(last_checked=time.time(), error='资源检查未完成，请检查链接或网络后重试。')
                    _write_json_atomic(_path(), state)
            raise ValueError('资源检查未完成，请检查链接、网络或改用本地导入。') from exc


def _item_error(source_id, item_id, revision, message):
    with _lock():
        state = _read()
        source = next((s for s in state['sources'] if s['id'] == source_id), None)
        if source and source['revision'] == revision:
            item = next((row for row in source['items'] if row['id'] == item_id), None)
            if item:
                item['error'] = message
                _write_json_atomic(_path(), state)


def handoff_completed(job_lookup, package_provider):
    """Only completed, quality-approved rendered jobs reach a publication draft."""
    from .publishing import prepare_publication, submit_publication, list_publications
    for source in list_sources():
        if not source['enabled'] or not source['auto_publish']:
            continue
        for item in source['items']:
            if not item.get('job_id'):
                continue
            try:
                job = job_lookup(item['job_id'])
                if not job or job['status'] != 'complete' or not job.get('episode_id'):
                    continue
                package, folder = package_provider(job['episode_id'])
                # The subscription lock is also the authorisation boundary.
                # Prepare, bind and claim submission before an edit/disable can
                # interleave and leave an unbound auto-publishing draft.
                with _lock():
                    state = _read()
                    live = next((s for s in state['sources'] if s['id'] == source['id']), None)
                    if not live or live['revision'] != source['revision'] or not live['enabled'] or not live['auto_publish']:
                        continue
                    target = next((r for r in live['items'] if r['id'] == item['id']), None)
                    if not target:
                        continue
                    if target.get('publication_id'):
                        if target.get('publication_revision') != live['revision']:
                            continue
                        draft = next((p for p in list_publications() if p['id'] == target['publication_id']), None)
                        if not draft:
                            target['error'] = '原投稿草稿不存在，请手动重新准备。'
                            _write_json_atomic(_path(), state)
                            continue
                    else:
                        options = {**live['publication'], 'source': target['url'], 'auto_publish': True}
                        options.setdefault('description', '英语精听学习视频，包含逐词高亮、简体中文释义与句型讲解。\n原始来源：' + target['url'])
                        draft = prepare_publication(package, folder, options)
                        target.update(publication_id=draft['id'], publication_revision=live['revision'], status='publication', error='')
                        # Persist the binding before the transport. Recovery
                        # reuses this ID and never blindly creates a new post.
                        _write_json_atomic(_path(), state)
                    if draft.get('status') in {'ready', 'draft'} and not draft.get('attempted_at'):
                        submit_publication(draft['id'], package_provider=package_provider, automatic=True)
                        target['error'] = ''
                    elif draft.get('status') == 'blocked':
                        target['error'] = '成片或同步检查尚未通过，请在投稿草稿中处理。'
                    elif draft.get('status') == 'uncertain':
                        target['error'] = 'B站提交结果待核对，请到创作中心确认，系统不会自动重投。'
                    _write_json_atomic(_path(), state)
            except Exception:
                # One failed account, file or draft cannot starve other feeds.
                # SDK exception text may contain credentials and is never kept.
                _item_error(source['id'], item['id'], source['revision'], '自动投稿暂未完成，请检查B站登录、成片状态或使用投稿包。')


def run_background(enqueue, job_lookup, package_provider, stop=None):
    from .publishing import process_ready_publications
    stop = stop or threading.Event()
    def loop():
        while not stop.is_set():
            try:
                for source in list_sources():
                    if source['enabled']:
                        try:
                            scan_source(source['id'], enqueue, force=False)
                        except Exception:
                            pass  # scan_source persists a safe, visible failure.
                handoff_completed(job_lookup, package_provider)
            except Exception:
                pass
            try:
                process_ready_publications(package_provider)
            except Exception:
                pass  # An account or draft failure cannot stop resource checks.
            stop.wait(60)
    threading.Thread(target=loop, daemon=True, name='english-resources').start()
    return stop
