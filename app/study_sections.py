"""Content navigation with boundaries resolved from measured sentence IDs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .providers import ProviderError, ask_deepseek
from .storage import get_settings, save_episode

VERSION = 1


def source_fingerprint(package: dict) -> str:
    value = [VERSION, package['episode']['duration_s'],
             [{k: sentence[k] for k in ('id', 'text', 'start_ms', 'end_ms')}
              for sentence in package['sentences']]]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def resolve_sections(package: dict, result: dict) -> list[dict]:
    """The provider chooses source anchors, never invents playback timestamps."""
    from opencc import OpenCC
    convert = OpenCC('t2s').convert
    sentences = package['sentences']
    positions = {sentence['id']: i for i, sentence in enumerate(sentences)}
    raw = result.get('sections') if isinstance(result, dict) else None
    if not isinstance(raw, list) or not 1 <= len(raw) <= min(60, len(sentences)):
        raise ValueError('主题目录数量无效')
    anchors = [item.get('start_sentence_id') if isinstance(item, dict) else None for item in raw]
    if any(not isinstance(anchor, str) or anchor not in positions for anchor in anchors):
        raise ValueError('主题目录引用了不存在的原句')
    starts = [positions[anchor] for anchor in anchors]
    if starts[0] != 0 or starts != sorted(set(starts)):
        raise ValueError('主题必须从第一句开始，且按原句顺序唯一排列')
    duration_ms = round(package['episode']['duration_s'] * 1000)
    sections = []
    for index, item in enumerate(raw):
        first = starts[index]
        last = starts[index + 1] if index + 1 < len(starts) else len(sentences)
        title, summary = item.get('title'), item.get('summary')
        if not isinstance(title, str) or not 2 <= len(title.strip()) <= 36:
            raise ValueError('主题标题缺失或过长')
        if not isinstance(summary, str) or not 8 <= len(summary.strip()) <= 160:
            raise ValueError('主题摘要缺失或过长')
        evidence = item.get('evidence_sentence_ids')
        allowed = {sentence['id'] for sentence in sentences[first:last]}
        if not isinstance(evidence, list) or not evidence or len(evidence) > 6 or any(not isinstance(s, str) or s not in allowed for s in evidence):
            raise ValueError('主题摘要的依据必须属于该主题内的原句')
        start = 0 if index == 0 else sentences[first]['start_ms']
        end = sentences[last]['start_ms'] if last < len(sentences) else duration_ms
        if start >= end:
            raise ValueError('主题时间区间无效')
        sections.append({'id': 'topic-' + hashlib.sha256(anchors[index].encode()).hexdigest()[:12],
                         'title': convert(title.strip()), 'summary': convert(summary.strip()),
                         'start_ms': start, 'end_ms': end, 'start_sentence_id': anchors[index],
                         'sentence_ids': [sentence['id'] for sentence in sentences[first:last]],
                         'evidence_sentence_ids': list(dict.fromkeys(evidence))})
    return sections


def ensure_study_sections(package: dict, folder: Path, progress=lambda *args: None) -> bool:
    """Optional, cached topic guide; unavailable annotation never hides a lesson."""
    if len(package.get('sentences', [])) < 8:
        return False
    fingerprint = source_fingerprint(package)
    existing = package.get('study_sections', {})
    if existing.get('source_fingerprint') == fingerprint and existing.get('status') == 'ready':
        try:
            existing['items'] = resolve_sections(package, {'sections': existing.get('items')})
            return True
        except ValueError:
            pass
    settings = get_settings()
    request_fingerprint = hashlib.sha256(json.dumps(
        [fingerprint, settings['model'], settings['base_url']], sort_keys=True).encode()).hexdigest()
    cache = folder / '.study-sections.json'
    result = None
    try:
        saved = json.loads(cache.read_text(encoding='utf-8'))
        if saved.get('request_fingerprint') == request_fingerprint:
            result = saved['result']
    except (OSError, ValueError, KeyError):
        pass
    # Short request-local IDs avoid transcription errors in long hash IDs.
    aliases = {f'S{i + 1:04d}': sentence['id'] for i, sentence in enumerate(package['sentences'])}
    payload = {'course_title': package['episode']['title'],
               'sentences': [{'id': alias, 'text': s['text']} for alias, s in zip(aliases, package['sentences'])]}
    def canonicalize(value):
        if not isinstance(value, dict) or not isinstance(value.get('sections'), list):
            return value
        mapped = []
        for section in value['sections']:
            if not isinstance(section, dict):
                mapped.append(section);continue
            section = dict(section)
            anchor = section.get('start_sentence_id')
            if isinstance(anchor, str):section['start_sentence_id'] = aliases.get(anchor, anchor)
            evidence = section.get('evidence_sentence_ids')
            if isinstance(evidence, list):section['evidence_sentence_ids'] = [aliases.get(item, item) if isinstance(item, str) else item for item in evidence]
            mapped.append(section)
        return {**value, 'sections': mapped}
    prompt = ('根据整课实际内容建立简体中文主题目录，帮助英语学习者按知识主题回听。'
              '先通读全部原句，按内容转折划分连续主题，不按固定分钟或处理批次划分。'
              '保留有意义的课程开场和结尾，每个主题只涵盖相近内容。长课通常10至22段，短视频可以更少。'
              '标题6至18字，摘要用一两句简体中文具体说明本段讲什么，通常30至80字。'
              '摘要不得加入原文没有的结论。只选择输入的原句id作为边界，不能输出或猜测时间。'
              '第一段必须从第一句开始，边界按原句顺序排列；每段给出1至6个本段依据句id。'
              '返回JSON {sections:[{start_sentence_id,title,summary,evidence_sentence_ids:[]}]}。')
    progress('sections', 91, '正在通读整课内容，整理主题时间段和摘要…')
    try:
        for attempt in range(2):
            if result is None:
                result = ask_deepseek(prompt, json.dumps(payload, ensure_ascii=False),
                                      max_tokens=6144, cache_dir=folder / '.teaching-cache')
            try:
                resolved_result = canonicalize(result)
                items = resolve_sections(package, resolved_result)
                result = resolved_result
                break
            except ValueError as error:
                if attempt:
                    raise ProviderError('主题目录未能可靠定位原句，保留原学习单元。') from error
                payload = {**payload, 'previous_result': result, 'validation_error': str(error)}
                result = None
        else:
            return False
        save_episode({'request_fingerprint': request_fingerprint, 'result': result}, folder, cache.name)
        package['study_sections'] = {'version': VERSION, 'status': 'ready',
                                     'source_fingerprint': fingerprint, 'items': items,
                                     'method': 'content_topics_with_measured_sentence_anchors'}
        package.setdefault('processing', {})['study_section_generation'] = {'status': 'ready', 'count': len(items)}
        return True
    except ProviderError as error:
        package.pop('study_sections', None)
        package.setdefault('processing', {})['study_section_generation'] = {'status': 'unavailable', 'reason': str(error)}
        progress('sections', 91, '主题目录暂未生成，已保留原学习单元和全部学习内容。')
        return False
