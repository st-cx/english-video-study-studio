"""Reconcile old review flags against current, signed acoustic observations.

This is a model-evidence gate, never an assertion of human timing accuracy.
All edits are returned as a new package; callers own backup and publication.
"""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path

from .acoustic_reference import _model_fingerprint, reference_issue_key, reference_words_fingerprint
from .reference_check import reference_issues
from .reference_resolution import (
    _continuous_support, _units, resolve_reference_findings, unresolved_alignment,
)
from .sync_audit import digest, file_sha256, normalize_ctc


def inventory(package):
    """Return unique source records, independent of overlapping chapter rows."""
    refs = package.get('processing', {}).get('reference_review', {}).get('unresolved')
    if refs is None:
        refs = [r for c in package.get('chapters', []) for r in c.get('reference_issues', [])]
    result, seen = [], set()
    for kind, rows in [('reference', refs), ('alignment', unresolved_alignment(package))]:
        for row in rows:
            key = digest([kind, row])
            if key not in seen:
                seen.add(key)
                result.append({'id': key, 'kind': kind, 'original': copy.deepcopy(row)})
    return result


def _indices(words, issue):
    return [i for i, w in enumerate(words)
            if issue['start_ms'] <= w['start_ms'] and w['end_ms'] <= issue['end_ms']]


def _values(text):
    return [u['value'] for u in _units([{'text': text, 'start_ms': 0, 'end_ms': 1}])]


def _preference(window, issue, actual, alternative, *, prefer_actual):
    """Recompute saved raw-score checks; no saved winning Boolean is trusted."""
    try:
        begin, end = int(window['clip_start_ms']), int(window['clip_end_ms'])
        comp = window['comparison']
        if not (window.get('status') == 'measured' and
                begin <= issue['start_ms'] < issue['end_ms'] <= end and end - begin <= 30000 and
                re.fullmatch('[0-9a-f]{64}', window['clip_sha256']) and
                comp.get('algorithm') == 'normalized-logsoftmax-ctc-full-context-nll-v1' and
                comp.get('source') == 'independent_ctc_candidate_likelihood' and
                comp.get('edge_wildcards') is False and comp.get('comparable')):
            return False
        rows = {r['id']: r for r in comp['observations']}
        a, b = rows['actual_recognized_context'], rows['reference_edited_context']
        if a['text'] != actual or b['text'] != alternative:
            return False
        for candidate in (a, b):
            text = normalize_ctc(candidate['text'])
            nll = float(candidate['negative_log_likelihood'])
            frames, tokens = int(candidate['frame_count']), int(candidate['target_token_count'])
            if (candidate.get('status') != 'measured' or not text or frames <= 0 or
                    tokens != len(text) or frames != comp.get('frame_count') or
                    not math.isfinite(nll) or nll < 0):
                return False
            if not all(math.isfinite(float(candidate[k])) for k in ('nll_per_frame', 'nll_per_target_token')):
                return False
            if not (math.isclose(candidate['nll_per_frame'], nll / frames, rel_tol=1e-5, abs_tol=1e-6) and
                    math.isclose(candidate['nll_per_target_token'], nll / tokens, rel_tol=1e-5, abs_tol=1e-6)):
                return False
        left, right = (a, b) if prefer_actual else (b, a)
        return all(left[k] < right[k] for k in ('nll_per_frame', 'nll_per_target_token'))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def rebind_ctc_sources(before, after, sources, audio_sha256):
    """Rebind only exact unchanged candidate contexts; never migrate a verdict."""
    old = before.get('words', [])
    by = {w['id']: i for i, w in enumerate(after.get('words', []))}
    words = after.get('words', [])
    findings = reference_issues(words, after.get('review', {}).get('official_transcript', ''))
    result = []
    for source in sources:
        if not isinstance(source, dict) or source.get('audio_sha256') != audio_sha256 or source.get('input_words_sha256') != reference_words_fingerprint(old):
            continue
        value = copy.deepcopy(source)
        kept, rejected = [], 0
        for row in source.get('rows', []):
            selected = row.get('global_word_indices', [])
            if not selected:
                # Source-only gaps require their own two-anchor rebinding route.
                rejected += 1
                continue
            if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(old) or
                   old[i]['id'] not in by or old[i]['text'] != words[by[old[i]['id']]]['text'] for i in selected):
                rejected += 1
                continue
            indices = [by[old[i]['id']] for i in selected]
            matches = [f for f in findings if all(f.get(k) == row.get(k)
                       for k in ('recognized', 'reference', 'start_ms', 'end_ms'))]
            if len(matches) != 1 or indices != list(range(indices[0], indices[-1] + 1)):
                rejected += 1
                continue
            win_kept = []
            for win in row.get('windows', []):
                begin, end = win.get('clip_start_ms', -1), win.get('clip_end_ms', -1)
                local = [i for i, w in enumerate(words) if begin <= w['start_ms'] and w['end_ms'] <= end]
                actual = ' '.join(words[i]['text'] for i in local)
                alternative = []
                for i in local:
                    if i == indices[0] and row.get('reference'):
                        alternative.append(row['reference'])
                    if i not in indices:
                        alternative.append(words[i]['text'])
                candidates = {c.get('id'): c for c in win.get('comparison', {}).get('observations', [])}
                if (candidates.get('actual_recognized_context', {}).get('text') == actual and
                        candidates.get('reference_edited_context', {}).get('text') == ' '.join(alternative)):
                    win_kept.append(copy.deepcopy(win))
            item = copy.deepcopy(row)
            item.update(global_word_indices=indices, windows=win_kept, issue_key=reference_issue_key(matches[0]))
            if win_kept:
                kept.append(item)
            else:
                rejected += 1
        value.update(rows=kept, input_words_sha256=reference_words_fingerprint(words),
                     input_words_rebound_from=source['input_words_sha256'],
                     rebinding_method='exact_full_candidate_context_and_retained_word_ids',
                     rejected_rebindings_count=rejected)
        result.append(value)
    return result


def _duplicate_repair(package, issue, observations, sources, audio_hash):
    """Only a repeated, nonnumeric adjacent token can use this narrow route.

    Two different ASR models must positively give the unique reduced continuous
    context (minimum .65 as in the existing full-context gate). The retained
    target itself must reach .85 in both models. Two full-context raw CTC
    comparisons must separately prefer exactly that reduction.
    """
    words = package['words']
    selected = _indices(words, issue)
    if issue.get('reference') or len(selected) != 1 or selected[0] <= 0:
        return None
    index = selected[0]
    prior, current = words[index - 1], words[index]
    target = _values(current['text'])
    if (len(target) != 1 or target != _values(prior['text']) or
            re.search(r'\d', current['text']) or current['start_ms'] - prior['end_ms'] > 1500):
        return None
    reduced = words[:index] + words[index + 1:]
    supports = {}
    for obs in observations:
        if obs.get('mode') != 'independent_unprompted_asr' or obs.get('audio_sha256') != audio_hash or not re.fullmatch('[0-9a-f]{64}', str(obs.get('model_fingerprint', ''))):
            continue
        core = _continuous_support(reduced, [index - 1], obs, issue)
        if not core:
            continue
        named = [w for w in core if _values(w['text']) == target]
        if len(named) != 1 or float(named[0].get('probability', 0)) < .85:
            continue
        supports.setdefault(obs['model_fingerprint'], []).append({
            'observation_id': obs['observation_id'], 'matched_words': copy.deepcopy(core),
            'observation_sha256': digest(obs)})
    if len(supports) < 2:
        return None
    windows = {}
    for source in sources:
        if (not isinstance(source, dict) or source.get('audio_sha256') != audio_hash or
                source.get('input_words_sha256') != reference_words_fingerprint(words) or
                not re.fullmatch('[0-9a-f]{64}', str(source.get('model_sha256', '')))):
            continue
        for row in source.get('rows', []):
            if row.get('issue_key') != reference_issue_key(issue):
                continue
            for window in row.get('windows', []):
                begin, end = window.get('clip_start_ms', -1), window.get('clip_end_ms', -1)
                local = [i for i, w in enumerate(words) if begin <= w['start_ms'] and w['end_ms'] <= end]
                if index not in local or index - 1 not in local:
                    continue
                actual = ' '.join(words[i]['text'] for i in local)
                alternative = ' '.join(words[i]['text'] for i in local if i != index)
                if _preference(window, issue, actual, alternative, prefer_actual=False):
                    windows[(begin, end)] = copy.deepcopy(window)
    if len(windows) < 2:
        return None
    return {'kind': 'duplicate_token_removed', 'removed_word_id': current['id'],
            'retained_word_id': prior['id'], 'removed_sentence_id': current['sentence_id'],
            'retained_sentence_id': prior['sentence_id'], 'before': copy.deepcopy(current),
            'asr_models': supports, 'ctc_windows': list(windows.values()),
            'audio_sha256': audio_hash, 'input_words_sha256': reference_words_fingerprint(words),
            'human_ground_truth': False,
            'reason': '两个不同无提示识别模型连续上下文均只出现一次该词，双窗以上完整原音候选均支持去掉重复。'}


def _apply_duplicate(package, repair):
    removed, kept = repair['removed_word_id'], repair['retained_word_id']
    old_sid, new_sid = repair['removed_sentence_id'], repair['retained_sentence_id']
    by = {w['id']: w for w in package['words']}
    package['words'] = [w for w in package['words'] if w['id'] != removed]
    aliases = package.setdefault('processing', {}).setdefault('identity_aliases', {'words': {}, 'sentences': {}})
    aliases['words'][removed] = kept
    emptied = False
    for sentence in package['sentences']:
        if removed not in sentence['word_ids']:
            continue
        sentence['word_ids'] = [k for k in sentence['word_ids'] if k != removed]
        if not sentence['word_ids']:
            emptied = True
        else:
            sentence.update(start_ms=by[sentence['word_ids'][0]]['start_ms'], end_ms=by[sentence['word_ids'][-1]]['end_ms'])
            sentence['text'] = ' '.join(by[k]['text'] for k in sentence['word_ids'])
    if emptied:
        package['sentences'] = [s for s in package['sentences'] if s['word_ids']]
        aliases['sentences'][old_sid] = new_sid
    for kind in ('cards', 'phrases'):
        for item in package.get(kind, []):
            if item.get('word_ids'):
                item['word_ids'] = list(dict.fromkeys(kept if k == removed else k for k in item['word_ids']))
            if item.get('sentence_ids'):
                item['sentence_ids'] = list(dict.fromkeys(new_sid if emptied and k == old_sid else k for k in item['sentence_ids']))
            if emptied and item.get('sentence_id') == old_sid:
                item['sentence_id'] = new_sid
    if package.get('layout'):
        pages = []
        for page in package['layout']['pages']:
            page['lines'] = [[k for k in line if k != removed] for line in page['lines']]
            page['lines'] = [line for line in page['lines'] if line]
            if page['lines']:
                pages.append(page)
        package['layout']['pages'] = pages


def _orthographic_repair(package, issue):
    """An apostrophe cannot be heard; use explicit reference spelling only.

    This does not collapse pronoun contractions or change numeric/name text.
    """
    actual, reference = issue.get('recognized', ''), issue.get('reference', '')
    if (not re.fullmatch('[a-z]{4,}s', actual) or not re.fullmatch("[a-z]{4,}'s", reference) or
            actual != reference.replace("'", '') or actual in {'its', 'thats', 'theres', 'heres'}):
        return None
    indices = _indices(package['words'], issue)
    if len(indices) != 1:
        return None
    word = package['words'][indices[0]]
    sentence = next((s for s in package['sentences'] if s['id'] == word['sentence_id']), None)
    if sentence is None or len(re.findall(r'\b' + re.escape(actual) + r'\b', sentence['text'])) != 1:
        return None
    before = word['text']
    word['text'] = re.sub(r'\b' + re.escape(actual) + r'\b', reference, before, count=1)
    sentence['text'] = re.sub(r'\b' + re.escape(actual) + r'\b', reference, sentence['text'], count=1)
    return {'kind': 'reference_orthographic_normalization', 'word_id': word['id'],
            'sentence_id': sentence['id'], 'before_text': before, 'after_text': word['text'],
            'before_ms': [word['start_ms'], word['end_ms']], 'after_ms': [word['start_ms'], word['end_ms']],
            'reference': reference, 'human_ground_truth': False,
            'reason': '规则复数与单数所有格在此读音相同，不能靠听辨区分；只依原参考稿规范撇号，保留词ID和原时钟。'}


def _boundary_support(package, row, surveys, audio_hash):
    """Retire an old ASR interval only with current stable positive CTC frames."""
    detail = row.get('details', {})
    aligned = detail.get('aligned_ms')
    if not isinstance(aligned, list) or len(aligned) != 2:
        return None
    words = [w for w in package['words'] if _values(w['text']) == _values(row.get('text', '')) and
             [w['start_ms'], w['end_ms']] == aligned]
    if len(words) != 1:
        return None
    word = words[0]
    measured = {}
    for survey in surveys:
        if (survey.get('audio_sha256') != audio_hash or
                not re.fullmatch('[0-9a-f]{64}', str(survey.get('model_sha256', '')))):
            continue
        for window in survey.get('windows', []):
            begin, end = window['start_ms'], window['end_ms']
            expected_context = [(w['id'], w['text']) for w in package['words']
                                if begin <= w['start_ms'] and w['end_ms'] <= end
                                and normalize_ctc(w.get('spoken_form') or w['text']) is not None]
            measured_context = [(w.get('id'), w.get('text')) for w in window.get('aligned', [])]
            if expected_context != measured_context:
                continue
            # This checks local row identity, not a stale global word index.
            for item in window.get('aligned', []):
                if item.get('id') != word['id'] or item.get('text') != word['text']:
                    continue
                values = [item.get('start_ms'), item.get('end_ms'), item.get('path_score_not_accuracy')]
                if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in values):
                    continue
                start, stop, score = values
                if not (begin + 800 <= start < stop <= end - 800 and 19 <= stop - start <= 1500 and .8 <= score <= 1):
                    continue
                greedy = [g for g in window.get('greedy_words', []) if
                          _values(g.get('text', '')) == _values(word['text']) and
                          abs(g['start_ms'] - start) <= 100 and abs(g['end_ms'] - stop) <= 100]
                if not greedy or max(abs(word['start_ms'] - start), abs(word['end_ms'] - stop)) > 200:
                    continue
                measured[(begin, end)] = {'window_ms': [begin, end], 'measurement': copy.deepcopy(item),
                                         'greedy': copy.deepcopy(greedy), 'context_sha256': digest(expected_context),
                                         'window_evidence_sha256': digest(window)}
    if len(measured) < 2:
        return None
    rows = list(measured.values())
    if any(max(r['measurement'][k] for r in rows) - min(r['measurement'][k] for r in rows) > 80 for k in ('start_ms', 'end_ms')):
        return None
    return {'kind': 'current_boundary_supported', 'word_id': word['id'],
            'current_ms': [word['start_ms'], word['end_ms']], 'audio_sha256': audio_hash,
            'observations': rows, 'human_ground_truth': False,
            'reason': '当前边界已获两窗高分声学路径和同位无提示CTC词帧支持；旧识别拖入停顿的区间归档。此结果不是人耳200ms验收。'}


def _assess(package, audio, paths, observations, sources):
    assessments = []
    for path in paths:
        assessments.append(resolve_reference_findings(package, package.get('review', {}).get('official_transcript', ''),
                            audio, model_path=str(path), observations=observations, ctc_observations=sources))
    return assessments


def reconcile_reviews(package, audio_path: Path, *, model_paths, observations,
                      ctc_observations=(), boundary_surveys=()):
    """Return reviewable draft, exact 47-style ledger and remaining short groups.

    Old source flags are archived with evidence, not emptied. The old 50-word
    human acceptance manifest is not certified or rewritten by this function.
    """
    original = copy.deepcopy(package)
    draft = copy.deepcopy(package)
    audio_path = Path(audio_path)
    audio_hash = file_sha256(audio_path)
    baseline = inventory(original)
    sources = list(ctc_observations)
    model_paths = list(model_paths)
    valid_models = set()
    for path in model_paths:
        try:
            valid_models.add(_model_fingerprint(str(path)))
        except (OSError, TypeError, ValueError):
            pass
    observations = [copy.deepcopy(o) for o in observations if o.get('audio_sha256') == audio_hash
                    and o.get('model_fingerprint') in valid_models]
    ledger, repairs, automatic = [], [], {}
    for record in baseline:
        if record['kind'] != 'reference':
            continue
        issue = record['original']
        repair = _duplicate_repair(original, issue, observations, sources, audio_hash)
        if repair:
            _apply_duplicate(draft, repair)
        else:
            repair = _orthographic_repair(draft, issue)
        if repair:
            repair.update(audio_sha256=audio_hash, input_words_sha256=reference_words_fingerprint(original['words']))
            repair['evidence_sha256'] = digest(repair)
            repairs.append(repair)
            automatic[record['id']] = repair
    rebound = rebind_ctc_sources(original, draft, sources, audio_hash) if repairs else sources
    assessments = _assess(draft, audio_path, model_paths, observations, rebound)
    current = reference_issues(draft['words'], draft.get('review', {}).get('official_transcript', ''))
    pending_baseline = {reference_issue_key(r['original']) for r in baseline if r['kind'] == 'reference'}
    combined, unresolved = [], []
    for issue in current:
        key = reference_issue_key(issue)
        choices = [d for a in assessments for d in a['resolutions'] if d.get('issue_key') == key]
        chosen = None
        # A recognizer supporting the other reading prevents automatic closure.
        if not any(d.get('status') == 'reference_supported' for d in choices):
            for candidate in choices:
                if candidate.get('status') != 'supported_recognition':
                    continue
                if key in pending_baseline:
                    indices = _indices(draft['words'], issue)
                    group = [o for o in observations if o.get('model_fingerprint') == candidate.get('model_fingerprint')]
                    if not indices or not any(_continuous_support(draft['words'], indices, o, issue) for o in group):
                        continue
                chosen = copy.deepcopy(candidate)
                break
        if chosen is None:
            chosen = {'issue_key': key, 'finding': copy.deepcopy(issue), 'status': 'unresolved',
                      'reason': '仍缺完整连续原声支持，或候选/模型互相冲突；保留供回听。'}
            unresolved.append(copy.deepcopy(issue))
        combined.append(chosen)
    by_issue = {reference_issue_key(d['finding']): d for d in combined}
    resolved_alignment, alignment_pending = {}, []
    removed_ids = {r['removed_word_id']: r for r in repairs if r['kind'] == 'duplicate_token_removed'}
    old_by = {w['id']: w for w in original['words']}
    for record in baseline:
        item = copy.deepcopy(record)
        detail = record['original']
        proof = automatic.get(record['id'])
        if record['kind'] == 'reference' and proof is None:
            candidate = by_issue.get(reference_issue_key(detail))
            if candidate and candidate.get('status') == 'supported_recognition':
                proof = candidate
        if record['kind'] == 'alignment':
            old_target = [w for w in old_by.values() if _values(w['text']) == _values(detail.get('text', '')) and
                          [w['start_ms'], w['end_ms']] == detail.get('details', {}).get('aligned_ms')]
            removed = [w for w in old_target if w['id'] in removed_ids]
            if len(removed) == 1:
                proof = removed_ids[removed[0]['id']]
            else:
                proof = _boundary_support(draft, detail, boundary_surveys, audio_hash)
            if proof:
                resolved_alignment[digest(detail)] = proof
            else:
                alignment_pending.append(copy.deepcopy(detail))
        item.update(status='auto_resolved' if proof else 'needs_listening',
                    audio_sha256=audio_hash, input_words_sha256=reference_words_fingerprint(original['words']),
                    output_words_sha256=reference_words_fingerprint(draft['words']), human_ground_truth=False)
        if proof:
            item['evidence'] = copy.deepcopy(proof)
        else:
            item['reason'] = '尚无足够且相互一致的证据，保留原始候选并合并相邻问题回听。'
        item['decision_sha256'] = digest(item)
        ledger.append(item)
    # Replace originating chunk metadata only for a traceable resolved record.
    for chunk in draft.get('processing', {}).get('speech_chunks', []):
        alignment = chunk.get('alignment', {})
        kept = []
        for dispute in alignment.get('alignment_disputes', []):
            matches = [r for r in baseline if r['kind'] == 'alignment' and r['original'].get('details') == dispute]
            proof = next((resolved_alignment[digest(r['original'])] for r in matches
                          if digest(r['original']) in resolved_alignment), None)
            if proof:
                alignment.setdefault('resolved_history', []).append({'original_dispute': copy.deepcopy(dispute),
                    'resolution': {'method': 'current_fingerprinted_evidence_reconciliation',
                                   'evidence': copy.deepcopy(proof), 'human_ground_truth': False}})
            else:
                kept.append(dispute)
        alignment['alignment_disputes'] = kept
    processing = draft.setdefault('processing', {})
    processing['reference_ctc_observations'] = rebound
    processing['reference_observations'] = observations
    processing['reference_review'] = {'findings': current, 'unresolved': unresolved, 'resolutions': combined,
        'audio_sha256': audio_hash, 'reference_available': bool(draft.get('review', {}).get('official_transcript')),
        'model_fingerprints': sorted({a['model_fingerprint'] for a in assessments if a.get('model_fingerprint')})}
    processing['review_reconciliation'] = {'version': 1, 'human_ground_truth': False,
        'audio_sha256': audio_hash, 'input_words_sha256': reference_words_fingerprint(original['words']),
        'output_words_sha256': reference_words_fingerprint(draft['words']), 'decisions': ledger, 'text_repairs': repairs}
    remaining_alignment = unresolved_alignment(draft)
    old_chapter_flags = {f"{c['title']}：{f}" for c in original.get('chapters', []) for f in c.get('flags', [])}
    for chapter in draft.get('chapters', []):
        old_source_flags = {r['reason'] for r in chapter.get('reference_issues', [])}
        old_source_flags.update(r['reason'] for r in chapter.get('alignment_issues', []))
        old_source_flags.update({'部分词边界仍需听辨。', '独立识别与对齐仍有差异，请回听本单元对应位置。'})
        retained_flags = [f for f in chapter.get('flags', []) + chapter.get('quality', {}).get('flags', [])
                          if f not in old_source_flags]
        chapter['reference_issues'] = [r for r in unresolved if r['start_ms'] < chapter['end_ms'] and r['end_ms'] > chapter['start_ms']]
        chapter['alignment_issues'] = [r for r in remaining_alignment if r['start_ms'] < chapter['end_ms'] and r['end_ms'] > chapter['start_ms']]
        flags = list(dict.fromkeys(retained_flags))
        semantic_review = chapter.get('teaching_review', {})
        if semantic_review.get('status') == 'needs_review' and chapter.get('editor_review', {}).get('status') != 'passed':
            flags += ['教学内容：' + r.get('reason', '教学复核尚未完成。') for r in semantic_review.get('issues', [])]
        flags += [r['reason'] for r in chapter['reference_issues']]
        if chapter['alignment_issues']:
            flags.append('部分词边界仍需听辨。')
        flags = list(dict.fromkeys(flags))
        chapter.update(flags=flags, status='review' if flags else 'ready',
                       teaching_status='text_reviewed_source_pending' if flags else 'ready')
    flags = [f for f in original.get('quality', {}).get('flags', []) if f not in old_chapter_flags]
    flags += [f"{c['title']}：{f}" for c in draft.get('chapters', []) for f in c['flags']]
    draft.setdefault('quality', {}).update(needs_review=bool(flags), flags=flags,
        review_chapter_ids=[c['id'] for c in draft.get('chapters', []) if c['status'] == 'review'])
    if repairs:
        draft.setdefault('review', {})['recognized_transcript'] = ' '.join(s['text'] for s in draft['sentences'])
        draft.pop('sync_review', None)
        draft.setdefault('artifacts', {}).update(video='not_requested', study='pending')
        draft.setdefault('render', {})['status'] = 'prepared'
        # Content summaries resolve their retained sentence anchors after edits.
        if draft.get('study_sections'):
            from .study_sections import resolve_sections, source_fingerprint
            draft['study_sections']['items'] = resolve_sections(draft, {'sections': draft['study_sections']['items']})
            draft['study_sections']['source_fingerprint'] = source_fingerprint(draft)
    remaining = [r for r in ledger if r['status'] == 'needs_listening']
    groups = []
    for record in sorted(remaining, key=lambda r: (r['original']['start_ms'], r['original']['end_ms'])):
        row = record['original']
        if groups and row['start_ms'] - groups[-1]['end_ms'] <= 1800 and row['end_ms'] - groups[-1]['start_ms'] <= 15000:
            groups[-1]['end_ms'] = max(groups[-1]['end_ms'], row['end_ms'])
            groups[-1]['issue_ids'].append(record['id'])
        else:
            groups.append({'start_ms': row['start_ms'], 'end_ms': row['end_ms'], 'issue_ids': [record['id']]})
    summary = {'input_review_records': len(baseline), 'automatic_records': sum(r['status'] == 'auto_resolved' for r in ledger),
               'reference_pending': len(unresolved), 'alignment_pending': len(remaining_alignment),
               'remaining_original_records': len(remaining), 'listening_groups': len(groups),
               'text_repairs': len(repairs), 'human_acceptance': 'not_completed',
               'changed_sentence_ids': sorted({r['sentence_id'] for r in repairs if r.get('sentence_id')} |
                   {r['retained_sentence_id'] for r in repairs if r.get('retained_sentence_id')} |
                   {r['removed_sentence_id'] for r in repairs if r.get('removed_sentence_id')}),
               'word_ids_retained': len({w['id'] for w in original['words']} & {w['id'] for w in draft['words']}),
               'sentence_ids_retained': len({s['id'] for s in original['sentences']} & {s['id'] for s in draft['sentences']})}
    return {'episode': draft, 'decisions': ledger, 'repairs': repairs, 'listening_groups': groups, 'summary': summary}
