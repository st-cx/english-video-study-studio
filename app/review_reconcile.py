"""Reconcile old review flags against current, signed acoustic observations.

This is a model-evidence gate, never an assertion of human timing accuracy.
All edits are returned as a new package; callers own backup and publication.
"""
from __future__ import annotations

import copy
import math
import re
import statistics
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


def merge_ctc_sources(sources, words, audio_sha256):
    """Join compatible measurement batches without turning copies into votes.

    A resumed survey can place two real crops in separate files. They may only
    be combined when their complete text/clock, audio and model identities all
    match. Conflicting copies of one crop are excluded, not arbitrarily chosen.
    Downstream gates still recompute every raw candidate likelihood check.
    """
    fingerprint = reference_words_fingerprint(words)
    groups = {}
    for source in sources:
        if (not isinstance(source, dict) or source.get('audio_sha256') != audio_sha256 or
                source.get('input_words_sha256') != fingerprint or
                source.get('input_words_fingerprint_version') != 'text-time-1' or
                source.get('source') != 'independent_full_context_ctc_likelihood' or
                not re.fullmatch('[0-9a-f]{64}', str(source.get('model_sha256', '')))):
            continue
        key = source['model_sha256']
        group = groups.setdefault(key, {'source': copy.deepcopy(source), 'rows': {}, 'batches': set()})
        group['batches'].add(digest(source))
        for row in source.get('rows', []):
            if not isinstance(row, dict):
                continue
            identity = {k: row[k] for k in ('issue_key', 'start_ms', 'end_ms',
                'recognized', 'reference', 'global_word_indices', 'global_anchor_indices',
                'global_insertion_boundary', 'anchors') if k in row}
            rkey = digest(identity)
            entry = group['rows'].setdefault(rkey, {'identity': identity, 'windows': {}, 'conflicts': set()})
            for window in row.get('windows', []):
                if not isinstance(window, dict):
                    continue
                bounds = (window.get('clip_start_ms'), window.get('clip_end_ms'))
                if (len(bounds) != 2 or any(not isinstance(n, (int, float)) or
                        isinstance(n, bool) or not math.isfinite(n) for n in bounds)):
                    continue
                previous = entry['windows'].get(bounds)
                if previous is not None and digest(previous) != digest(window):
                    entry['conflicts'].add(bounds)
                else:
                    entry['windows'][bounds] = copy.deepcopy(window)
    result = []
    for key in sorted(groups):
        group = groups[key]
        merged = group['source']
        merged['rows'] = []
        for rkey in sorted(group['rows']):
            entry = group['rows'][rkey]
            windows = [entry['windows'][bounds] for bounds in sorted(entry['windows'])
                       if bounds not in entry['conflicts']]
            merged['rows'].append({**entry['identity'], 'windows': windows})
        merged['merged_from_sha256'] = sorted(group['batches'])
        merged['merging_method'] = 'exact_audio_model_text_clock_with_conflicting_crop_rejection'
        result.append(merged)
    return result


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
    if issue.get('reference') or len(selected) != 1:
        return None
    index = selected[0]
    # A sequence matcher may label the *first* of two identical tokens extra.
    # Retain the earlier measured word and its ID in either representation.
    if (index + 1 < len(words) and
            _values(words[index]['text']) == _values(words[index + 1]['text'])):
        index += 1
    if index <= 0:
        return None
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


def _boundary_target(package, row):
    """Follow exact old spans or retained revision IDs, never nearest spelling."""
    detail = row.get('details', {})
    spans = [detail.get('aligned_ms'), detail.get('asr_ms')]
    by = {w['id']: w for w in package['words']}
    ids = {w['id'] for w in by.values() if _values(w['text']) == _values(row.get('text', '')) and
           [w['start_ms'], w['end_ms']] in spans}
    for history in package.get('processing', {}).get('timing_revision_history', []):
        for patch in history.get('patches', []):
            if (patch.get('before_ms') in spans and _values(patch.get('text', '')) == _values(row.get('text', '')) and
                    patch.get('word_id') in by and _values(by[patch['word_id']]['text']) == _values(row.get('text', ''))):
                ids.add(patch['word_id'])
    return by[next(iter(ids))] if len(ids) == 1 else None


def _interval_preference(comparison, window, word, before, after):
    """Recompute same-token acoustic localization on old and new raw frames."""
    try:
        if (comparison.get('algorithm') != 'same-target-two-interval-ctc-nll-v1' or
                comparison.get('word_id') != word['id'] or comparison.get('text') != word['text'] or
                comparison.get('before_ms') != before or comparison.get('after_ms') != after or
                comparison.get('emissions_sha256') != window.get('emissions_sha256') or
                not re.fullmatch('[0-9a-f]{64}', str(comparison.get('emissions_sha256', ''))) or
                comparison.get('padding_ms') != 100):
            return False
        values = comparison['intervals']
        for key, bounds in [('before', before), ('after', after)]:
            value = values[key]
            nll, frames, tokens = float(value['negative_log_likelihood']), int(value['frame_count']), int(value['target_token_count'])
            normalized = normalize_ctc(word['text'])
            if (not normalized or tokens != len(normalized) or frames <= 0 or
                    not math.isfinite(nll) or nll < 0 or value['interval_ms'] != bounds or
                    not window['start_ms'] <= bounds[0] - 100 < bounds[1] + 100 <= window['end_ms']):
                return False
            if not (math.isclose(value['nll_per_frame'], nll / frames, rel_tol=1e-5, abs_tol=1e-6) and
                    math.isclose(value['nll_per_target_token'], nll / tokens, rel_tol=1e-5, abs_tol=1e-6)):
                return False
        old, new = values['before'], values['after']
        return (all(new[k] < old[k] * .7 for k in ('nll_per_frame', 'nll_per_target_token')) and
                old['nll_per_target_token'] - new['nll_per_target_token'] >= 1)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _cross_boundary_evidence(package, row, observations, surveys, audio_hash):
    """An independent alternative to the unchanged high-path-score CTC gate.

    Two distinct unprompted ASR models must each positively recognize the exact
    target at >=.90 within a unique continuous context at >=.65. At least two
    full-context CTC crops must localize it within 80 ms of one another and
    within 300 ms of each ASR model. Raw same-token interval likelihoods must
    improve by >=30% in both normalizations and >=1 nat per target token.
    This repairs an old disputed clock; it never certifies human accuracy.
    """
    word = _boundary_target(package, row)
    before = row.get('details', {}).get('aligned_ms')
    if word is None or not isinstance(before, list) or len(before) != 2 or not normalize_ctc(word['text']):
        return None
    index = next(i for i, w in enumerate(package['words']) if w['id'] == word['id'])
    issue = {**row, 'recognized': word['text']}
    model_support = {}
    for observation in observations:
        if (observation.get('mode') != 'independent_unprompted_asr' or
                observation.get('audio_sha256') != audio_hash or
                not re.fullmatch('[0-9a-f]{64}', str(observation.get('model_fingerprint', '')))):
            continue
        core = _continuous_support(package['words'], [index], observation, issue)
        if not core:
            continue
        targets = [w for w in core if _values(w['text']) == _values(word['text'])]
        if len(targets) != 1 or targets[0].get('probability', 0) < .90:
            continue
        model_support.setdefault(observation['model_fingerprint'], {})[observation['observation_id']] = {
            'observation_id': observation['observation_id'], 'observation_sha256': digest(observation),
            'target': copy.deepcopy(targets[0]), 'matched_words': copy.deepcopy(core)}
    if len(model_support) < 2:
        return None
    grouped = {}
    for survey in surveys:
        if (survey.get('audio_sha256') != audio_hash or
                not re.fullmatch('[0-9a-f]{64}', str(survey.get('model_sha256', '')))):
            continue
        model = grouped.setdefault(survey['model_sha256'], {})
        for window in survey.get('windows', []):
            begin, end = window['start_ms'], window['end_ms']
            expected = [(w['id'], w['text']) for w in package['words']
                        if begin <= w['start_ms'] and w['end_ms'] <= end and
                        normalize_ctc(w.get('spoken_form') or w['text']) is not None]
            if expected != [(w.get('id'), w.get('text')) for w in window.get('aligned', [])]:
                continue
            targets = [w for w in window.get('aligned', []) if w.get('id') == word['id'] and w.get('text') == word['text']]
            if len(targets) != 1:
                continue
            target = targets[0]
            after = [target.get('start_ms'), target.get('end_ms')]
            if (any(not isinstance(n, (int, float)) or isinstance(n, bool) or not math.isfinite(n) for n in after) or
                    not begin + 800 <= after[0] < after[1] <= end - 800 or
                    not 19 <= after[1] - after[0] <= 1500):
                continue
            agreeing = {model_id: list(rows.values()) for model_id, rows in model_support.items()
                        if all(abs(statistics.median([r['target'][key] for r in rows.values()]) - after[k]) <= 300
                               for k, key in enumerate(('start_ms', 'end_ms')))}
            if len(agreeing) < 2:
                continue
            comparison = next((c for c in window.get('interval_comparisons', [])
                               if _interval_preference(c, window, word, before, after)), None)
            if comparison is not None:
                model[(begin, end)] = {'window_ms': [begin, end], 'measurement': copy.deepcopy(target),
                    'comparison': copy.deepcopy(comparison), 'asr_models': copy.deepcopy(agreeing),
                    'window_evidence_sha256': digest(window)}
    for model_id, windows in grouped.items():
        rows = list(windows.values())
        if len(rows) < 2 or any(max(r['measurement'][key] for r in rows) - min(r['measurement'][key] for r in rows) > 80
                                for key in ('start_ms', 'end_ms')):
            continue
        return {'kind': 'dual_asr_stable_ctc_interval_relocation', 'word_id': word['id'],
            'sentence_id': word['sentence_id'], 'text': word['text'], 'before_ms': before,
            'after_ms': [round(statistics.median(r['measurement'][key] for r in rows)) for key in ('start_ms', 'end_ms')],
            'audio_sha256': audio_hash, 'ctc_model_sha256': model_id, 'observations': rows,
            'human_ground_truth': False,
            'reason': '两个不同无提示识别模型正向词证据与双窗稳定CTC定位一致；同词原帧似然显著优于旧错位区间，实际修正时间而非仅清除提醒。'}
    return None


def propose_disputed_boundary_revisions(package, audio_path, *, model_paths, observations, boundary_surveys):
    """Only propose measured changes for currently disputed retained word IDs."""
    from .timing_revision import propose_timing_revision
    audio_hash = file_sha256(Path(audio_path))
    valid = {_model_fingerprint(str(path)) for path in model_paths}
    observations = [o for o in observations if o.get('model_fingerprint') in valid]
    disputed = unresolved_alignment(package)
    targets = {w['id']: row for row in disputed if (w := _boundary_target(package, row)) is not None}
    patches, evidence = {}, []
    for survey in boundary_surveys:
        if survey.get('input_timing_sha256') != digest(package['words']):
            continue
        proposal = propose_timing_revision(package['words'], survey, audio_hash)
        for patch in proposal['patches']:
            if patch['word_id'] in targets:
                patches[patch['word_id']] = copy.deepcopy(patch)
                evidence.append({'kind': 'current_disputed_word_remeasured', 'original': targets[patch['word_id']],
                    'patch': copy.deepcopy(patch), 'audio_sha256': audio_hash, 'human_ground_truth': False})
    for word_id, row in targets.items():
        if word_id in patches:
            continue
        proof = _cross_boundary_evidence(package, row, observations, boundary_surveys, audio_hash)
        if proof:
            word = next(w for w in package['words'] if w['id'] == word_id)
            if proof['after_ms'] == [word['start_ms'], word['end_ms']]:
                continue
            patches[word_id] = {'word_id': word_id, 'word_index': package['words'].index(word),
                'text': word['text'], 'before_ms': [word['start_ms'], word['end_ms']],
                'after_ms': proof['after_ms'], 'cross_evidence_sha256': digest(proof)}
            evidence.append({'kind': proof['kind'], 'original': copy.deepcopy(row), 'proof': proof})
    # Reject proposed shifts that create a new neighbor overlap as one transaction.
    while True:
        conflicts = set()
        for left, right in zip(package['words'], package['words'][1:]):
            a = patches.get(left['id'], {}).get('after_ms', [left['start_ms'], left['end_ms']])
            b = patches.get(right['id'], {}).get('after_ms', [right['start_ms'], right['end_ms']])
            if a[1] - b[0] > max(0, left['end_ms'] - right['start_ms']):
                conflicts.update(k for k in (left['id'], right['id']) if k in patches)
        if not conflicts:
            break
        for key in conflicts:
            patches.pop(key)
    return {'version': 1, 'algorithm': 'disputed-only-cross-evidence-localization-v1',
        'input_words_sha256': digest(package['words']), 'audio_sha256': audio_hash,
        'human_ground_truth': False, 'patches': sorted(patches.values(), key=lambda p: p['word_index']),
        'evidence': [e for e in evidence if (e.get('patch') or e.get('proof'))['word_id'] in patches]}


def _boundary_support(package, row, surveys, audio_hash):
    """Retire an old ASR interval only with current stable positive CTC frames."""
    detail = row.get('details', {})
    aligned = detail.get('aligned_ms')
    if not isinstance(aligned, list) or len(aligned) != 2:
        return None
    word = _boundary_target(package, row)
    if word is None:
        return None
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


def _substitution_retention(package, issue, observations, sources, audio_hash):
    """Keep measured speech over an editorial substitute with two ASR models.

    This cannot replace words, accept numbers/negations, or normalize names.
    It adds a second ASR model to the existing continuous-context+dual-CTC
    omission standard when the reference supplies a different noncritical word.
    """
    actual, reference = issue.get('recognized', ''), issue.get('reference', '')
    if (not actual or not reference or re.search(r'\d|[=<>+*/_]|\b(?:not|no|never|cannot|without|incorrect|correct|invalid|valid|impossible|possible)\b', actual + ' ' + reference, re.I)):
        return None
    indices = _indices(package['words'], issue)
    if not indices:
        return None
    alternative = (package['words'][:indices[0]] +
        [{**package['words'][indices[0]], 'text': reference,
          'start_ms': issue['start_ms'], 'end_ms': issue['end_ms']}] +
        package['words'][indices[-1] + 1:])
    support = {}
    for observation in observations:
        if (observation.get('mode') != 'independent_unprompted_asr' or observation.get('audio_sha256') != audio_hash or
                not re.fullmatch('[0-9a-f]{64}', str(observation.get('model_fingerprint', '')))):
            continue
        # An isolated nearby "it" is not evidence for replacing "so out".
        # A genuine alternative in its complete neighboring context still vetoes.
        if _continuous_support(alternative, [indices[0]], observation,
                               {**issue, 'recognized': reference}):
            return None
        core = _continuous_support(package['words'], indices, observation, issue)
        if core:
            support.setdefault(observation['model_fingerprint'], []).append({
                'observation_id': observation['observation_id'], 'observation_sha256': digest(observation),
                'matched_words': copy.deepcopy(core)})
    if len(support) < 2:
        return None
    windows = {}
    for source in sources:
        if (source.get('audio_sha256') != audio_hash or source.get('input_words_sha256') != reference_words_fingerprint(package['words'])):
            continue
        if not re.fullmatch('[0-9a-f]{64}', str(source.get('model_sha256', ''))):
            continue
        accepted = {}
        for row in source.get('rows', []):
            if row.get('issue_key') != reference_issue_key(issue):
                continue
            for window in row.get('windows', []):
                begin, end = window['clip_start_ms'], window['clip_end_ms']
                local = [i for i, w in enumerate(package['words']) if begin <= w['start_ms'] and w['end_ms'] <= end]
                current = ' '.join(package['words'][i]['text'] for i in local)
                edited = []
                for i in local:
                    if i == indices[0]:
                        edited.append(reference)
                    if i not in indices:
                        edited.append(package['words'][i]['text'])
                if _preference(window, issue, current, ' '.join(edited), prefer_actual=True):
                    accepted[(begin, end)] = copy.deepcopy(window)
        if len(accepted) >= 2:
            windows[source['model_sha256']] = list(accepted.values())
    if not windows:
        return None
    return {'kind': 'editorial_substitution_actual_speech_retained', 'status': 'supported_recognition',
        'issue_key': reference_issue_key(issue), 'finding': copy.deepcopy(issue),
        'audio_sha256': audio_hash, 'input_words_sha256': reference_words_fingerprint(package['words']),
        'asr_models': support, 'ctc_models': windows, 'human_ground_truth': False,
        'reason': '两种不同无提示识别模型的唯一连续上下文均支持当前口语，双窗完整CTC两种归一化也支持它；保留原参考而不拿编辑稿替换原声。'}


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
    sources = merge_ctc_sources(ctc_observations, original.get('words', []), audio_hash)
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
        actual_values, reference_values = _values(issue.get('recognized', '')), _values(issue.get('reference', ''))
        if (reference_values == ['yea'] and actual_values.count('yay') == 1 and
                all(value in {'yay', 'okay'} for value in actual_values)):
            automatic[record['id']] = {'kind': 'reference_interjection_spelling_equivalent',
                'recognized': issue['recognized'], 'reference': issue['reference'],
                'audio_sha256': audio_hash, 'input_words_sha256': reference_words_fingerprint(original['words']),
                'human_ground_truth': False,
                'reason': '原参考显式写yea，原音稿写同音感叹词yay；只统一比较拼写，保留所有原字幕、Okay和时间，未推广到yeah或yes。'}
            continue
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
            chosen = _substitution_retention(draft, issue, observations, rebound, audio_hash)
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
                if proof is None:
                    relocation = _cross_boundary_evidence(draft, detail, observations, boundary_surveys, audio_hash)
                    if relocation is not None:
                        current_word = next(w for w in draft['words'] if w['id'] == relocation['word_id'])
                        if relocation['after_ms'] == [current_word['start_ms'], current_word['end_ms']]:
                            proof = relocation
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
