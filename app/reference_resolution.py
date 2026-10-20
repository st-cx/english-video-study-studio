"""Bind reference findings to current, independent local audio observations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from .reference_check import reference_issues, _parts, _canonical
from .sync_audit import normalize_ctc
from .acoustic_reference import (
    _audio_digest, _model_fingerprint, assess_observations, reference_issue_key, reference_words_fingerprint,
    observe_disputes, reference_resolution_is_current,
)


def _current_observations(value, audio_sha256, model_fingerprint):
    if isinstance(value, dict):
        if value.get('audio_sha256', audio_sha256) != audio_sha256 or value.get('model_fingerprint', model_fingerprint) != model_fingerprint:
            return []
        value = value.get('observations', [])
    if not isinstance(value, list):
        return []
    result, seen = [], set()
    for observation in value:
        if not isinstance(observation, dict) or observation.get('mode') != 'independent_unprompted_asr' or observation.get('audio_sha256') != audio_sha256 or observation.get('model_fingerprint') != model_fingerprint:
            continue
        try:
            signature = hashlib.sha256(json.dumps(observation, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        except (TypeError, ValueError):
            continue
        if signature not in seen:
            result.append(copy.deepcopy(observation)); seen.add(signature)
    return result



def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()


def _units(words):
    parts = []
    for index, word in enumerate(words):
        parts.extend({**item, 'source_id': index} for item in _parts(str(word['text']),
            int(word['start_ms']), int(word['end_ms'])))
    return _canonical(parts)


def _continuous_support(words, indices, observation, issue):
    """Require actual contiguous context; CTC preference alone is not wording."""
    if observation.get('truncated'):
        return None
    observed = observation.get('words', [])
    expected = _units(words[max(0, indices[0] - 2):min(len(words), indices[-1] + 3)])
    actual = _units(observed)
    target = [unit['value'] for unit in expected]
    if not target or len(target) < len(_canonical(_parts(issue['recognized']))) + 2:
        return None
    matches = []
    for position in range(len(actual) - len(target) + 1):
        group = actual[position:position + len(target)]
        if [unit['value'] for unit in group] != target:
            continue
        core = observed[group[0]['source_id']:group[-1]['source_id'] + 1]
        if not core or any(not 0 <= int(word['start_ms']) < int(word['end_ms']) or
                not math.isfinite(float(word.get('probability', 0))) or
                not .65 <= float(word.get('probability', 0)) <= 1 for word in core):
            continue
        if core[0]['start_ms'] > issue['start_ms'] or core[-1]['end_ms'] < issue['end_ms']:
            continue
        # A repeated phrase elsewhere in the same listening window is ambiguous.
        matches.append(core)
    return matches[0] if len(matches) == 1 else None


def _dual_evidence(issue, words, observations, ctc, audio_sha256, model_fingerprint):
    """Narrow secondary method for editorial omissions, not text replacement.

    It combines two different crop windows, two length normalizations of an
    independent CTC model, and unique continuous unprompted ASR context. Raw
    scores and full candidate texts are checked rather than trusting a saved
    Boolean. Numbers, substitutions and missing ASR speech retain the stricter
    original route. This is model evidence, never human listening acceptance.
    """
    if issue.get('reference') or not issue.get('recognized') or not isinstance(ctc, dict):
        return None
    if ctc.get('audio_sha256') != audio_sha256 or ctc.get('input_words_sha256') != reference_words_fingerprint(words) or ctc.get('input_words_fingerprint_version') != 'text-time-1':
        return None
    if ctc.get('source') != 'independent_full_context_ctc_likelihood' or not re.fullmatch(r'[0-9a-f]{64}', str(ctc.get('model_sha256', ''))):
        return None
    if re.search(r'\d|[=<>+*/_]', issue['recognized']):
        return None
    key = reference_issue_key(issue)
    for row in ctc.get('rows', []):
        if row.get('issue_key') != key or any(row.get(field) != issue.get(field) for field in ('start_ms', 'end_ms', 'recognized', 'reference')):
            continue
        try:
            indices = row['global_word_indices']
            if not indices or indices != list(range(indices[0], indices[-1] + 1)) or indices[0] < 0 or indices[-1] >= len(words):
                continue
            selected = [words[index] for index in indices]
            if [x['value'] for x in _units(selected)] != [x['value'] for x in _canonical(_parts(issue['recognized']))]:
                continue
            if min(word['start_ms'] for word in selected) != issue['start_ms'] or max(word['end_ms'] for word in selected) != issue['end_ms']:
                continue
            windows, scores = set(), []
            for window in row.get('windows', []):
                begin, end = int(window['clip_start_ms']), int(window['clip_end_ms'])
                comparison = window.get('comparison', {})
                if window.get('status') != 'measured' or not begin <= issue['start_ms'] < issue['end_ms'] <= end or end - begin > 30000:
                    continue
                if comparison.get('algorithm') != 'normalized-logsoftmax-ctc-full-context-nll-v1' or comparison.get('edge_wildcards') is not False or not comparison.get('comparable'):
                    continue
                if not re.fullmatch(r'[0-9a-f]{64}', str(window.get('clip_sha256', ''))):
                    continue
                candidates = {value['id']: value for value in comparison['observations']}
                actual, reference = candidates['actual_recognized_context'], candidates['reference_edited_context']
                local_indices = [index for index, word in enumerate(words) if word['start_ms'] >= begin and word['end_ms'] <= end]
                actual_text = ' '.join(words[index]['text'] for index in local_indices)
                reference_text = ' '.join(words[index]['text'] for index in local_indices if index not in indices)
                if actual['text'] != actual_text or reference['text'] != reference_text or not reference_text:
                    continue
                valid = True
                for candidate in (actual, reference):
                    nll, frames, tokens = float(candidate['negative_log_likelihood']), int(candidate['frame_count']), int(candidate['target_token_count'])
                    normalized = normalize_ctc(candidate['text'])
                    if candidate.get('status') != 'measured' or frames <= 0 or not normalized or tokens != len(normalized) or frames != comparison.get('frame_count') or not math.isfinite(nll) or nll < 0:
                        valid = False; break
                    if any(not math.isfinite(float(candidate[metric])) for metric in ('nll_per_frame', 'nll_per_target_token')) or not math.isclose(candidate['nll_per_frame'], nll / frames, rel_tol=1e-5, abs_tol=1e-6) or not math.isclose(candidate['nll_per_target_token'], nll / tokens, rel_tol=1e-5, abs_tol=1e-6):
                        valid = False; break
                if not valid or any(float(actual[metric]) >= float(reference[metric]) for metric in ('nll_per_frame', 'nll_per_target_token')):
                    continue
                windows.add((begin, end)); scores.append(copy.deepcopy(window))
            if len(windows) < 2:
                continue
            supported = []
            for observation in observations:
                if observation.get('start_ms', math.inf) > issue['start_ms'] or observation.get('end_ms', -1) < issue['end_ms']:
                    continue
                core = _continuous_support(words, indices, observation, issue)
                if core:
                    supported.append((min(float(word['probability']) for word in core), observation, core))
            if not supported:
                continue
            confidence, observation, core = max(supported, key=lambda item: item[0])
            decision = {'version': 'reference-multievidence-1', 'issue_key': key,
                'status': 'supported_recognition', 'kind': 'reference_omission',
                'mode': 'independent_asr_and_dual_window_ctc', 'audio_sha256': audio_sha256,
                'model_fingerprint': model_fingerprint, 'ctc_model_sha256': ctc['model_sha256'],
                'input_words_sha256': ctc['input_words_sha256'], 'observation_id': observation['observation_id'],
                'matched_words': copy.deepcopy(core), 'candidate_words': [], 'confidence_min': confidence,
                'ctc_windows': scores, 'ctc_evidence_sha256': _digest(row),
                'reason': '独立无提示连续原声识别与双窗 CTC 两种归一化均支持实际口语；官方参考省略该片段。'}
            decision['evidence_signature'] = _digest(decision)
            return decision
        except (KeyError, TypeError, ValueError, OverflowError, IndexError):
            continue
    return None



def _gap_anchor_support(anchors, observation):
    if observation.get('truncated'):
        return None
    words = observation.get('words', [])
    units = _units(words)
    target = [item['value'] for item in _units(anchors)]
    matches = []
    for position in range(len(units) - len(target) + 1):
        group = units[position:position + len(target)]
        if [item['value'] for item in group] != target:
            continue
        core = words[group[0]['source_id']:group[-1]['source_id'] + 1]
        if not core or any(not _parts(word['text']) or not 0 <= word['start_ms'] < word['end_ms'] or
                not math.isfinite(float(word.get('probability', 0))) or
                not .85 <= float(word.get('probability', 0)) <= 1 for word in core):
            continue
        if any(right['start_ms'] < left['end_ms'] - 80 or right['start_ms'] - left['end_ms'] > 750 for left, right in zip(core, core[1:])):
            continue
        if abs(core[0]['start_ms'] - anchors[0]['start_ms']) > 600 or abs(core[-1]['end_ms'] - anchors[-1]['end_ms']) > 600:
            continue
        # No ignored filler or other source word may occupy the purported gap.
        if [item['value'] for item in _units(core)] != target:
            continue
        matches.append(core)
    return matches[0] if len(matches) == 1 else None


def _gap_ctc_window(window, issue, actual_text, alternative_text):
    comparison = window.get('comparison', {})
    begin, end = int(window['clip_start_ms']), int(window['clip_end_ms'])
    if window.get('status') != 'measured' or not begin <= issue['start_ms'] < issue['end_ms'] <= end or end - begin > 30000:
        return False
    if comparison.get('algorithm') != 'normalized-logsoftmax-ctc-full-context-nll-v1' or comparison.get('source') != 'independent_ctc_candidate_likelihood' or comparison.get('edge_wildcards') is not False or not comparison.get('comparable'):
        return False
    if not re.fullmatch(r'[0-9a-f]{64}', str(window.get('clip_sha256', ''))):
        return False
    candidates = {value['id']: value for value in comparison['observations']}
    actual, reference = candidates['actual_recognized_context'], candidates['reference_edited_context']
    if actual['text'] != actual_text or reference['text'] != alternative_text:
        return False
    for candidate in (actual, reference):
        nll, frames, tokens = float(candidate['negative_log_likelihood']), int(candidate['frame_count']), int(candidate['target_token_count'])
        normalized = normalize_ctc(candidate['text'])
        if candidate.get('status') != 'measured' or not normalized or tokens != len(normalized) or frames <= 0 or frames != comparison.get('frame_count') or not math.isfinite(nll) or nll < 0:
            return False
        if any(not math.isfinite(float(candidate[metric])) for metric in ('nll_per_frame', 'nll_per_target_token')) or not math.isclose(float(candidate['nll_per_frame']), nll / frames, rel_tol=1e-5, abs_tol=1e-6) or not math.isclose(float(candidate['nll_per_target_token']), nll / tokens, rel_tol=1e-5, abs_tol=1e-6):
            return False
    return all(float(actual[metric]) < float(reference[metric]) for metric in ('nll_per_frame', 'nll_per_target_token'))


def _reference_gap_evidence(issue, words, observations, ctc, audio_sha256, model_fingerprint):
    """Resolve source-only text only with two positive contiguous anchor runs.

    ASR silence, omitted tokens, saved approvals and CTC ranking booleans are
    insufficient. Both actual audio windows must positively recognize unique
    neighboring anchors with no intervening extra token, and independent raw
    CTC candidate likelihoods must support that exact continuous current text.
    """
    if issue.get('recognized') or not issue.get('reference') or not isinstance(ctc, dict):
        return None
    if ctc.get('audio_sha256') != audio_sha256 or ctc.get('input_words_fingerprint_version') != 'text-time-1' or ctc.get('input_words_sha256') != reference_words_fingerprint(words) or not re.fullmatch(r'[0-9a-f]{64}', str(ctc.get('model_sha256', ''))):
        return None
    key = reference_issue_key(issue)
    for row in ctc.get('rows', []):
        if row.get('issue_key') != key or any(row.get(field) != issue.get(field) for field in ('recognized', 'reference', 'start_ms', 'end_ms')):
            continue
        try:
            indices = row['global_anchor_indices']
            if len(indices) != 2 or indices[1] != indices[0] + 1 or row['global_insertion_boundary'] != indices[1] or not 0 <= indices[0] < indices[1] < len(words):
                continue
            anchors = [words[index] for index in indices]
            if [dict(text=w['text'], start_ms=w['start_ms'], end_ms=w['end_ms']) for w in row['anchors']] != [dict(text=w['text'], start_ms=w['start_ms'], end_ms=w['end_ms']) for w in anchors]:
                continue
            if anchors[0]['start_ms'] != issue['start_ms'] or anchors[-1]['end_ms'] != issue['end_ms']:
                continue
            positive = []
            for observation in observations:
                if observation.get('start_ms', math.inf) > issue['start_ms'] or observation.get('end_ms', -1) < issue['end_ms']:
                    continue
                core = _gap_anchor_support(anchors, observation)
                if core:
                    positive.append({'observation_id': observation['observation_id'], 'start_ms': observation['start_ms'], 'end_ms': observation['end_ms'], 'words': copy.deepcopy(core), 'observation_words_sha256': _digest(observation['words'])})
            if len({value['observation_id'] for value in positive}) < 2 or len({(value['start_ms'], value['end_ms']) for value in positive}) < 2:
                continue
            windows = []
            for window in row.get('windows', []):
                begin, end = int(window['clip_start_ms']), int(window['clip_end_ms'])
                local = [index for index, word in enumerate(words) if word['start_ms'] >= begin and word['end_ms'] <= end]
                if not all(index in local for index in indices):
                    continue
                actual_text = ' '.join(words[index]['text'] for index in local)
                inserted = []
                for index in local:
                    if index == indices[1]:
                        inserted.append(issue['reference'])
                    inserted.append(words[index]['text'])
                if _gap_ctc_window(window, issue, actual_text, ' '.join(inserted)):
                    windows.append(copy.deepcopy(window))
            if len({(value['clip_start_ms'], value['clip_end_ms']) for value in windows}) < 2:
                continue
            decision = {'version': 'reference-multievidence-1', 'issue_key': key,
                'status': 'supported_recognition', 'kind': 'reference_insertion_without_acoustic_support',
                'mode': 'independent_dual_asr_anchors_and_dual_window_ctc', 'audio_sha256': audio_sha256,
                'model_fingerprint': model_fingerprint, 'ctc_model_sha256': ctc['model_sha256'],
                'input_words_sha256': ctc['input_words_sha256'], 'matched_words': positive[0]['words'],
                'candidate_words': [], 'confidence_min': min(word['probability'] for value in positive for word in value['words']),
                'asr_anchor_observations': positive, 'ctc_windows': windows, 'ctc_evidence_sha256': _digest(row),
                'reason': '两份独立无提示原音在唯一前后锚点连续相接，双窗 CTC 两种归一化均支持实际连续口语；保留官方原参考差异。'}
            decision['evidence_signature'] = _digest(decision)
            return decision
        except (KeyError, TypeError, ValueError, OverflowError, IndexError):
            continue
    return None


def _current_decision(issue, decision, audio_sha256, model_fingerprint):
    if reference_resolution_is_current(issue, decision, audio_sha256=audio_sha256, model_fingerprint=model_fingerprint):
        return True
    if decision.get('version') != 'reference-multievidence-1' or decision.get('status') != 'supported_recognition' or decision.get('mode') not in {'independent_asr_and_dual_window_ctc', 'independent_dual_asr_anchors_and_dual_window_ctc'}:
        return False
    value = {key: item for key, item in decision.items() if key not in ('evidence_signature', 'finding')}
    return (decision.get('issue_key') == reference_issue_key(issue) and
            decision.get('audio_sha256') == audio_sha256 and decision.get('model_fingerprint') == model_fingerprint and
            decision.get('evidence_signature') == _digest(value))


def resolve_reference_findings(package: dict, reference: str, audio_path: Path, *,
                               model_path: str, observations=None, progress=None,
                               cache_dir: Path | None = None, observe=False,
                               device='cpu', compute_type='int8', ctc_observations=None) -> dict:
    """Return a conservative gate result without changing caller-owned state.

    Existing judgments or approved flags never substitute for current acoustic
    observations. A reference-supported candidate remains unresolved until the
    transcript is really corrected and aligned. Model failures retain pending
    findings and expose only a generic diagnostic, never local paths or input.
    """
    findings = reference_issues(package.get('words', []), reference or '')
    result = {'findings': findings, 'unresolved': list(findings), 'resolutions': [],
              'audio_sha256': '', 'model_fingerprint': '', 'observations': [],
              'reference_available': bool((reference or '').strip())}
    try:
        audio_sha256 = _audio_digest(Path(audio_path))
        model_fingerprint = _model_fingerprint(str(model_path))
    except (OSError, ValueError, TypeError):
        result['observation_error'] = '无法核对当前原声和模型指纹，已保留全部参考稿争议。'
        return result
    result.update(audio_sha256=audio_sha256, model_fingerprint=model_fingerprint)
    if observations is None:
        observations = package.get('processing', {}).get('reference_observations', [])
    current = _current_observations(observations, audio_sha256, model_fingerprint)
    decisions = assess_observations(findings, current, audio_sha256=audio_sha256, model_fingerprint=model_fingerprint)
    pending = [finding for finding, decision in zip(findings, decisions) if decision['status'] == 'unresolved']
    if observe and pending:
        progress_failure = []
        def report(*args):
            if progress is not None:
                try:
                    progress(*args)
                except BaseException as exc:
                    progress_failure.append(exc)
                    raise
        try:
            new = observe_disputes(Path(audio_path), pending, str(model_path), report,
                                   cache_dir=cache_dir, device=device, compute_type=compute_type)
            current = _current_observations(current + new, audio_sha256, model_fingerprint)
        except Exception:
            if progress_failure:
                raise
            result['observation_error'] = '独立原声复核未完成，已保留待检查项；可稍后重试。'
        decisions = assess_observations(findings, current, audio_sha256=audio_sha256, model_fingerprint=model_fingerprint)
    if ctc_observations is None:
        ctc_observations = package.get('processing', {}).get('reference_ctc_observations')
    for index, (finding, decision) in enumerate(zip(findings, decisions)):
        if decision['status'] == 'unresolved':
            sources = ctc_observations if isinstance(ctc_observations, list) else [ctc_observations]
            for ctc in sources:
                multi = (_dual_evidence(finding, package.get('words', []), current, ctc, audio_sha256, model_fingerprint)
                         or _reference_gap_evidence(finding, package.get('words', []), current, ctc, audio_sha256, model_fingerprint))
                if multi is not None:
                    decisions[index] = multi
                    break
    result['observations'] = current
    result['resolutions'] = [{**decision, 'finding': copy.deepcopy(finding)}
                             for finding, decision in zip(findings, decisions)]
    result['unresolved'] = [finding for finding, decision in zip(findings, decisions)
                            if not _current_decision(finding, decision, audio_sha256, model_fingerprint)]
    return result


def unresolved_alignment(package: dict) -> list[dict]:
    """Expose all current chunk disputes on the global audio clock.

    A free-standing resolved/approved Boolean is insufficient evidence to erase
    a disagreement. Actual re-alignment must replace the originating metadata.
    """
    result = []
    for index, chunk in enumerate(package.get('processing', {}).get('speech_chunks', [])):
        alignment = chunk.get('alignment', {})
        for dispute in alignment.get('alignment_disputes', []):
            intervals = []
            for field in ('aligned_ms', 'asr_ms'):
                value = dispute.get(field)
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    try:
                        intervals.append((int(value[0]), int(value[1])))
                    except (ValueError, TypeError, OverflowError):
                        pass
            global_clock = chunk.get('time_basis') in {'media', 'global'} or alignment.get('time_basis') in {'media', 'global'}
            offset = 0 if global_clock else int(chunk.get('clip_start_ms', chunk.get('start_ms', 0)))
            start = offset + min(interval[0] for interval in intervals) if intervals else int(chunk.get('start_ms', 0))
            end = offset + max(interval[1] for interval in intervals) if intervals else int(chunk.get('end_ms', start))
            result.append({'chunk_index': index, 'start_ms': start, 'end_ms': end,
                           'reason': '独立识别与对齐仍有差异，请回听对应位置。',
                           'text': dispute.get('text', ''), 'details': copy.deepcopy(dispute)})
    return result
