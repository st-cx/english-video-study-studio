"""Reproducible timing audits. Acoustic agreement never becomes human ground truth."""
from __future__ import annotations

import array
import bisect
import collections
import csv
import datetime as dt
import hashlib
import html
import importlib.metadata
import json
import math
import re
import subprocess
import wave
from pathlib import Path

VERSION = 1
MODEL_NAME = 'WAV2VEC2_ASR_BASE_960H'
MODEL_FILENAME = 'wav2vec2_fairseq_base_ls960_asr_ls960.pth'
MODEL_URL = 'https://download.pytorch.org/torchaudio/models/' + MODEL_FILENAME
MODEL_BYTES = 377664473
CTC_ALGORITHM = 'wav2vec2-base-960h-ctc-edge-wildcard-v2'
HUMAN_SOURCES = {'human_listening', 'human_listening_with_waveform'}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def timing_fingerprint(package: dict) -> str:
    return digest({'episode_id': package['episode']['id'], 'duration_s': package['episode']['duration_s'],
                   'words': [{k: w.get(k) for k in ('id', 'text', 'sentence_id', 'start_ms', 'end_ms', 'connected_words', 'spoken_words', 'spoken_form', 'phrase_unit', 'timing_unit')} for w in package['words']],
                   'chapters': [{k: c.get(k) for k in ('id', 'start_ms', 'end_ms')} for c in package.get('chapters', [])]})


def is_single_word(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9'’\.\-]*[.,!?;:]*", text.strip()))


def is_independent_timed_word(word: dict) -> bool:
    """A display token is not necessarily one acoustically timed spoken word."""
    if not is_single_word(word['text']) or word.get('phrase_unit') or word.get('timing_unit') == 'phrase':
        return False
    for key in ('connected_words', 'spoken_words'):
        value = word.get(key)
        if isinstance(value, (list, tuple)):
            if len(value) > 1:
                return False
            value = value[0] if value else ''
        if isinstance(value, dict):
            value = value.get('text', '')
        if isinstance(value, str) and len(value.split()) > 1:
            return False
    spoken = word.get('spoken_form')
    return not (isinstance(spoken, str) and len(spoken.split()) > 1)


def environment_inventory() -> dict:
    versions = {}
    for name in ('torch', 'torchaudio', 'transformers', 'faster-whisper', 'stable-ts', 'whisperx', 'ctc-forced-aligner', 'phonemizer', 'onnxruntime'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {'versions': versions, 'independent_ctc_model': MODEL_NAME, 'official_model_url': MODEL_URL,
              'reported_model_size_bytes': MODEL_BYTES, 'human_listening_performed_by_tool': False}
    try:
        import torch
        import torchaudio
        result.update(torch_runtime=torch.__version__, torchaudio_runtime=torchaudio.__version__, cuda_available=torch.cuda.is_available(),
                      forced_align_available=hasattr(torchaudio.functional, 'forced_align'))
    except Exception as error:
        result['runtime_error'] = type(error).__name__ + ': ' + str(error)[:800]
    return result


def build_manifest(package: dict, *, minimum: int = 48) -> dict:
    if minimum < 40:
        raise ValueError('同步验收至少需要 40 个不同的单词。')
    duration = round(float(package['episode']['duration_s']) * 1000)
    words = sorted(package['words'], key=lambda w: (w['start_ms'], w['end_ms'], str(w['id'])))
    eligible = [w for w in words if is_independent_timed_word(w)]
    if len(eligible) < 40:
        raise ValueError('素材不足 40 个可独立标注的单词，不能冒充完整同步验收。')
    sentence_by_id = {s['id']: s for s in package.get('sentences', [])}
    selected: dict[str, dict] = {}
    warnings = []
    chapters = sorted(package.get('chapters') or [{'id': 'all', 'start_ms': 0, 'end_ms': duration}], key=lambda c: c['start_ms'])

    def take(candidates, stratum: str, reason: str, **extra):
        candidate = next((w for w in candidates if w['id'] not in selected), None)
        if not candidate:
            return False
        sentence = sentence_by_id.get(candidate.get('sentence_id'), {})
        chapter = next((c for c in chapters if c['start_ms'] <= candidate['start_ms'] < c['end_ms']), None)
        selected[candidate['id']] = {'id': 'sample-' + str(candidate['id']), 'word_id': candidate['id'], 'text': candidate['text'],
                                    'sentence_id': candidate.get('sentence_id'), 'sentence': sentence.get('text', ''), 'speaker_label': sentence.get('speaker', ''),
                                    'chapter_id': chapter.get('id') if chapter else None, 'stratum': stratum, 'selection_reason': reason,
                                    'lexical_unit': 'word' if is_independent_timed_word(candidate) else 'connected_timed_unit',
                                    'system_start_ms': candidate['start_ms'], 'system_end_ms': candidate['end_ms'],
                                    'temporal_bin': min(5, max(0, int(candidate['start_ms'] / max(1, duration) * 6))),
                                    'human_annotation': None, **extra}
        return True

    take(eligible, 'opening_endpoint', '第一处实际计时单词，覆盖片头')
    take(reversed(eligible), 'ending_endpoint', '最后一处实际计时单词，覆盖片尾与累计漂移')
    # Eighteen fixed quantiles: six time bins, three positions in every bin.
    # No confidence filtering: uncertain and structurally bad words remain eligible.
    for bin_index in range(6):
        for fraction in (.25, .5, .75):
            target = duration * (bin_index + fraction) / 6
            take(sorted(eligible, key=lambda w: (abs(w['start_ms'] - target), w['id'])), 'time_coverage',
                 '六等分时间层内的固定四分位位置', target_ms=round(target))

    # Both sides of every chapter boundary; word IDs are unique across strata.
    for chapter in chapters[1:]:
        boundary = chapter['start_ms']
        take(sorted((w for w in eligible if w['end_ms'] <= boundary), key=lambda w: (boundary - w['end_ms'], w['id'])),
             'chapter_before', '章节边界前最近的尚未抽取单词', boundary_ms=boundary)
        take(sorted((w for w in eligible if w['start_ms'] >= boundary), key=lambda w: (w['start_ms'] - boundary, w['id'])),
             'chapter_after', '章节边界后最近的尚未抽取单词', boundary_ms=boundary)

    number_words = set('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion point'.split())
    numeric = [w for w in eligible if re.search(r'\d', w['text']) or w['text'].lower().strip('.,!?;:') in number_words]
    proper = [w for w in eligible if re.fullmatch(r'[A-Z]{2,8}[.,!?;:]?', w['text']) and w['text'].strip('.,!?;:') not in {'OK', 'THE', 'AND', 'SO', 'IT'}
              or w['text'].lower().strip('.,!?;:') in {'mit', 'ana', 'anna', 'bell', 'eric', 'grimson', 'guttag', 'python', 'anaconda', 'spyder', 'google', 'edx'}]
    for pool, stratum, reason in [(numeric, 'number', '数字或数字词候选，需原声核实实际说法'), (proper, 'proper_name', '缩写或课程专名候选，需原声核实')]:
        count = 0
        for fraction in (.1, .3, .5, .7, .9):
            count += take(sorted(pool, key=lambda w: (abs(w['start_ms'] - duration * fraction), w['id'])), stratum, reason)
        if count < 5:
            warnings.append(f'{stratum} 只有 {count} 个不同候选；不足部分用时间覆盖补齐，不伪称该层已满足。')

    sentences = sorted(package.get('sentences', []), key=lambda s: s['start_ms'])
    by_sentence = collections.defaultdict(list)
    for word in eligible:
        by_sentence[word.get('sentence_id')].append(word)
    turn_candidates = []
    for previous, sentence in zip(sentences, sentences[1:]):
        options = by_sentence[sentence['id']]
        if not options:
            continue
        labelled = bool(previous.get('speaker') and sentence.get('speaker') and previous['speaker'] != sentence['speaker'])
        gap = sentence['start_ms'] - previous['end_ms']
        question = '?' in previous.get('text', '')
        answer_cue = bool(re.match(r'^(yes|no|yeah|right|okay|ok|well|so)\b', sentence.get('text', ''), re.I))
        score = 100000 if labelled else min(max(gap, 0), 15000) + 3000 * question + 1000 * (question and answer_cue)
        turn_candidates.append((score, options[0], labelled, gap, question))
    turn_candidates.sort(key=lambda item: (-item[0], item[1]['start_ms']))
    # Spread four candidates across the lecture rather than taking four nearby pauses.
    turn_count = 0
    for fraction in (.05, .35, .65, .95):
        ranked = sorted(turn_candidates, key=lambda item: (-item[0] / 10000 + abs(item[1]['start_ms'] - duration * fraction) / max(1, duration / 4), item[1]['start_ms']))
        for score, word, labelled, gap, question in ranked:
            if word['id'] not in selected:
                take([word], 'speaker_change_candidate', '说话人标签变化候选' if labelled else '停顿或问答线索候选；尚未听辨，不能确认换人',
                     speaker_change_confirmed=None, turn_evidence={'label_changed': labelled, 'gap_ms': gap, 'question_before': question, 'acoustic_speaker_change_measured': False})
                turn_count += 1
                break
    if not any(s.get('speaker') for s in sentences):
        warnings.append('原数据没有说话人标签：四处换人层仅为停顿/问答候选，必须听辨确认；不能据此声称已覆盖真实换人。')
    if turn_count < 4:
        warnings.append(f'换人候选仅有 {turn_count} 处。')

    target_count = min(len(eligible), max(minimum, 48))
    for index in range(target_count * 3):
        if len(selected) >= target_count:
            break
        target = duration * ((index * 0.61803398875) % 1)
        take(sorted(eligible, key=lambda w: (abs(w['start_ms'] - target), w['id'])), 'coverage_fill', '固定黄金比例序列补足不同单词', target_ms=round(target))

    # Add existing zero-length/overlap problems as challenge cases, not silent exclusions.
    previous = None
    anomaly_count = 0
    for word in words:
        bad = word['end_ms'] <= word['start_ms'] or word['start_ms'] < 0 or word['end_ms'] > duration
        overlap = previous is not None and previous['end_ms'] - word['start_ms'] > 80
        if (bad or overlap) and is_independent_timed_word(word) and anomaly_count < 8:
            if word['id'] in selected:
                selected[word['id']]['structural_challenge'] = True
            else:
                take([word], 'structural_challenge', '现有零时长、越界或重叠问题，不得排除后宣称通过', structural_challenge=True)
            anomaly_count += 1
        previous = word
    samples = sorted(selected.values(), key=lambda sample: (sample['system_start_ms'], sample['word_id']))
    return {'version': VERSION, 'episode_id': package['episode']['id'], 'title': package['episode'].get('title', ''), 'duration_ms': duration,
            'timing_sha256': timing_fingerprint(package), 'sample_rule': 'six-bin-quantiles-v1 + both-sides-of-chapter-boundaries + number/name/turn-candidates + structural-challenges',
            'minimum_human_words': 40, 'threshold_ms': 200, 'required_within_fraction': .95,
            'sampling_counts': dict(collections.Counter(sample['stratum'] for sample in samples)), 'warnings': warnings,
            'independent_word_exclusion_rule': 'exclude-phrase-connected-and-multiple-spoken-words-v2',
            'excluded_units': [{'word_id': w['id'], 'text': w['text'], 'start_ms': w['start_ms'], 'end_ms': w['end_ms'],
                                'reason': '不是单个独立计时口语词；不纳入逐词95%验收分母'} for w in words if not is_independent_timed_word(w)],
            'human_listening_performed_by_tool': False, 'samples': samples}


def verify_manifest(package: dict, manifest: dict):
    if timing_fingerprint(package) != manifest.get('timing_sha256'):
        raise ValueError('当前词语或时间轴已改变；旧同步审查包失效，请重新生成，禁止混用旧标注。')


def waveform_envelope(wav_path: Path, points: int = 500) -> tuple[list[list[float]], int]:
    with wave.open(str(wav_path), 'rb') as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError('审查原音必须为单声道 16-bit PCM。')
        rate = stream.getframerate()
        samples = array.array('h', stream.readframes(stream.getnframes()))
    if not samples:
        raise ValueError('审查原音为空。')
    result = []
    for index in range(min(points, len(samples))):
        start = index * len(samples) // min(points, len(samples))
        end = (index + 1) * len(samples) // min(points, len(samples))
        chunk = samples[start:end]
        result.append([round(min(chunk) / 32768, 4), round(max(chunk) / 32768, 4)])
    return result, round(len(samples) / rate * 1000)


def prepare_evidence(package: dict, audio: Path, output: Path, *, minimum: int = 48, progress=print) -> dict:
    manifest = build_manifest(package, minimum=minimum)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'clips').mkdir(exist_ok=True)
    manifest['audio_sha256'] = file_sha256(audio)
    manifest['generated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    for index, sample in enumerate(manifest['samples']):
        start = max(0, sample['system_start_ms'] - 2500)
        end = min(manifest['duration_ms'], max(sample['system_end_ms'], sample['system_start_ms'] + 300) + 3000)
        stem = digest([manifest['audio_sha256'], start, end])[:18]
        target = output / 'clips' / (stem + '.wav')
        if not target.is_file():
            temporary = target.with_suffix('.part.wav')
            try:
                subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{start/1000:.3f}', '-i', str(audio), '-t', f'{(end-start)/1000:.3f}', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', '-y', str(temporary)], check=True, capture_output=True, timeout=45)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        envelope, measured_duration = waveform_envelope(target)
        if abs(measured_duration - (end-start)) > 50:
            raise ValueError(f'原音片段时长与请求范围不一致：{sample["id"]}')
        sample.update(clip_file='clips/' + target.name, clip_start_ms=start, clip_end_ms=end, clip_duration_ms=measured_duration,
                      clip_sha256=file_sha256(target), waveform=envelope, evidence_source='original_audio_ffmpeg_decoded_pcm', annotation_source=None)
        progress(f'原音与波形 {index+1}/{len(manifest["samples"])}：{sample["text"]}', flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'episode-snapshot.json').write_text(json.dumps(package, ensure_ascii=False), encoding='utf-8')
    write_review_page(manifest, output)
    (output / 'report.json').write_text(json.dumps(evaluate_annotations(manifest), ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def evaluate_annotations(manifest: dict, annotations: dict | None = None, ctc_results: dict | None = None) -> dict:
    if annotations and annotations.get('timing_sha256') != manifest['timing_sha256']:
        raise ValueError('人工标注来自另一版时间轴，不能计算验收结果。')
    if ctc_results and ctc_results.get('timing_sha256') != manifest['timing_sha256']:
        raise ValueError('CTC 结果来自另一版时间轴。')
    if annotations and manifest.get('audio_sha256') and annotations.get('audio_sha256') != manifest['audio_sha256']:
        raise ValueError('人工标注对应的原声音频不同，不能复用。')
    if ctc_results and manifest.get('audio_sha256') and ctc_results.get('audio_sha256') != manifest['audio_sha256']:
        raise ValueError('CTC 结果对应的原声音频不同。')
    supplied = {row.get('sample_id'): row for row in (annotations or {}).get('annotations', [])}
    model = {row.get('sample_id'): row for row in (ctc_results or {}).get('observations', [])}
    rows, human_eligible, passed = [], [], 0
    for sample in manifest['samples']:
        annotation = supplied.get(sample['id'])
        row = {key: sample.get(key) for key in ('id', 'word_id', 'text', 'stratum', 'temporal_bin', 'clip_file', 'clip_start_ms', 'clip_end_ms', 'clip_sha256', 'system_start_ms', 'system_end_ms')}
        row.update(annotation_source=None, human_error_ms=None, human_status='not_annotated', ctc_observation=model.get(sample['id']))
        if annotation:
            source = annotation.get('source')
            row['annotation_source'] = source
            if source not in HUMAN_SOURCES:
                row['human_status'] = 'not_human_source'
            elif not annotation.get('reviewer') or not annotation.get('listened'):
                row['human_status'] = 'missing_human_attestation'
            elif annotation.get('uncertain'):
                row['human_status'] = 'uncertain'
            else:
                start, end = annotation.get('observed_start_ms'), annotation.get('observed_end_ms')
                valid = isinstance(start, (int, float)) and not isinstance(start, bool) and isinstance(end, (int, float)) and not isinstance(end, bool) and math.isfinite(start) and math.isfinite(end)
                valid = valid and sample['clip_start_ms'] <= start < end <= sample['clip_end_ms']
                if not valid:
                    row['human_status'] = 'invalid_annotation'
                else:
                    start_error, end_error = sample['system_start_ms'] - start, sample['system_end_ms'] - end
                    within = max(abs(start_error), abs(end_error)) <= manifest['threshold_ms']
                    row.update(human_status='within_threshold' if within else 'outside_threshold',
                               human_error_ms={'start_signed': round(start_error, 2), 'end_signed': round(end_error, 2), 'maximum_absolute': round(max(abs(start_error), abs(end_error)), 2)},
                               human_observation={'start_ms': start, 'end_ms': end, 'reviewer': annotation['reviewer'], 'note': annotation.get('note', ''), 'speaker_change_confirmed': annotation.get('speaker_change_confirmed')})
                    if sample['lexical_unit'] == 'word':
                        human_eligible.append(row)
                        passed += within
        rows.append(row)
    expected = [sample for sample in manifest['samples'] if sample['lexical_unit'] == 'word']
    complete = len(human_eligible) == len(expected)
    rate = passed / len(human_eligible) if human_eligible else None
    human_status = 'not_completed'
    if complete and len(human_eligible) >= manifest['minimum_human_words']:
        human_status = 'passed' if rate >= manifest['required_within_fraction'] else 'failed'
    bins = sorted({row['temporal_bin'] for row in human_eligible})
    if human_status == 'passed' and bins != list(range(6)):
        human_status = 'insufficient_temporal_coverage'
    confirmed_turns = sum(row.get('stratum') == 'speaker_change_candidate' and row.get('human_observation', {}).get('speaker_change_confirmed') == 'yes' for row in human_eligible)
    if human_status == 'passed' and any(sample['stratum'] == 'speaker_change_candidate' for sample in expected) and not confirmed_turns:
        human_status = 'precision_passed_speaker_coverage_unconfirmed'
    return {'version': VERSION, 'episode_id': manifest['episode_id'], 'timing_sha256': manifest['timing_sha256'], 'sample_count': len(manifest['samples']),
            'human_acceptance': {'status': human_status, 'word_count': len(human_eligible), 'expected_word_count': len(expected), 'within_threshold_words': passed,
                                 'within_fraction': rate, 'threshold_ms': manifest['threshold_ms'], 'required_fraction': manifest['required_within_fraction'],
                                 'temporal_bins_covered': bins, 'all_selected_words_annotated': complete,
                                 'human_confirmed_speaker_change_samples': confirmed_turns,
                                 'provenance_note': '人工结果仅依据标注者自报已听辨的原音边界；程序不会把模型结果升级为人工标注，也不验证标注者身份。'},
            'acoustic_consistency': {'source': 'independent_ctc_forced_alignment' if ctc_results else None, 'model': (ctc_results or {}).get('model'),
                                     'human_ground_truth': False, 'observed_count': sum(row.get('status') == 'measured' for row in model.values())},
            'warnings': manifest.get('warnings', []), 'rows': rows}


def normalize_ctc(text: str) -> str | None:
    # Numeric spelling is deliberately not guessed: six point zero and course
    # numbers are different pronunciations. A reviewer may provide a spoken form.
    if re.search(r'\d', text):
        return None
    value = re.sub(r"[^A-Z' ]", ' ', text.upper().replace('’', "'"))
    return re.sub(r'\s+', ' ', value).strip() or None


def candidate_ctc_text(candidate: dict) -> str:
    """Require provenance and explicit spoken forms for numbers/code notation."""
    text = candidate.get('text')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('候选须提供完整连续上下文 text，不能只提交孤立争议词。')
    if not isinstance(candidate.get('source'), str) or not candidate['source'].strip():
        raise ValueError('每份候选必须记录 source，如 medium_asr 或 large_asr；不得伪称人工。')
    spoken = candidate.get('spoken_form')
    symbolic = bool(re.search(r'[\d=<>+*/\\_^%{}\[\]|&@#$~]|\s-\s', text))
    if symbolic and not spoken:
        raise ValueError('数字或技术符号须提供整段明确 spoken_form，程序不会猜测实际读法。')
    if spoken is not None and (not isinstance(spoken, str) or not spoken.strip()):
        raise ValueError('spoken_form 必须是完整且非空的连续口语文本。')
    chosen = spoken if spoken is not None else text
    if re.search(r'[\d=<>+*/\\_^%{}\[\]|&@#$~]|\s-\s', chosen):
        raise ValueError('spoken_form 仍含数字或技术符号，请用明确英文读法。')
    normalized = normalize_ctc(chosen)
    if normalized is None:
        raise ValueError('候选没有可供英文 CTC 比较的文本。')
    return normalized


def score_ctc_candidates(emissions, label_sequence, candidates: list[dict]) -> dict:
    """Compare full-context likelihoods on identical raw logits; never edit text.

    Only normalized log-softmax probabilities enter ctc_loss. The zero-score
    edge wildcard used by the timing audit MUST NOT enter this calculation.
    """
    import torch
    if len(candidates) < 2:
        raise ValueError('比较至少需要两份有来源的连续上下文候选。')
    identities = [c.get('id') for c in candidates]
    if any(not isinstance(identity, str) or not identity.strip() for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError('每份候选必须具有不同且非空的 id。')
    if emissions.ndim != 3 or emissions.shape[0] != 1 or emissions.shape[2] != len(label_sequence):
        raise ValueError('输入须为单片段的原始 CTC logits，且标签数必须匹配；不接受额外 wildcard。')
    if not torch.isfinite(emissions).all():
        raise ValueError('模型 logits 含非有限数。')
    labels = {label: index for index, label in enumerate(label_sequence)}
    if label_sequence[0] != '-' or '|' not in labels:
        raise ValueError('不支持的 CTC 标签表。')
    probabilities = torch.log_softmax(emissions.to(dtype=torch.float32), dim=-1).transpose(0, 1).contiguous()
    frames = probabilities.shape[0]
    rows = []
    for candidate in candidates:
        row = {key: candidate.get(key) for key in ('id', 'text', 'source', 'spoken_form')}
        row.update(human_ground_truth=False, score_is_calibrated_accuracy=False)
        try:
            normalized = candidate_ctc_text(candidate)
            target = [labels[char] for char in normalized.replace(' ', '|')]
            row.update(normalized_text=normalized, target_token_count=len(target), frame_count=frames)
            minimum_frames = len(target) + sum(a == b for a, b in zip(target, target[1:]))
            if minimum_frames > frames:
                raise ValueError('候选字符数与重复字符要求超过音频帧数，无法进行 CTC 比较。')
            with torch.inference_mode():
                loss = torch.nn.functional.ctc_loss(probabilities, torch.tensor(target, dtype=torch.int64),
                    torch.tensor([frames]), torch.tensor([len(target)]), blank=0, reduction='sum', zero_infinity=False)
            nll = float(loss.item())
            if not math.isfinite(nll):
                raise ValueError('CTC 路径概率不可计算；未用零值替代失败。')
            row.update(status='measured', negative_log_likelihood=round(nll, 6),
                       nll_per_frame=round(nll / frames, 8), nll_per_target_token=round(nll / len(target), 8))
        except (ValueError, KeyError) as error:
            row.update(status='not_measurable', error=str(error))
        rows.append(row)
    measured = [row for row in rows if row['status'] == 'measured']
    by_frame = sorted(measured, key=lambda row: (row['nll_per_frame'], row['id']))
    by_token = sorted(measured, key=lambda row: (row['nll_per_target_token'], row['id']))
    return {'algorithm': 'normalized-logsoftmax-ctc-full-context-nll-v1', 'source': 'independent_ctc_candidate_likelihood',
            'human_ground_truth': False, 'comparable': len(measured) >= 2, 'frame_count': frames,
            'lower_score_is_model_preference': True, 'edge_wildcards': False, 'observations': rows,
            'ranking_per_frame': [row['id'] for row in by_frame], 'ranking_per_target_token': [row['id'] for row in by_token],
            'ranking_agrees': len(measured) >= 2 and [row['id'] for row in by_frame] == [row['id'] for row in by_token],
            'caution': '同一原音连续上下文的 CTC 模型偏好，不是人工听辨或正确率。文本长度及裁剪边缘会影响分数；同时报告每帧和每字符归一化，禁止据此自动换稿。'}


def run_ctc_candidate_comparison(audio: Path, candidates: list[dict], cache: Path, *, allow_download: bool = False, threads: int = 4) -> dict:
    import torch
    import torchaudio
    weights = cache / MODEL_FILENAME
    if not weights.is_file() and not allow_download:
        raise FileNotFoundError('独立 CTC 权重尚未缓存；需显式 --allow-download 才下载约 360.2 MiB 官方权重。')
    cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, threads))
    with wave.open(str(audio), 'rb') as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
            raise ValueError('候选比较须使用相同的 16kHz 单声道 16-bit PCM 原音片段。')
        rate = stream.getframerate()
        pcm = array.array('h', stream.readframes(stream.getnframes()))
    if not pcm or len(pcm) > rate * 60:
        raise ValueError('候选比较仅支持非空且不超过 60 秒的局部片段，不能盲目重跑整片。')
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model(dl_kwargs={'model_dir': str(cache), 'progress': False}).eval()
    with torch.inference_mode():
        emissions, _ = model(torch.tensor(pcm, dtype=torch.float32).div_(32768).unsqueeze(0))
    result = score_ctc_candidates(emissions, bundle.get_labels(), candidates)
    result.update(version=VERSION, model=MODEL_NAME, model_source=MODEL_URL, model_sha256=file_sha256(weights),
                  audio_file=audio.name, audio_sha256=file_sha256(audio), duration_ms=round(len(pcm)/rate*1000),
                  sample_rate=rate, candidate_input_sha256=digest(candidates), generated_at=dt.datetime.now(dt.timezone.utc).isoformat())
    return result


def write_report_csv(report: dict, destination: Path):
    def safe(value):
        if isinstance(value, str) and value.startswith(('=', '+', '-', '@')):
            return "'" + value
        return value
    with destination.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['词语ID', '词语', '分层', '系统开始ms', '系统结束ms', 'CTC开始ms', 'CTC结束ms', 'CTC开始差ms', 'CTC结束差ms', 'CTC最大绝对差ms', 'CTC路径分数非准确率', 'CTC来源', 'CTC可靠性提示', '人工开始ms', '人工结束ms', '人工最大误差ms', '人工标注来源', '人工状态', '原音片段'])
        for row in report['rows']:
            ctc = row.get('ctc_observation') or {}
            human = row.get('human_observation') or {}
            error = row.get('human_error_ms') or {}
            writer.writerow([safe(value) for value in [row['word_id'], row['text'], row['stratum'], row['system_start_ms'], row['system_end_ms'],
                ctc.get('observed_start_ms'), ctc.get('observed_end_ms'), ctc.get('start_error_ms'), ctc.get('end_error_ms'), ctc.get('maximum_absolute_error_ms'),
                ctc.get('token_path_score'), ctc.get('source'), '; '.join(ctc.get('reliability_flags', [])), human.get('start_ms'), human.get('end_ms'),
                error.get('maximum_absolute'), row.get('annotation_source'), row['human_status'], row.get('clip_file')]])


def run_ctc_audit(package: dict, manifest: dict, output: Path, cache: Path, *, allow_download: bool = False, threads: int = 4, limit: int | None = None, progress=print) -> dict:
    verify_manifest(package, manifest)
    import torch
    import torchaudio
    cache.mkdir(parents=True, exist_ok=True)
    weights = cache / MODEL_FILENAME
    if not weights.is_file() and not allow_download:
        raise FileNotFoundError('独立 CTC 权重尚未缓存。需显式 --allow-download 才下载约 360.2 MiB 官方权重。')
    torch.set_num_threads(max(1, threads))
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model(dl_kwargs={'model_dir': str(cache), 'progress': False}).eval()
    labels = {label: index for index, label in enumerate(bundle.get_labels())}
    words = sorted(package['words'], key=lambda w: w['start_ms'])
    starts = [word['start_ms'] for word in words]
    result = {'version': VERSION, 'timing_sha256': manifest['timing_sha256'], 'audio_sha256': manifest.get('audio_sha256'), 'model': MODEL_NAME,
              'model_source': MODEL_URL, 'model_sha256': file_sha256(weights), 'algorithm': CTC_ALGORITHM, 'label_source': 'independent_ctc_forced_alignment',
              'target_transcript_source': 'episode words; text is not independently verified', 'human_ground_truth': False, 'observations': []}
    previous = {}
    old = output / 'ctc-results.json'
    if old.is_file():
        cached = json.loads(old.read_text(encoding='utf-8'))
        if cached.get('timing_sha256') == manifest['timing_sha256'] and cached.get('audio_sha256') == manifest.get('audio_sha256') and cached.get('model_sha256') == result['model_sha256'] and cached.get('algorithm') == CTC_ALGORITHM:
            previous = {row['sample_id']: row for row in cached.get('observations', [])}
    for index, sample in enumerate(manifest['samples'][:limit]):
        if sample['id'] in previous:
            result['observations'].append(previous[sample['id']])
            continue
        observation = {'sample_id': sample['id'], 'word_id': sample['word_id'], 'text': sample['text'], 'clip_file': sample['clip_file'],
                       'clip_sha256': sample['clip_sha256'], 'source': 'independent_ctc_forced_alignment', 'human_ground_truth': False}
        try:
            center = bisect.bisect_left(starts, sample['clip_start_ms'])
            context = [word for word in words[max(0, center-2):bisect.bisect_right(starts, sample['clip_end_ms'])]
                       if word['start_ms'] >= sample['clip_start_ms'] and word['end_ms'] <= sample['clip_end_ms']]
            # Like torchaudio's published MMS_FA wrapper, an extra zero-score
            # wildcard absorbs untranscribed audio. Put it at crop edges and
            # unsupported numeric words, so clipped speech cannot drag the first
            # actual word across a pause. It is not a new speech prediction.
            star = len(labels)
            target, ranges, omissions = [star], {}, []
            overrides = sample.get('spoken_forms', {})
            for word in context:
                normalized = normalize_ctc(overrides.get(word['id'], word.get('spoken_form', word['text'])))
                if normalized is None:
                    omissions.append(word['id'])
                    if target[-1] != star:
                        target.append(star)
                    continue
                if target[-1] != star:
                    target.append(labels['|'])
                begin = len(target)
                target.extend(labels[char] for char in normalized.replace(' ', '|'))
                ranges[word['id']] = (begin, len(target))
            if sample['word_id'] not in ranges:
                raise ValueError('目标含数字或不支持字符，未猜测实际读法；需填写 spoken_forms 后再做 CTC。')
            if target[-1] != star:
                target.append(star)
            with wave.open(str(output / sample['clip_file']), 'rb') as stream:
                rate = stream.getframerate()
                pcm = array.array('h', stream.readframes(stream.getnframes()))
            if rate != bundle.sample_rate:
                raise ValueError('CTC 原音采样率不正确。')
            waveform = torch.tensor(pcm, dtype=torch.float32).div_(32768).unsqueeze(0)
            with torch.inference_mode():
                emissions, _ = model(waveform)
                probabilities = torch.log_softmax(emissions, dim=-1)
                probabilities = torch.cat((probabilities, torch.zeros((*probabilities.shape[:2], 1), dtype=probabilities.dtype)), dim=-1)
                path, scores = torchaudio.functional.forced_align(probabilities, torch.tensor([target], dtype=torch.int64), blank=0)
                spans = torchaudio.functional.merge_tokens(path[0], scores[0], blank=0)
            if len(spans) != len(target):
                raise ValueError('CTC 字符路径没有完整对应目标文本。')
            a, b = ranges[sample['word_id']]
            frame_ms = len(pcm) / rate * 1000 / probabilities.shape[1]
            start = sample['clip_start_ms'] + spans[a].start * frame_ms
            end = sample['clip_start_ms'] + spans[b-1].end * frame_ms
            confidence = math.exp(sum(span.score * (span.end-span.start) for span in spans[a:b]) / sum(span.end-span.start for span in spans[a:b]))
            reliability_flags = []
            if confidence < .5:
                reliability_flags.append('原始CTC路径分低于0.5，文本或声学匹配可能不可靠；阈值不是校准准确率')
            if end-start > 1500:
                reliability_flags.append('单词CTC跨度超过1500ms，可能存在被迫拉长或错误文本')
            observation.update(status='measured', observed_start_ms=round(start, 2), observed_end_ms=round(end, 2),
                               start_error_ms=round(sample['system_start_ms']-start, 2), end_error_ms=round(sample['system_end_ms']-end, 2),
                               maximum_absolute_error_ms=round(max(abs(sample['system_start_ms']-start), abs(sample['system_end_ms']-end)), 2),
                               token_path_score=round(confidence, 5), score_is_calibrated_accuracy=False, frame_resolution_ms=round(frame_ms, 3),
                               omitted_context_word_ids=omissions, normalized_target=normalize_ctc(overrides.get(sample['word_id'], next(w for w in context if w['id']==sample['word_id']).get('spoken_form', sample['text']))),
                               reliability_flags=reliability_flags, edge_wildcards=True,
                               caution='仅独立声学模型与系统时间的差异；非人工真值。低分或被迫对齐的错误文本都需要回听。')
        except Exception as error:
            observation.update(status='not_measurable', error=type(error).__name__ + ': ' + str(error)[:900])
        result['observations'].append(observation)
        old.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        progress(f'独立 CTC {index+1}/{len(manifest["samples"][:limit])}：{sample["text"]} / {observation["status"]}', flush=True)
    old.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_review_page(manifest, output, result)
    return result


def serve_review(output: Path, *, port: int = 8781):
    """Expose only this pack's review page and original clips on loopback."""
    import http.server
    import urllib.parse
    root = output.resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    allowed = {'/review.html'} | {'/' + row['clip_file'] for row in manifest['samples']}

    class ReviewHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def send_head(self):
            requested = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            if requested == '/':
                requested = '/review.html'
            target = (root/requested.lstrip('/')).resolve()
            if requested not in allowed or not target.is_relative_to(root) or not target.is_file():
                self.send_error(404, 'Only the review page and its original clips are available.')
                return None
            self.path = requested
            return super().send_head()

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

    with http.server.ThreadingHTTPServer(('127.0.0.1', port), ReviewHandler) as server:
        print(f'同步原音审查页：http://127.0.0.1:{port}/review.html（仅本机可访问；Ctrl+C 结束）', flush=True)
        server.serve_forever()


def write_review_page(manifest: dict, output: Path, ctc_results: dict | None = None):
    data = json.dumps({'manifest': manifest, 'ctc': ctc_results}, ensure_ascii=False).replace('<', '\\u003c')
    page = REVIEW_HTML.replace('__TITLE__', html.escape(manifest.get('title', '同步审查'))).replace('__DATA__', data)
    (output / 'review.html').write_text(page, encoding='utf-8')


REVIEW_HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐词同步审查 · __TITLE__</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f9;color:#263b58;font:15px/1.7 Arial,"Microsoft YaHei",sans-serif}header,main{max-width:1160px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 8px}h2{font-size:21px;margin:0}.notice{padding:14px 18px;border:1px solid #e0ca99;background:#fff7e7;border-radius:9px}.toolbar{position:sticky;top:0;z-index:4;background:#f9fbffff;box-shadow:0 4px 20px #18305218;padding:13px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar audio{height:38px;flex:1;min-width:230px}.toolbar label{font-size:12px}button,select,input,textarea{font:inherit}button{cursor:pointer;border:1px solid #c6d4e7;background:#fff;color:#365a87;border-radius:6px;padding:7px 11px}button:disabled{opacity:.4;cursor:default}input[type=text],input[type=number],select,textarea{padding:8px;border:1px solid #cdd8e8;border-radius:6px;background:white;min-width:0}input[type=number]{width:155px}.toolbar input[type=text]{width:130px}.sample{background:white;border:1px solid #dbe4ef;border-radius:12px;padding:22px;margin-bottom:18px}.sample.active{border-color:#6c9bcb;box-shadow:0 0 0 2px #739bc522}.sample-head{display:flex;align-items:baseline;gap:15px;justify-content:space-between}.sample-head small{font-size:11px;color:#8296af}.context{font-size:14px;color:#8593a8;margin:10px 0}.wave{width:100%;height:150px;background:#f7faff;border:1px solid #e2eaf4;border-radius:7px;cursor:crosshair;touch-action:manipulation}.wave .peak{stroke:#7596b8;stroke-width:1}.wave .human-start{stroke:#29845c;stroke-width:2}.wave .human-end{stroke:#ad7843;stroke-width:2}.wave .system{stroke:#bb5862;stroke-width:1.5;stroke-dasharray:5 4;display:none}.wave .ctc{stroke:#816cb7;stroke-width:1.5;stroke-dasharray:2 3;display:none}.wave .playhead{stroke:#24394f;stroke-width:2}.show-system .system,.show-model .ctc{display:block}.times{font-size:11px;color:#8b9db3;display:flex;justify-content:space-between}.fields{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:15px 0}.fields label{font-size:12px;display:flex;align-items:center;gap:7px}.sample textarea{width:100%;min-height:50px;margin-top:9px;font-size:12px}.attest{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#69819e}.model{display:none;background:#f6f3fd;color:#7e6d9f;padding:10px 13px;margin-top:12px;font-size:12px}.show-model .model{display:block}.small{font-size:12px;color:#8595ab}.sample .state{font-size:12px;color:#8196b2}.sample.done .state{color:#358463}#summary{font-size:12px;color:#7c8da4}details{margin:13px 0}summary{cursor:pointer;color:#6b81a0;font-size:13px}@media(max-width:650px){header,main{padding:17px 12px}.sample{padding:15px}.sample-head{display:block}.toolbar{padding:10px 12px;gap:8px}.toolbar audio{width:100%;flex-basis:100%}.wave{height:130px}h1{font-size:22px}.fields input[type=number]{width:128px}.fields label{width:100%}}
</style><header><h1>逐词同步审查</h1><h2>__TITLE__</h2><p class="notice">本页没有替你完成听辨。原音和波形来自实际素材；模型边界只是辅助证据。先听原音、独立标起止，再展开系统与 CTC 边界比较。未听清可标“待确认”，不得猜时间。</p><details><summary>抽样与验收规则</summary><div id="rules"></div><p>每词误差取开始与结束边界绝对误差的较大者；至少 40 个单词、所有选定单词标注完整、六个时间层均覆盖后，才计算 95% 是否在 200ms 内。换人候选须另外听辨确认；停顿不等于换人。导出的“人工”来源只是标注者自报，不是程序证明。</p></details></header>
<div class="toolbar"><audio id="audio" controls preload="none"></audio><label>语速 <select id="rate"><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.25">1.25×</option></select></label><label>标注者 <input type="text" id="reviewer" placeholder="姓名或代号"></label><label><input type="checkbox" id="showSystem">显示系统边界</label><label><input type="checkbox" id="showModel">显示 CTC</label><button id="export">导出标注 JSON</button><span id="summary"></span></div><main id="samples"></main>
<script>const DATA=__DATA__;const manifest=DATA.manifest;const observations=new Map((DATA.ctc?.observations||[]).map(o=>[o.sample_id,o]));const root=document.getElementById('samples'),audio=document.getElementById('audio');const key='sync-audit:'+manifest.timing_sha256;let state={reviewer:'',rows:{}},active=null;try{state=JSON.parse(localStorage.getItem(key))||state}catch{}document.getElementById('reviewer').value=state.reviewer||'';function save(){try{localStorage.setItem(key,JSON.stringify(state))}catch{}updateSummary()}function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function clock(ms){let s=Math.max(0,ms)/1000;return Math.floor(s/60).toString().padStart(2,'0')+':'+(s%60).toFixed(3).padStart(6,'0')}function x(sample,ms){return (ms-sample.clip_start_ms)/(sample.clip_end_ms-sample.clip_start_ms)*1000}function row(s){return state.rows[s.id]||(state.rows[s.id]={observed_start_ms:null,observed_end_ms:null,listened:false,uncertain:false,note:'',speaker_change_confirmed:null})}function draw(s){const el=document.getElementById(s.id),r=row(s);for(const side of ['start','end']){const line=el.querySelector('.human-'+side);const value=r['observed_'+side+'_ms'];line.setAttribute('x1',value===null?-20:x(s,value));line.setAttribute('x2',value===null?-20:x(s,value));}const valid=r.observed_start_ms!==null&&r.observed_end_ms>r.observed_start_ms;el.classList.toggle('done',valid&&r.listened&&!r.uncertain);el.querySelector('.state').textContent=r.uncertain?'待确认':valid&&r.listened?'已自报听辨':'尚未完成听辨'}function updateSummary(){document.getElementById('summary').textContent=manifest.samples.filter(s=>{const r=row(s);return r.listened&&!r.uncertain&&r.observed_start_ms!==null&&r.observed_end_ms>r.observed_start_ms}).length+' / '+manifest.samples.length+' 条已标注'}
document.getElementById('rules').textContent=manifest.samples.length+' 个样本。分层：'+JSON.stringify(manifest.sampling_counts)+'。'+manifest.warnings.join(' ');for(const s of manifest.samples){const r=row(s),o=observations.get(s.id);const card=document.createElement('section');card.id=s.id;card.className='sample';const peaks=s.waveform.map((v,i)=>'<line class="peak" x1="'+(i+.5)/s.waveform.length*1000+'" x2="'+(i+.5)/s.waveform.length*1000+'" y1="'+(75-v[1]*65)+'" y2="'+(75-v[0]*65)+'"/>').join('');const marker=(cls,pos)=>'<line class="'+cls+'" x1="'+pos+'" x2="'+pos+'" y1="6" y2="144"/>';card.innerHTML='<div class="sample-head"><h2>'+esc(s.text)+'</h2><small>'+esc(s.word_id)+' · '+esc(s.stratum)+' · '+esc(s.chapter_id||'')+'</small></div><p class="context">'+esc(s.sentence)+'</p><svg class="wave" viewBox="0 0 1000 150" preserveAspectRatio="none" aria-label="原音波形，点击选择边界">'+peaks+marker('system',x(s,s.system_start_ms))+marker('system',x(s,s.system_end_ms))+marker('ctc',o?.observed_start_ms?x(s,o.observed_start_ms):-20)+marker('ctc',o?.observed_end_ms?x(s,o.observed_end_ms):-20)+marker('human-start',-20)+marker('human-end',-20)+marker('playhead',-20)+'</svg><div class="times"><span>'+clock(s.clip_start_ms)+'</span><span>原音全局时间</span><span>'+clock(s.clip_end_ms)+'</span></div><div class="fields"><button data-action="play">回听片段</button><button data-action="start-now">当前声位置作开始</button><button data-action="end-now">当前声位置作结束</button><label>开始（ms）<input type="number" data-field="observed_start_ms" value="'+(r.observed_start_ms??'')+'"></label><label>结束（ms）<input type="number" data-field="observed_end_ms" value="'+(r.observed_end_ms??'')+'"></label><label>波形点击设 <select data-click-mode><option value="observed_start_ms">开始</option><option value="observed_end_ms">结束</option></select></label></div><div class="attest"><label><input type="checkbox" data-field="listened" '+(r.listened?'checked':'')+'>我已听辨此词并确认边界</label><label><input type="checkbox" data-field="uncertain" '+(r.uncertain?'checked':'')+'>听不清／边界有争议，待确认</label>'+(s.stratum==='speaker_change_candidate'?'<label>确有换人 <select data-field="speaker_change_confirmed"><option value="">尚未确认</option><option value="yes" '+(r.speaker_change_confirmed==='yes'?'selected':'')+'>是</option><option value="no" '+(r.speaker_change_confirmed==='no'?'selected':'')+'>否</option></select></label>':'')+'<span class="state"></span></div><textarea data-field="note" placeholder="听辨说明、实际说法或争议原因">'+esc(r.note)+'</textarea><div class="model">系统：'+s.system_start_ms+' → '+s.system_end_ms+'ms；CTC：'+esc(o?.status==='measured'?o.observed_start_ms+' → '+o.observed_end_ms+'ms，差异最大 '+o.maximum_absolute_error_ms+'ms，原始路径分 '+o.token_path_score:o?.error||'未运行')+'。CTC 非人工真值。</div>';root.appendChild(card);card.querySelectorAll('[data-field]').forEach(input=>input.addEventListener('change',()=>{const f=input.dataset.field;r[f]=input.type==='checkbox'?input.checked:input.type==='number'?(input.value===''?null:Number(input.value)):input.value;draw(s);save()}));card.querySelector('[data-action="play"]').onclick=()=>{document.querySelectorAll('.sample.active').forEach(c=>c.classList.remove('active'));active=s;card.classList.add('active');audio.src=s.clip_file;audio.playbackRate=Number(document.getElementById('rate').value);audio.play().catch(()=>{})};function current(f){if(active?.id!==s.id){alert('请先回听此片段，再选当前声位置。');return}r[f]=Math.round(s.clip_start_ms+audio.currentTime*1000);card.querySelector('[data-field="'+f+'"]').value=r[f];draw(s);save()}card.querySelector('[data-action="start-now"]').onclick=()=>current('observed_start_ms');card.querySelector('[data-action="end-now"]').onclick=()=>current('observed_end_ms');card.querySelector('.wave').onclick=e=>{const box=e.currentTarget.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(e.clientX-box.left)/box.width));const f=card.querySelector('[data-click-mode]').value;r[f]=Math.round(s.clip_start_ms+fraction*(s.clip_end_ms-s.clip_start_ms));card.querySelector('[data-field="'+f+'"]').value=r[f];draw(s);save()};draw(s)}document.getElementById('rate').onchange=e=>audio.playbackRate=Number(e.target.value);document.getElementById('reviewer').onchange=e=>{state.reviewer=e.target.value.trim();save()};document.getElementById('showSystem').onchange=e=>document.body.classList.toggle('show-system',e.target.checked);document.getElementById('showModel').onchange=e=>document.body.classList.toggle('show-model',e.target.checked);audio.ontimeupdate=()=>{if(active){const line=document.getElementById(active.id).querySelector('.playhead'),pos=x(active,active.clip_start_ms+audio.currentTime*1000);line.setAttribute('x1',pos);line.setAttribute('x2',pos)}};document.getElementById('export').onclick=()=>{if(!state.reviewer.trim()){alert('请填写标注者姓名或代号。');return}const output={version:1,timing_sha256:manifest.timing_sha256,audio_sha256:manifest.audio_sha256,annotations:manifest.samples.map(s=>({sample_id:s.id,word_id:s.word_id,...row(s),source:'human_listening_with_waveform',reviewer:state.reviewer,exported_at:new Date().toISOString()}))};const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='sync-human-annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000)};updateSummary();</script></html>'''
