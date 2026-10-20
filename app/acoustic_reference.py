"""Resolve a reference spelling dispute only with another acoustic observation."""
from __future__ import annotations
import copy,hashlib,json,math,os,re,subprocess,tempfile
from pathlib import Path
from .reference_check import reference_issues, _parts, _canonical

_OBSERVATION_VERSION = 'unprompted-reference-audio-1'
_RESOLUTION_VERSION = 'reference-audio-evidence-1'



def reference_words_fingerprint(words: list[dict]) -> str:
    """Bind only audible text/clock, independent of lesson IDs and glosses."""
    value = [{'text': str(word['text']), 'start_ms': int(word['start_ms']),
              'end_ms': int(word['end_ms'])} for word in words]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def reference_issue_key(issue: dict) -> str:
    """Bind a decision to exact current wording and measured listening bounds."""
    value = {'version': _RESOLUTION_VERSION, 'recognized': str(issue.get('recognized', '')),
             'reference': str(issue.get('reference', '')), 'start_ms': int(issue['start_ms']),
             'end_ms': int(issue['end_ms'])}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _evidence_signature(value):
    fields = ('version', 'issue_key', 'status', 'kind', 'mode', 'audio_sha256', 'model_fingerprint',
              'observation_id', 'observation_words_sha256', 'matched_words', 'candidate_words', 'confidence_min')
    return hashlib.sha256(json.dumps({key: value.get(key) for key in fields}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def reference_resolution_is_current(issue: dict, resolution: dict, *, audio_sha256: str, model_fingerprint: str) -> bool:
    """Only a signed, current supported-recognition result can resolve a flag."""
    return (resolution.get('version') == _RESOLUTION_VERSION
            and resolution.get('status') == 'supported_recognition'
            and resolution.get('mode') == 'independent_unprompted_asr'
            and resolution.get('issue_key') == reference_issue_key(issue)
            and resolution.get('audio_sha256') == audio_sha256
            and resolution.get('model_fingerprint') == model_fingerprint
            and bool(resolution.get('matched_words'))
            and resolution.get('evidence_signature') == _evidence_signature(resolution))


def _acoustic_matches(text, words, issue, threshold, tolerance_ms):
    target = [item['value'] for item in _canonical(_parts(text))]
    if not target:
        return []
    timed = []
    for index, word in enumerate(words):
        try:
            start, end = int(word['start_ms']), int(word['end_ms'])
            probability = float(word['probability'])
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
        if not math.isfinite(probability):
            continue
        timed.extend({**item, 'source_id': index} for item in _parts(str(word.get('text', '')), start, end))
    units = _canonical(timed)
    values = [item['value'] for item in units]
    result = []
    for index in range(len(values) - len(target) + 1):
        if values[index:index + len(target)] != target:
            continue
        group = units[index:index + len(target)]
        start, end = min(item['start_ms'] for item in group), max(item['end_ms'] for item in group)
        if start < issue['start_ms'] - tolerance_ms or end > issue['end_ms'] + tolerance_ms:
            continue
        core = [copy.deepcopy(word) for word in words if
                (word['start_ms'] < end and word['end_ms'] > start)
                or start <= word['start_ms'] <= word['end_ms'] <= end]
        if not core or any(word['end_ms'] <= word['start_ms'] or not math.isfinite(float(word.get('probability', 0))) or not threshold <= float(word.get('probability', 0)) <= 1 for word in core):
            continue
        result.append({'start_ms': start, 'end_ms': end, 'words': core,
                       'confidence_min': min(float(word['probability']) for word in core)})
    return result


def assess_observations(issues: list[dict], observations: list[dict], *, audio_sha256: str,
                        model_fingerprint: str, min_probability=.85, tolerance_ms=600) -> list[dict]:
    """Classify positive independent audio evidence, never infer silence from ASR.

    'supported_recognition' keeps actual speech when a reference is editorial.
    'reference_supported' provides measured correction candidates and still
    requires real transcript replacement/alignment. Everything else remains
    unresolved, including a reference-only word absent from both recognizers.
    The caller supplies current audio/model fingerprints to reject stale data.
    """
    if not .85 <= min_probability <= 1 or not 0 <= tolerance_ms <= 1000:
        raise ValueError('原声证据判定阈值无效。')
    results = []
    for issue in issues:
        verdict = {'version': _RESOLUTION_VERSION, 'issue_key': reference_issue_key(issue),
                   'status': 'unresolved', 'kind': 'unresolved', 'reason': '尚无足够的独立正向声学证据'}
        eligible = [observation for observation in observations if
                    observation.get('mode') == 'independent_unprompted_asr'
                    and observation.get('audio_sha256') == audio_sha256
                    and observation.get('model_fingerprint') == model_fingerprint
                    and not observation.get('truncated')
                    and observation.get('start_ms', math.inf) <= issue['start_ms']
                    and observation.get('end_ms', -1) >= issue['end_ms']]
        supported, ambiguous = [], False
        for observation in eligible:
            try:
                raw_words = observation.get('words', [])
                if not raw_words or any(not isinstance(word.get('text'), str) for word in raw_words):
                    continue
                words = [{**word, 'start_ms': int(word['start_ms']), 'end_ms': int(word['end_ms']), 'probability': float(word['probability'])} for word in raw_words]
                if any(not math.isfinite(word['probability']) for word in words):
                    continue
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                continue
            critical = bool(re.search(r'\d|\b(?:not|no|never|cannot|incorrect|correct|invalid|valid|impossible|possible)\b', issue.get('recognized', '') + ' ' + issue.get('reference', ''), re.I))
            threshold = max(min_probability, .9 if critical else min_probability)
            actual = _acoustic_matches(issue.get('recognized', ''), words, issue, threshold, tolerance_ms)
            expected = _acoustic_matches(issue.get('reference', ''), words, issue, threshold, tolerance_ms)
            if actual and expected:
                ambiguous = True
                continue
            if not actual and not expected:
                continue
            match = max(actual or expected, key=lambda item: item['confidence_min'])
            status = 'supported_recognition' if actual else 'reference_supported'
            kind = ('reference_omission' if not issue.get('reference') else 'reference_substitution') if actual else ('transcription_omission' if not issue.get('recognized') else 'transcription_substitution')
            supported.append({**verdict, 'status': status, 'kind': kind,
                              'mode': observation['mode'], 'audio_sha256': audio_sha256,
                              'model_fingerprint': model_fingerprint, 'observation_id': observation['observation_id'],
                              'observation_words_sha256': hashlib.sha256(json.dumps(words, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                              'matched_words': match['words'], 'candidate_words': match['words'] if expected else [],
                              'confidence_min': match['confidence_min'],
                              'reason': '独立无提示识别在当前实测窗口明确支持原口语' if actual else '独立无提示识别支持参考词，须先实际修正并重新对齐'})
        if ambiguous:
            verdict['reason'] = '同一实测窗口同时出现双方词语，无法据此消除争议'
        elif supported and len({item['status'] for item in supported}) == 1:
            verdict = max(supported, key=lambda item: item['confidence_min'])
            verdict['evidence_signature'] = _evidence_signature(verdict)
        elif supported:
            verdict['reason'] = '不同独立观测互相冲突，仍需回听'
        results.append(verdict)
    return results


def _audio_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprint(model_path: str) -> str:
    path = Path(model_path).resolve()
    identity = [str(path)]
    for name in ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json', 'vocabulary.txt'):
        item = path / name
        if item.is_file():
            stat = item.stat()
            identity.append([name, stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def observe_disputes(audio: Path, issues: list[dict], model_path: str, progress,
                     *, cache_dir: Path | None = None, model=None, device='cpu',
                     compute_type='int8', margin_ms=2500) -> list[dict]:
    """Collect independent timed audio observations without approving or editing.

    Adjacent listening windows share one ASR call. The reference and prior ASR
    text are deliberately never supplied as model prompts. Returned tokens are
    acoustic measurements and can disagree with either input; callers must keep
    unresolved differences pending. Audio/model/algorithm fingerprints protect
    resume caches. A caller may reuse its already loaded local model.
    """
    if not issues:
        return []
    audio = Path(audio).resolve(strict=True)
    if not audio.is_file() or not 0 <= margin_ms <= 10000:
        raise ValueError('原声复核参数无效。')
    windows = []
    for issue in sorted(issues, key=lambda value: (value['start_ms'], value['end_ms'])):
        start, end = int(issue['start_ms']), int(issue['end_ms'])
        if start < 0 or end < start:
            raise ValueError('原声复核时间无效。')
        lo, hi = max(0, start - margin_ms), end + margin_ms
        # Do not turn a corrupt long dispute into an unbounded media operation.
        truncated = hi - lo > 30000
        if truncated:
            lo = max(0, (start + end) // 2 - 15000)
            hi = lo + 30000
        if windows and lo <= windows[-1]['end_ms'] + 500 and hi - windows[-1]['start_ms'] <= 30000:
            windows[-1]['end_ms'] = max(windows[-1]['end_ms'], hi)
            windows[-1]['issues'].append(copy.deepcopy(issue))
            windows[-1]['truncated'] |= truncated
        else:
            windows.append({'start_ms': lo, 'end_ms': hi, 'issues': [copy.deepcopy(issue)], 'truncated': truncated})
    audio_hash, model_id = _audio_digest(audio), _model_fingerprint(model_path)
    cache_root = Path(cache_dir) if cache_dir is not None else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)
    observations = []
    for index, window in enumerate(windows):
        progress('reference', 74, f'正在独立回听原声差异 {index + 1} / {len(windows)}')
        signature = [_OBSERVATION_VERSION, audio_hash, model_id, device, compute_type, window['start_ms'], window['end_ms'], 5]
        key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        cache = cache_root / (key + '.json') if cache_root else None
        saved = None
        if cache and cache.is_file():
            try:
                candidate = json.loads(cache.read_text(encoding='utf-8'))
                if candidate.get('observation_id') == key and candidate.get('status') == 'observed' and isinstance(candidate.get('words'), list) and candidate['words']:
                    saved = candidate
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        if saved is None:
            if model is None:
                from faster_whisper import WhisperModel
                model = WhisperModel(model_path, device=device, compute_type=compute_type)
            measured, warnings = [], []
            with tempfile.TemporaryDirectory(prefix='english-unprompted-reference-') as directory:
                clip = Path(directory) / 'observation.wav'
                command = ['ffmpeg', '-nostdin', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-ss', str(window['start_ms'] / 1000), '-i', str(audio), '-t', str((window['end_ms'] - window['start_ms']) / 1000), '-ar', '16000', '-ac', '1', '-y', str(clip)]
                try:
                    subprocess.run(command, check=True, capture_output=True, timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except (subprocess.SubprocessError, OSError) as exc:
                    raise RuntimeError('局部原声提取失败；已保留现有文稿和待检查状态。') from exc
                segments, _ = model.transcribe(str(clip), language='en', word_timestamps=True, vad_filter=False, beam_size=5, condition_on_previous_text=False)
                for segment in segments:
                    for word in segment.words or []:
                        start, end, probability = float(word.start), float(word.end), float(word.probability)
                        if not all(math.isfinite(value) for value in (start, end, probability)):
                            warnings.append('识别返回非有限时间或概率')
                            continue
                        token = {'text': word.word.strip(), 'start_ms': window['start_ms'] + round(start * 1000), 'end_ms': window['start_ms'] + round(end * 1000), 'probability': probability}
                        measured.append(token)
                        if start < 0 or end <= start or token['end_ms'] > window['end_ms'] + 200:
                            warnings.append('独立识别词语时间无效或越界')
                        if probability < .65 or not 0 <= probability <= 1:
                            warnings.append('独立识别词语置信度不足')
            if not measured:
                warnings.append('独立识别没有返回可核对的词语')
            saved = {'observation_id': key, 'version': _OBSERVATION_VERSION, 'mode': 'independent_unprompted_asr', 'audio_sha256': audio_hash, 'model_fingerprint': model_id, 'start_ms': window['start_ms'], 'end_ms': window['end_ms'], 'words': measured, 'text': ' '.join(word['text'] for word in measured), 'status': 'needs_check' if warnings else 'observed', 'warnings': list(dict.fromkeys(warnings))}
            if cache and not warnings:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=cache_root, suffix='.tmp', delete=False) as stream:
                        temporary = Path(stream.name)
                        json.dump(saved, stream, ensure_ascii=False)
                    os.replace(temporary, cache)
                finally:
                    if temporary:
                        temporary.unlink(missing_ok=True)
        observations.append({**saved, 'issues': window['issues'], 'truncated': window['truncated']})
    return observations

def reconcile(audio:Path,words:list[dict],reference:str,model_path:str,progress):
    issues=reference_issues(words,reference) if reference.strip() else []
    possible=[x for x in issues if len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",x['recognized']))==1 and len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",x['reference']))==1]
    if not possible:return words,[]
    from faster_whisper import WhisperModel
    model=WhisperModel(model_path,device='cpu',compute_type='int8')
    result=copy.deepcopy(words);changes=[]
    norm=lambda text:re.sub(r'[^a-z0-9]','',text.lower())
    for issue in possible:
        target=norm(issue['reference']);old=norm(issue['recognized'])
        index=next((i for i,w in enumerate(result) if norm(w['text'])==old and abs(w['start_ms']-issue['start_ms'])<100),None)
        if index is None:continue
        progress('reference',74,'正在对疑似错词进行局部原声复核…')
        word=result[index];offset=max(0,word['start_ms']/1000-2)
        with tempfile.TemporaryDirectory(prefix='english-reference-') as d:
            clip=Path(d)/'review.wav'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-protocol_whitelist','file,pipe','-ss',str(offset),'-i',str(audio),'-t','6','-ar','16000','-ac','1','-y',str(clip)],check=True,capture_output=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            # A reference answer must not bias its own acoustic verification.
            # Accept only a separately recognized, timed, confident token.
            segments,_=model.transcribe(str(clip),language='en',word_timestamps=True,vad_filter=False,beam_size=5,condition_on_previous_text=False)
            candidates=[w for s in segments for w in s.words or [] if norm(w.word)==target and w.end>w.start and w.probability>=.65]
        candidate=min(candidates,key=lambda w:abs((offset+w.start)*1000-word['start_ms']),default=None)
        if candidate is None:continue
        start,end=round((offset+candidate.start)*1000),round((offset+candidate.end)*1000)
        if abs(start-word['start_ms'])>800 or index and result[index-1]['end_ms']>start+80 or index+1<len(result) and end>result[index+1]['start_ms']+80:continue
        spelling=re.search(r"[A-Za-z]+(?:'[A-Za-z]+)?",issue['reference']).group(0)
        punctuation=re.search(r'[^A-Za-z0-9]*$',word['text']).group(0)
        changes.append({'before':word['text'],'after':spelling+punctuation,'original_ms':[word['start_ms'],word['end_ms']],'measured_ms':[start,end],'method':'reference_spelling_confirmed_by_local_asr','probability':float(candidate.probability)})
        word.update(text=spelling+punctuation,start_ms=start,end_ms=end,probability=float(candidate.probability))
    del model
    return result,changes
