from __future__ import annotations

import difflib
import re
import subprocess
import tempfile
from pathlib import Path


def group_connected_speech(words: list[dict]) -> tuple[list[dict],list[dict]]:
    """Keep measured spans for inseparable weak function words; never invent a boundary."""
    result,groups=[],[]
    for index,source in enumerate(words):
        word=dict(source)
        weak=word['text'].lower().strip('.,!?;:') in {'a','an','the','to','of','and','or'}
        if word['end_ms']<=word['start_ms'] and weak and result and index<len(words)-1:
            previous=result[-1]
            following=words[index+1]
            if following['end_ms']>following['start_ms'] and abs(following['start_ms']-word['end_ms'])<=120 and previous['end_ms']>previous['start_ms'] and abs(previous['end_ms']-word['start_ms'])<=100 and not re.search(r'[.!?]$',previous['text']):
                groups.append({'texts':[previous['text'],word['text']],'measured_ms':[previous['start_ms'],max(previous['end_ms'],word['end_ms'])],'reason':'inseparable_weak_function_word'})
                previous['text']+=' '+word['text']
                previous['end_ms']=max(previous['end_ms'],word['end_ms'])
                previous['connected_words']=groups[-1]['texts']
                continue
        result.append(word)
    return result,groups


def _norm(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())


def _positive(word: dict) -> bool:
    return word['start_ms'] >= 0 and word['end_ms'] > word['start_ms']


def _large_zero_shift(before: dict, observed: dict) -> bool:
    return (before['end_ms'] <= before['start_ms']
            and max(abs(before[key]-observed[key]) for key in ('start_ms', 'end_ms')) > 1200)


_WEAK_FUNCTION_WORDS = {'a','an','the','to','of','and','or','in','on','at','as','for','by','with','from',
                        'is','am','are','was','were','be','been','being','it','that'}


def _weak_shift_requires_independent_acoustics(before: dict, observed: dict) -> bool:
    """Whisper attention timestamps can expand short weak words across pauses.

    A high ASR spelling probability does not verify a new boundary. Keep a
    positive stable interval disputed until independent CTC can corroborate a
    large move; zero-duration words do not have an interval to protect.
    """
    return (_positive(before) and _norm(before['text']) in _WEAK_FUNCTION_WORDS
            and max(abs(before[key]-observed[key]) for key in ('start_ms','end_ms')) > 400)


def _compatible_patch(words: list[dict], patch: dict[int, dict]) -> bool:
    """Validate a transaction of measured spans without moving any boundary."""
    if any(not _positive(w) for w in patch.values()):
        return False
    for i in {j for k in patch for j in (k-1, k) if 0 <= j < len(words)-1}:
        left, right = patch.get(i, words[i]), patch.get(i+1, words[i+1])
        if left['start_ms'] > right['start_ms'] or left['end_ms']-right['start_ms'] > 80:
            return False
    return True


def _contextual_matches(expected: list[dict], observed: list[dict]) -> dict[int, dict]:
    """A same-spelling token alone is insufficient evidence, especially for I/So.

    Match spoken sequences, requiring at least two nearby positive anchor words.
    The candidate intervals are the acoustic observations verbatim, not shares
    of a sentence span. An unrelated repeated occurrence cannot pass on spelling.
    """
    mapping = {}
    matcher = difflib.SequenceMatcher(None, [_norm(w['text']) for w in expected],
                                      [_norm(w['text']) for w in observed], autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.size < 3:
            continue
        anchors = [j for j in range(block.size)
                   if _positive(expected[block.a+j]) and _positive(observed[block.b+j])
                   and max(abs(expected[block.a+j][key]-observed[block.b+j][key])
                           for key in ('start_ms', 'end_ms')) <= 650]
        if len(anchors) < 2:
            continue
        for j in range(block.size):
            candidate = observed[block.b+j]
            if _positive(candidate):
                if _large_zero_shift(expected[block.a+j], candidate):
                    # Two anchors elsewhere in a long matching sentence do not
                    # corroborate an isolated word hallucinated in its silence.
                    # Require the candidate's actual acoustic neighbourhood;
                    # spelling confidence alone cannot bridge several seconds.
                    nearby = sum(max(candidate['start_ms']-observed[block.b+k]['end_ms'],
                                     observed[block.b+k]['start_ms']-candidate['end_ms'], 0) <= 650
                                 for k in anchors if k != j)
                    if nearby < 2:
                        continue
                mapping[block.a+j] = dict(candidate)
    return mapping


def _probe_window(words: list[dict], indices: list[int], originals: dict[int, dict]) -> tuple[int, int]:
    spans = [words[i] for i in indices] + [originals[i] for i in indices if i in originals]
    return max(0, min(w['start_ms'] for w in spans)-3000), max(w['end_ms'] for w in spans)+3500


def _source_mapping(recognized: list[dict], words: list[dict]) -> dict[int, dict]:
    result = {}
    for block in difflib.SequenceMatcher(None, [_norm(w['text']) for w in recognized],
                                        [_norm(w['text']) for w in words], autojunk=False).get_matching_blocks():
        for i in range(block.size):
            result[block.b+i] = recognized[block.a+i]
    return result


def _acoustic_observation(audio: Path, start_ms: int, end_ms: int, model) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix='english-alignment-') as temp:
        clip = Path(temp)/'audit.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-ss', str(start_ms/1000),
                        '-i', str(audio), '-t', str((end_ms-start_ms)/1000), '-ar', '16000',
                        '-ac', '1', '-y', str(clip)], check=True, capture_output=True, timeout=30,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        transcribe = getattr(model, 'transcribe_original', model.transcribe)
        segments, _ = transcribe(str(clip), language='en', word_timestamps=True,
                                       vad_filter=False, beam_size=5, condition_on_previous_text=False)
        return [{'text':w.word.strip(), 'start_ms':start_ms+round(w.start*1000),
                 'end_ms':start_ms+round(w.end*1000), 'probability':float(w.probability)}
                for segment in segments for w in segment.words or [] if w.word.strip()]


def refine_word_boundaries(audio: Path, source_words: list[dict], model_path: str, progress,
                           recognized: list[dict] | None = None, disputes: list[dict] | None = None,
                           probe_model=None) -> tuple[list[dict], dict]:
    """Remeasure only anomalous regions; unresolved observations remain explicit.

    All inputs/outputs use the same time basis as ``audio``. This also supports
    auditing cached words without pretending the original ASR was persisted.
    """
    words = [dict(w) for w in source_words]
    originals = _source_mapping(recognized or [], words)
    disputes = [dict(d) for d in disputes or []]
    targets = {i for i,w in enumerate(words) if not _positive(w)}
    for i in range(1,len(words)):
        if words[i-1]['end_ms']-words[i]['start_ms']>80 or words[i-1]['start_ms']>words[i]['start_ms']:
            targets.update((i-1,i))
    for item in disputes:
        index = item.get('word_index')
        if (not isinstance(index,int) or not 0<=index<len(words)
                or _norm(words[index]['text'])!=_norm(item['text'])
                or abs(words[index]['start_ms']-item['aligned_ms'][0])>100):
            index=None
        if index is None:
            matches=[i for i,w in enumerate(words) if _norm(w['text'])==_norm(item['text'])
                     and abs(w['start_ms']-item['aligned_ms'][0])<=100]
            index=matches[0] if len(matches)==1 else None
        if index is not None:
            item['word_index']=index; targets.add(index)
            if item.get('asr_ms') and index not in originals:
                originals[index]={'text':item['text'],'start_ms':item['asr_ms'][0],'end_ms':item['asr_ms'][1]}
    if not targets:
        return words, {'alignment_repairs':[], 'alignment_disputes':disputes, 'alignment_window_audits':[]}
    regions=[]
    for i in sorted(targets):
        lo,hi=_probe_window(words,[i],originals)
        if regions and lo<=regions[-1]['end_ms']:
            regions[-1]['indices'].append(i);regions[-1]['start_ms']=min(lo,regions[-1]['start_ms']);regions[-1]['end_ms']=max(hi,regions[-1]['end_ms'])
        else:regions.append({'start_ms':lo,'end_ms':hi,'indices':[i]})
    owns_model=probe_model is None
    if owns_model:
        from faster_whisper import WhisperModel
        probe_model=WhisperModel(model_path,device='cpu',compute_type='int8')
    repairs=[];audits=[];resolved=set()
    try:
        for number,region in enumerate(regions):
            progress('align',73,f'正在核对局部原声边界 {number+1}/{len(regions)}…')
            if region['end_ms']-region['start_ms']>20000:
                audits.append({**region,'method':'unresolved_window_exceeds_20_seconds'});continue
            windows=[(region['start_ms'],region['end_ms'])]
            wider=(max(0,region['start_ms']-3000),region['end_ms']+2500)
            if wider[1]-wider[0]<=20000:windows.append(wider)
            for attempt,(lo,hi) in enumerate(windows):
                observed=_acoustic_observation(audio,lo,hi,probe_model)
                # Include all actual speech in the window. A fixed six-word
                # margin can omit the neighbour required to repair a cluster.
                context=[i for i,w in enumerate(words) if w['start_ms']<hi and w['end_ms']>=lo]
                first=max(0,min(context+region['indices'])-2)
                last=min(len(words),max(context+region['indices'])+3)
                local=_contextual_matches(words[first:last],observed)
                candidates={first+i:w for i,w in local.items() if w.get('probability',1)>=.35}
                blocked={i:w for i,w in candidates.items() if _weak_shift_requires_independent_acoustics(words[i],w)}
                measured={i:w for i,w in candidates.items() if i not in blocked}
                for i in set(blocked) & set(region['indices']):
                    existing=[d for d in disputes if d.get('word_index')==i]
                    if not existing:
                        item={'word_index':i,'text':words[i]['text'],
                              'aligned_ms':[words[i]['start_ms'],words[i]['end_ms']],
                              'asr_ms':[blocked[i]['start_ms'],blocked[i]['end_ms']],
                              'reason':'weak_function_word_large_asr_shift_requires_independent_ctc'}
                        disputes.append(item);existing=[item]
                    for item in existing:
                        item['independent_acoustic_confirmation_required']=True
                        item['blocked_window_asr_ms']=[blocked[i]['start_ms'],blocked[i]['end_ms']]
                patch={i:measured[i] for i in region['indices'] if i not in resolved and i in measured}
                # If a neighbour conflicts, its own measured span must join
                # the transaction. Never trim, clamp, or distribute duration.
                for _ in range(last-first+1):
                    additions=set()
                    for i in {j for k in patch for j in (k-1,k) if 0<=j<len(words)-1}:
                        a,b=patch.get(i,words[i]),patch.get(i+1,words[i+1])
                        if a['end_ms']-b['start_ms']>80 or a['start_ms']>b['start_ms']:
                            additions.update(j for j in (i,i+1) if j not in patch)
                    if not additions:break
                    if any(i not in measured for i in additions):break
                    patch.update({i:measured[i] for i in additions})
                accepted=bool(patch) and _compatible_patch(words,patch)
                if accepted:
                    for i,value in sorted(patch.items()):
                        before=[words[i]['start_ms'],words[i]['end_ms']]
                        after=[value['start_ms'],value['end_ms']]
                        if before!=after:
                            repairs.append({'word_index':i,'text':words[i]['text'],'aligned_ms':before,
                                            'window_asr_ms':after,'method':'contextual_window_asr_transaction'})
                            words[i].update(start_ms=after[0],end_ms=after[1],probability=value.get('probability'))
                        resolved.add(i)
                audits.append({**region,'start_ms':lo,'end_ms':hi,'attempt':attempt+1,
                               'method':'contextual_window_asr','accepted':accepted,
                               'observed_words':observed,'matched_indices':sorted(measured),
                               'blocked_weak_function_word_indices':sorted(blocked),
                               'applied_indices':sorted(patch) if accepted else [],
                               'unresolved_indices':[i for i in region['indices'] if i not in resolved]})
                if all(i in resolved for i in region['indices']):break
    finally:
        if owns_model:del probe_model
    unresolved=[d for d in disputes if d.get('word_index') not in resolved]
    return words, {'alignment_repairs':repairs,'alignment_disputes':unresolved,'alignment_window_audits':audits}


def align_words(audio: Path, recognized: list[dict], model_path: str, progress,
                alignment_model=None) -> tuple[list[dict], dict]:
    import stable_whisper
    progress('align', 73, '正在用 stable-ts 核对逐词边界…')
    owns_model=alignment_model is None
    model = alignment_model if alignment_model is not None else stable_whisper.load_faster_whisper(model_path, device='cpu', compute_type='int8')
    text = ' '.join(w['text'] for w in recognized)
    result = model.align(str(audio), text, language='en', regroup=False, verbose=None)
    if owns_model:del model
    if result is None:
        raise RuntimeError('逐词对齐没有返回结果，已保留识别检查点。')
    words = []
    for segment in result.segments:
        for token in segment.words or []:
            spelling = token.word.strip()
            if not spelling:
                continue
            if not re.search(r'[A-Za-z0-9]', spelling) and words:
                words[-1]['text'] += spelling
                continue
            words.append({'text': spelling, 'start_ms': round(token.start*1000), 'end_ms': round(token.end*1000), 'probability': None, 'whisper_segment': round(segment.start, 2)})
    original = _source_mapping(recognized, words)
    repairs, disputed = [], []
    proposed={}
    for i, word in enumerate(words):
        source = original.get(i)
        if not source:
            continue
        word['probability'] = source.get('probability')
        if word['end_ms'] <= word['start_ms'] and source['end_ms'] > source['start_ms']:
            proposed[i]=source
        elif word['end_ms']-word['start_ms'] > source['end_ms']-source['start_ms']+900 and source['end_ms']>source['start_ms']:
            proposed[i]=source
        elif max(abs(word['start_ms']-source['start_ms']), abs(word['end_ms']-source['end_ms'])) > 1200:
            disputed.append({'word_index':i,'text': word['text'], 'aligned_ms': [word['start_ms'], word['end_ms']], 'asr_ms': [source['start_ms'], source['end_ms']]})
    for i in range(1,len(words)):
        if words[i-1]['end_ms']-words[i]['start_ms']>80:
            left,right=original.get(i-1),original.get(i)
            if left and right and left['end_ms']>left['start_ms'] and right['end_ms']>right['start_ms'] and left['end_ms']<=right['start_ms']+40:
                proposed.update({i-1:left,i:right})
    # Apply the same protection to an initial ASR fallback, not only to a
    # contextual retry; both are observations from the same attention timing.
    contextual = _contextual_matches(words, recognized)
    for i in list(proposed):
        if _large_zero_shift(words[i], proposed[i]) and i not in contextual:
            source = proposed.pop(i)
            disputed.append({'word_index': i, 'text': words[i]['text'],
                             'aligned_ms': [words[i]['start_ms'], words[i]['end_ms']],
                             'asr_ms': [source['start_ms'], source['end_ms']],
                             'reason': 'unanchored_large_zero_duration_asr_shift_requires_contextual_remeasurement'})
        elif _weak_shift_requires_independent_acoustics(words[i],proposed[i]):
            source=proposed.pop(i)
            disputed.append({'word_index':i,'text':words[i]['text'],
                             'aligned_ms':[words[i]['start_ms'],words[i]['end_ms']],
                             'asr_ms':[source['start_ms'],source['end_ms']],
                             'reason':'weak_function_word_large_asr_shift_requires_independent_ctc',
                             'independent_acoustic_confirmation_required':True})
    # Commit connected proposed ASR spans together. A neighbouring zero-duration
    # token can otherwise make a positive fallback reverse spoken word order.
    pending=set(proposed)
    while pending:
        start=min(pending);group={start};pending.remove(start)
        while start+1 in pending:start+=1;group.add(start);pending.remove(start)
        patch={i:proposed[i] for i in group}
        if _compatible_patch(words,patch):
            for i,source in sorted(patch.items()):
                repairs.append({'word_index':i,'text':words[i]['text'],'aligned_ms':[words[i]['start_ms'],words[i]['end_ms']],
                                'asr_ms':[source['start_ms'],source['end_ms']],'method':'compatible_independent_asr_transaction'})
                words[i].update(start_ms=source['start_ms'],end_ms=source['end_ms'])
        else:
            for i,source in sorted(patch.items()):
                disputed.append({'word_index':i,'text':words[i]['text'],
                                 'aligned_ms':[words[i]['start_ms'],words[i]['end_ms']],
                                 'asr_ms':[source['start_ms'],source['end_ms']],
                                 'reason':'independent_asr_interval_conflicts_with_neighbours'})
    words,local=refine_word_boundaries(audio,words,model_path,progress,recognized,disputed,probe_model=alignment_model)
    return words, {'alignment':'stable-ts + faster-whisper','alignment_repairs':repairs+local['alignment_repairs'],
                   'alignment_disputes':local['alignment_disputes'],'alignment_window_audits':local['alignment_window_audits'],
                   'aligned_word_count':len(words)}
