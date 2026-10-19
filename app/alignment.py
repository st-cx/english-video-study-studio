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


def align_words(audio: Path, recognized: list[dict], model_path: str, progress) -> tuple[list[dict], dict]:
    import stable_whisper
    progress('align', 73, '正在用 stable-ts 核对逐词边界…')
    model = stable_whisper.load_faster_whisper(model_path, device='cpu', compute_type='int8')
    text = ' '.join(w['text'] for w in recognized)
    result = model.align(str(audio), text, language='en', regroup=False, verbose=None)
    del model
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
    norm = lambda value: re.sub(r'[^a-z0-9]', '', value.lower())
    previous = [norm(w['text']) for w in recognized]
    aligned = [norm(w['text']) for w in words]
    original = {}
    for block in difflib.SequenceMatcher(None, previous, aligned, autojunk=False).get_matching_blocks():
        for i in range(block.size):
            original[block.b+i] = recognized[block.a+i]
    repairs, disputed = [], []
    for i, word in enumerate(words):
        source = original.get(i)
        if not source:
            continue
        word['probability'] = source.get('probability')
        if word['end_ms'] <= word['start_ms'] and source['end_ms'] > source['start_ms']:
            repairs.append({'text': word['text'], 'aligned_ms': [word['start_ms'], word['end_ms']], 'asr_ms': [source['start_ms'], source['end_ms']], 'method': 'independent_asr_positive_interval'})
            word.update(start_ms=source['start_ms'], end_ms=source['end_ms'])
        elif word['end_ms']-word['start_ms'] > source['end_ms']-source['start_ms']+900 and source['end_ms']>source['start_ms']:
            repairs.append({'text':word['text'],'aligned_ms':[word['start_ms'],word['end_ms']],'asr_ms':[source['start_ms'],source['end_ms']],'method':'independent_asr_for_extended_word'})
            word.update(start_ms=source['start_ms'],end_ms=source['end_ms'])
        elif max(abs(word['start_ms']-source['start_ms']), abs(word['end_ms']-source['end_ms'])) > 1200:
            disputed.append({'text': word['text'], 'aligned_ms': [word['start_ms'], word['end_ms']], 'asr_ms': [source['start_ms'], source['end_ms']]})
    for i in range(1,len(words)):
        if words[i-1]['end_ms']-words[i]['start_ms']>80:
            left,right=original.get(i-1),original.get(i)
            if left and right and left['end_ms']>left['start_ms'] and right['end_ms']>right['start_ms'] and left['end_ms']<=right['start_ms']+40:
                for index,source in ((i-1,left),(i,right)):
                    repairs.append({'text':words[index]['text'],'aligned_ms':[words[index]['start_ms'],words[index]['end_ms']],'asr_ms':[source['start_ms'],source['end_ms']],'method':'independent_asr_for_overlap'})
                    words[index].update(start_ms=source['start_ms'],end_ms=source['end_ms'])
    # Re-measure zero-duration tokens in a local window. Repeating a full-file
    # alignment can reproduce the same degenerate boundary; a second acoustic
    # observation is required before any positive interval is accepted.
    zero_indices=[i for i,w in enumerate(words) if w['end_ms']<=w['start_ms']]
    if zero_indices:
        from faster_whisper import WhisperModel
        zero_probe=WhisperModel(model_path,device='cpu',compute_type='int8')
        for i in zero_indices:
            word=words[i]
            previous_end=words[i-1]['end_ms'] if i else word['start_ms']
            offset=max(0,min(word['start_ms'],previous_end)/1000-1)
            with tempfile.TemporaryDirectory(prefix='english-word-review-') as temp:
                clip=Path(temp)/'word.wav'
                subprocess.run(['ffmpeg','-v','error','-ss',str(offset),'-i',str(audio),'-t','7','-ar','16000','-ac','1','-y',str(clip)],check=True,capture_output=True,timeout=30)
                segments,_=zero_probe.transcribe(str(clip),language='en',word_timestamps=True,vad_filter=False,beam_size=5,condition_on_previous_text=False)
                candidates=[w for segment in segments for w in segment.words or [] if norm(w.word)==norm(word['text']) and w.end>w.start]
            candidate=min(candidates,key=lambda w:abs((offset+w.start)*1000-word['start_ms']),default=None)
            if candidate is None:continue
            start,end=round((offset+candidate.start)*1000),round((offset+candidate.end)*1000)
            before_ok=i==0 or words[i-1]['end_ms']<=start+80
            after_ok=i==len(words)-1 or end<=words[i+1]['start_ms']+80
            if before_ok and after_ok and abs(start-word['start_ms'])<=1600:
                repairs.append({'text':word['text'],'aligned_ms':[word['start_ms'],word['end_ms']],'window_asr_ms':[start,end],'method':'independent_window_asr_for_zero'})
                word.update(start_ms=start,end_ms=end,probability=float(candidate.probability))
        del zero_probe
    audits=[]
    if disputed:
        from faster_whisper import WhisperModel
        probe=WhisperModel(model_path,device='cpu',compute_type='int8')
        unresolved=[]
        for item in disputed:
            offset=max(0,item['aligned_ms'][0]/1000-2)
            with tempfile.TemporaryDirectory(prefix='bbc-alignment-') as temp:
                clip=Path(temp)/'audit.wav'
                subprocess.run(['ffmpeg','-v','error','-ss',str(offset),'-t','7','-i',str(audio),'-ar','16000','-ac','1','-y',str(clip)],check=True,capture_output=True,timeout=30)
                segments,_=probe.transcribe(str(clip),language='en',word_timestamps=True,vad_filter=False,beam_size=5)
                candidates=[w for segment in segments for w in segment.words or [] if norm(w.word)==norm(item['text'])]
            closest=min(candidates,key=lambda w:abs((offset+w.start)*1000-item['aligned_ms'][0]),default=None)
            if closest and max(abs((offset+closest.start)*1000-item['aligned_ms'][0]),abs((offset+closest.end)*1000-item['aligned_ms'][1]))<=300:
                audits.append({**item,'window_asr_ms':[round((offset+closest.start)*1000),round((offset+closest.end)*1000)],'method':'window_asr_confirms_alignment'})
            elif closest and closest.end>closest.start and max(abs((offset+closest.start)*1000-item['asr_ms'][0]),abs((offset+closest.end)*1000-item['asr_ms'][1]))<=300:
                start,end=round((offset+closest.start)*1000),round((offset+closest.end)*1000)
                index=next((i for i,w in enumerate(words) if norm(w['text'])==norm(item['text']) and [w['start_ms'],w['end_ms']]==item['aligned_ms']),None)
                if index is not None and (index==0 or words[index-1]['end_ms']<=start+80) and (index==len(words)-1 or end<=words[index+1]['start_ms']+80):
                    words[index].update(start_ms=start,end_ms=end)
                    audits.append({**item,'window_asr_ms':[start,end],'method':'window_asr_confirms_independent_asr'})
                else:unresolved.append(item)
            else:unresolved.append(item)
        disputed=unresolved
        del probe
    return words, {'alignment': 'stable-ts + faster-whisper', 'alignment_repairs': repairs, 'alignment_disputes': disputed, 'alignment_window_audits':audits, 'aligned_word_count': len(words)}
