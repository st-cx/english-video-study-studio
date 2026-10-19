"""Resolve a reference spelling dispute only with another acoustic observation."""
from __future__ import annotations
import copy,re,subprocess,tempfile
from pathlib import Path
from .reference_check import reference_issues

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
            subprocess.run(['ffmpeg','-v','error','-ss',str(offset),'-i',str(audio),'-t','6','-ar','16000','-ac','1','-y',str(clip)],check=True,capture_output=True,timeout=30)
            # The short reference is a spelling hint, not an instruction to
            # create words. We accept only an actually recognized timed token.
            segments,_=model.transcribe(str(clip),language='en',word_timestamps=True,vad_filter=False,beam_size=5,condition_on_previous_text=False,initial_prompt=issue['reference'])
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
