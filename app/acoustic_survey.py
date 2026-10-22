"""Local overlapping CTC measurements with content-addressed resumable caches."""
from __future__ import annotations
import array,itertools,json,math,subprocess,wave
from pathlib import Path
from .sync_audit import MODEL_FILENAME,digest,file_sha256,normalize_ctc

VERSION='overlapping-ctc-survey-1'

def greedy_words(logp,labels,begin,end):
    tokens=logp[0].argmax(-1).tolist();frame=(end-begin)/len(tokens)
    rows=[];position=0;word='';first=last=0
    for token,group in itertools.groupby(tokens):
        a=position;position+=sum(1 for _ in group)
        if token==0:continue
        if labels[token]=='|':
            if word:rows.append({'text':word,'start_ms':round(begin+first*frame,2),'end_ms':round(begin+last*frame,2)});word=''
        else:
            if not word:first=a
            word+=labels[token];last=position
    if word:rows.append({'text':word,'start_ms':round(begin+first*frame,2),'end_ms':round(begin+last*frame,2)})
    return rows

def align_window(logp,labels,words,begin,end):
    import torch,torchaudio
    lookup={c:i for i,c in enumerate(labels)};star=len(labels);target=[star];ranges={}
    for i,w in enumerate(words):
        if not (w['start_ms']>=begin and w['end_ms']<=end):continue
        normalized=normalize_ctc(w.get('spoken_form') or w['text'])
        if normalized is None:
            if target[-1]!=star:target.append(star)
            continue
        if target[-1]!=star:target.append(lookup['|'])
        a=len(target);target.extend(lookup[c] for c in normalized.replace(' ','|'));ranges[i]=(a,len(target))
    if target[-1]!=star:target.append(star)
    wild=torch.cat((logp,torch.zeros((*logp.shape[:2],1))),dim=-1)
    path,scores=torchaudio.functional.forced_align(wild,torch.tensor([target],dtype=torch.int64),blank=0)
    spans=torchaudio.functional.merge_tokens(path[0],scores[0],blank=0)
    if len(spans)!=len(target):raise ValueError('CTC字符路径不完整。')
    frame=(end-begin)/logp.shape[1];rows=[]
    for i,(a,b) in ranges.items():
        score=math.exp(sum(s.score*(s.end-s.start) for s in spans[a:b])/sum(s.end-s.start for s in spans[a:b]))
        rows.append({'index':i,'id':words[i]['id'],'text':words[i]['text'],
            'start_ms':round(begin+spans[a].start*frame,2),'end_ms':round(begin+spans[b-1].end*frame,2),'path_score_not_accuracy':round(score,5)})
    return rows

def survey_audio(package,audio,output,model_cache,progress=print):
    """No network or config edits. Failed windows stay explicit and cannot patch."""
    import numpy as np,torch,torchaudio
    audio=Path(audio);output=Path(output);model_cache=Path(model_cache)
    weight=model_cache/MODEL_FILENAME
    if not weight.is_file():raise FileNotFoundError('需先缓存独立CTC模型；本步骤不会自动下载或更改配置。')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    audio_hash=file_sha256(audio);model_hash=file_sha256(weight)
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H;labels=bundle.get_labels();model=None
    wav=output/('audio-'+audio_hash[:16]+'.wav')
    if not wav.is_file():
        part=wav.with_suffix('.part.wav')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(audio),'-ar','16000','-ac','1','-c:a','pcm_s16le','-y',str(part)],check=True,capture_output=True,timeout=300)
        part.replace(wav)
    with wave.open(str(wav),'rb') as stream:
        rate=stream.getframerate();pcm=np.frombuffer(stream.readframes(stream.getnframes()),dtype=np.int16).copy()
    duration=len(pcm)/rate*1000
    if abs(duration-package['episode']['duration_s']*1000)>250:raise ValueError('原声时长与课程不一致，已停止测量。')
    windows=[(begin,min(duration,begin+20000)) for begin in range(0,math.ceil(duration),10000)]
    if duration>20000:windows.extend([(0,15000),(max(0,duration-15000),duration)])
    result={'version':VERSION,'audio_sha256':audio_hash,'model_sha256':model_hash,
        'input_timing_sha256':digest(package['words']),'human_ground_truth':False,'windows':[]}
    for number,(begin,end) in enumerate(windows):
        signature=digest([VERSION,audio_hash,model_hash,begin,end]);cache=output/(signature+'.npz');metadata=cache.with_suffix('.json');logp=None
        if cache.is_file() and metadata.is_file():
            try:
                info=json.loads(metadata.read_text(encoding='utf-8'))
                if info.get('signature')==signature and info.get('sha256')==file_sha256(cache):logp=torch.from_numpy(np.load(cache,allow_pickle=False)['logp'])
            except (ValueError,OSError,KeyError):pass
        if logp is None:
            if model is None:model=bundle.get_model(dl_kwargs={'model_dir':str(model_cache),'progress':False}).eval()
            x=torch.from_numpy(pcm[round(begin/1000*rate):round(end/1000*rate)]).float().div_(32768).unsqueeze(0)
            with torch.inference_mode():emissions,_=model(x);logp=torch.log_softmax(emissions,dim=-1)
            np.savez_compressed(cache,logp=logp.numpy());metadata.write_text(json.dumps({'signature':signature,'sha256':file_sha256(cache)}),encoding='utf-8')
        if logp.ndim!=3 or logp.shape[0]!=1 or logp.shape[2]!=len(labels) or not bool(logp.isfinite().all()):raise ValueError('声学缓存格式无效。')
        prediction=greedy_words(logp,labels,begin,end)
        row={'start_ms':begin,'end_ms':end,'greedy_words':prediction,'greedy_unprompted_text':' '.join(w['text'] for w in prediction),'emissions_file':cache.name,'emissions_sha256':file_sha256(cache)}
        try:row['aligned']=align_window(logp,labels,package['words'],begin,end)
        except Exception as error:row['error']=type(error).__name__
        result['windows'].append(row)
        temporary=output/'survey.part.json';temporary.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');temporary.replace(output/'survey.json')
        progress(f'整课原声测量 {number+1}/{len(windows)}')
    return result
