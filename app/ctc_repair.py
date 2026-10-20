"""Conservative local CTC timing repair. No network and no transcript deletion."""
from __future__ import annotations
import array
import copy
import json
import math
import re
import subprocess
import wave
from pathlib import Path
from .sync_audit import MODEL_FILENAME, MODEL_NAME, MODEL_URL, file_sha256, digest, normalize_ctc

MIN_PATH_SCORE = .8  # Operational rejection threshold, NOT calibrated accuracy.
MAX_WINDOW_DIFFERENCE_MS = 80


def _spoken(text):
    if re.search(r'[\d=<>+*/\\_^%{}\[\]|&@#$~]|\s-\s', text):
        return None
    return normalize_ctc(text)


def _context_is_unique(words, index):
    normalized=[_spoken(w['text']) for w in words]
    a=max(0,index-2); b=min(len(words),index+3)
    key=normalized[a:b]
    if len(key)<3 or any(value is None for value in key):
        return False
    count=sum(normalized[start:start+len(key)]==key for start in range(len(words)-len(key)+1))
    return count==1


def build_measured_transaction(words, index, measurements):
    """Validate both acoustic observations and preserve untouched neighbors."""
    reasons=[]
    if len(measurements)!=2 or any(m.get('status')!='measured' for m in measurements):
        return {'status':'rejected','reasons':['两种原音窗口未均完成独立CTC测量'],'patch':[]}
    observations=[{w['index']:w for w in m['aligned_context']} for m in measurements]
    if not _context_is_unique(words,index):
        reasons.append('目标前后连续上下文非唯一或含不支持的数字/代码，需独立复核')
    local=[_spoken(w['text']) for w in words[max(0,index-4):index+5]]
    if any(a is not None and a==b for a,b in zip(local,local[1:])):
        reasons.append('相邻重复词须独立ASR共识或人工听辨，本工具不自动删词或拉长')
    if index not in observations[0] or index not in observations[1]:
        return {'status':'rejected','reasons':reasons+['目标没有完整CTC字符路径'],'patch':[]}
    if index==0 or index==len(words)-1:
        reasons.append('结构修复须有目标两侧的邻词实测锚')
    start=max(0,index-1); end=min(len(words)-1,index+1)
    wider=observations[1]
    # Expand the joint transaction until its measured ends cannot overlap any
    # unchanged word. Never clip a measured boundary to manufacture order.
    while start>0:
        if start not in wider:break
        if wider[start]['measured_start_ms']>=words[start-1]['end_ms']:break
        start-=1
    while end<len(words)-1:
        if end not in wider:break
        if wider[end]['measured_end_ms']<=words[end+1]['start_ms']:break
        end+=1
    patch=[]
    for i in range(start,end+1):
        first=observations[0].get(i); second=observations[1].get(i)
        if first is None or second is None:
            reasons.append(f'邻词{i}缺少双窗实测，不能人为截断其他词')
            continue
        if _spoken(words[i]['text']) is None:
            reasons.append(f'邻词{i}含未确认读法的数字/代码')
        numeric=[first.get(k) for k in ['measured_start_ms','measured_end_ms','raw_path_score_not_accuracy']]+[second.get(k) for k in ['measured_start_ms','measured_end_ms','raw_path_score_not_accuracy']]
        if any(not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) for value in numeric):
            reasons.append(f'邻词{i}CTC证据包含无效数值')
            continue
        span=second['measured_end_ms']-second['measured_start_ms']
        difference=max(abs(first['measured_start_ms']-second['measured_start_ms']),abs(first['measured_end_ms']-second['measured_end_ms']))
        if difference>MAX_WINDOW_DIFFERENCE_MS:reasons.append(f'邻词{i}双窗差{difference:.2f}ms超过80ms')
        if min(first['raw_path_score_not_accuracy'],second['raw_path_score_not_accuracy'])<MIN_PATH_SCORE:reasons.append(f'邻词{i}CTC路径低分；0.8不是准确率阈值')
        if not 19<=span<=1500:reasons.append(f'邻词{i}CTC跨度异常{span:.2f}ms')
        patch.append({'word_index':i,'word_id':words[i].get('id'),'text':words[i]['text'],
                      'original_start_ms':words[i]['start_ms'],'original_end_ms':words[i]['end_ms'],
                      'start_ms':round(second['measured_start_ms']),'end_ms':round(second['measured_end_ms']),
                      'two_window_max_difference_ms':round(difference,2),'raw_path_score_not_accuracy':second['raw_path_score_not_accuracy']})
    if patch and len(patch)==end-start+1:
        for previous,current in zip(patch,patch[1:]):
            if previous['end_ms']>current['start_ms']:reasons.append('实际测量交易内部仍有重叠')
        if start>0 and patch[0]['start_ms']<words[start-1]['end_ms']:reasons.append('交易左侧与未修邻词重叠')
        if end+1<len(words) and patch[-1]['end_ms']>words[end+1]['start_ms']:reasons.append('交易右侧与未修邻词重叠')
    if reasons:return {'status':'rejected','reasons':list(dict.fromkeys(reasons)),'patch':[],'rejected_measured_candidate':patch}
    return {'status':'measured_transaction','reasons':[],'patch':patch,'source':'independent_ctc_two_window_measurement','human_ground_truth':False}


def _target_indices(words,disputes):
    if disputes is None:
        return [i for i,w in enumerate(words) if w['end_ms']<=w['start_ms'] or i>0 and words[i-1]['end_ms']-w['start_ms']>40]
    result=[]
    for dispute in disputes:
        if isinstance(dispute,int):index=dispute
        elif isinstance(dispute,dict):
            index=dispute.get('word_index',dispute.get('index'))
            if index is None and dispute.get('word_id') is not None:
                index=next((i for i,w in enumerate(words) if w.get('id')==dispute['word_id']),None)
            if index is None and 'start_ms' in dispute:
                matches=[i for i,w in enumerate(words) if w['start_ms']==dispute['start_ms'] and (not dispute.get('text') or w['text']==dispute['text'])]
                index=matches[0] if len(matches)==1 else None
        else:index=None
        if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<len(words):
            raise ValueError('CTC争议必须用唯一有效的word_index、word_id或start_ms+text定位。')
        if index not in result:result.append(index)
    return result


def _measure_context(audio,words,index,padding,folder,torch,torchaudio,model,bundle,labels,audio_hash):
    center=words[index]
    begin=max(0,center['start_ms']-padding); end=max(center['start_ms']+100,center['end_ms'])+padding
    clip=folder/(digest([audio_hash,begin,end])[:20]+'.wav')
    if not clip.is_file():
        subprocess.run(['ffmpeg','-v','error','-ss',str(begin/1000),'-i',str(audio),'-t',str((end-begin)/1000),'-ar','16000','-ac','1','-c:a','pcm_s16le','-y',str(clip)],check=True,capture_output=True,timeout=45)
    observation={'padding_ms':padding,'clip_file':str(clip),'clip_start_ms':begin,'clip_end_ms':end,'clip_sha256':file_sha256(clip),'human_ground_truth':False}
    try:
        with wave.open(str(clip),'rb') as stream:
            rate=stream.getframerate();pcm=array.array('h',stream.readframes(stream.getnframes()))
        context=[i for i,w in enumerate(words) if w['start_ms']>=begin and w['end_ms']<=end]
        target=[len(labels)];ranges={}
        for i in context:
            normalized=_spoken(words[i]['text'])
            if normalized is None:
                raise ValueError('连续上下文含数字或代码且无明确读法，拒绝强制省略后修正')
            if target[-1]!=len(labels):target.append(labels['|'])
            a=len(target);target.extend(labels[c] for c in normalized.replace(' ','|'));ranges[i]=(a,len(target))
        target.append(len(labels))
        with torch.inference_mode():
            emissions,_=model(torch.tensor(pcm,dtype=torch.float32).div_(32768).unsqueeze(0))
            probabilities=torch.log_softmax(emissions,dim=-1)
            probabilities=torch.cat((probabilities,torch.zeros((*probabilities.shape[:2],1),dtype=probabilities.dtype)),dim=-1)
            path,scores=torchaudio.functional.forced_align(probabilities,torch.tensor([target],dtype=torch.int64),blank=0)
            spans=torchaudio.functional.merge_tokens(path[0],scores[0],blank=0)
        if len(spans)!=len(target):raise ValueError('CTC路径没有完整对应连续上下文')
        frame_ms=len(pcm)/rate*1000/probabilities.shape[1]
        aligned=[]
        for i,(a,b) in ranges.items():
            confidence=math.exp(sum(s.score*(s.end-s.start) for s in spans[a:b])/sum(s.end-s.start for s in spans[a:b]))
            aligned.append({'index':i,'text':words[i]['text'],'measured_start_ms':round(begin+spans[a].start*frame_ms,2),'measured_end_ms':round(begin+spans[b-1].end*frame_ms,2),'raw_path_score_not_accuracy':round(confidence,5)})
        observation.update(status='measured',frame_resolution_ms=round(frame_ms,3),normalized_context=' '.join(_spoken(words[i]['text']) for i in context),aligned_context=aligned)
    except Exception as error:observation.update(status='not_measurable',error=type(error).__name__+': '+str(error)[:900])
    return observation


def repair_with_ctc(audio, words, disputes=None, progress=print, *, cache_dir):
    """Return a copy plus evidence. Rejected/uncached cases keep original timings.

    disputes: indices, or dictionaries with word_index / word_id / start_ms+text.
    Only timing fields change. No network, ASR guesses, timestamp interpolation,
    or deletion. Callers should retain evidence with their versioned package.
    """
    audio=Path(audio);cache=Path(cache_dir);weights=cache/MODEL_FILENAME
    original=copy.deepcopy(words);new_words=copy.deepcopy(words)
    targets=_target_indices(words,disputes)
    evidence={'version':1,'algorithm':'conservative-ctc-joint-neighbor-transaction-v1','model':MODEL_NAME,'model_source':MODEL_URL,
              'human_ground_truth':False,'score_is_calibrated_accuracy':False,'input_words_sha256':digest(words),
              'thresholds':{'minimum_path_score_not_accuracy':MIN_PATH_SCORE,'maximum_window_boundary_difference_ms':MAX_WINDOW_DIFFERENCE_MS},'rows':[],'applied_transactions':0}
    if not targets:evidence['status']='no_disputes';return new_words,evidence
    if not weights.is_file():evidence.update(status='model_not_cached',reason='未缓存独立CTC权重；不发起网络请求，原词保持待检查');return new_words,evidence
    try:
        import torch,torchaudio
        torch.set_num_threads(4)
        bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model=bundle.get_model(dl_kwargs={'model_dir':str(cache),'progress':False}).eval()
        labels={label:i for i,label in enumerate(bundle.get_labels())}
        audio_hash=file_sha256(audio)
        evidence.update(model_sha256=file_sha256(weights),audio_sha256=audio_hash)
        folder=cache/'repair-evidence'/digest([audio_hash,evidence['input_words_sha256']])[:16]
        folder.mkdir(parents=True,exist_ok=True)
    except Exception as error:evidence.update(status='runtime_unavailable',reason=type(error).__name__+': '+str(error)[:900]);return new_words,evidence
    for index in targets:
        progress(f'独立CTC结构复核：{index} {words[index]["text"]}')
        measured=[]
        for padding in [2500,4500]:
            try:
                observation=_measure_context(audio,original,index,padding,folder,torch,torchaudio,model,bundle,labels,audio_hash)
            except Exception as error:
                observation={'status':'not_measurable','padding_ms':padding,'error':type(error).__name__+': '+str(error)[:900]}
            measured.append(observation)
        transaction=build_measured_transaction(original,index,measured)
        row={'word_index':index,'text':words[index]['text'],'measurements':measured,'transaction':transaction,'human_ground_truth':False}
        if transaction['status']=='measured_transaction':
            changed=[patch for patch in transaction['patch'] if new_words[patch['word_index']]['start_ms']!=patch['original_start_ms'] or new_words[patch['word_index']]['end_ms']!=patch['original_end_ms']]
            if changed:
                transaction.update(status='rejected',reasons=['与此前修正交易冲突；保留已有结果，不能混用旧锚'],rejected_measured_candidate=transaction['patch'],patch=[])
            else:
                for patch in transaction['patch']:
                    new_words[patch['word_index']].update(start_ms=patch['start_ms'],end_ms=patch['end_ms'])
                evidence['applied_transactions']+=1
        evidence['rows'].append(row)
    evidence['status']='measured'
    evidence['output_words_sha256']=digest(new_words)
    return new_words,evidence
