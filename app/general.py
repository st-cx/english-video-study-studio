"""Source-independent, resumable English media production.

Only measured media timestamps enter the package. Provider output is teaching
content; it never supplies timing or silently replaces a disputed transcript.
"""
from __future__ import annotations
import copy,csv,difflib,hashlib,json,math,os,re,shutil,subprocess,tempfile
from datetime import datetime,timezone
from pathlib import Path
from .storage import output_root,save_episode,get_settings

PROCESSING_VERSION='general-4.4'

def _signature(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def _read(path:Path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return default

def _global_speech_record(record:dict)->dict:
    """Migrate diagnostics once; the selected word intervals are already global."""
    if record.get('time_basis')=='media':return record
    result=copy.deepcopy(record)
    offset=int(result.get('clip_start_ms',max(0,int(result['start_ms'])-1500)))
    span_keys={'aligned_ms','asr_ms','window_asr_ms','measured_ms','original_ms'}
    def shift(value):
        if isinstance(value,dict):
            return {key:([int(v)+offset for v in item] if key in span_keys and isinstance(item,list)
                         and all(isinstance(v,(int,float)) for v in item)
                         else int(item)+offset if key in {'start_ms','end_ms'} and isinstance(item,(int,float))
                         else shift(item)) for key,item in value.items()}
        if isinstance(value,list):return [shift(item) for item in value]
        return value
    result['alignment']=shift(result.get('alignment',{}))
    for word in result.get('words',[]):
        if isinstance(word.get('whisper_segment'),(int,float)):
            word['whisper_segment']=round(word['whisper_segment']+offset/1000,3)
    for word in result.get('raw_asr_words',[]):
        word['start_ms']+=offset;word['end_ms']+=offset
        if isinstance(word.get('whisper_segment'),(int,float)):
            word['whisper_segment']=round(word['whisper_segment']+offset/1000,3)
    result.update(clip_start_ms=offset,time_basis='media',index_basis='clip')
    return result

def _boundaries(audio:Path,duration:float,folder:Path) -> list[float]:
    """Prefer quiet boundaries near five minutes, without deleting pauses."""
    cached=_read(folder/'cuts.json')
    if cached and cached.get('duration')==duration:return cached['cuts']
    quiet=[]
    if duration>300:
        result=subprocess.run(['ffmpeg','-hide_banner','-i',str(audio),'-af','silencedetect=noise=-35dB:d=0.4','-f','null','-'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=max(120,int(duration)))
        starts=[float(x) for x in re.findall(r'silence_start: ([\d.]+)',result.stderr)]
        ends=[float(x) for x in re.findall(r'silence_end: ([\d.]+)',result.stderr)]
        quiet=[(a+b)/2 for a,b in zip(starts,ends) if b>=a]
    cuts=[0.0]
    while duration-cuts[-1]>330:
        target=cuts[-1]+300
        choices=[q for q in quiet if target-15<=q<=target+15]
        cuts.append(min(choices,key=lambda q:abs(q-target)) if choices else target)
    cuts.append(duration)
    save_episode({'duration':duration,'cuts':cuts},folder,'cuts.json')
    return cuts


def _alignment_implementation_fingerprint()->str:
    """Invalidate only speech checkpoints when the installed aligner changes."""
    from . import alignment
    return hashlib.sha256(Path(alignment.__file__).read_bytes()).hexdigest()


def _cpu_runtime_identity(model_path:str)->dict:
    path=Path(model_path)
    if str(model_path)==path.name:
        name=str(model_path);snapshot=''
    else:
        model_folder=next((p.name for p in path.parents if p.name.startswith('models--Systran--faster-whisper-')),'')
        name=model_folder.removeprefix('models--Systran--faster-whisper-') if model_folder else 'local_snapshot'
        snapshot=path.name
    return {'model':name,'snapshot':snapshot,'device':'cpu','compute_type':'int8'}


def _transcribe_chunks(audio:Path,duration:float,folder:Path,fingerprint:str,progress):
    from .pipeline import _local_whisper_path
    from .alignment import align_words,group_connected_speech
    import stable_whisper
    from .speech_runtime import speech_runtime_plan
    plan=speech_runtime_plan()
    alignment_fingerprint=_alignment_implementation_fingerprint()
    runtime={**{k:v for k,v in plan.items() if k!='path'},'alignment_implementation_sha256':alignment_fingerprint}
    chunks=folder/'.speech';chunks.mkdir(exist_ok=True)
    cuts=_boundaries(audio,duration,chunks)
    output=[];records=[];model=None;backend='CPU'
    for index,(start,end) in enumerate(zip(cuts,cuts[1:])):
        progress('transcribe',10+int(index/max(1,len(cuts)-1)*60),f'识别并对齐原声 {index+1} / {len(cuts)-1} 段')
        key=_signature([PROCESSING_VERSION,fingerprint,start,end,runtime])
        checkpoint=chunks/f'{index:04}.json';saved=_read(checkpoint)
        valid=(isinstance(saved,dict) and isinstance(saved.get('runtime'),dict)
               and saved['runtime'].get('alignment_implementation_sha256')==alignment_fingerprint
               and saved.get('key')==key)
        if not valid:
            if model is None:
                try:
                    model=stable_whisper.load_faster_whisper(plan['path'],device=plan['device'],compute_type=plan['compute_type'])
                    backend='GPU' if plan['device']=='cuda' else 'CPU'
                    runtime={**{k:v for k,v in plan.items() if k!='path'},'alignment_implementation_sha256':alignment_fingerprint}
                except Exception:
                    cpu_path=_local_whisper_path()
                    model=stable_whisper.load_faster_whisper(cpu_path,device='cpu',compute_type='int8');backend='CPU'
                    runtime={**_cpu_runtime_identity(cpu_path),'alignment_implementation_sha256':alignment_fingerprint}
            clip_start=max(0,start-1.5);clip_end=min(duration,end+1.5)
            clip=chunks/f'{index:04}.wav'
            subprocess.run(['ffmpeg','-v','error','-ss',str(clip_start),'-i',str(audio),'-t',str(clip_end-clip_start),'-ar','16000','-ac','1','-y',str(clip)],check=True,capture_output=True,timeout=120)
            def run_asr():
                segments,info=model.transcribe_original(str(clip),language=None,word_timestamps=True,vad_filter=True,beam_size=5,condition_on_previous_text=False)
                words=[];suspects=[]
                for segment in segments:
                    if segment.no_speech_prob>.8 and segment.avg_logprob<-1:
                        suspects.append({'start_ms':round((clip_start+segment.start)*1000),'end_ms':round((clip_start+segment.end)*1000),'reason':'低可信语音片段，未写入字幕'})
                        continue
                    if segment.compression_ratio>2.8:
                        suspects.append({'start_ms':round((clip_start+segment.start)*1000),'end_ms':round((clip_start+segment.end)*1000),'reason':'可能重复识别，请回听'})
                    for w in segment.words or []:
                        if w.word.strip():words.append({'text':w.word.strip(),'start_ms':round(w.start*1000),'end_ms':round(w.end*1000),'probability':round(w.probability,4),'whisper_segment':round(segment.start,2)})
                return words,info,suspects
            try:words,info,suspects=run_asr()
            except Exception:
                if backend!='GPU':raise
                cpu_path=_local_whisper_path()
                model=stable_whisper.load_faster_whisper(cpu_path,device='cpu',compute_type='int8');backend='CPU'
                runtime={**_cpu_runtime_identity(cpu_path),'alignment_implementation_sha256':alignment_fingerprint}
                words,info,suspects=run_asr()
            language=info.language;lp=info.language_probability
            raw_asr_words=copy.deepcopy(words)
            alignment={}
            if words and (language=='en' or lp<.6):
                words,alignment=align_words(clip,words,plan['path'],progress,alignment_model=model)
                words,groups=group_connected_speech(words);alignment['connected_speech_groups']=groups
            elif language!='en' and lp>=.6:
                suspects.append({'start_ms':round(start*1000),'end_ms':round(end*1000),'reason':f'本段主要识别语言为 {language}，需要确认英语范围'})
            selected=[]
            for w in words:
                a=w['start_ms']+round(clip_start*1000);b=w['end_ms']+round(clip_start*1000)
                if start*1000<=(a+b)/2<end*1000 or index==len(cuts)-2 and (a+b)/2==end*1000:
                    selected.append({**w,'start_ms':a,'end_ms':b})
            key=_signature([PROCESSING_VERSION,fingerprint,start,end,runtime])
            saved={'key':key,'runtime':dict(runtime),'words':selected,'raw_asr_words':raw_asr_words,'backend':backend,'language':language,'language_probability':lp,'suspects':suspects,'alignment':alignment,'start_ms':round(start*1000),'end_ms':round(end*1000),'clip_start_ms':round(clip_start*1000)}
            saved=_global_speech_record(saved)
            save_episode(saved,chunks,checkpoint.name)
            clip.unlink(missing_ok=True)
        elif saved.get('time_basis')!='media':
            saved=_global_speech_record(saved)
            save_episode(saved,chunks,checkpoint.name)
        # Overlap belongs to one core span. Remove only matching boundary words
        # observed at effectively the same time; never flatten genuine repeats.
        incoming=saved['words']
        if output and incoming:
            for n in range(min(6,len(output),len(incoming)),0,-1):
                left=output[-n:];right=incoming[:n]
                norm=lambda w:re.sub(r'\W+','',w['text'].lower())
                def same_measurement(a,b):
                    overlap=max(0,min(a['end_ms'],b['end_ms'])-max(a['start_ms'],b['start_ms']))
                    shortest=min(a['end_ms']-a['start_ms'],b['end_ms']-b['start_ms'])
                    return shortest>0 and overlap>=shortest*.6 and abs(a['start_ms']-b['start_ms'])<200
                if [norm(w) for w in left]==[norm(w) for w in right] and all(same_measurement(a,b) for a,b in zip(left,right)):
                    incoming=incoming[n:];break
        output.extend(incoming);records.append({k:v for k,v in saved.items() if k not in {'key','words'}})
    del model
    import gc
    gc.collect()
    if not output:raise RuntimeError('没有识别到可用英语人声。请检查音轨或选择包含英语的片段。')
    if any(w['end_ms']<=w['start_ms'] for w in output) or any(a['end_ms']-b['start_ms']>40 for a,b in zip(output,output[1:])) or any(record.get('alignment',{}).get('alignment_disputes') for record in records):
        from .speech_repair import repair_speech_words
        output,records=repair_speech_words(audio,output,records,folder,lambda message:progress('align',74,message))
    return output,records

def _chapters(sentences:list[dict],duration:float,source_chapters=None) -> list[dict]:
    starts=[0]
    # Media chapter metadata is optional; only boundaries aligned to a measured
    # sentence are accepted, so a lesson never cuts a sentence in half.
    targets=[]
    for c in source_chapters or []:
        if not isinstance(c,dict):continue
        try:
            value=float(c.get('start_time',c.get('start',0)))*1000
            if 1000<value<duration*1000:targets.append(value)
        except (ValueError,TypeError):continue
    if targets:
        for target in targets:
            sentence=min(sentences,key=lambda s:abs(s['start_ms']-target))
            if sentence['start_ms']>starts[-1]+1000:starts.append(sentence['start_ms'])
    else:
        for s in sentences:
            if s['start_ms']-starts[-1]>=300000:starts.append(s['start_ms'])
    if len(starts)>1 and duration*1000-starts[-1]<30000:starts.pop()
    result=[{'id':f'ch{i+1:03}','title':f'学习单元 {i+1}','start_ms':start,'end_ms':starts[i+1] if i+1<len(starts) else round(duration*1000),'status':'pending'} for i,start in enumerate(starts)]
    for chapter in result:
        candidates=[]
        for original in source_chapters or []:
            try:
                distance=abs(float(original.get('start_time',original.get('start',0)))*1000-chapter['start_ms'])
                if original.get('title') and distance<30000:candidates.append((distance,str(original['title'])[:200]))
            except (ValueError,TypeError,AttributeError):continue
        if candidates:chapter['title']=min(candidates)[1]
    return result

def _chapter_package(package:dict,chapter:dict)->dict:
    result=copy.deepcopy(package)
    result['sentences']=[s for s in result['sentences'] if chapter['start_ms']<=s['start_ms']<chapter['end_ms']]
    ids={s['id'] for s in result['sentences']}
    result['words']=[w for w in result['words'] if w['sentence_id'] in ids]
    result['cards']=[];result.pop('phrases',None);result.pop('layout',None)
    return result


def _teaching_source_identity(package:dict)->str:
    return _signature({
        'sentences':[(sentence['id'],sentence['text'],sentence['start_ms'],sentence['end_ms'],sentence.get('word_ids',[])) for sentence in package['sentences']],
        'words':[(word['id'],word['sentence_id'],word['text'],word['start_ms'],word['end_ms']) for word in package['words']],
    })


def _cached_teaching(part:dict,chapter_folder:Path,revision:str):
    """Reuse teaching only for the same English IDs and measured intervals.

    Alignment revisions have separate finished output folders. Their unchanged
    chapters may share teaching, while current ASR probabilities, source metadata
    and audio diagnostics always remain from the current package.
    """
    candidates=[chapter_folder/'teaching.json']
    base_id=part['episode']['id'].split('-r',1)[0]
    if re.fullmatch(r'media-[0-9a-f]{20}',base_id):
        root=output_root()
        relatives=[root/base_id]+sorted(root.glob(base_id+'-r*'))
        for folder in relatives:
            candidate=folder/'.chapters'/chapter_folder.name/'teaching.json'
            if candidate not in candidates:candidates.append(candidate)
    identity=_teaching_source_identity(part)
    for path in candidates:
        cached=_read(path)
        if not isinstance(cached,dict) or cached.get('_revision')!=revision:
            continue
        try:
            if _teaching_source_identity(cached)!=identity:continue
        except (KeyError,TypeError):
            continue
        result=copy.deepcopy(part)
        sentence_by_id={sentence['id']:sentence for sentence in cached['sentences']}
        for sentence in result['sentences']:
            for field in ('translation','card_ids'):
                if field in sentence_by_id[sentence['id']]:sentence[field]=copy.deepcopy(sentence_by_id[sentence['id']][field])
        word_by_id={word['id']:word for word in cached['words']}
        for word in result['words']:
            for field in ('gloss_zh','focus'):
                if field in word_by_id[word['id']]:word[field]=copy.deepcopy(word_by_id[word['id']][field])
        for field in ('cards','phrases','layout','teaching_review','editor_review','editor_corrections'):
            if field in cached:result[field]=copy.deepcopy(cached[field])
        result['episode']['title_zh']=cached.get('episode',{}).get('title_zh',part['episode'].get('title_zh',''))
        for field in ('example_excerpts','teaching_format_repairs','phrase_id_repairs','review_adjudications'):
            if field in cached.get('processing',{}):result.setdefault('processing',{})[field]=copy.deepcopy(cached['processing'][field])
        result['_revision']=revision
        if path!=candidates[0]:save_episode(result,chapter_folder,'teaching.json')
        return result
    return None


def _schedule_chapter_cards(cards:list[dict],chapter:dict)->list[dict]:
    """Schedule reading time inside one chapter; every card stays in notes."""
    result=copy.deepcopy(cards)
    begin,end=int(chapter['start_ms']),int(chapter['end_ms'])
    visible=[]
    for card in sorted(result,key=lambda value:int(value['start_ms'])):
        card.setdefault('original_display_start_ms',card['start_ms'])
        card.setdefault('original_display_end_ms',card['end_ms'])
        start=max(begin,int(card['start_ms']))
        spaced=not visible or start-visible[-1]['start_ms']>=8000
        if card.get('display',True) and spaced and end-start>=8000:
            card.update(start_ms=start,display=True);visible.append(card)
        else:
            # A late or already-hidden card remains available in the study notes.
            card['display']=False
            card['start_ms']=min(end,max(begin,int(card['start_ms'])))
            card['end_ms']=min(end,max(card['start_ms'],int(card['end_ms'])))
    for index,card in enumerate(visible):
        card['end_ms']=visible[index+1]['start_ms'] if index+1<len(visible) else end
    return result


def _merge_chapter(package,part,chapter):
    ids={s['id'] for s in part['sentences']}
    originals={s['id']:s for s in part['sentences']}
    package['sentences']=[originals.get(s['id'],s) for s in package['sentences']]
    words={w['id']:w for w in part['words']}
    package['words']=[words.get(w['id'],w) for w in package['words']]
    prefix=chapter['id']+'-'
    for s in part['sentences']:s['card_ids']=[prefix+x for x in s.get('card_ids',[])]
    package['cards']=[c for c in package['cards'] if not set(c['sentence_ids'])&ids]
    package['cards'].extend({**c,'id':prefix+c['id']} for c in _schedule_chapter_cards(part['cards'],chapter))
    package['phrases']=[p for p in package.get('phrases',[]) if p['sentence_id'] not in ids]+[{**p,'id':prefix+p['id']} for p in part.get('phrases',[])]
    pages=package.setdefault('layout',{'version':1,'pages':[]})['pages']
    pages[:]=[p for p in pages if p['sentence_id'] not in ids]+[{**p,'id':prefix+p['id']} for p in part.get('layout',{}).get('pages',[])]
    pages.sort(key=lambda p:p['start_ms'])
    package['episode']['title_zh']=part['episode'].get('title_zh') or package['episode'].get('title_zh','')

def _write_bundle(package,folder,progress):
    original_package=package
    package=copy.deepcopy(package)
    from .study_sections import ensure_study_sections
    ensure_study_sections(package,folder,progress)
    from .pipeline import make_notes
    from .subtitles import make_captions
    from .study_page import write_study_page
    from .media import _publish
    from .renderer import audit_layout
    with tempfile.TemporaryDirectory(prefix='.bundle-',dir=folder) as temporary:
        staged=Path(temporary)
        en,zh,ass=make_captions(package,staged)
        (staged/'study-notes.md').write_text(make_notes(package),encoding='utf-8-sig')
        with (staged/'vocabulary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['词语','语境释义','原声秒数']);seen=set()
            for w in package['words']:
                if w.get('focus') and w.get('gloss_zh') and w['text'].lower() not in seen:
                    seen.add(w['text'].lower());writer.writerow([("'" if re.match(r'^[=+@-]',w['text']) else '')+w['text'],w['gloss_zh'],w['start_ms']/1000])
        package.setdefault('render',{}).update(english_subtitles=en.name,chinese_subtitles=zh.name,ass_file=ass.name,study_notes='study-notes.md')
        review_files=[]
        if any(c.get('reference_issues') or c.get('alignment_issues') for c in package.get('chapters',[])):
            from .issue_review import write_issue_review
            package['issue_review']=write_issue_review(package,staged,media_folder=folder)
            review_files=['issue-review/review.html']
        else:
            package.pop('issue_review',None)
        package.setdefault('artifacts',{})['study']='pending'
        save_episode(package,staged)
        package.setdefault('processing',{})['layout_audit']=audit_layout(staged/'episode.json',folder,progress)
        package['artifacts']['study']='ready'
        write_study_page(package,staged)
        save_episode(package,staged)
        _publish(staged,folder,[en.name,zh.name,ass.name,'study-notes.md','vocabulary.csv','study.html',*review_files,'episode.json'])
    original_package.clear();original_package.update(package)
    progress('study',94,'离线学习页、字幕和精读笔记已生成')

def _process_chapters(package,folder,reference,cues,progress,only=None):
    from .pipeline import _annotate,_quality
    from .teaching_review import review_teaching
    from .layout import prepare_layout
    from .providers import ProviderError
    # Compare the whole spoken stream once. A subtitle cue may contain several
    # sentences across a chapter boundary; its complete text is context, not a
    # second copy of words missing from each neighbouring chapter.
    from .reference_resolution import resolve_reference_findings,unresolved_alignment
    from .speech_runtime import speech_runtime_plan
    plan=speech_runtime_plan()
    reference_review=resolve_reference_findings(package,reference,folder/'audio.mp3',model_path=plan['path'])
    package['processing']['reference_review']={k:v for k,v in reference_review.items() if k!='observations'}
    reference_findings=reference_review['unresolved']
    alignment_findings=unresolved_alignment(package)
    for chapter in package['chapters']:
        if only and chapter['id']!=only:continue
        if chapter['status']=='ready':continue
        progress('teaching',78,f"正在处理{chapter['title']}")
        part=_chapter_package(package,chapter)
        local_ref=' '.join(c['text'] for c in cues if c['start_ms']<chapter['end_ms'] and c['end_ms']>chapter['start_ms']) if cues else (reference if len(package['chapters'])==1 else '')
        quality=_quality(part,'')
        quality.update(reference_available=bool(reference),reference_scope='global_audio_anchors')
        lexical=[issue for issue in reference_findings
                 if issue['start_ms']<chapter['end_ms'] and issue['end_ms']>chapter['start_ms']]
        chapter['reference_issues']=lexical
        issues=[s['reason'] for record in package['processing'].get('speech_chunks',[]) for s in record.get('suspects',[]) if s['start_ms']<chapter['end_ms'] and s['end_ms']>chapter['start_ms']]+[x['reason'] for x in lexical]
        if package['processing'].get('global_reference_dispute'):
            issues.append('全片参考稿与识别内容差异较大，请核对稿件后确认本单元。')
        disputes=[d for d in alignment_findings if d['start_ms']<chapter['end_ms'] and d['end_ms']>chapter['start_ms']]
        chapter['alignment_issues']=disputes
        if disputes:issues.append('独立识别与对齐仍有差异，请回听本单元对应位置。')
        if quality['structural_flags'] or not chapter.get('approved') and (quality['needs_review'] or issues):
            chapter.update(status='review',flags=quality['flags']+issues,quality=quality)
            continue
        chapter_folder=folder/'.chapters'/chapter['id'];chapter_folder.mkdir(parents=True,exist_ok=True)
        revision=_signature([PROCESSING_VERSION,part['sentences'],[(w['id'],w['text'],w['start_ms'],w['end_ms']) for w in part['words']],get_settings()['model'],get_settings()['base_url']])
        cached=_cached_teaching(part,chapter_folder,revision)
        if cached:
            part=cached
        else:
            try:
                request_cache=output_root()/package['episode']['id'].split('-r')[0]/'.teaching-cache'
                _annotate(part,local_ref,progress,request_cache)
                review_teaching(part,progress,request_cache)
                prepare_layout(part,folder)
            except ProviderError as exc:
                save_episode(part,chapter_folder,'teaching.partial.json')
                if part.get('teaching_review',{}).get('status')=='needs_review':
                    chapter.update(status='review',flags=[str(exc)],quality=quality);continue
                save_episode(package,folder);raise
            part['_revision']=revision;save_episode(part,chapter_folder,'teaching.json')
        _merge_chapter(package,part,chapter)
        chapter.update(status='ready',flags=[],quality=quality,teaching_review=part.get('teaching_review',{}))
        save_episode(package,folder)
    global_quality=_quality(package,reference)
    structural=[];duration_ms=round(package['episode']['duration_s']*1000)
    for w in package['words']:
        if w['end_ms']<=w['start_ms'] or w['start_ms']<0 or w['end_ms']>duration_ms+200:
            structural.append({'start_ms':w['start_ms'],'end_ms':max(w['start_ms']+1,w['end_ms']),'reason':'词语时间无效或越界'})
    for a,b in zip(package['words'],package['words'][1:]):
        if a['end_ms']-b['start_ms']>80:structural.append({'start_ms':b['start_ms'],'end_ms':a['end_ms'],'reason':'相邻词语时间重叠，含章节边界'})
    for chapter in package['chapters']:
        reasons=[x['reason'] for x in structural if x['start_ms']<chapter['end_ms'] and x['end_ms']>chapter['start_ms']]
        if reasons:chapter.update(status='review',flags=list(dict.fromkeys(chapter.get('flags',[])+reasons)))
    flags=[f"{c['title']}：{f}" for c in package['chapters'] for f in c.get('flags',[])]
    package['quality']={'needs_review':bool(flags),'flags':flags,'structural_flags':global_quality['structural_flags'],'structural_issues':structural,'source_similarity':global_quality['source_similarity'],'reference_available':bool(reference),'review_chapter_ids':[c['id'] for c in package['chapters'] if c['status']=='review']}
    covered={p['sentence_id'] for p in package.get('layout',{}).get('pages',[])}
    missing=[s for s in package['sentences'] if s['id'] not in covered]
    if missing:
        # Unapproved English remains inspectable but is marked as such. Do not
        # fabricate translations or teaching cards just to make a page ready.
        raw_package=copy.deepcopy(package);raw_package.pop('layout',None)
        raw_package['sentences']=missing
        raw=folder/'layout.preparing.json';save_episode(raw_package,folder,raw.name)
        subprocess.run(['node',str(Path(__file__).resolve().parents[1]/'scripts/prepare-layout.mjs'),str(raw)],check=True,capture_output=True,timeout=120)
        pages=package.setdefault('layout',{'version':1,'pages':[]})['pages']
        pages.extend(_read(raw)['layout']['pages']);pages.sort(key=lambda p:p['start_ms']);raw.unlink(missing_ok=True)
    _write_bundle(package,folder,progress)
    if flags:
        from .pipeline import ReviewRequired
        raise ReviewRequired('部分学习单元需要检查，其他单元可先学习。',package)
    return package

def create_media_episode(spec:dict,job_folder:Path,progress)->dict:
    from .media import prepare_media
    from .pipeline import _build_sentences,_quality
    from .subtitle_input import parse_subtitles
    job_folder.mkdir(parents=True,exist_ok=True)
    source_spec=dict(spec)
    if source_spec.get('url') and not source_spec.get('media_path'):
        from .sources import acquire_source
        saved=_read(job_folder/'source.json')
        if saved and Path(saved.get('media_path','')).is_file():source_spec.update(saved)
        else:
            acquired=acquire_source(source_spec['url'],job_folder/'source',progress)
            source_spec.update(acquired);save_episode(acquired,job_folder,'source.json')
    path=Path(source_spec.get('media_path',''))
    if not path.is_file():raise ValueError('找不到导入的媒体，请重新选择文件。')
    media=prepare_media(path,job_folder/'normalized',audio_stream_index=source_spec.get('audio_stream_index'),start_ms=source_spec.get('start_ms',0),end_ms=source_spec.get('end_ms'),progress=progress)
    episode_id='media-'+str(media['fingerprint'])[:20]
    folder=output_root()/episode_id;folder.mkdir(parents=True,exist_ok=True)
    speech_folder=folder
    reference=str(source_spec.get('transcript') or '')
    cues=[];ref_kind=source_spec.get('transcript_kind') or ('provided_text' if reference else 'none');ref_format='text' if reference else ''
    if source_spec.get('subtitle_path'):
        parsed=parse_subtitles(Path(source_spec['subtitle_path']));reference=parsed['text'];cues=parsed.get('cues',[])
        ref_format=parsed.get('kind','text');ref_kind=source_spec.get('transcript_kind') or 'provided_subtitle'
        offset=int(media.get('source_offset_ms',0));end=offset+round(media['duration_s']*1000)
        if cues:
            cues=[{**c,'start_ms':max(0,c['start_ms']-offset),'end_ms':min(end,c['end_ms'])-offset} for c in cues if c['start_ms']<end and c['end_ms']>offset]
            reference=' '.join(c['text'] for c in cues)
    if source_spec.get('source_kind')=='bbc':
        from .transcript import spoken_text
        reference=spoken_text(reference)
    alignment_fingerprint=_alignment_implementation_fingerprint()
    revision=_signature([PROCESSING_VERSION,media['fingerprint'],reference,cues,ref_kind,ref_format,source_spec.get('chapters') or media.get('chapters',[]),get_settings()['model'],get_settings()['base_url'],{'alignment_implementation_sha256':alignment_fingerprint}])
    package=_read(folder/'episode.json')
    if package and package.get('processing',{}).get('revision')!=revision and package.get('artifacts',{}).get('study')=='ready':
        episode_id+='-r'+revision[:8];folder=output_root()/episode_id;folder.mkdir(parents=True,exist_ok=True)
        package=_read(folder/'episode.json')
    if not package or package.get('processing',{}).get('revision')!=revision:
        progress('prepare',8,'正在准备原视频与统一时间轴')
        for source,name in [(media['audio_path'],'audio.mp3'),(media.get('video_path'),'source.mp4'),(media.get('cover_path') or source_spec.get('cover_path'),'cover.jpg')]:
            if source and Path(source).is_file():
                tmp=folder/(name+'.part');shutil.copy2(source,tmp);os.replace(tmp,folder/name)
        raw,records=_transcribe_chunks(folder/'audio.mp3',media['duration_s'],speech_folder,media['fingerprint'],progress)
        from .transcript import normalize_orthography
        raw,orthography=normalize_orthography(raw,reference) if reference else (raw,[])
        repairs=[]
        if reference:
            from .acoustic_reference import reconcile
            from .pipeline import _local_whisper_path
            raw,repairs=reconcile(folder/'audio.mp3',raw,reference,_local_whisper_path(),progress)
        sentences,words=_build_sentences(raw)
        from .privacy import public_source_url
        kind=source_spec.get('source_kind','local');url=public_source_url(source_spec.get('source_url') or source_spec.get('url') or '')
        title=str(source_spec.get('title') or path.stem)[:200]
        package={'version':4,'episode':{'id':episode_id,'title':title,'title_zh':'','source_url':url,'audio_url':'','duration_s':media['duration_s'],'source_date':'','cover':'cover.jpg' if (folder/'cover.jpg').exists() else '', 'audio_file':'audio.mp3','fingerprint':media['fingerprint'],'created_at':datetime.now(timezone.utc).isoformat()},'source':{'kind':kind,'name':{'bbc':'BBC Learning English','youtube':'YouTube','bilibili':'哔哩哔哩','direct':'媒体链接'}.get(kind,'本地视频' if media.get('video_path') else '本地音频'),'url':url,'transcript_kind':ref_kind},'media':{'video_file':'source.mp4' if media.get('video_path') else '', 'audio_file':'audio.mp3','duration_s':media['duration_s'],'source_offset_ms':media.get('source_offset_ms',0),'audio_stream_index':source_spec.get('audio_stream_index')},'sentences':sentences,'words':words,'cards':[],'quality':{},'review':{'official_transcript':reference,'recognized_transcript':' '.join(s['text'] for s in sentences),'approved':False},'render':{'width':1920,'height':1080,'fps':30,'status':'prepared'},'artifacts':{'study':'pending','video':'not_requested'},'processing':{'version':PROCESSING_VERSION,'revision':revision,'speech_chunks':records},'chapters':_chapters(sentences,media['duration_s'],source_spec.get('chapters'))}
        package['source']['reference_format']=ref_format
        if isinstance(source_spec.get('source_metadata'),dict):
            package['source']['metadata']={k:str(v)[:500] for k,v in source_spec['source_metadata'].items() if k in {'title','uploader','channel','license','upload_date'}}
        package['processing']['alignment_implementation_sha256']=alignment_fingerprint
        package['processing']['reference_orthography']=orthography
        if source_spec.get('chapters') or media.get('chapters'):
            package['chapters']=_chapters(sentences,media['duration_s'],source_spec.get('chapters') or media.get('chapters'))
        package['processing']['reference_repairs']=repairs
        package['media']['audio_stream_index']=media.get('audio_stream_index')
        if kind in {'media','direct'}:package['source']['name']='媒体链接'
        package['quality']=_quality(package,reference)
        if reference and not cues and len(package['chapters'])>1 and (package['quality'].get('source_similarity') or 0)<.68:
            package['processing']['global_reference_dispute']=True
        save_episode(package,folder)
        save_episode({'reference':reference,'cues':cues},folder,'.reference.json')
    package.setdefault('processing',{})['render_mode']=spec.get('render_mode','auto')
    if reference:
        from .reference_resolution import resolve_reference_findings
        from .speech_runtime import speech_runtime_plan
        plan=speech_runtime_plan()
        measured=resolve_reference_findings(package,reference,folder/'audio.mp3',model_path=plan['path'],
            progress=progress,cache_dir=folder/'.reference-observations',observe=True,
            device=plan['device'],compute_type=plan['compute_type'])
        package['processing']['reference_observations']=measured['observations']
        save_episode(package,folder)
    _process_chapters(package,folder,reference,cues,progress)
    mode=spec.get('render_mode','auto')
    if mode=='video' or mode=='auto' and package['episode']['duration_s']<=1200:
        if not (folder/'lesson.mp4').is_file() or package['artifacts'].get('video')!='ready':
            from .renderer import export_video
            export_video(folder/'episode.json',folder,progress)
            package=_read(folder/'episode.json')
    progress('complete',100,'学习包已生成，可打开精读精听')
    return package

def continue_media_review(package,folder,progress,corrected_transcript='',chapter_id=''):
    from .pipeline import _realign_corrected,_quality,ReviewRequired
    candidates=[c for c in package['chapters'] if c['status']=='review' and (not chapter_id or c['id']==chapter_id)]
    if not candidates:raise ValueError('未找到需要检查的学习单元。')
    if corrected_transcript.strip() and len(candidates)!=1:
        raise ValueError('修改文字时请先选择一个学习单元，避免覆盖其他章节。')
    for chapter in candidates:
        part=_chapter_package(package,chapter)
        original=' '.join(s['text'] for s in part['sentences'])
        if corrected_transcript.strip() and corrected_transcript.strip()!=original:
            old_ids={s['id'] for s in part['sentences']}
            clip=folder/'.review.wav';start=chapter['start_ms']/1000;end=chapter['end_ms']/1000
            subprocess.run(['ffmpeg','-v','error','-ss',str(start),'-i',str(folder/'audio.mp3'),'-t',str(end-start),'-ar','16000','-ac','1','-y',str(clip)],check=True,capture_output=True,timeout=120)
            for w in part['words']:
                w['start_ms']-=chapter['start_ms'];w['end_ms']-=chapter['start_ms']
            _realign_corrected(part,clip,corrected_transcript.strip(),progress);clip.unlink(missing_ok=True)
            for i,s in enumerate(part['sentences']):
                old=s['id'];s['id']=chapter['id']+f'-s{i:04}';s['start_ms']+=chapter['start_ms'];s['end_ms']+=chapter['start_ms']
                ws=[w for w in part['words'] if w['sentence_id']==old]
                for w in ws:w.update(id=chapter['id']+'-'+w['id'],sentence_id=s['id'],start_ms=w['start_ms']+chapter['start_ms'],end_ms=w['end_ms']+chapter['start_ms'])
                s['word_ids']=[w['id'] for w in ws]
            kept_s=[s for s in package['sentences'] if not chapter['start_ms']<=s['start_ms']<chapter['end_ms']]
            ids={s['id'] for s in kept_s};package['words']=[w for w in package['words'] if w['sentence_id'] in ids]+part['words'];package['words'].sort(key=lambda w:w['start_ms'])
            package['sentences']=sorted(kept_s+part['sentences'],key=lambda s:s['start_ms'])
            package['cards']=[c for c in package['cards'] if not set(c.get('sentence_ids',[]))&old_ids]
            package['phrases']=[p for p in package.get('phrases',[]) if p['sentence_id'] not in old_ids]
            if package.get('layout'):package['layout']['pages']=[p for p in package['layout']['pages'] if p['sentence_id'] not in old_ids]
            package['review'].setdefault('correction_history',[]).append({'chapter_id':chapter['id'],'before':original,'after':corrected_transcript.strip(),'at':datetime.now(timezone.utc).isoformat()})
            package['artifacts']['video']='not_requested'
            package['artifacts']['study']='pending'
            package['render']['status']='prepared'
        checked=_quality(_chapter_package(package,chapter),'')
        save_episode(package,folder)
        if checked['structural_flags']:raise ReviewRequired('；'.join(checked['structural_flags']),package)
        chapter.update(approved=True,status='pending')
    package['review']['recognized_transcript']=' '.join(s['text'] for s in package['sentences'])
    save_episode(package,folder)
    reference=_read(folder/'.reference.json',{'reference':package['review'].get('official_transcript',''),'cues':[]})
    result=_process_chapters(package,folder,reference['reference'],reference['cues'],progress,chapter_id or None)
    mode=result.get('processing',{}).get('render_mode','auto')
    if mode=='video' or mode=='auto' and result['episode']['duration_s']<=1200:
        from .renderer import export_video
        export_video(folder/'episode.json',folder,progress)
        result=_read(folder/'episode.json')
    return result
