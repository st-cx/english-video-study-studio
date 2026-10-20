"""Versioned production wrapper for conservative independent CTC repairs."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from .ctc_repair import repair_with_ctc, MIN_PATH_SCORE, MAX_WINDOW_DIFFERENCE_MS
from .sync_audit import digest, file_sha256, MODEL_FILENAME

ROOT = Path(__file__).resolve().parents[1]


def _model_cache():
    # The first version never downloads automatically. A packaged deployment may
    # prepopulate this isolated directory; missing weights leave review intact.
    return ROOT/'work/sync-audit/model-cache'


def _implementation_hash():
    return digest({name:file_sha256(Path(__file__).with_name(name)) for name in ['speech_repair.py','ctc_repair.py','sync_audit.py']})


def _unique_global_index(words, dispute):
    text=dispute.get('text');interval=dispute.get('aligned_ms')
    if not isinstance(text,str) or not isinstance(interval,list) or len(interval)!=2:
        return None
    # A clip-local word_index is deliberately ignored, even when in range.
    matches=[i for i,w in enumerate(words) if w['text']==text and [w['start_ms'],w['end_ms']]==interval]
    return matches[0] if len(matches)==1 else None


def _write_atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.part.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def repair_speech_words(audio, words, records, folder, progress):
    """Repair measured timing only; keep all unresolved evidence and raw words.

    Expects records in media/global time. Never converts clip indices to global
    word indices by arithmetic. Unmappable, ambiguous, legacy-local, and failed
    targets stay in alignment_disputes for the existing review gate.
    """
    audio=Path(audio);folder=Path(folder)
    cache=_model_cache();weight=cache/MODEL_FILENAME
    signature={'audio_sha256':file_sha256(audio),'input_words_sha256':digest(words),'input_records_sha256':digest(records),
               'implementation_sha256':_implementation_hash(),'model_cache_sha256':file_sha256(weight) if weight.is_file() else 'not_cached'}
    fingerprint=digest(signature)
    location=folder/'.speech-repair'/fingerprint
    saved=location/'result.json'
    if saved.is_file():
        try:
            existing=json.loads(saved.read_text(encoding='utf-8'))
            if existing.get('signature')==signature and existing.get('output_words_sha256')==digest(existing['words']) and existing.get('output_records_sha256')==digest(existing['records']) and (location/'evidence.json').is_file() and existing.get('evidence_sha256')==file_sha256(location/'evidence.json'):
                progress('已复用独立CTC复核缓存；未完成的争议仍保留待检查。')
                return copy.deepcopy(existing['words']),copy.deepcopy(existing['records'])
        except (ValueError,KeyError,OSError,TypeError):pass
    original_words=copy.deepcopy(words);new_records=copy.deepcopy(records)
    bindings=[];targets=set()
    for record_index,record in enumerate(records):
        for dispute_index,dispute in enumerate(record.get('alignment',{}).get('alignment_disputes',[])):
            # Timestamp migration belongs to the upstream chunk reader. Guessing
            # an offset here could resolve the wrong repeated phrase.
            index=_unique_global_index(words,dispute) if record.get('time_basis')=='media' else None
            bindings.append({'record_index':record_index,'dispute_index':dispute_index,'word_index':index,'mapped_uniquely':index is not None})
            if index is not None:targets.add(index)
    structural=[i for i,w in enumerate(words) if w['end_ms']<=w['start_ms'] or i>0 and words[i-1]['end_ms']-w['start_ms']>40]
    targets.update(structural)
    progress_failure=[]
    def report(*args):
        try:
            progress(*args)
        except BaseException as error:
            progress_failure.append(error)
            raise
    try:
        new_words,evidence=repair_with_ctc(audio,words,sorted(targets),report,cache_dir=cache)
    except Exception as error:
        if progress_failure:
            # Pause/cancel is control flow, never a cached acoustic failure.
            # Preserve the original callback exception even if a lower layer
            # wrapped it while unwinding its model runtime.
            raise progress_failure[0]
        new_words=copy.deepcopy(words)
        evidence={'status':'measurement_failed','human_ground_truth':False,'rows':[],'applied_transactions':0,'reason':type(error).__name__+': '+str(error)[:900]}
    applied={}
    for row in evidence.get('rows',[]):
        transaction=row.get('transaction',{})
        if transaction.get('status')!='measured_transaction':continue
        measurements=row.get('measurements',[])
        if len(measurements)!=2 or any(m.get('status')!='measured' for m in measurements):continue
        # Inclusion in a transaction is insufficient by itself: every changed
        # patch word needs its own complete, stable, high-score measurements.
        for patch in transaction.get('patch',[]):
            index=patch.get('word_index')
            if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<len(words):continue
            before=original_words[index];after=new_words[index]
            if before['text']!=patch.get('text') or after['text']!=before['text']:continue
            if [before['start_ms'],before['end_ms']]!=[patch.get('original_start_ms'),patch.get('original_end_ms')]:continue
            if [after['start_ms'],after['end_ms']]!=[patch.get('start_ms'),patch.get('end_ms')] or after['end_ms']<=after['start_ms']:continue
            if [before['start_ms'],before['end_ms']]==[after['start_ms'],after['end_ms']]:continue
            observations=[]
            for measurement in measurements:
                matched=[w for w in measurement.get('aligned_context',[]) if w.get('index')==index and w.get('text')==before['text']]
                if len(matched)!=1:break
                observed=matched[0]
                values=[observed.get(key) for key in ['measured_start_ms','measured_end_ms','raw_path_score_not_accuracy']]
                if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in values):break
                if observed['raw_path_score_not_accuracy']<MIN_PATH_SCORE or not 19<=observed['measured_end_ms']-observed['measured_start_ms']<=1500:break
                observations.append(observed)
            if len(observations)!=2:continue
            first,second=observations
            difference=max(abs(first['measured_start_ms']-second['measured_start_ms']),abs(first['measured_end_ms']-second['measured_end_ms']))
            if difference>MAX_WINDOW_DIFFERENCE_MS:continue
            if [round(second['measured_start_ms']),round(second['measured_end_ms'])]!=[after['start_ms'],after['end_ms']]:continue
            applied[index]=row
    resolved=0
    for record_index,record in enumerate(new_records):
        alignment=record.get('alignment')
        if not isinstance(alignment,dict):continue
        keep=[]
        for dispute_index,dispute in enumerate(alignment.get('alignment_disputes',[])):
            binding=next(item for item in bindings if item['record_index']==record_index and item['dispute_index']==dispute_index)
            target=binding['word_index']
            if target not in applied:
                keep.append(dispute);continue
            row=applied[target]
            alignment.setdefault('resolved_history',[]).append({'original_dispute':copy.deepcopy(dispute),
                'resolution':{'method':'independent_ctc_joint_neighbor_transaction','global_word_index':target,'fingerprint':fingerprint,
                              'evidence_file':str(location/'evidence.json'),'measured_ms':[new_words[target]['start_ms'],new_words[target]['end_ms']],
                              'human_ground_truth':False,'transaction':copy.deepcopy(row['transaction'])}})
            resolved+=1
        alignment['alignment_disputes']=keep
        alignment['ctc_repair_audit']={'fingerprint':fingerprint,'evidence_file':str(location/'evidence.json'),'status':evidence.get('status'),
                                       'human_ground_truth':False,'remaining_dispute_count':len(keep)}
    evidence.update(signature=signature,fingerprint=fingerprint,dispute_bindings=bindings,structural_target_indices=structural,
                    resolved_dispute_count=resolved,unmapped_dispute_count=sum(not b['mapped_uniquely'] for b in bindings),
                    human_ground_truth=False)
    _write_atomic(location/'evidence.json',evidence)
    _write_atomic(saved,{'signature':signature,'fingerprint':fingerprint,'words':new_words,'records':new_records,
                        'output_words_sha256':digest(new_words),'output_records_sha256':digest(new_records),'evidence_file':'evidence.json','evidence_sha256':file_sha256(location/'evidence.json')})
    return new_words,new_records
