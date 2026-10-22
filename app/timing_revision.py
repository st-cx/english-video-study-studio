"""Apply measured timing revisions without mixing clocks or claiming human truth."""
from __future__ import annotations
import collections,copy,math,re,statistics
from .sync_audit import digest,normalize_ctc


def _tokens(text):
    normalized=normalize_ctc(text)
    return normalized.split() if normalized else []


def propose_timing_revision(words, survey, audio_sha256):
    if survey.get('input_timing_sha256')!=digest(words) or survey.get('audio_sha256')!=audio_sha256:
        raise ValueError('声学测量对应的原声或词语版本已改变。')
    if not re.fullmatch('[0-9a-f]{64}',str(survey.get('model_sha256',''))):
        raise ValueError('缺少声学模型指纹。')
    observations=collections.defaultdict(dict)
    for window in survey.get('windows',[]):
        begin,end=window['start_ms'],window['end_ms']
        greedy=_tokens(window.get('greedy_unprompted_text',''))
        for row in window.get('aligned',[]):
            i=row.get('index')
            if not isinstance(i,int) or isinstance(i,bool) or not 0<=i<len(words):continue
            w=words[i]
            if row.get('id')!=w['id'] or row.get('text')!=w['text']:continue
            values=[row.get('start_ms'),row.get('end_ms'),row.get('path_score_not_accuracy')]
            if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in values):continue
            start,stop,score=values
            if not 0<=begin<=start<stop<=end or not 19<=stop-start<=1500 or not .8<=score<=1:continue
            # Interior observations protect against clipped speech at crop edges.
            # The actual beginning/end of the whole recording can have one side.
            if begin>0 and start-begin<800:continue
            if end-stop<800:continue
            context_start=min(max(0,i-1),max(0,len(words)-3))
            context=words[context_start:min(len(words),context_start+3)]
            phrase=_tokens(' '.join(x.get('spoken_form',x['text']) for x in context))
            if len(phrase)<3:continue
            continuous=any(greedy[j:j+len(phrase)]==phrase for j in range(len(greedy)-len(phrase)+1))
            # Unprompted greedy CTC words carry their own acoustic frame spans.
            # A matching word at the measured interval is stronger evidence than
            # finding the same spelling somewhere else in a twenty-second crop.
            timed_match=any(_tokens(g.get('text',''))==_tokens(w.get('spoken_form',w['text']))
                and abs(g['start_ms']-start)<=100 and abs(g['end_ms']-stop)<=100
                for g in window.get('greedy_words',[]))
            if not continuous and not timed_match:continue
            observations[i][(begin,end)]=row
    candidates={}
    for i,windows in observations.items():
        rows=list(windows.values())
        if len(rows)<2:continue
        if max(max(r[k] for r in rows)-min(r[k] for r in rows) for k in ['start_ms','end_ms'])>80:continue
        start,stop=(round(statistics.median(r[k] for r in rows)) for k in ['start_ms','end_ms'])
        if start<stop:candidates[i]={'word_id':words[i]['id'],'word_index':i,'text':words[i]['text'],
            'before_ms':[words[i]['start_ms'],words[i]['end_ms']],'after_ms':[start,stop],
            'observations':[{'window_ms':list(window),'measurement':copy.deepcopy(row)} for window,row in windows.items()]}
    # Resolve the full proposal as one transaction. A rejected neighbor restores
    # its old span, which can expose another conflict; reach a stable fixed point.
    while True:
        conflicts=set()
        for i in range(1,len(words)):
            a=candidates.get(i-1,{}).get('after_ms',[words[i-1]['start_ms'],words[i-1]['end_ms']])
            b=candidates.get(i,{}).get('after_ms',[words[i]['start_ms'],words[i]['end_ms']])
            if a[1]>b[0]:
                if i-1 in candidates:conflicts.add(i-1)
                if i in candidates:conflicts.add(i)
        if not conflicts:break
        for i in conflicts:candidates.pop(i)
    return {'version':1,'algorithm':'overlapping-ctc-with-unprompted-context-v1','input_words_sha256':digest(words),
        'audio_sha256':audio_sha256,'model_sha256':survey['model_sha256'],'human_ground_truth':False,
        'caution':'两窗为同一CTC模型的稳定性测量；不是两个独立模型，也不是人工准确率。',
        'patches':[candidates[i] for i in sorted(candidates) if candidates[i]['before_ms']!=candidates[i]['after_ms']]}


def retime_package(package, proposal):
    if proposal.get('input_words_sha256')!=digest(package['words']):raise ValueError('修复提案对应旧版词语。')
    result=copy.deepcopy(package);original={w['id']:w for w in package['words']};by={w['id']:w for w in result['words']}
    for patch in proposal['patches']:
        w=by[patch['word_id']]
        if patch['text']!=w['text'] or patch['before_ms']!=[w['start_ms'],w['end_ms']]:raise ValueError('修复提案词语锚点不匹配。')
        start,end=patch['after_ms']
        if not isinstance(start,int) or not isinstance(end,int) or start<0 or start>=end:raise ValueError('实测边界无效。')
        w.update(start_ms=start,end_ms=end)
    for a,b in zip(result['words'],result['words'][1:]):
        olda,oldb=original[a['id']],original[b['id']]
        if a['end_ms']>b['start_ms'] and a['end_ms']-b['start_ms']>max(0,olda['end_ms']-oldb['start_ms']):raise ValueError('修复会产生新的邻词重叠。')
    for sentence in result.get('sentences',[]):
        ws=[by[k] for k in sentence['word_ids']];sentence.update(start_ms=ws[0]['start_ms'],end_ms=ws[-1]['end_ms'])
    sentences={s['id']:s for s in result.get('sentences',[])}
    old_sentences={s['id']:s for s in package.get('sentences',[])}
    for card in result.get('cards',[]):
        anchors=[k for k in card.get('word_ids',[]) if k in by]
        if anchors:
            key=min(anchors,key=lambda k:abs(original[k]['start_ms']-card['start_ms']));delta=by[key]['start_ms']-original[key]['start_ms']
        else:
            ids=[k for k in card.get('sentence_ids',[]) if k in sentences]
            delta=sentences[ids[0]]['start_ms']-old_sentences[ids[0]]['start_ms'] if ids else 0
        card['start_ms']+=delta;card['end_ms']+=delta
    for page in result.get('layout',{}).get('pages',[]):
        ids=[k for line in page['lines'] for k in line];page.update(start_ms=by[ids[0]]['start_ms'],end_ms=by[ids[-1]]['end_ms'])
    if result.get('chapters'):
        from .general import _schedule_chapter_cards
        scheduled=[]
        for chapter in result['chapters']:
            ids={s['id'] for s in package.get('sentences',[]) if chapter['start_ms']<=s['start_ms']<chapter['end_ms']}
            cards=[c for c in result.get('cards',[]) if set(c.get('sentence_ids',[]))&ids]
            scheduled.extend(_schedule_chapter_cards(cards,chapter))
        if len(scheduled)==len(result.get('cards',[])) and len({c['id'] for c in scheduled})==len(scheduled):result['cards']=scheduled
    # Content summaries remain valid for timing-only changes. Their original
    # sentence anchors resolve the new clock without asking a model for times.
    if result.get('study_sections'):
        from .study_sections import resolve_sections,source_fingerprint
        sections=result['study_sections'];sections['items']=resolve_sections(result,{'sections':sections['items']});sections['source_fingerprint']=source_fingerprint(result)
    result.pop('sync_review',None)
    result.setdefault('processing',{}).setdefault('timing_revision_history',[]).append(copy.deepcopy(proposal))
    result['processing']['timing_revision_sha256']=digest(proposal)
    result.setdefault('artifacts',{}).update(video='not_requested',study='pending')
    result.setdefault('render',{})['status']='prepared'
    return result
