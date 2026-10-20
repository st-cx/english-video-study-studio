from __future__ import annotations

import json
from pathlib import Path
from opencc import OpenCC
from .providers import ask_deepseek, ProviderError


def review_teaching(package: dict, progress, cache_dir: Path | None = None) -> None:
    """Second pass checks meaning, source claims and attribution, never timestamps."""
    converter = OpenCC('t2s')
    history = [change for change in package.get('teaching_review', {}).get('corrections', [])
               if isinstance(change, dict) and change.get('before') != change.get('value')]
    targets = {
        'sentence': ({s['id']: s for s in package['sentences']}, {'translation'}),
        'word': ({w['id']: w for w in package['words']}, {'gloss_zh'}),
        'phrase': ({p['id']: p for p in package.get('phrases', [])}, {'gloss_zh'}),
        'card': ({c['id']: c for c in package['cards']}, {'meaning', 'structure', 'translation', 'points'}),
    }
    def prepared_changes(result, ids):
        changes = result.get('corrections', [])
        if not isinstance(changes, list):
            raise ProviderError('教学复核返回格式不正确。')
        effective, unchanged = [], []
        for change in changes:
            if not isinstance(change, dict):
                raise ProviderError('教学复核返回格式不正确。')
            kind, field = change.get('type'), change.get('field')
            mapping, allowed = targets.get(kind, ({}, set()))
            target = mapping.get(change.get('id'))
            value = change.get('value')
            points_ok = field == 'points' and isinstance(value, list) and 0 < len(value) <= 3 and all(isinstance(point, dict) and set(point) == {'label', 'text'} and all(isinstance(point[k], str) and 0 < len(point[k]) <= 40 for k in ('label', 'text')) for point in value)
            text_ok = isinstance(value, str) and bool(value.strip())
            if target is None or field not in allowed or not (points_ok or (field != 'points' and text_ok)):
                raise ProviderError('教学复核试图修改未授权字段或引用无效 ID，已暂停。')
            if kind == 'sentence' and target['id'] not in ids or kind in {'word', 'phrase'} and target['sentence_id'] not in ids:
                raise ProviderError('教学复核跨组修改，已暂停。')
            if kind == 'card' and not set(target['sentence_ids']) & ids:
                raise ProviderError('教学复核卡片引用跨组，已暂停。')
            value = [{k: converter.convert(v) for k, v in point.items()} for point in value] if points_ok else converter.convert(value.strip())
            normalized = {**change, 'value': value}
            (unchanged if target.get(field) == value else effective).append(normalized)
        return effective, unchanged

    def pause(issues):
        package['teaching_review'] = {'status': 'needs_review', 'issues': issues, 'corrections': history}
        raise ProviderError('教学内容复核发现尚不能确认的问题，已保存结果并暂停：' + json.dumps(issues, ensure_ascii=False)[:500])
    system = ('你是独立的英语教学校对。只纠正实质错误：数字、否定、专名、代词指代、语境词义、短语范围、句型分析、凭空增加事实。'
              '逐条检查所有ID，一次性列出发现的全部实质错误；不要分多次挑选小部分。对“原义也可接受”的同义措辞不得提出correction。'
              '音频句子允许跨句承接、插话和省略；利用context_before/context_after判断含义，不要仅因句子是片段就判为无法确认。'
              '所有中文用简体中文。不要为措辞偏好改写，不把自然意译判错。功能词的释义允许简短解释语法作用。'
              '核对不定式to、关系代词和从句引导词，不应机械解释为去、那、谁；means doing不能误判为宾语从句，副词不决定动词形态。be+过去分词构成被动时be是助动词；Python口述type of不等于一段可执行代码，type()是内置函数。公式口述须区分两个数取平均与整个和再相除。what从句里的what可作句末介词of的宾语，不是凭空省略一个先行词。口头截断或重启不能被翻译成无依据的技术结论。'
              '固定短语按整体解释；同一个词在不同视频中应按实际语境取义，不套用其他节目的词义。'
              '不要更改英文、ID、时间或例句。只返回 JSON {ok:true,corrections:[],issues:[]}。'
              '有可直接修正的错误，corrections返回[{type:"sentence|word|phrase|card",id,field,value,reason}]；'
              '不能确认的重大问题放issues:[{sentence_id,reason}]，并设ok:false。sentence仅能改translation；word/phrase仅gloss_zh；card仅meaning/structure/translation/points。points必须整体返回最多3项的[{label,text}]，每项文字不超过40字符。')
    batches = [package['sentences'][i:i+7] for i in range(0,len(package['sentences']),7)]
    for i, batch in enumerate(batches):
        ids = {s['id'] for s in batch}
        progress('teaching_review', 90, f'正在复核译文、词义和原句出处 {i+1} / {len(batches)}')
        for attempt in range(5):
            payload = {'sentences': [{k:s[k] for k in ('id','text','translation','speaker')} for s in batch],
                       'context_before':[s['text'] for s in package['sentences'][max(0,i*7-3):i*7]],
                       'context_after':[s['text'] for s in package['sentences'][(i+1)*7:(i+1)*7+3]],
                       'words': [{k:w.get(k) for k in ('id','text','sentence_id','gloss_zh')} for w in package['words'] if w['sentence_id'] in ids],
                       'phrases': [p for p in package.get('phrases',[]) if p['sentence_id'] in ids],
                       'cards': [{k:c.get(k) for k in ('id','type','title','meaning','structure','points','example','translation','sentence_ids')} for c in package['cards'] if set(c['sentence_ids']) & ids]}
            result = ask_deepseek(system, json.dumps(payload,ensure_ascii=False), max_tokens=4096, cache_dir=cache_dir)
            if result.get('issues'):
                original_issues=result['issues']
                decision=ask_deepseek(system+' 这次仅复核上次列出的issues是否实质成立。若意见本身写“可接受、无重大错误”等，应作为措辞建议并从issues移除；可明确修正的错误给corrections。只有根据上下文仍无法判断的重大问题留在issues，不要凭空添加新问题。上次已给出的corrections仍会执行，无需重复；若其中某项需撤回或替换，必须以相同type/id/field返回正确value，撤回时返回input中的当前值。',json.dumps({'input':payload,'previous_review':result},ensure_ascii=False),max_tokens=4096,cache_dir=cache_dir)
                if not isinstance(result.get('corrections', []), list) or not isinstance(decision.get('corrections', []), list):
                    raise ProviderError('教学复核返回格式不正确。')
                combined = []
                for change in result.get('corrections', []) + decision.get('corrections', []):
                    if isinstance(change, dict):
                        combined = [c for c in combined if not isinstance(c, dict) or any(c.get(k) != change.get(k) for k in ('type', 'id', 'field'))]
                    combined.append(change)
                decision = {**decision, 'corrections': combined}
                package.setdefault('processing',{}).setdefault('review_adjudications',[]).append({'batch':i+1,'original_issues':original_issues,'remaining_issues':decision.get('issues',[])})
                result=decision
            if result.get('issues'):
                pause(result['issues'])
            changes = result.get('corrections',[])
            if not isinstance(changes,list):
                raise ProviderError('教学复核返回格式不正确。')
            def invalid_scope(change):
                if not isinstance(change, dict):return True
                kind=change.get('type');mapping,allowed=targets.get(kind,({},set()));target=mapping.get(change.get('id'))
                if target is None or change.get('field') not in allowed:return True
                if kind=='sentence':return target['id'] not in ids
                if kind in {'word','phrase'}:return target['sentence_id'] not in ids
                if kind=='card':return not bool(set(target['sentence_ids'])&ids)
                return True
            if any(invalid_scope(c) for c in changes):
                allowed_ids={kind:[key for key,value in mapping.items() if (kind=='sentence' and key in ids) or (kind in {'word','phrase'} and value['sentence_id'] in ids) or (kind=='card' and set(value['sentence_ids'])&ids)] for kind,(mapping,_) in targets.items()}
                result=ask_deepseek(system+' 上次结果引用了不在当前批次的ID或字段。严格只使用allowed_ids及许可字段；上下文仅用于理解，不能修改上下文的条目。已经在其他批次修正的内容不要重复修正。重新返回当前批次的完整复核结果。',json.dumps({'input':payload,'previous_review':result,'allowed_ids':allowed_ids},ensure_ascii=False),max_tokens=4096,cache_dir=cache_dir)
                changes=result.get('corrections',[])
                if not isinstance(changes,list) or any(invalid_scope(c) for c in changes):raise ProviderError('教学复核修复后仍跨组修改，已暂停。')
                if result.get('issues') or result.get('ok') is not True and not changes:raise ProviderError('教学复核修复后仍有待确认内容。')
            try:
                changes, unchanged = prepared_changes(result, ids)
            except ProviderError:
                result=ask_deepseek(system+' 上次修正值未通过格式校验。每项correction必须有type/id/field/value，value必须为非空字符串；points是完整的[{label,text}]，最多3项，每项label/text均不超过40字符。严格仅引用当前input中的ID和许可字段。修复格式并返回当前批次完整结论，不能把错误原样重复。',json.dumps({'input':payload,'previous_review':result},ensure_ascii=False),max_tokens=4096,cache_dir=cache_dir)
                changes,unchanged=prepared_changes(result,ids)
                if result.get('issues'):pause(result['issues'])

            if not changes and unchanged and result.get('ok') is not True:
                # Identical values cannot fix anything. Obtain an explicit verdict
                # instead of consuming all correction rounds on cached no-ops.
                previous = result
                result = ask_deepseek(system + ' 上次corrections中的下列值与当前内容完全相同，实际未作修改。请核对这些意见：正确或可接受的原义不是实质错误；若无真实疑点，明确返回ok:true、corrections:[]、issues:[]。若仍有实质错误，请给出不同且可验证的修正值，或在issues说明具体未解决的问题。不得仅为填充列表重复正确内容。', json.dumps({'input': payload, 'previous_review': previous, 'unchanged_corrections': unchanged}, ensure_ascii=False), max_tokens=4096, cache_dir=cache_dir)
                package.setdefault('processing', {}).setdefault('review_adjudications', []).append({'batch': i + 1, 'unchanged_corrections': unchanged, 'ok': result.get('ok'), 'remaining_issues': result.get('issues', [])})
                changes, _ = prepared_changes(result, ids)
            if result.get('issues'):
                pause(result['issues'])
            if not changes:
                if result.get('ok') is not True:
                    pause([{'reason': '模型没有明确通过教学复核；未提供可生效的修正或具体疑点'}])
                break
            oscillating=any(any(all(old.get(k)==change.get(k) for k in ('type','id','field','value')) for old in history) for change in changes)
            for change in changes:
                kind, field = change.get('type'),change.get('field')
                target = targets[kind][0][change['id']]
                history.append({**change,'before':target.get(field)})
                target[field] = change['value']
            if attempt==4 or oscillating:
                package['teaching_review']={'status':'needs_review','issues':[{'reason':'多轮复核仍提出实质修正，需要检查最终修正内容', 'targets': [{'type': c['type'], 'id': c['id'], 'field': c['field']} for c in changes]}],'corrections':history}
                raise ProviderError('教学复核修正后仍未明确通过，已保留内容等待检查。')
    package['teaching_review'] = {'status':'passed','sentence_count':len(package['sentences']),'corrections':history,'note':'模型复核通过仍不能替代人工事实核验。'}
