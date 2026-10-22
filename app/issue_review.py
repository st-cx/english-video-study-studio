"""Compact replay/feedback page for unresolved source and alignment findings.

Human feedback is separate from word-boundary ground truth. Selecting a text
alternative records a correction request; it never silently invents alignment.
"""
from __future__ import annotations
import hashlib
import html
import json
import math
from pathlib import Path
from .storage import data_root, _file_lock, _write_json_atomic


def revision(package):
    provenance = package.get('processing', {})
    facts = {'episode': {k: package['episode'].get(k) for k in ('id', 'fingerprint', 'audio_file', 'duration_s', 'source_sha256')},
             'audio_sha256': provenance.get('review_reconciliation', {}).get('audio_sha256') or provenance.get('reference_review', {}).get('audio_sha256'),
             'media': package.get('media'),
             'words': [{k: w.get(k) for k in ('id', 'text', 'start_ms', 'end_ms')} for w in package.get('words', [])],
             'findings': [{k: c.get(k, []) for k in ('reference_issues', 'alignment_issues')} for c in package.get('chapters', [])]}
    return hashlib.sha256(json.dumps(facts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_manifest(package):
    duration = round(float(package['episode']['duration_s']) * 1000)
    issues, seen = [], set()
    for chapter in package.get('chapters', []):
        for field, kind in [('reference_issues', '文稿差异'), ('alignment_issues', '边界争议')]:
            for original in chapter.get(field, []):
                row = {'kind': kind, 'start_ms': original.get('start_ms'), 'end_ms': original.get('end_ms'),
                       'current': original.get('recognized', original.get('text', '')),
                       'reference': original.get('reference', ''), 'reason': original.get('reason', '')}
                if any(isinstance(row[k], bool) or not isinstance(row[k], (int, float)) or not math.isfinite(row[k]) for k in ('start_ms', 'end_ms')):
                    raise ValueError('待核对区间包含无效时间。')
                if not 0 <= row['start_ms'] <= row['end_ms'] <= duration:
                    raise ValueError('待核对区间超出课程。')
                identity = json.dumps(row, sort_keys=True, ensure_ascii=False)
                if identity in seen:
                    continue
                seen.add(identity)
                row['id'] = hashlib.sha256(identity.encode()).hexdigest()[:24]
                if kind == '边界争议':
                    details = original.get('details', {})
                    pairs = [(label, details.get(key)) for key, label in [('aligned_ms', '原对齐器候选'), ('asr_ms', '独立识别候选')]]
                    valid = [(label, pair) for label, pair in pairs if isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pair)]
                    offset = row['start_ms'] - min(pair[0] for _, pair in valid) if valid else 0
                    row['candidates'] = [{'label': label, 'start_ms': pair[0] + offset, 'end_ms': pair[1] + offset} for label, pair in valid]
                issues.append(row)
    issues.sort(key=lambda r: (r['start_ms'], r['end_ms'], r['kind']))
    groups = []
    for issue in issues:
        if groups and issue['start_ms'] <= groups[-1]['end_ms'] + 2400 and max(groups[-1]['end_ms'], issue['end_ms']) - groups[-1]['start_ms'] <= 18000:
            group = groups[-1]
            group['end_ms'] = max(group['end_ms'], issue['end_ms'])
            group['issues'].append(issue)
        else:
            groups.append({'start_ms': issue['start_ms'], 'end_ms': issue['end_ms'], 'issues': [issue]})
    for group in groups:
        group['id'] = hashlib.sha256('|'.join(i['id'] for i in group['issues']).encode()).hexdigest()[:24]
        group['play_start_ms'] = max(0, group['start_ms'] - 1200)
        group['play_end_ms'] = min(duration, max(group['end_ms'] + 1200, group['play_start_ms'] + 3500))
        group['context'] = ' '.join(w['text'] for w in package['words'] if w['end_ms'] > group['play_start_ms'] and w['start_ms'] < group['play_end_ms'])
    return {'version': 1, 'episode_id': package['episode']['id'], 'revision': revision(package),
            'title': package['episode'].get('title_zh') or package['episode']['title'], 'groups': groups,
            'issue_count': len(issues), 'group_count': len(groups),
            'listening_seconds': round(sum(g['play_end_ms'] - g['play_start_ms'] for g in groups) / 1000, 1)}


def save_feedback(package, payload):
    manifest = build_manifest(package)
    if payload.get('revision') != manifest['revision']:
        raise ValueError('课程已修订，请刷新检查页后重新确认；旧标注仍保留。')
    ids = {i['id'] for g in manifest['groups'] for i in g['issues']}
    incoming = payload.get('decisions')
    if not isinstance(incoming, dict) or not incoming or not set(incoming).issubset(ids):
        raise ValueError('确认记录与当前待核对问题不匹配。')
    choices = {'current', 'reference', 'uncertain'}
    for row in incoming.values():
        if not isinstance(row, dict) or row.get('choice') not in choices or not isinstance(row.get('note', ''), str) or len(row.get('note', '')) > 1000:
            raise ValueError('请使用当前稿、参考稿或不确定，并将备注限制为1000字。')
    folder = data_root() / 'review-feedback'
    path = folder / (hashlib.sha256(manifest['episode_id'].encode()).hexdigest()[:24] + '-' + manifest['revision'] + '.json')
    with _file_lock(path.with_suffix('.lock'), 20, '确认记录正在保存，请稍后重试。', interval=.03):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            record = {'version': 1, 'episode_id': manifest['episode_id'], 'revision': manifest['revision'], 'decisions': {}}
        record['decisions'].update({k: {'choice': v['choice'], 'note': v.get('note', ''), 'human_listening_self_report': True} for k, v in incoming.items()})
        _write_json_atomic(path, record)
    return {'saved': True, 'recorded': len(record['decisions']), 'word_boundary_acceptance': False,
            'message': '已保存听辨判断。替换英文或边界须重新对齐；本记录不代替50词边界验收。'}


def feedback(package):
    current = revision(package)
    path = data_root() / 'review-feedback' / (hashlib.sha256(package['episode']['id'].encode()).hexdigest()[:24] + '-' + current + '.json')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {'revision': current, 'decisions': {}}


def write_issue_review(package, folder, media_folder=None):
    manifest = build_manifest(package)
    media = package.get('media', {})
    filename = media.get('video_file') or media.get('audio_file') or package['episode']['audio_file']
    media_folder = Path(media_folder or folder).resolve()
    target = (media_folder / filename).resolve()
    if not target.is_relative_to(media_folder) or not target.is_file():
        raise ValueError('原声文件不在当前课程目录中。')
    manifest['media'] = '../' + target.relative_to(media_folder).as_posix()
    raw = json.dumps(manifest, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = PAGE.replace('__TITLE__', html.escape(manifest['title'])).replace('__DATA__', raw)
    output = folder / 'issue-review'
    output.mkdir(exist_ok=True)
    (output / 'review.html').write_text(page, encoding='utf-8')
    return {'page': 'issue-review/review.html', 'issue_count': manifest['issue_count'], 'group_count': manifest['group_count'],
            'listening_seconds': manifest['listening_seconds'], 'revision': manifest['revision']}


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>回听确认 · __TITLE__</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#243953;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:960px;margin:auto;padding:22px}h1{font-size:26px}a{color:#2861a5}button,select{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid #cdd8e6;background:white;color:#24466d;cursor:pointer}button.active{background:#e3f2e8;border-color:#67a87c}button.primary{background:#245a95;color:white}button:focus-visible{outline:3px solid #eda63b}.player{position:sticky;top:0;background:#f4f6faee;padding:12px 0;z-index:2}video{display:block;width:100%;max-height:280px;background:#172735;border-radius:10px}.tools{display:flex;gap:8px;flex-wrap:wrap;margin:9px 0}.card{background:white;border:1px solid #dce3ed;border-radius:12px;padding:18px;margin:15px 0;overflow-wrap:anywhere}.issue{padding:12px 0;border-top:1px solid #e4e9ef}.muted{color:#64788e}textarea{font:inherit;width:100%;min-height:55px;border:1px solid #ccd8e7;border-radius:7px;padding:8px}p{margin:9px 0}#status{min-height:26px;font-weight:600}time{font-variant-numeric:tabular-nums}@media(max-width:500px){main{padding:12px}.card{padding:13px}h1{font-size:22px}button{padding:8px}}
</style><main><a href="../study.html">← 返回学习页</a><h1>只听需要你判断的短片段</h1><p id="summary"></p><p class="muted">相邻问题已合并回听。点击「已听清」记录你的判断；选择参考稿或识别候选会记录修订请求，须重新对齐后生效。此处的判断与逐词边界精度验收分别保存。</p>
<div class="player"><video id="player" controls playsinline preload="metadata"></video><div class="tools"><button class="primary" id="replay">▶ 重听当前片段</button><button id="next">下一处未确认</button><select id="chooser" aria-label="选择回听片段"></select><label><input id="loop" type="checkbox" checked> 循环片段</label><select id="rate" aria-label="回听速度"><option value="0.75">0.75 倍</option><option value="1" selected>1 倍</option><option value="1.25">1.25 倍</option></select></div><div id="status" role="status"></div></div>
<div class="tools"><label><input type="checkbox" id="hide"> 隐藏已确认</label><button id="download">导出听辨记录</button><a href="../sync-review/review.html">50词精度检查 ↗</a></div><div id="groups"></div></main>
<script>const DATA=__DATA__;const key='english-issue-feedback:'+DATA.episode_id+':'+DATA.revision;let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch{}const el=id=>document.getElementById(id),player=el('player');player.src=DATA.media;let active=DATA.groups[0]||null;const notes={};const clock=ms=>Math.floor(ms/60000)+':'+String(Math.floor(ms/1000)%60).padStart(2,'0')+'.'+Math.floor(ms%1000/100);function msg(s){el('status').textContent=s}function persist(){try{localStorage.setItem(key,JSON.stringify(saved));return true}catch{return false}}function play(g){active=g;render();player.currentTime=g.play_start_ms/1000;player.play().catch(()=>msg('请点击播放器播放原声。'));msg('正在回听 '+clock(g.play_start_ms)+'–'+clock(g.play_end_ms));}player.addEventListener('seeking',()=>{if(active&&(player.currentTime<active.play_start_ms/1000-.15||player.currentTime>active.play_end_ms/1000+.15))active=null});player.addEventListener('timeupdate',()=>{if(active&&player.currentTime>=active.play_end_ms/1000){if(el('loop').checked){player.currentTime=active.play_start_ms/1000;player.play().catch(()=>{})}else player.pause()}});el('replay').onclick=()=>{if(active)play(active);else if(DATA.groups[0])play(DATA.groups[0])};el('rate').onchange=e=>player.playbackRate=Number(e.target.value);function next(){const start=active?DATA.groups.findIndex(g=>g.id===active.id):-1;const ordered=[...DATA.groups.slice(start+1),...DATA.groups.slice(0,start+1)];const g=ordered.find(g=>g.issues.some(i=>!saved[i.id]||saved[i.id].choice==='uncertain'));if(g){play(g);document.getElementById(g.id)?.scrollIntoView({block:'nearest'})}else msg('所有问题均已记录；仍可重新回听。')}el('next').onclick=next;
async function choose(issue,choice,note){saved[issue.id]={choice,note:note.value,human_listening_self_report:true};const localSaved=persist();render();msg(localSaved?'已保存在本浏览器。':'浏览器未能保存，请导出记录；正在尝试同步本机。');if(location.protocol!=='file:'){try{const r=await fetch('/api/review-feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode_id:DATA.episode_id,revision:DATA.revision,decisions:{[issue.id]:saved[issue.id]}})});const v=await r.json();if(!r.ok)throw Error(v.error||'保存失败');msg(v.message)}catch(e){msg('浏览器已保存，本机同步失败：'+e.message)}}}
function render(){el('summary').textContent=DATA.issue_count+' 条检查记录合并为 '+DATA.group_count+' 段，共约 '+Math.ceil(DATA.listening_seconds)+' 秒原声。已记录 '+Object.keys(saved).filter(id=>saved[id].choice!=='uncertain').length+' 条判断。';el('groups').replaceChildren();if(active)el('chooser').value=active.id;for(const g of DATA.groups){if(active&&g.id!==active.id)continue;if(el('hide').checked&&g.issues.every(i=>saved[i.id]&&saved[i.id].choice!=='uncertain'))continue;const card=document.createElement('section');card.className='card';card.id=g.id;const title=document.createElement('button');title.className='primary';title.textContent='▶ '+clock(g.play_start_ms)+'–'+clock(g.play_end_ms)+' · '+g.issues.length+' 项';title.onclick=()=>play(g);card.append(title);const context=document.createElement('p');context.textContent=g.context;card.append(context);for(const i of g.issues){const area=document.createElement('div');area.className='issue';for(const line of [i.kind+' · '+i.reason,'当前：'+(i.current||'（此处无词）'),...(i.kind==='边界争议'?(i.candidates||[]).map(c=>c.label+' '+clock(c.start_ms)+'–'+clock(c.end_ms)+'（仅供定位）'):['参考：'+(i.reference||'（参考稿省略）')])]){const p=document.createElement('p');p.textContent=line;area.append(p)}const note=document.createElement('textarea');note.placeholder='听到的其他内容／备注（可选）';note.maxLength=1000;note.value=notes[i.id]??saved[i.id]?.note??'';note.oninput=()=>notes[i.id]=note.value;area.append(note);const tools=document.createElement('div');tools.className='tools';for(const [value,label] of (i.kind==='边界争议'?[['current','已回听：当前可用'],...((i.candidates||[]).length>1?[['reference','已回听：识别候选更接近']]:[]),['uncertain','仍不确定']]:[['current','已听清：当前稿'],['reference','已听清：参考稿'],['uncertain','仍不确定']])){const b=document.createElement('button');b.textContent=label;b.className=saved[i.id]?.choice===value?'active':'';b.onclick=()=>choose(i,value,note);tools.append(b)}area.append(tools);card.append(area)}el('groups').append(card)}}for(const [index,g] of DATA.groups.entries()){const option=document.createElement('option');option.value=g.id;option.textContent=(index+1)+' / '+DATA.group_count+' · '+clock(g.start_ms)+' · '+g.issues.length+'项';el('chooser').append(option)}el('chooser').onchange=e=>{const g=DATA.groups.find(g=>g.id===e.target.value);if(g)play(g)};el('hide').onchange=render;el('download').onclick=()=>{const blob=new Blob([JSON.stringify({version:1,episode_id:DATA.episode_id,revision:DATA.revision,decisions:saved},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='听辨确认.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();if(location.protocol!=='file:')fetch('/api/review-feedback?episode_id='+encodeURIComponent(DATA.episode_id)).then(r=>r.json()).then(v=>{if(v.revision===DATA.revision){saved={...(v.decisions||{}),...saved};persist();render()}}).catch(()=>{});
</script></html>'''
