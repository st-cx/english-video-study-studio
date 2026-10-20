from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def prepare_layout(package: dict, folder: Path) -> None:
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=folder,prefix='.layout-',suffix='.json',delete=False) as stream:
        path=Path(stream.name)
        json.dump(package,stream,ensure_ascii=False)
    try:
        subprocess.run(['node',str(ROOT/'scripts/prepare-layout.mjs'),str(path)],cwd=ROOT,check=True,capture_output=True,text=True,timeout=90)
        package['layout'] = json.loads(path.read_text(encoding='utf-8'))['layout']
        _page_translations(package, folder)
    finally:
        path.unlink(missing_ok=True)

def _page_translations(package: dict, folder: Path) -> None:
    from opencc import OpenCC
    from .providers import ask_deepseek, ProviderError
    convert = OpenCC('t2s')
    sentences = {s['id']:s for s in package['sentences']}
    words = {w['id']:w['text'] for w in package['words']}
    pages = package['layout']['pages']
    multiple = {p['sentence_id'] for p in pages if sum(q['sentence_id']==p['sentence_id'] for q in pages)>1}
    pending = [p for p in pages if p['sentence_id'] in multiple or len(sentences[p['sentence_id']]['translation'])>72]
    for p in pages:
        if p not in pending:p['translation']=sentences[p['sentence_id']]['translation']
    for start in range(0,len(pending),8):
        batch=pending[start:start+8]
        payload=[{'id':p['id'],'english':' '.join(words[wid] for line in p['lines'] for wid in line),'full_sentence':sentences[p['sentence_id']]['text'],'full_translation':sentences[p['sentence_id']]['translation']} for p in batch]
        result=ask_deepseek('为英文字幕页写对应的简体中文译文。参考完整句子，但只翻译当前页覆盖的内容，不重复整句。不得漏掉否定、数字或关系。每页不超过68字，不加解释。返回 JSON {pages:[{id,zh}]}。',json.dumps(payload,ensure_ascii=False),max_tokens=4096,cache_dir=folder/'.teaching-cache')
        translations={x.get('id'):x.get('zh') for x in result.get('pages',[]) if isinstance(x,dict)}
        for attempt in range(3):
            invalid=[p for p in batch if not isinstance(translations.get(p['id']),str) or not translations[p['id']].strip() or len(translations[p['id']])>72]
            if not invalid:break
            required={p['id'] for p in invalid}
            repair=ask_deepseek('只输出JSON object，为指定字幕页补齐或精简对应的简体中文译文。逐字保留每个输入id，必须翻译当前页英文，不能重复整句，不漏否定数字。每页68字以内。返回 {"pages":[{"id":"原id","zh":"实际译文"}]}，不得占位、不得返回其他id。',json.dumps({'required_pages':[p for p in payload if p['id'] in required],'previous':[{'id':p['id'],'zh':translations.get(p['id'])} for p in invalid],'repair_attempt':attempt+1},ensure_ascii=False),max_tokens=4096,cache_dir=folder/'.teaching-cache')
            for item in repair.get('pages',[]):
                if isinstance(item,dict) and item.get('id') in required:translations[item['id']]=item.get('zh')
        for p in batch:
            value=translations.get(p['id'])
            if not isinstance(value,str) or not value.strip() or len(value)>72:
                raise ProviderError('分页译文多次补齐后仍缺失或超出两行，已暂停避免裁切。')
            p['translation']=convert.convert(value.strip())
