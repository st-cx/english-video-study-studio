from __future__ import annotations
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def prepare_layout(package: dict, folder: Path) -> None:
    path = folder / 'layout.preparing.json'
    path.write_text(json.dumps(package, ensure_ascii=False),encoding='utf-8')
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
        translations={x.get('id'):x.get('zh') for x in result.get('pages',[])}
        for p in batch:
            value=translations.get(p['id'])
            if not isinstance(value,str) or not value.strip() or len(value)>72:
                raise ProviderError('分页译文缺失或超出两行，已暂停避免裁切。')
            p['translation']=convert.convert(value.strip())
