from __future__ import annotations
import json
import re
from pathlib import Path
from opencc import OpenCC
from .providers import ask_deepseek,ProviderError

def translate_excerpts(package: dict,cache_dir: Path | None=None):
    excerpts=package.get('processing',{}).get('example_excerpts',[])
    for card in package['cards']:
        if any(p.get('label') in {'主干','修饰','从句','句子主干'} for p in card.get('points',[])) or re.search(r'时态|完成时|过去时|从句|句型|最高级|被动|定语|状语|不定式|动名词',card.get('title','')):
            card['type']='grammar'
        if card['type']=='grammar' and len(card.get('points',[]))>=2 and len(card.get('meaning',''))>40:
            # The detailed breakdown already explains the meaning. Keep the
            # full summary in notes instead of repeating it above three rows.
            card['show_meaning']=False
        excerpt=next((e for e in excerpts if e['title']==card['title'] and e['excerpt']==card['example']),None)
        if not excerpt:continue
        result=ask_deepseek('将给定英文原句片段准确翻译为简体中文，只翻译excerpt，不补上全文未显示的后半部分。中文不超过36字。返回JSON {translation:"..."}。',json.dumps(excerpt,ensure_ascii=False),max_tokens=512,cache_dir=cache_dir)
        value=result.get('translation')
        if not isinstance(value,str) or not value.strip() or len(value)>40:raise ProviderError('原句片段的译文未通过长度检查。')
        card['translation']=OpenCC('t2s').convert(value.strip())
