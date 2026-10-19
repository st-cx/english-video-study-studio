from __future__ import annotations

from pathlib import Path


def _timestamp(seconds: float, ass: bool = False) -> str:
    ticks = max(0, round(seconds * (100 if ass else 1000)))
    scale = 100 if ass else 1000
    hours, rest = divmod(ticks, 3600 * scale)
    minutes, rest = divmod(rest, 60 * scale)
    secs, fraction = divmod(rest, scale)
    return f"{hours}:{minutes:02}:{secs:02}.{fraction:02}" if ass else f"{hours:02}:{minutes:02}:{secs:02},{fraction:03}"


def _join(words):
    import re
    return ''.join((' ' if i and not re.match(r'^[.,!?;:]',w['text']) else '')+w['text'] for i,w in enumerate(words))


def _pages(package):
    by_id={w['id']:w for w in package['words']}
    sentences={s['id']:s for s in package['sentences']}
    if package.get('layout',{}).get('pages'):
        result=[]
        for page in package['layout']['pages']:
            lines=[[by_id[wid] for wid in line] for line in page['lines']]
            result.append({**page,'lines':lines,'words':[w for line in lines for w in line],'translation':page.get('translation') or sentences[page['sentence_id']]['translation']})
        return result
    return [{**s,'lines':[[by_id[i] for i in s['word_ids']]],'words':[by_id[i] for i in s['word_ids']]} for s in package['sentences']]


def write_srt(segments: list[dict], language: str, destination: Path) -> None:
    rows=[]
    for item in segments:
        text=item.get('translation','') if language=='zh' else item.get('text','')
        if not text.strip():continue
        rows.extend([str(len(rows)//4+1),f"{_timestamp(item['start_ms']/1000)} --> {_timestamp(item['end_ms']/1000)}",text.strip(),''])
    destination.write_text('\n'.join(rows),encoding='utf-8-sig')


def _escape(text: str) -> str:
    return text.replace('\\','\\\\').replace('{','\\{').replace('}','\\}').replace('\n','\\N')


def write_ass(package: dict, destination: Path) -> None:
    rows=['[Script Info]','ScriptType: v4.00+','PlayResX: 1920','PlayResY: 1080','WrapStyle: 2','ScaledBorderAndShadow: yes','','[V4+ Styles]',
          'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
          'Style: English,Arial,48,&H00FFFFFF,&H0000A5FF,&H500B111C,&HA0000000,0,0,0,0,100,100,0,0,1,3,0,2,226,226,108,1',
          'Style: Chinese,Microsoft YaHei,36,&H00FFFFFF,&H0000A5FF,&H500B111C,&H950B111C,0,0,0,0,100,100,0,0,1,2,0,8,344,65,25,1','',
          '[Events]','Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
    pages=_pages(package)
    for index,page in enumerate(pages):
        end=pages[index+1]['start_ms'] if index+1<len(pages) else min(round(package['episode']['duration_s']*1000),page['end_ms']+1200)
        edges=sorted({page['start_ms'],end,*[max(page['start_ms'],min(end,w[k])) for w in page['words'] for k in ('start_ms','end_ms')]})
        for start,finish in zip(edges,edges[1:]):
            if finish<=start:continue
            lines=[]
            for line in page['lines']:
                parts=[]
                for i,w in enumerate(line):
                    color='&H0057ADFF&' if w['start_ms']<=start<w['end_ms'] else '&H0071D2F0&' if start>=w['end_ms'] else '&H00FFFFFF&'
                    parts.append('{\\1c'+color+'}'+_escape(w['text']))
                lines.append(' '.join(parts))
            rows.append(f"Dialogue: 0,{_timestamp(start/1000,True)},{_timestamp(finish/1000,True)},English,,0,0,0,,"+'\\N'.join(lines))
        rows.append(f"Dialogue: 1,{_timestamp(page['start_ms']/1000,True)},{_timestamp(end/1000,True)},Chinese,,0,0,0,,"+_escape(page.get('translation','')))
    destination.write_text('\n'.join(rows),encoding='utf-8-sig')


def make_captions(package: dict, folder: Path) -> tuple[Path,Path,Path]:
    en,zh,ass=folder/'subtitles.en.srt',folder/'subtitles.zh.srt',folder/'captions.ass'
    pages=_pages(package)
    segments=[]
    for i,p in enumerate(pages):
        end=pages[i+1]['start_ms'] if i+1<len(pages) else min(round(package['episode']['duration_s']*1000),p['end_ms']+1200)
        segments.append({**p,'end_ms':end,'text':'\n'.join(_join(line) for line in p['lines'])})
    write_srt(segments,'en',en);write_srt(segments,'zh',zh);write_ass(package,ass)
    return en,zh,ass
