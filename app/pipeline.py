from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .bbc import episode_fingerprint, fetch_episode
from .providers import ProviderError, ask_deepseek
from .storage import output_root, save_episode
from .subtitles import make_captions
from .transcript import assign_speakers, normalize_orthography, official_vocabulary, spoken_text, tokens

Progress = Callable[[str, int, str], None]


def _ffprobe_duration(path: Path) -> float:
    probe = shutil.which("ffprobe")
    if not probe:
        raise RuntimeError("找不到 ffprobe。请安装 FFmpeg 并将其加入 PATH。")
    result = subprocess.run(
        [probe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return float(result.stdout.strip())


def _local_whisper_path() -> str:
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-medium" / "snapshots"
    snapshots = sorted(root.glob("*")) if root.exists() else []
    return str(snapshots[-1]) if snapshots else "medium.en"


def transcribe_words(audio_path: Path, progress: Progress) -> tuple[list[dict], float, str]:
    from .speech_runtime import configure_cuda_runtime
    configure_cuda_runtime()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("缺少 faster-whisper。请运行 install.bat 安装本工具依赖。") from exc
    local_path = _local_whisper_path()
    backend = "GPU"
    try:
        progress("transcribe", 15, "正在加载本机 GPU 语音模型…")
        model = WhisperModel(local_path, device="cuda", compute_type="float16")
        segments, info = model.transcribe(str(audio_path), language="en", word_timestamps=True, vad_filter=True, beam_size=3, condition_on_previous_text=False)
        words, previous_start = [], 0.0
        duration = float(info.duration or _ffprobe_duration(audio_path))
        for spoken in segments:
            for token in spoken.words or []:
                text = token.word.strip()
                if not text:
                    continue
                start, end = float(token.start), float(token.end)
                words.append({"text": text, "start_ms": round(start * 1000), "end_ms": round(end * 1000), "probability": round(float(token.probability), 4), "whisper_segment": round(float(spoken.start), 2)})
            previous_start = float(spoken.end)
            pct = min(77, 15 + int(previous_start / max(duration, 1) * 62))
            progress("transcribe", pct, f"逐词识别中 {int(previous_start)} / {int(duration)} 秒")
        del model
    except Exception as gpu_error:
        # Missing CUDA runtime libraries are common even when an NVIDIA GPU is present.
        try:
            del model
        except UnboundLocalError:
            pass
        backend = "CPU"
        progress("transcribe", 16, f"GPU 运行库不可用，正在改用 CPU：{str(gpu_error)[:120]}")
        try:
            model = WhisperModel(local_path, device="cpu", compute_type="int8")
            segments, info = model.transcribe(str(audio_path), language="en", word_timestamps=True, vad_filter=True, beam_size=3, condition_on_previous_text=False)
            words, previous_start = [], 0.0
            duration = float(info.duration or _ffprobe_duration(audio_path))
            for spoken in segments:
                for token in spoken.words or []:
                    text = token.word.strip()
                    if not text:
                        continue
                    start, end = float(token.start), float(token.end)
                    words.append({"text": text, "start_ms": round(start * 1000), "end_ms": round(end * 1000), "probability": round(float(token.probability), 4), "whisper_segment": round(float(spoken.start), 2)})
                previous_start = float(spoken.end)
                pct = min(77, 15 + int(previous_start / max(duration, 1) * 62))
                progress("transcribe", pct, f"CPU 逐词识别中 {int(previous_start)} / {int(duration)} 秒")
            del model
        except Exception as cpu_error:
            raise RuntimeError(f"本机语音识别失败（GPU 与 CPU）：{str(cpu_error)[:500]}") from cpu_error
    if not words:
        raise RuntimeError("音频中未识别到英文内容。")
    return words, duration, backend


def _build_sentences(words: list[dict]) -> tuple[list[dict], list[dict]]:
    buckets: list[list[dict]] = []
    active: list[dict] = []
    for word in words:
        gap_ms = word["start_ms"] - active[-1]["end_ms"] if active else 0
        if active and (word.get('speaker', '') != active[-1].get('speaker', '') or gap_ms > 2500 or (gap_ms > 900 and len(active) >= 8)):
            buckets.append(active)
            active = []
        active.append(word)
        if re.search(r"[.!?][\"']?$", word["text"]) and not re.fullmatch(r"(?:Mr|Mrs|Ms|Dr|Prof)\.", word['text']):
            buckets.append(active)
            active = []
    if active:
        buckets.append(active)
    sentences: list[dict] = []
    all_words: list[dict] = []
    for idx, group in enumerate(buckets):
        if not group:
            continue
        sentence_id = f"s{idx:04d}"
        ids = []
        for wi, word in enumerate(group):
            wid = f"w{len(all_words):05d}"
            ids.append(wid)
            all_words.append({"id": wid, "sentence_id": sentence_id, "index": wi, "text": word["text"], "start_ms": word["start_ms"], "end_ms": word["end_ms"], "gloss_zh": "", "probability": word["probability"]})
            for field in ('connected_words','spoken_form','timing_unit','phrase_unit','spoken_words','original_zero_diagnostics','timing_evidence'):
                if field in word:all_words[-1][field]=word[field]
        text = " ".join(word["text"] for word in group)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        sentences.append({"id": sentence_id, "start_ms": group[0]["start_ms"], "end_ms": group[-1]["end_ms"], "text": text, "translation": "", "speaker": group[0].get('speaker', ''), "word_ids": ids, "card_ids": []})
    return sentences, all_words


def _official_vocabulary(transcript: str) -> str:
    return official_vocabulary(transcript)


def _annotate(package: dict, raw_transcript: str, progress: Progress, cache_dir: Path | None = None) -> None:
    from opencc import OpenCC
    converter = OpenCC('t2s')
    def simplify(value):
        if isinstance(value, dict): return {k: simplify(v) for k, v in value.items()}
        if isinstance(value, list): return [simplify(v) for v in value]
        return converter.convert(value) if isinstance(value, str) else value
    sentences, words = package["sentences"], package["words"]
    context_sentences=package.get('_annotation_context',sentences)
    context_positions={s['id']:i for i,s in enumerate(context_sentences)}
    for sentence in sentences:
        sentence['card_ids'] = []
    word_by_id = {word["id"]: word for word in words}
    source_vocab = _official_vocabulary(raw_transcript) if package.get('source',{}).get('kind','bbc')=='bbc' else ''
    all_cards: list[dict] = []
    all_phrases: list[dict] = []
    system = (
        "你是英语视频的双语精读编辑，服务中国四级入门学习者。所有中文必须使用简体中文。只依据输入的英文句子和可靠参考词汇编写，"
        "不虚构事实或错误标记词性。自然、准确地翻译每句；重点解释固定搭配、习语和有学习价值的词，"
        "不定式to、关系代词that/who、宾语从句that须按语法作用解释，不机械译为去、那、谁。副词不决定动词必须用原形；means doing中的doing是动名词短语；be+过去分词构成被动时be是助动词。编程口述中的type of不能教成Python代码，须按上下文区分type()函数与自然语言of；公式口述保留运算顺序与括号。口头截断和重启不构成技术定义，不能编造命题。人名与术语在整期保持一致，未核实的中文书名保留英文。"
        "每个输入词都必须给出本句语境里的简短中文义，包括功能词、助动词和专名，每条通常不超过24字；不能用‘结合上下文理解’等占位句。"
        "每句只把1到3个值得记忆的实词标为focus:true，其他词为false；功能词和人名照常给义但不要挤进重点词表。"
        "固定短语和常用搭配另放入 phrases，必须关联同一句中连续的2到8个词语ID，解释整体语境义，不拆开解释。"
        "值得积累的固定搭配标study:true；仅为保持意群完整而组合的普通语块、机构名标study:false。"
        "只为确有学习价值的句子生成讲解卡，每句最多一张；优先官方词汇，之后才考虑搭配或长难句。"
        "难句卡用points分成最多3项（主干/修饰/理解等）；例句只能逐字摘自输入原句，不要另造例句。"
        "只挑重点上卡，普通短句不要硬讲解。卡片title尽量短于35字符；meaning不超过30字，structure不超过55字符，example截取原句中不超过100字符的连续片段，translation不超过36字，points最多3项，每项text不超过40字符。"
        "严格输出 JSON object，字段为 title_zh、translations、glosses、phrases、cards。title_zh 是本期标题的自然中文译名；translations 是 [{id,zh}]；"
        "glosses 是 [{word_id,zh,focus}]；phrases 是 [{word_ids,zh,study}]；cards 是 [{type,title,meaning,structure,example,translation,word_ids,sentence_ids,points:[{label,text}]}]。"
        "没有值得解释的句子可省略 card。不要返回时间、字幕或输入中不存在的句子。"
    )
    # Bound output size by words as well as sentences. Compatible gateways may
    # time out while generating hundreds of glossary entries in one response.
    batches = []
    batch, word_count = [], 0
    for sentence in sentences:
        count=len(sentence['word_ids'])
        if batch and (word_count+count>48 or len(batch)>=4):
            batches.append(batch);batch=[];word_count=0
        batch.append(sentence);word_count+=count
    if batch:batches.append(batch)
    for batch_index, batch in enumerate(batches):
        ids = {sentence["id"] for sentence in batch}
        batch_words = [word for word in words if word["sentence_id"] in ids]
        payload = {"episode_title": package["episode"]["title"], "official_vocabulary": source_vocab, "context_before": [s['text'] for s in context_sentences[max(0,context_positions[batch[0]['id']]-2):context_positions[batch[0]['id']]]], "context_after": [s['text'] for s in context_sentences[context_positions[batch[-1]['id']]+1:context_positions[batch[-1]['id']]+3]], "sentences": [{"id": s["id"], "english": s["text"], "focus_word_ids": [wid for wid in s["word_ids"] if len(re.sub(r"[^A-Za-z]", "", word_by_id[wid]["text"])) >= 4][:5], "words": [{"id": word_by_id[wid]["id"], "text": word_by_id[wid]["text"]} for wid in s["word_ids"]]} for s in batch]}
        progress("teaching", 78 + int(batch_index / max(1, len(batches)) * 13), f"正在编写双语讲解 {batch_index + 1} / {len(batches)}")
        try:
            result = simplify(ask_deepseek(system, json.dumps(payload, ensure_ascii=False), max_tokens=8192, cache_dir=cache_dir))
            def complete_function_words(candidate):
                for card in candidate.get('cards',[]):
                    example=card.get('example','')
                    if isinstance(example,str) and len(example)>80:
                        boundaries=[m.end() for m in re.finditer(r'[,;:](?:\s|$)',example) if 35<=m.end()<=80]
                        if boundaries:
                            card['example_full']=example
                            card['example']=example[:min(boundaries)].rstrip()
                            package.setdefault('processing',{}).setdefault('example_excerpts',[]).append({'title':card.get('title'),'full':example,'excerpt':card['example'],'reason':'natural_clause_boundary'})
                for phrase in candidate.get('phrases',[]):
                    linked=[word_by_id.get(wid) for wid in phrase.get('word_ids',[])]
                    if not linked or any(w is None for w in linked) or len({w['sentence_id'] for w in linked})!=1: continue
                    first,last=min(w['index'] for w in linked),max(w['index'] for w in linked)
                    full=[w for w in batch_words if w['sentence_id']==linked[0]['sentence_id'] and first<=w['index']<=last]
                    missing=[w for w in full if w not in linked]
                    if missing and len(full)<=8 and all(w['text'].lower().strip('.,!?;:') in {'to','a','an','the','of','and','or','for'} for w in missing):
                        before=list(phrase['word_ids']);phrase['word_ids']=[w['id'] for w in full]
                        package.setdefault('processing',{}).setdefault('phrase_id_repairs',[]).append({'before':before,'after':phrase['word_ids'],'reason':'restored_omitted_function_words'})
            complete_function_words(result)
            def validate(candidate):
                for phrase in candidate.get('phrases',[]):
                    linked=[word_by_id.get(wid) for wid in dict.fromkeys(phrase.get('word_ids',[]))]
                    if not 2<=len(linked)<=8 or any(w is None or w['sentence_id'] not in ids for w in linked):
                        return 'phrases 中有无效词语 ID 或长度不在2到8词。'
                    linked.sort(key=lambda w:w['index'])
                    if len({w['sentence_id'] for w in linked})!=1 or any(b['index']!=a['index']+1 for a,b in zip(linked,linked[1:])):
                        return 'phrases 必须引用同一句中连续的所有词语 ID，不能跳过中间词；错误词组为'+json.dumps(phrase,ensure_ascii=False)+'。不能连续解释的搭配放在cards，删除该phrase。'
                for card in candidate.get('cards',[]):
                    if not isinstance(card.get('title'),str) or not card['title'].strip():
                        return 'cards中的可选讲解卡须有非空title；不能给出可靠标题时请删除该card，不得编造。'
                    linked=[s for s in batch if s['id'] in card.get('sentence_ids',[])]
                    example=card.get('example','')
                    if not linked or not isinstance(example,str) or example and example.lower() not in ' '.join(s['text'] for s in linked).lower():
                        return 'cards 必须有有效sentence_ids；example必须是对应输入英文中逐字连续的片段，不得添加省略号或改写。'
                    if any(len(str(card.get(k,'')))>limit for k,limit in {'title':55,'meaning':60,'structure':85,'example':160,'translation':65}.items()) or any(len(str(p.get('text','')))>55 for p in card.get('points',[])):
                        return '卡片'+str(card.get('title'))+'内容过长，当前各字段字符数为'+json.dumps({k:len(str(card.get(k,''))) for k in ('title','meaning','structure','example','translation')})+'。限制：title<=55字符，meaning<=60字，structure<=85字符，example<=160字符，translation<=65字，points每项<=55字符。请精简；example可只保留连续原句片段。'
                return ''
            problem=validate(result)
            if problem:
                repair_system=system+' 上次结果未通过格式校验，请仅修复指出的问题并返回整组完整JSON。'
                result=simplify(ask_deepseek(repair_system,json.dumps({'input':payload,'previous_output':result,'validation_error':problem},ensure_ascii=False),max_tokens=8192,cache_dir=cache_dir))
                complete_function_words(result)
                if validate(result):
                    # Phrase suggestions are optional. A model repeatedly
                    # linking non-contiguous words must not teach that unsafe
                    # phrase or block otherwise complete translations/glosses.
                    valid_phrases=[]
                    for phrase in result.get('phrases',[]):
                        reason=validate({'phrases':[phrase],'cards':[]})
                        if reason:
                            package.setdefault('processing',{}).setdefault('omitted_phrase_suggestions',[]).append({'candidate':phrase,'reason':reason})
                        else:valid_phrases.append(phrase)
                    result['phrases']=valid_phrases
                    valid_cards=[]
                    for card in result.get('cards',[]):
                        reason=validate({'phrases':[],'cards':[card]})
                        if reason:
                            package.setdefault('processing',{}).setdefault('omitted_card_suggestions',[]).append({'candidate':card,'reason':reason})
                        else:valid_cards.append(card)
                    result['cards']=valid_cards
                    if validate(result):raise ProviderError('教学数据自动修复后仍不满足关联和长度规则：'+validate(result))
                package.setdefault('processing',{}).setdefault('teaching_format_repairs',[]).append({'batch':batch_index+1,'reason':problem})
            if not package["episode"].get("title_zh") and isinstance(result.get("title_zh"), str):
                package["episode"]["title_zh"] = result["title_zh"].strip()[:120]
            translations = result.get("translations", [])
            for item in translations:
                target = next((s for s in batch if s["id"] == item.get("id")), None)
                if target and isinstance(item.get("zh"), str):
                    target["translation"] = item["zh"].strip()
            for item in result.get("glosses", []):
                target = word_by_id.get(item.get("word_id"))
                if target and target["sentence_id"] in ids and isinstance(item.get("zh"), str):
                    target["gloss_zh"] = item["zh"].strip()
                    target["focus"] = item.get("focus") is True
            for item in result.get("phrases", []):
                phrase_ids = list(dict.fromkeys(item.get("word_ids", [])))
                linked_words = [word_by_id.get(wid) for wid in phrase_ids]
                if not 2 <= len(linked_words) <= 8 or any(w is None for w in linked_words):
                    raise ProviderError("词组关联了无效的词语 ID，已暂停等待重新生成。")
                linked_words.sort(key=lambda w: w["index"])
                sid = linked_words[0]["sentence_id"]
                if sid not in ids or any(w["sentence_id"] != sid for w in linked_words) or any(b["index"] != a["index"] + 1 for a, b in zip(linked_words, linked_words[1:])):
                    raise ProviderError("词组跨句或不连续，已暂停以免错误解释搭配。")
                meaning = str(item.get("zh", "")).strip()
                if not meaning or len(meaning) > 60:
                    raise ProviderError("词组释义为空或过长，需要重新生成。")
                all_phrases.append({"id": f"p{len(all_phrases):04d}", "sentence_id": sid, "word_ids": [w["id"] for w in linked_words], "text": " ".join(w["text"] for w in linked_words).strip(".,!?;:"), "gloss_zh": meaning, "study": item.get("study") is not False})
            for raw_card in result.get("cards", []):
                word_ids = [wid for wid in raw_card.get("word_ids", []) if wid in word_by_id and word_by_id[wid]["sentence_id"] in ids]
                sentence_ids = [sid for sid in raw_card.get("sentence_ids", []) if sid in ids]
                if not sentence_ids and word_ids:
                    sentence_ids = list(dict.fromkeys(word_by_id[wid]["sentence_id"] for wid in word_ids))
                if not sentence_ids:
                    continue
                linked = [sentence for sentence in batch if sentence["id"] in sentence_ids]
                card = {
                    "id": f"c{len(all_cards):04d}",
                    "type": "grammar" if raw_card.get("type") == "grammar" else "vocab",
                    "title": str(raw_card.get("title", ""))[:100],
                    "meaning": str(raw_card.get("meaning", ""))[:500],
                    "structure": str(raw_card.get("structure", ""))[:800],
                    "example": str(raw_card.get("example", ""))[:500],
                    "translation": str(raw_card.get("translation", ""))[:500],
                    "word_ids": word_ids,
                    "sentence_ids": sentence_ids,
                    "start_ms": min(s["start_ms"] for s in linked),
                    "end_ms": max(max(s["end_ms"] for s in linked), min(s["start_ms"] for s in linked) + 8000),
                    "points": [{"label": str(point.get("label", ""))[:12], "text": str(point.get("text", ""))} for point in raw_card.get("points", [])[:3]],
                }
                if any(point['label'] in {'主干','修饰','从句','句子主干'} for point in card['points']):
                    card['type']='grammar'
                if not card["title"] or len(card["meaning"]) > 80 or len(card["structure"]) > 120 or len(card["example"]) > 190 or any(len(point["text"]) > 80 for point in card["points"]):
                    raise ProviderError("讲解卡内容超出一屏的阅读量，已暂停而不是裁切文字。")
                original_context = " ".join(s["text"] for s in linked)
                if card["example"] and card["example"].lower() not in original_context.lower():
                    raise ProviderError("讲解例句不能定位到输入原句，已暂停以免编造出处。")
                all_cards.append(card)
                for sentence in linked:
                    sentence["card_ids"].append(card["id"])
            for repair_attempt in range(3):
                invalid_words=[w for w in batch_words if not w['gloss_zh'] or len(w['gloss_zh'])>70]
                if not invalid_words:break
                repair=simplify(ask_deepseek(
                    '只输出JSON object。为每个指定word_id补齐或精简本句语境义，所有中文简体，通常24字以内；功能词说明语法作用。严格返回 {"glosses":[{"word_id":"输入中的word_id","zh":"实际中文义"}]}，不要返回id或word_ids，不得占位。',
                    json.dumps({'sentences':payload['sentences'],'required_words':[{'word_id':w['id'],'text':w['text'],'previous':w['gloss_zh']} for w in invalid_words],'repair_attempt':repair_attempt+1},ensure_ascii=False),
                    max_tokens=8192,cache_dir=cache_dir))
                requested={w['id'] for w in invalid_words}
                for item in repair.get('glosses',[]):
                    target=word_by_id.get(item.get('word_id'))
                    if target and target['id'] in requested and isinstance(item.get('zh'),str):
                        target['gloss_zh']=item['zh'].strip()
            if any(not w['gloss_zh'] or len(w['gloss_zh'])>70 for w in batch_words):
                raise ProviderError('当前词义多次补齐后仍有缺漏或过长内容，已暂停，避免成片出现占位提示。')
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"教学内容生成失败（第 {batch_index + 1} 组）：{str(exc)[:300]}") from exc
    for attempt in range(2):
        missing=[s for s in sentences if not s['translation']]
        if not missing:break
        repaired=simplify(ask_deepseek('只输出JSON object，使用简体中文补齐各句自然译文，必须使用输入id。返回 {"translations":[{"id":"原id","zh":"译文"}]}。口语片段根据上下文理解，不凭空补事实。',json.dumps({'missing_sentences':[{'id':s['id'],'english':s['text']} for s in missing],'attempt':attempt+1},ensure_ascii=False),max_tokens=8192,cache_dir=cache_dir))
        required={s['id']:s for s in missing}
        for item in repaired.get('translations',[]):
            if item.get('id') in required and isinstance(item.get('zh'),str):required[item['id']]['translation']=item['zh'].strip()
    missing=sum(not sentence['translation'] for sentence in sentences)
    if missing:raise ProviderError(f'有 {missing} 句多次补齐后仍没有中文翻译，已暂停以免输出不完整材料。')
    package["phrases"] = all_phrases
    # Persist display intervals: cards too close together remain in notes only.
    display_cards = []
    duration_ms = round(package["episode"]["duration_s"] * 1000)
    for card in sorted(all_cards, key=lambda c: c["start_ms"]):
        card["display"] = not display_cards or card["start_ms"] - display_cards[-1]["start_ms"] >= 8000
        if duration_ms - card["start_ms"] < 8000:
            card["display"] = False
        if card["display"]:
            display_cards.append(card)
    for index, card in enumerate(display_cards):
        card["end_ms"] = display_cards[index + 1]["start_ms"] if index + 1 < len(display_cards) else duration_ms
    package["cards"] = all_cards
    from .presentation import translate_excerpts
    translate_excerpts(package,cache_dir)


def _quality(package: dict, transcript: str) -> dict:
    recognized = tokens(' '.join(w['text'] for w in package['words']))
    source = tokens(spoken_text(transcript))
    ratio = difflib.SequenceMatcher(None, source, recognized, autojunk=False).ratio() if source and recognized else None
    invalid = [w for w in package["words"] if w["end_ms"] <= w["start_ms"]]
    probs = [w["probability"] for w in package["words"] if w["probability"] is not None]
    average = sum(probs) / len(probs) if probs else 1.0
    flags = []
    if ratio is not None and ratio < 0.68:
        flags.append("参考文稿与音频识别结果差异较大；请检查是否选错素材或字幕。")
    if invalid:
        flags.append(f"有 {len(invalid)} 个词没有可靠的正时长时间戳，请检查原声后重新对齐。")
    overlap = sum(a['end_ms']-b['start_ms']>80 for a,b in zip(package['words'],package['words'][1:]))
    if overlap:
        flags.append(f'有 {overlap} 处词语时间重叠超过 80ms，需要核对。')
    if package.get('processing',{}).get('alignment_disputes'):
        flags.append('独立识别和强制对齐仍有明显差异，请核对对应原声片段。')
    outside = sum(w['start_ms'] < 0 or w['end_ms'] > package['episode']['duration_s']*1000+200 for w in package['words'])
    if outside:
        flags.append('存在超出原声音频范围的时间戳。')
    if average < 0.42:
        flags.append("语音识别置信度偏低，请抽听检查专名和数字。")
    if not package["sentences"]:
        flags.append("没有可用字幕句子。")
    low_confidence = int(sum(bool(w["probability"] is not None and w["probability"] < 0.18) for w in package["words"]))
    if low_confidence > max(4, len(package["words"]) * 0.06):
        flags.append(f"有 {low_confidence} 个单词的识别置信度偏低。")
    structural = []
    if invalid: structural.append('存在零时长或负时长的词语。')
    if overlap: structural.append('存在词语时间重叠。')
    if outside: structural.append('存在音频范围之外的词语。')
    if not package['sentences']: structural.append('没有可用字幕句子。')
    return {"needs_review": bool(flags), "flags": flags, "structural_flags": structural, "source_similarity": round(ratio, 3) if ratio is not None else None, "reference_available":bool(source), "mean_asr_confidence": round(average, 3), "low_confidence_word_count": low_confidence, "timed_word_count": len(package["words"]), "bad_timestamp_count": len(invalid)}


def create_bbc_episode(url: str, job_folder: Path, progress: Progress, allow_review_bypass: bool = False) -> dict:
    episode = fetch_episode(url, job_folder)
    return _create_episode(episode, job_folder, progress, allow_review_bypass)


def create_imported_episode(title: str, audio_path: Path, transcript: str, job_folder: Path, progress: Progress, source_url: str = "local import", cover_path: str = "", allow_review_bypass: bool = False) -> dict:
    if not audio_path.is_file() or not transcript.strip():
        raise ValueError("导入需要音频文件和英文文稿。")
    from hashlib import sha1
    episode = {"id": sha1((title + str(audio_path)).encode()).hexdigest()[:12], "title": title.strip() or "Imported listening", "source_url": source_url, "audio_url": "", "transcript_url": "", "audio_path": str(audio_path), "transcript_path": "", "transcript": transcript, "cover_path": cover_path, "duration_s": 0, "source_date": ""}
    return _create_episode(episode, job_folder, progress, allow_review_bypass)


def _create_episode(episode: dict, job_folder: Path, progress: Progress, allow_review_bypass: bool, prepare_only: bool = False) -> dict:
    folder = output_root() / episode["id"]
    if (folder / "episode.json").is_file():
        existing = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
        if (existing.get("episode", {}).get("source_url") == episode.get("source_url") or episode.get("source_url") == "local import") and existing.get("render", {}).get("status") == "ready" and all((folder / name).is_file() for name in ('lesson.mp4','study.html','audio.mp3','study-notes.md','subtitles.en.srt','subtitles.zh.srt')):
            progress("complete", 100, "这一期已经生成过，已载入现有学习包。")
            return existing
    folder.mkdir(parents=True, exist_ok=True)
    audio_path = Path(episode["audio_path"])
    fingerprint = episode_fingerprint(episode, audio_path) if episode.get("source_url") != "local import" else ""
    shutil.copy2(audio_path, folder / "audio.mp3")
    if episode.get("cover_path") and Path(episode["cover_path"]).is_file():
        shutil.copy2(episode["cover_path"], folder / "cover.jpg")
    checkpoint = folder / "episode.checkpoint.json"
    package = None
    if checkpoint.is_file():
        try:
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if cached.get("episode", {}).get("fingerprint") == fingerprint and cached.get("words") and cached.get("sentences"):
                package = cached
                progress("transcribe", 72, "已恢复逐词识别结果，继续生成教学内容…")
        except (OSError, ValueError):
            package = None
    if package is None:
        progress("transcribe", 10, "素材已准备，开始本机逐词识别…")
        asr_cache = folder / 'asr.checkpoint.json'
        try:
            cached_asr = json.loads(asr_cache.read_text(encoding='utf-8'))
            if cached_asr.get('fingerprint') != fingerprint: raise ValueError('素材不同')
            words, duration, backend = cached_asr['words'], cached_asr['duration'], cached_asr['asr_backend']
            progress('transcribe',72,'已恢复本期原声的识别结果，继续对齐…')
        except (OSError,ValueError,KeyError):
            words, duration, backend = transcribe_words(audio_path, progress)
        words, corrections = normalize_orthography(words, episode.get('transcript', ''))
        from .alignment import align_words,group_connected_speech
        raw_checkpoint = {'words': words, 'duration': duration, 'asr_backend': backend, 'fingerprint': fingerprint}
        save_episode(raw_checkpoint, folder, 'asr.checkpoint.json')
        words, alignment_info = align_words(audio_path, words, _local_whisper_path(), progress)
        words, connected=group_connected_speech(words)
        alignment_info['connected_speech_groups']=connected
        assign_speakers(words, episode.get('transcript', ''))
        sentences, timed_words = _build_sentences(words)
        package = {
            "version": 3,
            "episode": {"id": episode["id"], "title": episode["title"], "title_zh": "", "source_url": episode["source_url"], "audio_url": episode.get("audio_url", ""), "duration_s": round(duration, 3), "source_date": episode.get("source_date", ""), "cover": "cover.jpg" if episode.get("cover_path") else "", "audio_file": "audio.mp3", "fingerprint": fingerprint, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "sentences": sentences,
            "words": timed_words,
            "cards": [],
            "quality": {},
            "review": {"official_transcript": episode.get("transcript", ""), "recognized_transcript": " ".join(sentence["text"] for sentence in sentences), "approved": False},
            "render": {"width": 1920, "height": 1080, "fps": 30, "status": "pending"},
            "processing": {"asr_backend": backend, 'orthography_corrections': corrections, **alignment_info},
        }
    episode["duration_s"] = package['episode']['duration_s']
    episode["transcript"] = episode.get("transcript", "")
    review = _quality(package, episode["transcript"])
    package["quality"] = review
    package["review"]["official_transcript"] = episode.get("transcript", "")
    save_episode(package, folder, "episode.checkpoint.json")
    if review["needs_review"] and (review['structural_flags'] or not (allow_review_bypass or package['review'].get('approved'))):
        save_episode(package, folder)
        raise ReviewRequired("; ".join(review["flags"]), package)
    try:
        _annotate(package, episode["transcript"], progress, folder / '.teaching-cache')
        from .teaching_review import review_teaching
        review_teaching(package, progress, folder / '.teaching-cache')
    except ProviderError as exc:
        save_episode(package,folder)
        if package.get('teaching_review',{}).get('status')=='needs_review':
            package['quality']['needs_review']=True
            package['quality']['flags'].append(str(exc))
            save_episode(package,folder)
            raise ReviewRequired(str(exc),package) from exc
        raise
    from .layout import prepare_layout
    prepare_layout(package, folder)
    progress("render", 94, "教学内容已完成，正在渲染逐词高亮视频…")
    en, zh, ass = make_captions(package, folder)
    package["render"]["ass_file"] = ass.name
    package["render"]["english_subtitles"] = en.name
    package["render"]["chinese_subtitles"] = zh.name
    notes = make_notes(package)
    (folder / "study-notes.md").write_text(notes, encoding="utf-8-sig")
    package["render"].update({"status": "rendering", "video_file": "lesson.mp4", "study_notes": "study-notes.md", "english_subtitles": en.name, "chinese_subtitles": zh.name})
    package_path = save_episode(package, folder)
    if prepare_only:
        package['render']['status']='prepared'
        save_episode(package,folder)
        return package
    from .renderer import render_video
    render_video(package_path, folder, progress)
    if (folder/'layout-audit.json').is_file():
        audit=json.loads((folder/'layout-audit.json').read_text(encoding='utf-8'))
        package['processing']['layout_audit']={k:audit[k] for k in ('status','cases','checks')}
    package['render']['status'] = 'ready'
    save_episode(package, folder)
    from .study_page import write_study_page
    write_study_page(package, folder)
    return package


class ReviewRequired(RuntimeError):
    def __init__(self, message: str, package: dict):
        super().__init__(message)
        self.package = package


def continue_after_review(package: dict, folder: Path, progress: Progress, corrected_transcript: str = "", chapter_id: str = '') -> dict:
    if package.get('version',0)>=4:
        from .general import continue_media_review
        return continue_media_review(package,folder,progress,corrected_transcript,chapter_id)
    episode = package["episode"]
    original = package["review"].get("recognized_transcript", "")
    if corrected_transcript.strip() and re.sub(r"\W+", " ", corrected_transcript.lower()).strip() != re.sub(r"\W+", " ", original.lower()).strip():
        progress("align", 84, "正在用 stable-ts 将你校正的文稿重新对齐到原声…")
        _realign_corrected(package, folder / "audio.mp3", corrected_transcript.strip(), progress)
        package["review"].setdefault("correction_history", []).append({"before": original, "after": corrected_transcript.strip(), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        package["review"]["recognized_transcript"] = corrected_transcript.strip()
    checked = _quality(package, package['review'].get('official_transcript', ''))
    package['quality'] = checked
    # Persist corrected source before any network processing can fail.
    package['review']['approved'] = False
    save_episode(package, folder)
    save_episode(package, folder, 'episode.checkpoint.json')
    if checked['structural_flags']:
        raise ReviewRequired('；'.join(checked['structural_flags']), package)
    package['quality']['acknowledged_flags'] = checked['flags']
    package['quality']['flags'] = []
    package['quality']['needs_review'] = False
    package["review"]["approved"] = True
    package["review"]["approved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_episode(package, folder)
    save_episode(package, folder, 'episode.checkpoint.json')
    # Rebuild an annotated language package from the held source and current recognized transcript.
    _annotate(package, package["review"].get("official_transcript", ""), progress, folder / '.teaching-cache')
    from .teaching_review import review_teaching
    review_teaching(package, progress, folder / '.teaching-cache')
    from .layout import prepare_layout
    prepare_layout(package, folder)
    make_captions(package, folder)
    (folder / "study-notes.md").write_text(make_notes(package), encoding="utf-8-sig")
    package["render"].update({"status": "rendering", "video_file": "lesson.mp4", "study_notes": "study-notes.md"})
    path = save_episode(package, folder)
    from .renderer import render_video
    render_video(path, folder, progress)
    if (folder/'layout-audit.json').is_file():
        audit=json.loads((folder/'layout-audit.json').read_text(encoding='utf-8'))
        package.setdefault('processing',{})['layout_audit']={k:audit[k] for k in ('status','cases','checks')}
    package['render']['status'] = 'ready'
    save_episode(package, folder)
    from .study_page import write_study_page
    write_study_page(package, folder)
    return package


def _realign_corrected(package: dict, audio_path: Path, corrected: str, progress: Progress) -> None:
    previous_words=[dict(w) for w in package.get('words',[])]
    try:
        import stable_whisper
    except ImportError as exc:
        raise RuntimeError("修改了英文稿，但当前环境没有 stable-ts。请先运行 install.bat 安装完整依赖。") from exc
    model = stable_whisper.load_faster_whisper(_local_whisper_path(), device="cpu", compute_type="int8")
    result = model.align(str(audio_path), corrected, language="English")
    words = []
    for segment in result.segments:
        for token in (segment.words or []):
            text = getattr(token, "word", "") or ""
            if not text.strip():
                continue
            start = float(token.start)
            end = float(token.end)
            words.append({"text": text.strip(), "start_ms": round(start * 1000), "end_ms": round(end * 1000), "probability": getattr(token, "probability", None), "whisper_segment": round(float(segment.start), 2)})
    del model
    if not words:
        raise RuntimeError("stable-ts 没有找到可用的校正稿时间戳，原学习包仍保留。")
    # A textual correction invalidates the affected tokens. Unchanged tokens
    # retain their already measured spans instead of letting a fresh full
    # alignment introduce regressions far away from the edited word.
    normalize=lambda text:re.sub(r'[^a-z0-9]','',text.lower())
    matcher=difflib.SequenceMatcher(None,[normalize(w['text']) for w in previous_words],[normalize(w['text']) for w in words],autojunk=False)
    preserved=[]
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            original=previous_words[block.a+offset];target=words[block.b+offset]
            if original['end_ms']>original['start_ms']:
                target.update(start_ms=original['start_ms'],end_ms=original['end_ms'],probability=original.get('probability'))
                preserved.append(original.get('id',original['text']))
    package.setdefault('processing',{})['preserved_unchanged_measured_tokens']=preserved
    from .alignment import group_connected_speech
    words,connected=group_connected_speech(words)
    package.setdefault('processing',{})['corrected_connected_speech']=connected
    if any(w['end_ms']<=w['start_ms'] for w in words):
        from .alignment import align_words
        words,info=align_words(audio_path,words,_local_whisper_path(),progress)
        package.setdefault('processing',{})['corrected_alignment']=info
        words,more_connected=group_connected_speech(words)
        package['processing']['corrected_connected_speech'].extend(more_connected)
        if any(w['end_ms']<=w['start_ms'] for w in words):
            raise RuntimeError('校正稿仍有无法可靠分界的词语，已保留原学习包。请核对原声或修改对应意群。')
    sentences, timed_words = _build_sentences(words)
    package["sentences"] = sentences
    package["words"] = timed_words
    package["cards"] = []
    package.pop('layout', None)
    package.pop('phrases', None)
    package.pop('teaching_review', None)


def make_notes(package: dict) -> str:
    from .privacy import public_source_url
    episode = package["episode"]
    source_url = public_source_url(episode.get("source_url", ""))
    rows = [f"# {episode.get('title_zh') or episode['title']}", "", f"- 原题：{episode['title']}", f"- 来源：{source_url}", f"- 时长：{int(episode['duration_s']//60)} 分 {round(episode['duration_s']%60)} 秒", "", "## 逐句精读", ""]
    words = {word["id"]: word for word in package["words"]}
    cards = {card["id"]: card for card in package["cards"]}
    for sentence in package["sentences"]:
        rows.extend([f"### {sentence['text']}", sentence["translation"], f"[回听 {int(sentence['start_ms']//60000):02}:{int(sentence['start_ms']//1000)%60:02}](study.html?t={sentence['start_ms']/1000:.3f})", ""])
        for card in package["cards"]:
            if sentence["id"] in card.get("sentence_ids", []):
                rows.append(f"- **{card['title']}**：{card['meaning']}")
                if card.get("structure"):
                    rows.append(f"  - 句子结构：{card['structure']}")
                for point in card.get('points',[]):
                    rows.append(f"  - {point['label']}：{point['text']}")
                if card.get("example"):
                    rows.append(f"  - 原句片段：{card['example']}")
                    rows.append(f"  - 片段译文：{card.get('translation','')}")
        phrases=[p for p in package.get('phrases',[]) if p['sentence_id']==sentence['id'] and p.get('study') is not False]
        if phrases:rows.append('- 搭配：'+'；'.join(f"{p['text']}：{p['gloss_zh']}" for p in phrases))
        glosses = [f"{words[wid]['text']}：{words[wid]['gloss_zh']}" for wid in sentence["word_ids"] if words[wid]["gloss_zh"] and words[wid].get('focus',True)]
        if glosses:
            rows.append("- 词义：" + "；".join(glosses))
        rows.append("")
    del cards
    rows.extend(["## 原声与使用说明", "", "字幕与教学内容依据原声及可用参考稿生成。识别和模型复核可能出错；遇到专名、口头语、数字和否定时，请结合原声复核。", ""])
    return "\n".join(rows)

def create_media_episode(spec: dict, job_folder: Path, progress: Progress) -> dict:
    from .general import create_media_episode as create
    return create(spec,job_folder,progress)
