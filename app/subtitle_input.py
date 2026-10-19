"""Import spoken English from common subtitle/document formats.

Only overlapping rolling captions are deduplicated. Repeated dialogue in
separate time spans, hesitations and repeated words remain in the transcript.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path

SUPPORTED_SUBTITLE_SUFFIXES = {".srt", ".vtt", ".ass", ".ssa", ".txt", ".pdf"}
_STAMP = r"(?:\d{1,3}:)?\d{1,2}:\d{2}[.,]\d{1,3}"
_TIMING = re.compile(rf"^\s*({_STAMP})\s*-->\s*({_STAMP})(?:\s+.*)?$")
_STAGE_LABEL = re.compile(
    r"(?:(?:soft|background|instrumental|audience)\s+)?"
    r"(?:music(?:\s+playing)?|laughter|laughing|laughs?|applause|applauding|clapping|"
    r"inaudible|unintelligible|coughs?|coughing|chuckles?|chuckling|cheering|cheers|"
    r"sighs?|sighing|gasps?|gasping|silence|background\s+noise)"
    r"(?:\s*(?:and|,|/|&|\+)\s*(?:music|laughter|applause|clapping|cheering))*[.!…]*",
    re.I,
)


def _remove_stage_notes(value: str) -> str:
    def replace(match):
        content = (match.group(1) if match.group(1) is not None else match.group(2)).strip()
        if _STAGE_LABEL.fullmatch(content):
            return " "
        sound = re.sub(r"[!?.…\s]", "", content).lower()
        # A single phonetic animal/sound token, not ordinary parenthesized
        # English or arbitrary words with stretched spelling such as sooo.
        phonetic = re.fullmatch(r"(?:b+a+h*|m+e+o+w*|m+o+|w+o+f+|h+a+|[aeiou]+|h+m+|z+|g+r+)", sound)
        if phonetic and re.search(r"([a-z])\1{3,}", sound):
            return " "
        return match.group(0)
    return re.sub(r"\(([^()\n]*)\)|\[([^\[\]\n]*)\]", replace, value)


def _milliseconds(value: str) -> int:
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) == 2:
        hours, minutes, tail = 0, int(parts[0]), parts[1]
    elif len(parts) == 3:
        hours, minutes, tail = int(parts[0]), int(parts[1]), parts[2]
    else:
        raise ValueError("字幕时间格式无效。")
    seconds, fraction = tail.split(".", 1)
    seconds = int(seconds)
    if seconds >= 60 or minutes >= 60 or min(hours, minutes, seconds) < 0:
        raise ValueError("字幕分钟或秒钟超出范围。")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + int(fraction.ljust(3, "0")[:3])


def _clean_line(value: str) -> str:
    value = re.sub(r"\{[^{}]*\}", "", value)
    value = re.sub(r"<[^>]*>", "", value)
    value = html.unescape(value).replace("\u200b", "").replace("\ufeff", "")
    value = _remove_stage_notes(value)
    return re.sub(r"\s+", " ", value).strip()


def _join_rolling_lines(lines: list[str], *, rolling: bool = False) -> str:
    if not rolling:
        return " ".join(line for line in (_clean_line(raw) for raw in lines) if line)
    result = ""
    for raw in lines:
        line = _clean_line(raw)
        if not line:
            continue
        if line == result and len(line.split()) >= 2:
            continue
        # YouTube can repeat the previous complete line inside one cue.
        words, new = result.split(), line.split()
        overlap = 0
        for size in range(min(len(words), len(new)), 1, -1):
            if words[-size:] == new[:size]:
                overlap = size
                break
        result = " ".join([*words, *new[overlap:]])
    return result


def _deduplicate_cues(cues: list[dict], *, rolling: bool = False) -> list[dict]:
    result = []
    for cue in sorted(cues, key=lambda item: (item["start_ms"], item["end_ms"])):
        if not cue["text"]:
            continue
        item = dict(cue)
        if result:
            previous = result[-1]
            if previous["start_ms"] == item["start_ms"] and previous["end_ms"] == item["end_ms"] and previous["text"] == item["text"]:
                continue
            # Only time-overlapping rolling cues can contain already spoken
            # context. Do not globally remove naturally repeated dialogue.
            if rolling and item["start_ms"] < previous["end_ms"]:
                old_words, new_words = previous["text"].split(), item["text"].split()
                for count in range(min(len(old_words), len(new_words)), 0, -1):
                    if old_words[-count:] == new_words[:count]:
                        if count == len(new_words) == 1 and previous["start_ms"] != item["start_ms"]:
                            # A short overlapping answer can be real dialogue,
                            # not rolling caption context. Preserve uncertainty.
                            break
                        item["text"] = " ".join(new_words[count:])
                        break
                if not item["text"]:
                    previous["end_ms"] = max(previous["end_ms"], item["end_ms"])
                    continue
        result.append(item)
    return result


def _timed_text(text: str, *, rolling: bool = False) -> list[dict]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    cues = []
    for index, line in enumerate(lines):
        match = _TIMING.match(line)
        if not match:
            continue
        start, end = _milliseconds(match[1]), _milliseconds(match[2])
        if end <= start:
            raise ValueError(f"字幕第 {index + 1} 行结束时间没有晚于开始时间。")
        body, cursor = [], index + 1
        while cursor < len(lines) and lines[cursor].strip() and not _TIMING.match(lines[cursor]):
            # Index numbers immediately preceding another timing line are
            # metadata, even in files without a blank separator.
            if lines[cursor].strip().isdigit() and cursor + 1 < len(lines) and _TIMING.match(lines[cursor + 1]):
                break
            body.append(lines[cursor])
            cursor += 1
        cues.append({"start_ms": start, "end_ms": end, "text": _join_rolling_lines(body, rolling=rolling)})
    if not cues and text.strip() and re.search(r"\d\s*-->", text):
        raise ValueError("未找到可读取的字幕时间，支持 SRT／WebVTT 标准时间格式。")
    return _deduplicate_cues(cues, rolling=rolling)


def _ass_text(text: str) -> list[dict]:
    in_events, fields, cues = False, [], []
    for line in text.splitlines():
        value = line.strip()
        if value.startswith("["):
            in_events = value.lower() == "[events]"
            continue
        if not in_events:
            continue
        if value.lower().startswith("format:"):
            fields = [field.strip().lower() for field in value.split(":", 1)[1].split(",")]
        elif value.lower().startswith("dialogue:"):
            names = fields or ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]
            values = value.split(":", 1)[1].lstrip().split(",", len(names) - 1)
            if len(values) != len(names):
                raise ValueError("ASS 对白字段不完整。")
            entry = dict(zip(names, values))
            if not all(name in entry for name in ("start", "end", "text")):
                raise ValueError("ASS 缺少 Start、End 或 Text 字段。")
            start, end = _milliseconds(entry["start"]), _milliseconds(entry["end"])
            if end <= start:
                raise ValueError("ASS 对白结束时间没有晚于开始时间。")
            body = re.sub(r"\\[Nn]", "\n", entry["text"]).replace(r"\h", " ")
            # ASS may deliberately repeat a full sentence on every karaoke
            # event; exact/overlapping duplicate cues are merged below.
            spoken = _join_rolling_lines(body.splitlines())
            styled_roll = bool(re.search(r"\\(?:[1-4]?c|k[fo]?)(?:[(&\\0-9])", entry["text"], re.I))
            if styled_roll and cues and cues[-1]["text"] == spoken and start <= cues[-1]["end_ms"] + 50:
                cues[-1]["end_ms"] = max(cues[-1]["end_ms"], end)
            else:
                cues.append({"start_ms": start, "end_ms": end, "text": spoken})
    return _deduplicate_cues(cues)


def _pdf_text(path: Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要 PyMuPDF，请运行安装程序。") from exc
    with fitz.open(path) as document:
        if document.needs_pass:
            raise ValueError("PDF 已加密，请先提供可以读取的英文文稿。")
        pages = [[line.strip() for line in page.get_text("text").splitlines() if line.strip()] for page in document]
    if not pages or not any(pages):
        raise ValueError("PDF 没有可读取文字；扫描文档请先转成英文 TXT。")
    furniture = Counter()
    for lines in pages:
        furniture.update(set(lines[:3] + lines[-3:]))
    result = []
    for lines in pages:
        for index, line in enumerate(lines):
            near_edge = index < 3 or index >= len(lines) - 3
            if len(pages) > 1 and near_edge and furniture[line] >= 2:
                # Remove repeated page furniture only. A substantial spoken
                # paragraph is never removed merely because it recurs.
                if len(line) <= 100 and (re.search(r"(?:page\s+\d|©|copyright|https?://|www\.|BBC LEARNING ENGLISH|6 Minute English)", line, re.I)
                                          or re.fullmatch(r"\d+(?:\s*/\s*\d+)?", line)):
                    continue
            result.append(line)
    return "\n".join(result).strip()


def parse_subtitles(path: str | Path) -> dict:
    """Return {text, cues, kind}; an absent optional subtitle is valid input."""
    if path is None or str(path).strip() == "":
        return {"text": "", "cues": [], "kind": "none"}
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in SUPPORTED_SUBTITLE_SUFFIXES:
        raise ValueError("字幕／文稿支持 SRT、VTT、ASS、TXT 和 PDF。")
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return {"text": _pdf_text(source), "cues": [], "kind": "pdf"}
    data = source.read_bytes()
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("字幕／文稿超过 32MB，请分段导入。")
    text = None
    encodings = ("utf-8-sig", "utf-16", "gb18030") if data.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except UnicodeError:
            continue
    if text is None:
        raise ValueError("字幕编码无法识别，请保存为 UTF-8 后重试。")
    if suffix == ".txt":
        # Preserve repeated lines and oral repetitions in untimed prose.
        return {"text": text.replace("\r\n", "\n").strip(), "cues": [], "kind": "txt"}
    cues = _ass_text(text) if suffix in {".ass", ".ssa"} else _timed_text(text, rolling=suffix == ".vtt")
    if not cues and text.strip():
        raise ValueError("该字幕文件没有可读取的对白。")
    return {"text": "\n".join(cue["text"] for cue in cues), "cues": cues, "kind": suffix[1:]}


parse_subtitle_file = parse_subtitles
