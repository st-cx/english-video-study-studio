"""Separate spoken BBC transcript turns from repeated PDF furniture and vocabulary notes."""
from __future__ import annotations

import difflib
import re
from collections import Counter


def spoken_turns(raw: str) -> list[dict]:
    lines = [line.strip() for line in raw.splitlines()]
    body = []
    for line in lines:
        if re.fullmatch(r"(?:VOCABULARY|This week's vocabulary|Vocabulary)", line):
            break
        if line.startswith('*'):
            break
        if not line or re.search(r'^(?:BBC LEARNING ENGLISH|6 Minute English|©|Page \d+ of \d+|bbclearningenglish\.com$|This is not a word-for-word transcript)', line, re.I):
            continue
        body.append(line)
    def name(line):
        return bool(re.fullmatch(r"[A-Z][A-Za-zÀ-ž’'-]+(?: [A-Z][A-Za-zÀ-ž’'-]+){0,3}", line)) and not line.isupper()
    counts = Counter(line for line in body if name(line))
    speakers = {line for line, n in counts.items() if n >= 2 or ' ' in line}
    turns, current = [], None
    for line in body:
        if line in speakers:
            current = {'speaker': line, 'text': ''}
            turns.append(current)
        elif current is not None:
            current['text'] += (' ' if current['text'] else '') + line
    return [turn for turn in turns if turn['text']]


def spoken_text(raw: str) -> str:
    turns = spoken_turns(raw)
    return ' '.join(turn['text'] for turn in turns) if turns else raw.strip()


def official_vocabulary(raw: str) -> str:
    match = re.search(r'(?:^|\n)\s*(?:VOCABULARY|Vocabulary|This week[’\']s vocabulary)\s*\n(.*)', raw, re.S)
    return re.sub(r'\s+', ' ', match.group(1)).strip()[:7000] if match else ''


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.'’-][a-z0-9]+)*", text.lower().replace('’', "'"))


def assign_speakers(words: list[dict], raw: str) -> None:
    official, labels = [], []
    for turn in spoken_turns(raw):
        pieces = tokens(turn['text'])
        official.extend(pieces)
        labels.extend([turn['speaker']] * len(pieces))
    recognized, owner = [], []
    for i, word in enumerate(words):
        for token in tokens(word['text']):
            recognized.append(token)
            owner.append(i)
    matched = {}
    for block in difflib.SequenceMatcher(None, official, recognized, autojunk=False).get_matching_blocks():
        for j in range(block.size):
            matched[owner[block.b+j]] = labels[block.a+j]
    previous = ''
    for i, word in enumerate(words):
        speaker = matched.get(i, previous)
        if not speaker:
            speaker = next((matched[k] for k in sorted(matched) if k > i), '')
        word['speaker'] = speaker
        previous = speaker


def normalize_orthography(words: list[dict], raw: str) -> tuple[list[dict], list[dict]]:
    """Restore written spelling only; keep the measured audio span and log each edit."""
    official = spoken_text(raw)
    vocabulary = {term.lower() for term in re.findall(r"[A-Za-z0-9]+(?:[.'’-][A-Za-z0-9]+)*", official)}
    vocabulary.update(re.findall(r'\d+(?:[,\.]\d+)+',official))
    names = {piece for turn in spoken_turns(raw) for piece in turn['speaker'].split()}
    result, changes = [], []
    for original in words:
        word = dict(original)
        if result and re.match(r'^[.,\-][A-Za-z0-9]', word['text']):
            candidate = result[-1]['text'] + word['text']
            if candidate.strip('.,!?;:').lower() in vocabulary:
                previous = result[-1]
                changes.append({'before': previous['text']+' '+word['text'], 'after': candidate, 'reason': 'official_joined_spelling', 'start_ms': previous['start_ms']})
                previous['text'] = candidate
                previous['end_ms'] = max(previous['end_ms'], word['end_ms'])
                continue
        clean = word['text'].strip('.,!?;:')
        if clean and clean[0].isupper() and clean not in names:
            candidates = [(difflib.SequenceMatcher(None, clean.lower(), name.lower()).ratio(), name) for name in names]
            if candidates:
                score, name = max(candidates)
                if score >= .84:
                    replacement = word['text'].replace(clean, name, 1)
                    changes.append({'before': word['text'], 'after': replacement, 'reason': 'official_speaker_spelling', 'start_ms': word['start_ms']})
                    word['text'] = replacement
        result.append(word)
    return result, changes
