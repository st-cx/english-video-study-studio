"""Locate meaningful reference/ASR disagreements without rewriting speech."""
from __future__ import annotations

import difflib
import re
from decimal import Decimal

_PATTERN = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z]+(?:['’\-][A-Za-z]+)*")
_SMALL = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
_TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
_SCALES = {'thousand': 1000, 'million': 1_000_000, 'billion': 1_000_000_000}
_NUMBER_WORDS = set(_SMALL) | set(_TENS) | set(_SCALES) | {'hundred', 'point'}
_FILLERS = {'um', 'umm', 'uh', 'uhh', 'er', 'erm', 'hmm', 'hmmm', 'mhm', 'mmhmm', 'uhhuh', 'ah', 'oh'}
_NEGATIVE = {'not', 'no', 'never', 'neither', 'nor', 'without', 'nothing', 'nobody', 'nowhere'}
_FUNCTION = set('a an the this that these those i me my mine myself you your yours yourself he him his himself she her hers herself it its itself we us our ours ourselves they them their theirs themselves and or but so as if than then because while when where which who whom whose how what why at by for from in into of on onto to with about through before after between during under over is am are was were be been being do does did have has had having can could will would shall should may might must all any both each either every other some such there here yes okay please'.split())
_CONTRACTIONS = {
    "can't": 'can not', 'cannot': 'can not', "won't": 'will not', "shan't": 'shall not', "ain't": 'is not',
    "i'm": 'i am', "let's": 'let us', "it's": 'it is', "that's": 'that is',
    "there's": 'there is', "here's": 'here is', "what's": 'what is',
    "he's": 'he is', "she's": 'she is', 'gonna': 'going to', 'wanna': 'want to', 'ok': 'okay',
}
# Keep this allowlist narrow: fuzzy matching names would hide real errors.
_ABBREVIATIONS = {'mt': 'mount'}


def _parts(text, start_ms=0, end_ms=0):
    result = []
    for match in _PATTERN.finditer(text):
        raw = match.group()
        token = raw.replace('’', "'").lower()
        expanded = _CONTRACTIONS.get(token, _ABBREVIATIONS.get(token))
        if expanded is None:
            if token.endswith("n't"):
                expanded = token[:-3] + ' not'
            elif token.endswith("'re"):
                expanded = token[:-3] + ' are'
            elif token.endswith("'ve"):
                expanded = token[:-3] + ' have'
            elif token.endswith("'ll"):
                expanded = token[:-3] + ' will'
            elif token.endswith("'d"):
                expanded = token[:-2] + ' would'
            else:
                expanded = token.replace('-', ' ') if '-' in token and all(part in _NUMBER_WORDS for part in token.split('-')) else token.replace('-', '')
        for value in expanded.split():
            if value not in _FILLERS:
                result.append({'value': value, 'raw': raw, 'start_ms': start_ms, 'end_ms': end_ms,
                               'piece_id': (match.start(), match.end()),
                               'proper': len(raw) > 1 and raw.isupper() and (token not in _FUNCTION or raw in {'US', 'UK'})})
    return result


def _number(token):
    return token in _NUMBER_WORDS or bool(re.fullmatch(r'\d+(?:[.,]\d+)*', token))


def _number_value(parts):
    total, current, index = Decimal(0), Decimal(0), 0
    while index < len(parts):
        part = parts[index]
        if part == 'and':
            pass
        elif part in _SMALL:
            current += _SMALL[part]
        elif part in _TENS:
            current += _TENS[part]
        elif part == 'hundred':
            current = max(current, Decimal(1)) * 100
        elif part in _SCALES:
            total += max(current, Decimal(1)) * _SCALES[part]
            current = Decimal(0)
        elif part == 'point':
            digits = ''
            while index + 1 < len(parts):
                candidate = parts[index + 1]
                digit = str(_SMALL[candidate]) if candidate in _SMALL and _SMALL[candidate] < 10 else candidate if candidate.isdigit() else ''
                if not digit:
                    break
                digits += digit
                index += 1
            if digits:
                current += Decimal('0.' + digits)
        else:
            current += Decimal(part.replace(',', ''))
        index += 1
    return format(total + current, 'f').rstrip('0').rstrip('.') if '.' in format(total + current, 'f') else str(total + current)


def _join_raw(items):
    # Expanded contractions share one source token and must appear only once.
    result, previous = [], None
    for item in items:
        identity = (item.get('source_id', id(item)), item.get('piece_id'))
        if identity != previous:
            result.append(item['raw'])
        previous = identity
    return ' '.join(result)


def _canonical(items):
    result, index = [], 0
    while index < len(items):
        item = items[index]
        value = item['value']
        if value == 'all' and index + 1 < len(items) and items[index + 1]['value'] == 'right':
            group = items[index:index + 2]
            result.append({**item, 'value': 'alright', 'raw': _join_raw(group), 'end_ms': group[-1]['end_ms']})
            index += 2
        elif _number(value):
            end = index + 1
            while end < len(items):
                next_value, previous = items[end]['value'], items[end - 1]['value']
                values = [word['value'] for word in items[index:end]]
                if next_value == 'and':
                    if not any(part in _SCALES or part == 'hundred' for part in values) or end + 1 >= len(items) or not _number(items[end + 1]['value']):
                        break
                elif not _number(next_value):
                    break
                elif next_value in _SCALES or next_value == 'hundred' or previous in _SCALES or previous in {'hundred', 'and'}:
                    pass
                elif next_value == 'point' and 'point' not in values:
                    pass
                elif 'point' in values and (next_value.isdigit() or next_value in _SMALL and _SMALL[next_value] < 10):
                    pass
                elif previous in _TENS and next_value in _SMALL and 0 < _SMALL[next_value] < 10:
                    pass
                else:
                    break
                end += 1
            group = items[index:end]
            result.append({**item, 'value': _number_value([word['value'] for word in group]),
                           'raw': _join_raw(group), 'end_ms': group[-1]['end_ms'], 'numeric': True})
            index = end
        else:
            result.append(item)
            index += 1
    return result


def reference_issues(words: list[dict], reference: str) -> list[dict]:
    """Return measured listening windows for material disagreements.

    Wording remains ASR-owned. A reference can be editorial rather than verbatim,
    so every item is a request to listen, never an automatic substitution.
    """
    recognized = []
    for index, word in enumerate(words):
        recognized.extend({**item, 'source_id': index} for item in _parts(str(word.get('text', '')),
                           int(word.get('start_ms') or 0), int(word.get('end_ms') or 0)))
    expected = []
    for index, match in enumerate(_PATTERN.finditer(reference or '')):
        expected.extend({**item, 'source_id': index} for item in _parts(match.group()))
    recognized, expected = _canonical(recognized), _canonical(expected)
    if not recognized or not expected:
        return []
    result = []
    matcher = difflib.SequenceMatcher(None, [word['value'] for word in recognized], [word['value'] for word in expected], autojunk=False)
    for operation, left, right, target_start, target_end in matcher.get_opcodes():
        if operation == 'equal':
            continue
        actual, wanted = recognized[left:right], expected[target_start:target_end]
        if ''.join(word['value'] for word in actual) == ''.join(word['value'] for word in wanted):
            continue
        changed = actual + wanted
        meaningful = [word for word in changed if word.get('numeric') or word.get('proper') or word['value'] in _NEGATIVE or word['value'] not in _FUNCTION]
        if not meaningful:
            continue
        categories = []
        if any(word.get('numeric') for word in changed):
            categories.append('数字')
        if any(word['value'] in _NEGATIVE for word in changed):
            categories.append('否定')
        if any(word.get('proper') for word in changed):
            categories.append('专名')
        if not categories:
            categories.append('实词')
        anchored = actual or recognized[max(0, left - 1):min(len(recognized), left + 1)]
        result.append({'reason': '参考稿与识别存在' + '、'.join(categories) + '差异，请回听核对。',
                       'start_ms': min(word['start_ms'] for word in anchored),
                       'end_ms': max(word['end_ms'] for word in anchored),
                       'recognized': _join_raw(actual), 'reference': _join_raw(wanted)})
    return result
