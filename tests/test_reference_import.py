import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.subtitle_input import _timed_text


class ReferenceImportTests(unittest.TestCase):
    def test_explicit_named_speaker_is_metadata_not_spoken_words(self):
        text = 'WEBVTT\n\n00:00:29.957 --> 00:00:30.790\nANA BELL: All right.\n\n00:00:58.960 --> 00:01:00.610\nMy name is Ana Bell.\n'
        cues = _timed_text(text, rolling=True)
        self.assertEqual(cues[0]['text'], 'All right.')
        self.assertEqual(cues[0]['speaker'], 'ANA BELL')
        self.assertEqual(cues[1]['text'], 'My name is Ana Bell.')
        self.assertEqual(cues[0]['start_ms'], 29957)

    def test_voice_tags_and_roles_are_metadata_without_removing_colon_prose(self):
        cues = _timed_text('00:00:00.000 --> 00:00:01.000\n<v Ana Bell>Hello.</v>\n\n00:00:01.000 --> 00:00:02.000\nSTUDENT: Yes.\n\n00:00:02.000 --> 00:00:03.000\nPYTHON: print(x)\n\n00:00:03.000 --> 00:00:04.000\nRemember: keep this.\n')
        self.assertEqual(cues[0]['speaker'], 'Ana Bell')
        self.assertEqual(cues[0]['text'], 'Hello.')
        self.assertEqual(cues[1]['text'], 'Yes.')
        self.assertEqual(cues[2]['text'], 'PYTHON: print(x)')
        self.assertEqual(cues[3]['text'], 'Remember: keep this.')

    def test_youtube_adjacent_hold_cues_do_not_repeat_displayed_context(self):
        text = '''WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:02.990 align:start position:0%
__EMPTY_DISPLAY_ROW__
We<00:00:01.200><c> can</c><00:00:01.500><c> learn.</c>

00:00:02.990 --> 00:00:03.000 align:start position:0%
We can learn.

00:00:03.000 --> 00:00:04.990 align:start position:0%
We can learn.
Then<00:00:03.200><c> share</c><00:00:03.500><c> ideas.</c>

00:00:04.990 --> 00:00:05.000 align:start position:0%
Then share ideas.

00:00:05.000 --> 00:00:07.000 align:start position:0%
Then share ideas.
And<00:00:05.200><c> create</c><00:00:06.000><c> together.</c>
'''
        cues = _timed_text(text.replace('__EMPTY_DISPLAY_ROW__', ' '), rolling=True)
        self.assertEqual([cue['text'] for cue in cues],
                         ['We can learn.', 'Then share ideas.', 'And create together.'])
        self.assertEqual([(cue['start_ms'], cue['end_ms']) for cue in cues],
                         [(1000, 2990), (3000, 4990), (5000, 7000)])
        self.assertFalse(any(key.startswith('_') for cue in cues for key in cue))

    def test_new_inline_timestamped_repetition_is_retained(self):
        text = '''WEBVTT

00:00:00.000 --> 00:00:01.000
Try<00:00:00.400><c> again.</c>

00:00:01.000 --> 00:00:02.000
Try again.
Try<00:00:01.400><c> again.</c>
'''
        self.assertEqual([cue['text'] for cue in _timed_text(text, rolling=True)], ['Try again.', 'Try again.'])

    def test_caption_header_or_line_wrapping_alone_never_deduplicates_repetition(self):
        text = '''WEBVTT
Kind: captions

00:00:00.000 --> 00:00:01.000
I think so.
I think so.

00:00:01.000 --> 00:00:01.010
I think so.

00:00:01.010 --> 00:00:02.000
I think so.
Say it again.
'''
        self.assertEqual([cue['text'] for cue in _timed_text(text, rolling=True)],
                         ['I think so. I think so.', 'I think so.', 'I think so. Say it again.'])

    def test_srt_inline_markup_does_not_activate_youtube_rolling_rules(self):
        text = '''1
00:00:00,000 --> 00:00:01,000
Try<00:00:00.400> again.

2
00:00:01,000 --> 00:00:01,010
Try again.
'''
        self.assertEqual([cue['text'] for cue in _timed_text(text)], ['Try again.', 'Try again.'])

    def test_rolling_evidence_cannot_cross_gap_or_speaker_change(self):
        for second_start, first_speaker, second_speaker in (
                ('00:00:01.100', '', ''),
                ('00:00:01.000', '<v First>', '<v Second>')):
            text = (f'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{first_speaker}Try<00:00:00.400> again.\n\n'
                    f'{second_start} --> 00:00:02.000\n{second_speaker}Try again.\nThen<00:00:01.500> continue.\n')
            with self.subTest(start=second_start, speaker=second_speaker):
                self.assertEqual(_timed_text(text, rolling=True)[1]['text'], 'Try again. Then continue.')

    def test_overlapping_distinct_speakers_do_not_erase_the_same_answer(self):
        text = ('WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<v First>I think so.\n\n'
                '00:00:00.900 --> 00:00:02.000\n<v Second>I think so.\n')
        self.assertEqual(len(_timed_text(text, rolling=True)), 2)
        same_time = text.replace('00:00:00.900 --> 00:00:02.000', '00:00:00.000 --> 00:00:01.000')
        self.assertEqual(len(_timed_text(same_time, rolling=True)), 2)

    def test_out_of_range_inline_timestamp_is_not_carryover_evidence(self):
        text = ('WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTry again.\n\n'
                '00:00:01.000 --> 00:00:02.000\nTry again.\nThen<00:00:09.000> continue.\n')
        self.assertEqual(_timed_text(text, rolling=True)[1]['text'], 'Try again. Then continue.')

    def test_single_word_without_inline_stamp_uses_proven_raw_hold_bridge(self):
        text = '''WEBVTT

00:00:00.000 --> 00:00:01.990
We<00:00:00.500> agree.

00:00:01.990 --> 00:00:02.000
We agree.

00:00:02.000 --> 00:00:02.990
We agree.
Yes.

00:00:02.990 --> 00:00:03.000
Yes.

00:00:03.000 --> 00:00:04.000
Yes.
Let<00:00:03.500> us continue.
'''
        cues = _timed_text(text, rolling=True)
        self.assertEqual([cue['text'] for cue in cues], ['We agree.', 'Yes.', 'Let us continue.'])
        self.assertEqual((cues[1]['start_ms'], cues[1]['end_ms']), (2000, 2990))


if __name__ == '__main__':
    unittest.main()
