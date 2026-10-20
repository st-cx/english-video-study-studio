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


if __name__ == '__main__':
    unittest.main()
