import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.general import _process_chapters
from app.pipeline import ReviewRequired


def package(second='More learning.'):
    texts = ['Hello world.', second]
    words, sentences = [], []
    for index, text in enumerate(texts):
        ids = []
        for offset, token in enumerate(text.split()):
            wid = f'w{len(words)}'
            ids.append(wid)
            words.append({'id': wid, 'index': len(words), 'sentence_id': f's{index}', 'text': token,
                          'start_ms': index * 3000 + offset * 500 + 100,
                          'end_ms': index * 3000 + offset * 500 + 450, 'probability': .99, 'gloss_zh': ''})
        sentences.append({'id': f's{index}', 'text': text, 'translation': '', 'speaker': '',
                          'start_ms': index * 3000 + 100, 'end_ms': index * 3000 + 950,
                          'word_ids': ids, 'card_ids': []})
    return {'version': 4, 'episode': {'id': 'scope-test', 'title': 'Scope', 'duration_s': 6},
            'sentences': sentences, 'words': words, 'cards': [], 'review': {},
            'layout': {'version': 1, 'pages': [{'id': s['id'], 'sentence_id': s['id'],
                                               'start_ms': s['start_ms']} for s in sentences]},
            'processing': {'speech_chunks': []}, 'render': {}, 'artifacts': {},
            'chapters': [{'id': f'ch{i}', 'title': f'Unit {i}', 'start_ms': i * 3000,
                          'end_ms': (i + 1) * 3000, 'status': 'pending'} for i in range(2)]}


class ChapterReferenceScopeTests(unittest.TestCase):
    def run_package(self, value):
        reference = 'Hello world. More learning.'
        cues = [{'text': reference, 'start_ms': 0, 'end_ms': 6000}]
        def layout(part, folder):
            part['layout'] = {'version': 1, 'pages': [{'id': s['id'], 'sentence_id': s['id'],
                                                     'start_ms': s['start_ms']} for s in part['sentences']]}
        with tempfile.TemporaryDirectory() as temporary, \
             patch('app.pipeline._annotate'), patch('app.teaching_review.review_teaching'), \
             patch('app.layout.prepare_layout', layout), patch('app.general._write_bundle'), \
             patch('app.general.save_episode'):
            return _process_chapters(value, Path(temporary), reference, cues, lambda *a: None)

    def test_cue_crossing_chapters_does_not_create_missing_words_in_both_units(self):
        value = self.run_package(package())
        self.assertFalse(value['quality']['needs_review'])
        self.assertEqual([c['status'] for c in value['chapters']], ['ready', 'ready'])

    def test_real_replacement_is_still_anchored_to_affected_unit(self):
        value = package('More earning.')
        with self.assertRaises(ReviewRequired):
            self.run_package(value)
        self.assertEqual(value['chapters'][0]['status'], 'ready')
        self.assertEqual(value['chapters'][1]['status'], 'review')
        self.assertEqual(value['chapters'][1]['reference_issues'][0]['recognized'], 'earning')


if __name__ == '__main__':
    unittest.main()
