import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import general


class SpeechImplementationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / 'alignment-fixture.py'
        self.source.write_text('rule = "first"\n', encoding='utf-8')
        self.plan = {'path': 'large-fixture', 'model': 'large-v3', 'snapshot': 'fixture',
                     'device': 'cuda', 'compute_type': 'int8_float16'}
        self.raw = [{'text': 'Hello.', 'start_ms': 100, 'end_ms': 800,
                     'probability': .99, 'whisper_segment': .1}]

    def model(self):
        segment = SimpleNamespace(no_speech_prob=0, avg_logprob=0, compression_ratio=1,
            start=.1, words=[SimpleNamespace(word='Hello.', start=.1, end=.8, probability=.99)])
        model = Mock()
        model.transcribe_original.return_value = ([segment], SimpleNamespace(language='en', language_probability=1))
        return model

    def transcribe(self, loader):
        with patch('app.alignment.__file__', str(self.source)), \
             patch('app.speech_runtime.speech_runtime_plan', return_value=self.plan), \
             patch('app.pipeline._local_whisper_path', return_value='medium-fixture'), \
             patch('stable_whisper.load_faster_whisper', side_effect=loader), \
             patch('app.general._boundaries', return_value=[0, 2]), \
             patch('app.general.subprocess.run'), \
             patch('app.alignment.align_words', side_effect=lambda *a, **k: (copy.deepcopy(self.raw), {})):
            return general._transcribe_chunks(self.folder / 'audio.mp3', 2, self.folder, 'media', lambda *a: None)

    def saved(self):
        return json.loads((self.folder / '.speech/0000.json').read_text(encoding='utf-8'))

    def test_unchanged_implementation_reuses_audio_but_rule_change_reprocesses(self):
        model = self.model()
        first = self.transcribe(lambda *a, **k: model)
        key = self.saved()['key']
        self.transcribe(lambda *a, **k: (_ for _ in ()).throw(AssertionError('unchanged cache must not load model')))
        self.assertEqual(model.transcribe_original.call_count, 1)
        self.source.write_text('rule = "second"\n', encoding='utf-8')
        second = self.transcribe(lambda *a, **k: model)
        self.assertEqual(model.transcribe_original.call_count, 2)
        self.assertNotEqual(self.saved()['key'], key)
        self.assertEqual(first[0], second[0])

    def test_cpu_fallback_keeps_the_same_alignment_identity(self):
        for failure in ('load', 'transcribe'):
            with self.subTest(failure=failure):
                checkpoint = self.folder / '.speech/0000.json'
                checkpoint.unlink(missing_ok=True)
                cpu = self.model(); gpu = self.model()
                if failure == 'transcribe': gpu.transcribe_original.side_effect = RuntimeError('fixture GPU failure')
                loads = [RuntimeError('fixture GPU load failure'), cpu] if failure == 'load' else [gpu, cpu]
                self.transcribe(Mock(side_effect=loads))
                saved = self.saved()
                self.assertEqual(saved['backend'], 'CPU')
                self.assertEqual(saved['runtime']['model'], 'medium-fixture')
                self.assertEqual(saved['runtime']['alignment_implementation_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
                self.assertEqual(saved['key'], general._signature([general.PROCESSING_VERSION, 'media', 0, 2, saved['runtime']]))

    def test_unfingerprinted_legacy_speech_checkpoint_cannot_bypass_new_rules(self):
        speech = self.folder / '.speech'; speech.mkdir()
        for runtime in (None, {k: v for k, v in self.plan.items() if k != 'path'}):
            with self.subTest(runtime=runtime):
                fields = [general.PROCESSING_VERSION, 'media', 0, 2]
                if runtime is not None: fields.append(runtime)
                saved = {'key': general._signature(fields), 'words': copy.deepcopy(self.raw),
                    'backend': 'CPU', 'language': 'en', 'language_probability': 1, 'suspects': [],
                    'alignment': {}, 'start_ms': 0, 'end_ms': 2000, 'time_basis': 'media'}
                if runtime is not None: saved['runtime'] = runtime
                (speech / '0000.json').write_text(json.dumps(saved), encoding='utf-8')
                model = self.model(); self.transcribe(lambda *a, **k: model)
                model.transcribe_original.assert_called_once()
                self.assertIn('alignment_implementation_sha256', self.saved()['runtime'])

    def test_alignment_source_changes_do_not_invalidate_unchanged_teaching(self):
        sentence = {'id': 's1', 'text': 'Hello.', 'translation': '', 'start_ms': 100,
                    'end_ms': 800, 'word_ids': ['w1'], 'card_ids': []}
        package = {'version': 4, 'episode': {'id': 'fixture', 'duration_s': 2},
            'sentences': [sentence], 'words': [{'id': 'w1', 'sentence_id': 's1', 'text': 'Hello.',
                'start_ms': 100, 'end_ms': 800, 'probability': .99, 'gloss_zh': ''}],
            'cards': [], 'quality': {}, 'review': {}, 'render': {}, 'processing': {'speech_chunks': []},
            'chapters': [{'id': 'ch001', 'title': 'unit', 'start_ms': 0, 'end_ms': 2000, 'status': 'pending'}]}
        part = general._chapter_package(package, package['chapters'][0])
        revision = general._signature([general.PROCESSING_VERSION, part['sentences'],
            [(w['id'], w['text'], w['start_ms'], w['end_ms']) for w in part['words']], 'fixture-model', 'https://example.invalid/v1'])
        part['_revision'] = revision; part['sentences'][0]['translation'] = '你好。'
        part['teaching_review'] = {'status': 'passed'}
        part['layout'] = {'version': 1, 'pages': [{'id': 'p1', 'sentence_id': 's1', 'start_ms': 100, 'end_ms': 800}]}
        chapter = self.folder / '.chapters/ch001'; chapter.mkdir(parents=True)
        (chapter / 'teaching.json').write_text(json.dumps(part), encoding='utf-8')
        quality = {'needs_review': False, 'flags': [], 'structural_flags': [], 'source_similarity': None}
        for body in ('rule="first"', 'rule="second"'):
            self.source.write_text(body, encoding='utf-8')
            current = copy.deepcopy(package)
            with patch('app.alignment.__file__', str(self.source)), \
                 patch('app.general.get_settings', return_value={'model': 'fixture-model', 'base_url': 'https://example.invalid/v1'}), \
                 patch('app.speech_runtime.speech_runtime_plan', return_value=self.plan), \
                 patch('app.reference_resolution.resolve_reference_findings', return_value={'unresolved': []}), \
                 patch('app.pipeline._quality', return_value=quality), \
                 patch('app.pipeline._annotate', side_effect=AssertionError('unchanged teaching must stay cached')), \
                 patch('app.teaching_review.review_teaching', side_effect=AssertionError('must not call model')), \
                 patch('app.general.save_episode'), patch('app.general._write_bundle'):
                general._process_chapters(current, self.folder, '', [], lambda *a: None)
            self.assertEqual(current['chapters'][0]['status'], 'ready')
            self.assertEqual(current['sentences'][0]['translation'], '你好。')



    def test_episode_checkpoint_cannot_bypass_changed_aligner_or_overwrite_finished_output(self):
        output = self.folder / 'outputs'; output.mkdir()
        fingerprint = 'a' * 64; episode_id = 'media-' + fingerprint[:20]
        completed = output / episode_id; completed.mkdir()
        source = self.folder / 'input.wav'; source.write_bytes(b'fixture input')
        normalized = self.folder / 'normalized.mp3'; normalized.write_bytes(b'fixture audio')
        settings = {'model': 'fixture-model', 'base_url': 'https://example.invalid/v1'}
        legacy_revision = general._signature([general.PROCESSING_VERSION, fingerprint, '', [], 'none', '', [], settings['model'], settings['base_url']])
        old = {'episode': {'id': episode_id, 'duration_s': 2}, 'processing': {'revision': legacy_revision},
               'artifacts': {'study': 'ready'}, 'words': [], 'sentences': [], 'chapters': []}
        old_bytes = json.dumps(old).encode(); (completed / 'episode.json').write_bytes(old_bytes)
        (completed / 'lesson.mp4').write_bytes(b'finished video')
        media = {'fingerprint': fingerprint, 'duration_s': 2, 'audio_path': normalized}
        transcribe = Mock(return_value=(copy.deepcopy(self.raw), []))
        with patch('app.alignment.__file__', str(self.source)), \
             patch('app.general.output_root', return_value=output), \
             patch('app.general.get_settings', return_value=settings), \
             patch('app.media.prepare_media', return_value=media), \
             patch('app.general._transcribe_chunks', transcribe), \
             patch('app.general._process_chapters'):
            first = general.create_media_episode({'media_path': str(source), 'render_mode': 'study'}, self.folder / 'job', lambda *a: None)
            self.assertEqual(transcribe.call_count, 1)
            first_folder = output / first['episode']['id']
            self.assertNotEqual(first_folder, completed)
            first['artifacts']['study'] = 'ready'
            first_bytes = json.dumps(first).encode(); (first_folder / 'episode.json').write_bytes(first_bytes)
            self.source.write_text('rule = "changed again"', encoding='utf-8')
            second = general.create_media_episode({'media_path': str(source), 'render_mode': 'study'}, self.folder / 'job', lambda *a: None)
            self.assertEqual(transcribe.call_count, 2)
            self.assertNotEqual(second['episode']['id'], first['episode']['id'])
        self.assertEqual((completed / 'episode.json').read_bytes(), old_bytes)
        self.assertEqual((completed / 'lesson.mp4').read_bytes(), b'finished video')
        self.assertEqual((first_folder / 'episode.json').read_bytes(), first_bytes)

    def test_new_alignment_revision_reuses_matching_sibling_teaching(self):
        output = self.folder / 'outputs'; output.mkdir()
        base_id = 'media-' + 'b' * 20
        old_folder = output / base_id; old_chapter = old_folder / '.chapters/ch001'; old_chapter.mkdir(parents=True)
        current_folder = output / (base_id + '-r12345678'); current_folder.mkdir()
        sentence = {'id': 's1', 'text': 'Hello.', 'translation': '', 'start_ms': 100, 'end_ms': 800, 'word_ids': ['w1'], 'card_ids': []}
        package = {'version': 4, 'episode': {'id': current_folder.name, 'duration_s': 2},
            'sentences': [sentence], 'words': [{'id': 'w1', 'sentence_id': 's1', 'text': 'Hello.', 'start_ms': 100, 'end_ms': 800, 'probability': .99, 'gloss_zh': ''}],
            'cards': [], 'quality': {}, 'review': {}, 'render': {}, 'processing': {'speech_chunks': []},
            'chapters': [{'id': 'ch001', 'title': 'unit', 'start_ms': 0, 'end_ms': 2000, 'status': 'pending'}]}
        part = general._chapter_package(package, package['chapters'][0])
        part['_revision'] = general._signature([general.PROCESSING_VERSION, part['sentences'],
            [(w['id'], w['text'], w['start_ms'], w['end_ms']) for w in part['words']], 'fixture-model', 'https://example.invalid/v1'])
        part['sentences'][0]['translation'] = '已复核的你好。'; part['episode']['id'] = base_id
        part['words'][0]['probability'] = .42
        part['editor_review'] = {'status': 'passed', 'scope': 'text_semantics'}
        part['editor_corrections'] = [{'before': '旧译', 'after': '已复核的你好。', 'sentence_id': 's1'}]
        part['teaching_review'] = {'status': 'passed'}
        part['layout'] = {'version': 1, 'pages': [{'id': 'p1', 'sentence_id': 's1', 'start_ms': 100, 'end_ms': 800}]}
        cache_path = old_chapter / 'teaching.json'; old_bytes = json.dumps(part).encode(); cache_path.write_bytes(old_bytes)
        quality = {'needs_review': False, 'flags': [], 'structural_flags': [], 'source_similarity': None}
        with patch('app.general.output_root', return_value=output), \
             patch('app.general.get_settings', return_value={'model': 'fixture-model', 'base_url': 'https://example.invalid/v1'}), \
             patch('app.speech_runtime.speech_runtime_plan', return_value=self.plan), \
             patch('app.reference_resolution.resolve_reference_findings', return_value={'unresolved': []}), \
             patch('app.pipeline._quality', return_value=quality), \
             patch('app.pipeline._annotate', side_effect=AssertionError('same English must reuse prior teaching')), \
             patch('app.teaching_review.review_teaching', side_effect=AssertionError('must not call model')), \
             patch('app.general.save_episode') as save, patch('app.general._write_bundle'):
            general._process_chapters(package, current_folder, '', [], lambda *a: None)
        self.assertEqual(package['sentences'][0]['translation'], '已复核的你好。')
        self.assertEqual(package['episode']['id'], current_folder.name)
        self.assertEqual(package['words'][0]['probability'], .99)
        reused = next(call.args[0] for call in save.call_args_list if len(call.args) > 2 and call.args[2] == 'teaching.json')
        self.assertEqual(reused['editor_review'], part['editor_review'])
        self.assertEqual(reused['editor_corrections'], part['editor_corrections'])
        self.assertEqual(cache_path.read_bytes(), old_bytes)


if __name__ == '__main__': unittest.main()
