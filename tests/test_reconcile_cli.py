import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.sync_audit import digest, file_sha256


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'reconcile-review.py'
spec = importlib.util.spec_from_file_location('reconcile_review_cli', SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ReconcileCliTests(unittest.TestCase):
    def fixture(self, folder):
        source = folder / 'live'
        source.mkdir()
        audio = source / 'audio.mp3'
        audio.write_bytes(b'unchanged source audio')
        package = {'episode': {'id': 'lesson', 'duration_s': 3},
            'words': [{'id': 'w1', 'sentence_id': 's1', 'text': 'Hello', 'start_ms': 1000, 'end_ms': 1200},
                      {'id': 'w2', 'sentence_id': 's1', 'text': 'world.', 'start_ms': 2500, 'end_ms': 2700}],
            'sentences': [{'id': 's1', 'text': 'Hello world.', 'word_ids': ['w1', 'w2'],
                           'start_ms': 1000, 'end_ms': 2700}],
            'quality': {'needs_review': True, 'flags': ['real source and teaching flags']},
            'processing': {}, 'sync_review': {'old': True}}
        episode = source / 'episode.json'
        episode.write_text(json.dumps(package), encoding='utf8')
        survey = folder / 'survey.json'
        survey.write_text(json.dumps({'audio_sha256': file_sha256(audio),
            'input_timing_sha256': digest(package['words']), 'model_sha256': 'd' * 64,
            'windows': []}), encoding='utf8')
        output = folder / 'draft'
        args = ['--episode', str(episode), '--audio', str(audio), '--output', str(output),
                '--model-path', str(folder / 'already-cached-model')]
        return episode, audio, package, survey, output, args

    @staticmethod
    def result(package, *args, **kwargs):
        return {'episode': copy.deepcopy(package), 'summary': {'human_acceptance': 'not_completed'},
                'decisions': [], 'repairs': [], 'listening_groups': []}

    def test_default_and_existing_survey_do_not_enable_timing_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            episode, audio, package, survey, output, args = self.fixture(Path(directory))
            before = episode.read_bytes()
            with patch.object(cli, 'propose_disputed_boundary_revisions') as propose, \
                    patch.object(cli, 'reconcile_reviews', side_effect=self.result) as reconcile, \
                    patch('builtins.print'):
                cli.main(args + ['--survey', str(survey)])
            propose.assert_not_called()
            self.assertEqual(reconcile.call_args.args[0], package)
            self.assertEqual(json.loads((output / 'episode-draft.json').read_text(encoding='utf8')), package)
            self.assertFalse((output / 'boundary-proposal.json').exists())
            self.assertFalse((output / 'summary.json').exists())
            self.assertEqual(episode.read_bytes(), before)

    def test_explicit_repeated_boundary_surveys_create_only_draft_proposal_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            episode, audio, package, survey, output, args = self.fixture(Path(directory))
            before, audio_before = episode.read_bytes(), audio.read_bytes()
            proposal = {'version': 1, 'input_words_sha256': digest(package['words']),
                'audio_sha256': file_sha256(audio), 'human_ground_truth': False,
                'patches': [{'word_id': 'w1', 'text': 'Hello', 'before_ms': [1000, 1200], 'after_ms': [1500, 1700]}]}
            with patch.object(cli, 'propose_disputed_boundary_revisions', return_value=proposal) as propose, \
                    patch.object(cli, 'reconcile_reviews', side_effect=self.result) as reconcile, \
                    patch('builtins.print'):
                cli.main(args + ['--boundary-survey', str(survey), '--boundary-survey', str(survey)])
            self.assertEqual(len(propose.call_args.kwargs['boundary_surveys']), 2)
            revised = reconcile.call_args.args[0]
            self.assertEqual(revised['words'][0]['start_ms'], 1500)
            self.assertEqual(revised['sentences'][0]['start_ms'], 1500)
            self.assertEqual(revised['quality'], package['quality'])
            self.assertNotIn('sync_review', revised)
            self.assertEqual(json.loads((output / 'boundary-proposal.json').read_text(encoding='utf8')), proposal)
            summary = json.loads((output / 'summary.json').read_text(encoding='utf8'))
            self.assertEqual(summary['timing_repairs'], 1)
            self.assertEqual(summary['changed_timing_sentence_ids'], ['s1'])
            self.assertEqual(summary['human_acceptance'], 'not_completed')
            self.assertEqual(episode.read_bytes(), before)
            self.assertEqual(audio.read_bytes(), audio_before)

    def test_mismatched_audio_or_word_version_stops_before_any_output(self):
        for field in ('audio_sha256', 'input_timing_sha256'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                episode, audio, package, survey, output, args = self.fixture(Path(directory))
                data = cli.read(survey)
                data[field] = 'f' * 64
                survey.write_text(json.dumps(data), encoding='utf8')
                before = episode.read_bytes()
                with patch.object(cli, 'propose_disputed_boundary_revisions') as propose, \
                        self.assertRaisesRegex(ValueError, 'SHA不一致'):
                    cli.main(args + ['--boundary-survey', str(survey)])
                propose.assert_not_called()
                self.assertFalse(output.exists())
                self.assertEqual(episode.read_bytes(), before)

    def test_explicit_measurement_without_patch_leaves_original_review_and_clock_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            episode, audio, package, survey, output, args = self.fixture(Path(directory))
            proposal = {'patches': [], 'human_ground_truth': False}
            with patch.object(cli, 'propose_disputed_boundary_revisions', return_value=proposal), \
                    patch.object(cli, 'reconcile_reviews', side_effect=self.result) as reconcile, \
                    patch.object(cli, 'retime_package') as retime, patch('builtins.print'):
                cli.main(args + ['--boundary-survey', str(survey)])
            retime.assert_not_called()
            self.assertEqual(reconcile.call_args.args[0], package)
            self.assertEqual(json.loads((output / 'summary.json').read_text(encoding='utf8'))['timing_repairs'], 0)

    def test_actual_lesson_output_directory_remains_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            episode, audio, package, survey, output, args = self.fixture(Path(directory))
            args[args.index('--output') + 1] = str(episode.parent)
            with self.assertRaises(SystemExit) as raised, patch('sys.stderr'):
                cli.main(args + ['--boundary-survey', str(survey)])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse((episode.parent / 'episode-draft.json').exists())

    def test_real_cli_functions_export_an_unmeasured_draft_without_certifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            episode, audio, package, survey, output, args = self.fixture(Path(directory))
            original = episode.read_bytes()
            with patch('builtins.print'):
                cli.main(args + ['--boundary-survey', str(survey)])
            exported = cli.read(output / 'episode-draft.json')
            summary = cli.read(output / 'summary.json')
            self.assertEqual(exported['words'], package['words'])
            self.assertTrue(exported['quality']['needs_review'])
            self.assertIn('real source and teaching flags', exported['quality']['flags'])
            self.assertEqual(summary['timing_repairs'], 0)
            self.assertEqual(summary['human_acceptance'], 'not_completed')
            self.assertEqual(episode.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
