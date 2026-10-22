"""A ready flag cannot bind an old MP4 to revised teaching or media."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import render_identity
from app.render_identity import (content_fingerprint, create_manifest, input_identity,
                                 manifest_path, verify_render)


def lesson():
    return {
        'episode': {'id': 'identity-fixture', 'title': 'Share ideas', 'title_zh': '分享想法',
                    'duration_s': 4, 'audio_file': 'audio.mp3', 'cover': 'cover.jpg'},
        'source': {'kind': 'youtube', 'name': 'Creative Commons', 'url': 'https://example.org/old'},
        'media': {'audio_file': 'audio.mp3', 'video_file': 'source.mp4'},
        'sentences': [{'id': 's1', 'start_ms': 0, 'end_ms': 1500, 'text': 'Share ideas.',
                       'translation': '分享想法。', 'speaker': 'Narrator'}],
        'words': [{'id': 'w1', 'sentence_id': 's1', 'index': 0, 'text': 'Share',
                   'start_ms': 0, 'end_ms': 500, 'gloss_zh': '分享', 'probability': .9}],
        'phrases': [],
        'cards': [{'id': 'c1', 'type': 'vocab', 'title': 'share', 'meaning': '分享',
                   'start_ms': 0, 'end_ms': 4000}],
        'layout': {'version': 1, 'pages': [{'id': 'p1', 'sentence_id': 's1', 'start_ms': 0,
                   'end_ms': 1500, 'lines': [['w1']], 'translation': '分享想法。'}]},
        'render': {'status': 'ready', 'width': 1920, 'height': 1080, 'fps': 30},
        'quality': {'needs_review': False}, 'artifacts': {'video': 'ready'},
    }


class RenderIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='render-identity-')
        self.folder = Path(self.temporary.name)
        for name in ('audio.mp3', 'source.mp4', 'cover.jpg', 'lesson.mp4'):
            (self.folder / name).write_bytes(('unique bytes for ' + name).encode())
        self.package = lesson()

    def tearDown(self):
        self.temporary.cleanup()

    def publish_manifest(self, name='lesson.mp4', start=0, end=4000):
        result = create_manifest(self.package, self.folder, self.folder / name, name, start, end)
        manifest_path(self.folder, name).write_text(json.dumps(result), encoding='utf-8')
        return result

    def test_ready_flag_without_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '凭据'):
            verify_render(self.package, self.folder, 'lesson.mp4')

    def test_real_draft_preparation_rejects_unbound_and_revised_exports(self):
        from app import publishing
        with patch.object(publishing, '_mutate', side_effect=lambda callback: callback([])):
            options = {'rights_confirmed': True, 'source': 'https://example.org/original'}
            unbound = publishing.prepare_publication(self.package, self.folder, options)
            self.assertEqual(unbound['status'], 'blocked')
            self.assertTrue(any('版本凭据' in reason for reason in unbound['blockers']))
            self.publish_manifest()
            bound = publishing.prepare_publication(self.package, self.folder, options)
            self.assertEqual(bound['status'], 'draft')
            self.package['words'][0]['end_ms'] = 650
            revised = publishing.prepare_publication(self.package, self.folder, options)
            self.assertEqual(revised['status'], 'blocked')
            self.assertTrue(any('修订' in reason for reason in revised['blockers']))

    def test_valid_export_and_operational_updates_keep_same_identity(self):
        original = self.publish_manifest()
        self.package['quality'] = {'needs_review': True, 'flags': ['Human review retained']}
        self.package['artifacts']['video'] = 'pending'
        self.package['render']['status'] = 'prepared'
        self.package['processing'] = {'revision': 'new audit run', 'layout_audit': {'status': 'passed'}}
        self.package['review'] = {'approved': True, 'feedback': 'browser annotations'}
        self.package['study_sections'] = {'version': 2, 'items': []}
        self.package['source']['url'] = 'https://another.example/source-provenance-only'
        self.package['words'][0]['probability'] = .4
        self.assertEqual(verify_render(self.package, self.folder, 'lesson.mp4'), original)

    def test_each_audible_or_visible_revision_rejects_previous_export(self):
        self.publish_manifest()
        mutations = [
            lambda p: p['words'][0].update(start_ms=80),
            lambda p: p['words'][0].update(gloss_zh='分给别人'),
            lambda p: p['sentences'][0].update(translation='共享创意。'),
            lambda p: p['cards'][0].update(meaning='与他人分享'),
            lambda p: p['layout']['pages'][0].update(translation='分享你的想法。'),
            lambda p: p['source'].update(name='Different displayed source'),
            lambda p: p['episode'].update(title='Different lesson'),
            lambda p: p['episode'].update(source_date='2026-10-22'),
        ]
        for mutate in mutations:
            revised = copy.deepcopy(self.package)
            mutate(revised)
            with self.subTest(revised=revised), self.assertRaisesRegex(ValueError, '修订'):
                verify_render(revised, self.folder, 'lesson.mp4')

    def test_all_source_assets_and_the_export_are_bound_by_file_bytes(self):
        for name in ('audio.mp3', 'source.mp4', 'cover.jpg', 'lesson.mp4'):
            with self.subTest(name=name):
                self.publish_manifest()
                path = self.folder / name
                before = path.read_bytes()
                path.write_bytes(before + b' changed')
                with self.assertRaises(ValueError):
                    verify_render(self.package, self.folder, 'lesson.mp4')
                path.write_bytes(before)

    def test_manifest_range_dimensions_identity_and_version_are_validated(self):
        cases = [('version', 99), ('width', 1280), ('height', 720), ('fps', 60),
                 ('episode_id', 'another-lesson'), ('video_file', 'old.mp4'),
                 ('range', {'start_ms': 0, 'end_ms': 3000}),
                 ('range', {'start_ms': True, 'end_ms': 4000})]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                record = self.publish_manifest()
                record[key] = value
                manifest_path(self.folder, 'lesson.mp4').write_text(json.dumps(record), encoding='utf-8')
                with self.assertRaises(ValueError):
                    verify_render(self.package, self.folder, 'lesson.mp4')

    def test_partial_export_records_exact_range(self):
        (self.folder / 'exports').mkdir()
        (self.folder / 'exports' / 'clip.mp4').write_bytes(b'rendered clip')
        self.publish_manifest('exports/clip.mp4', 750, 2500)
        result = verify_render(self.package, self.folder, 'exports/clip.mp4')
        self.assertEqual(result['range'], {'start_ms': 750, 'end_ms': 2500})

    def test_missing_input_and_escaped_paths_fail_closed(self):
        self.publish_manifest()
        (self.folder / 'cover.jpg').unlink()
        with self.assertRaisesRegex(ValueError, '缺失'):
            verify_render(self.package, self.folder, 'lesson.mp4')
        for name in ('../escape.mp4', '/escape.mp4', 'C:/escape.mp4'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                manifest_path(self.folder, name)

    def test_concurrent_input_revision_prevents_new_manifest(self):
        before = input_identity(self.package, self.folder)
        self.package['words'][0]['start_ms'] = 80
        with self.assertRaisesRegex(ValueError, '渲染期间'):
            create_manifest(self.package, self.folder, self.folder / 'lesson.mp4',
                            'lesson.mp4', 0, 4000, expected_inputs=before)

    def test_display_source_fallback_is_bound_but_same_named_source_url_is_not(self):
        base = content_fingerprint(self.package)
        self.package['source']['url'] = 'https://different.example/new'
        self.assertEqual(base, content_fingerprint(self.package))
        self.package['source'].pop('name')
        fallback = content_fingerprint(self.package)
        self.package['source']['url'] = 'https://renamed.example/new'
        self.assertNotEqual(fallback, content_fingerprint(self.package))

    def test_scene_styles_and_render_dependencies_expire_video_but_app_version_does_not(self):
        runtime = self.folder / 'render-source-fixture'
        for name in render_identity._RENDER_FILES:
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original rendering source: ' + name, encoding='utf-8')
        locked = {'name': 'fixture', 'version': '0.6.0', 'packages': {
            '': {'name': 'fixture', 'version': '0.6.0'},
            **{'node_modules/' + name: {'version': '1.0', 'integrity': 'test-hash'}
               for name in render_identity._RENDER_DEPENDENCIES}}}
        lock = runtime / 'package-lock.json'
        lock.write_text(json.dumps(locked), encoding='utf-8')
        with patch.object(render_identity, 'ROOT', runtime):
            self.publish_manifest()
            locked['version'] = locked['packages']['']['version'] = '0.6.1'
            locked['packages']['node_modules/vite'] = {'version': 'different dev dependency'}
            lock.write_text(json.dumps(locked), encoding='utf-8')
            verify_render(self.package, self.folder, 'lesson.mp4')
            scene_css = runtime / 'src/lesson.css'
            original_css = scene_css.read_bytes()
            scene_css.write_bytes(original_css + b' .scene-word{font-size:56px}')
            with self.assertRaisesRegex(ValueError, '画面样式'):
                verify_render(self.package, self.folder, 'lesson.mp4')
            scene_css.write_bytes(original_css)
            locked['packages']['node_modules/remotion']['version'] = '2.0'
            lock.write_text(json.dumps(locked), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '制作组件'):
                verify_render(self.package, self.folder, 'lesson.mp4')

    def test_renderer_publishes_manifest_and_rolls_back_failed_sidecar_replacement(self):
        from app import renderer, storage, study_page
        runtime = self.folder / 'runtime'
        cli = runtime / 'node_modules' / '@remotion' / 'cli' / 'remotion-cli.js'
        cli.parent.mkdir(parents=True)
        cli.write_text('fixture', encoding='utf-8')
        storage.save_episode(self.package, self.folder)
        (self.folder / 'study.html').write_text('previous study', encoding='utf-8')
        self.publish_manifest()

        class FinishedRender:
            returncode = 0

            def __init__(self, command, **kwargs):
                Path(command[command.index('Lesson') + 1]).write_bytes(b'rendered media' * 10000)

            def poll(self):
                return 0

        def staged_page(package, folder):
            path = Path(folder) / 'study.html'
            path.write_text('new study', encoding='utf-8')
            return path

        real_replace = os.replace
        probe = {'streams': [{'codec_type': 'video', 'width': 1920, 'height': 1080, 'r_frame_rate': '30/1'},
                             {'codec_type': 'audio'}], 'format': {'duration': '4.000'}}
        before = {name: (self.folder / name).read_bytes() for name in
                  ('lesson.mp4', 'lesson.mp4.manifest.json', 'study.html', 'episode.json')}

        def fail_new_manifest(source, destination):
            if Path(source).name == 'render-manifest.json':
                raise OSError('simulated manifest publication failure')
            return real_replace(source, destination)

        for fail in (True, False):
            with self.subTest(fail=fail), patch.object(renderer, 'ROOT', runtime), \
                    patch.object(renderer.shutil, 'which', return_value='node'), \
                    patch.object(renderer, '_browser_path', return_value=''), \
                    patch.object(renderer, 'audit_layout', return_value={'status': 'passed'}), \
                    patch.object(renderer.subprocess, 'Popen', FinishedRender), \
                    patch.object(renderer, '_probe', return_value=probe), \
                    patch.object(study_page, 'write_study_page', side_effect=staged_page), \
                    patch.object(renderer.os, 'replace', side_effect=fail_new_manifest if fail else real_replace):
                if fail:
                    with self.assertRaisesRegex(OSError, 'manifest publication failure'):
                        renderer.export_video(self.folder / 'episode.json', self.folder, lambda *args: None)
                    for name, content in before.items():
                        self.assertEqual((self.folder / name).read_bytes(), content, name)
                else:
                    renderer.export_video(self.folder / 'episode.json', self.folder, lambda *args: None)
                    current = json.loads((self.folder / 'episode.json').read_text(encoding='utf-8'))
                    self.assertEqual(current['render']['video_file'], 'lesson.mp4')
                    self.assertEqual(verify_render(current, self.folder, 'lesson.mp4')['range']['end_ms'], 4000)
                    self.assertEqual((self.folder / 'study.html').read_text(encoding='utf-8'), 'new study')
            self.assertFalse((self.folder / 'lesson.rendering.mp4').exists())
            self.assertFalse(list(self.folder.glob('.publish-*')))


if __name__ == '__main__':
    unittest.main()
