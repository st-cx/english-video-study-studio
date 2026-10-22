"""Publishing boundaries use real files; expensive render/model work is stubbed."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import general, pipeline, renderer, storage, study_page

ROOT = Path(__file__).resolve().parents[1]


def episode_package() -> dict:
    return {
        "version": 4,
        "episode": {"id": "publication-fixture", "title": "Replacement edition", "title_zh": "新版学习内容",
                    "source_url": "https://example.com/public-lesson.mp4", "duration_s": 2, "audio_file": "audio.mp3"},
        "sentences": [{"id": "s1", "start_ms": 100, "end_ms": 600, "text": "Hello.", "translation": "你好。",
                       "speaker": "", "word_ids": ["w1"], "card_ids": []}],
        "words": [{"id": "w1", "index": 0, "sentence_id": "s1", "text": "Hello.", "start_ms": 100,
                   "end_ms": 600, "probability": 1, "focus": True, "gloss_zh": "你好"}],
        "cards": [], "phrases": [],
        "layout": {"version": 1, "pages": [{"id": "p1", "sentence_id": "s1", "start_ms": 100,
                                               "end_ms": 600, "lines": [["w1"]], "translation": "你好。"}]},
        "chapters": [{"id": "ch1", "title": "学习单元 1", "start_ms": 0, "end_ms": 2000, "status": "ready", "flags": []}],
        "render": {"status": "prepared", "width": 1920, "height": 1080, "fps": 30},
        "artifacts": {"study": "pending", "video": "not_requested"},
        "quality": {"needs_review": False, "flags": [], "structural_flags": []}, "processing": {},
    }


class PublishIntegrityTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="publish-integrity-", dir=ROOT / "work")
        self.base = Path(self.temporary.name)
        self.folder = self.base / "episode"
        self.folder.mkdir()
        (self.folder / 'audio.mp3').write_bytes(b'fixture source audio')
        # Standalone static assets keep tests independent of an npm build.
        self.runtime = self.base / "runtime"
        assets = self.runtime / "dist" / "offline"
        assets.mkdir(parents=True)
        (assets / "lesson.js").write_text("window.testLesson=true;", encoding="utf-8")
        (assets / "lesson.css").write_text("body{margin:0}", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def seed_previous(self):
        previous = episode_package()
        previous["episode"]["title"] = "Previous published edition"
        previous["artifacts"] = {"study": "ready", "video": "ready"}
        previous["render"]["status"] = "ready"
        storage.save_episode(previous, self.folder)
        for name in ("study.html", "lesson.mp4", "lesson.mp4.manifest.json", "subtitles.en.srt", "subtitles.zh.srt", "captions.ass", "study-notes.md", "vocabulary.csv"):
            (self.folder / name).write_bytes(("previous content: " + name).encode())
        return {path.name: path.read_bytes() for path in self.folder.iterdir() if path.is_file()}

    def test_page_generation_failure_preserves_published_files_and_not_ready(self):
        previous = self.seed_previous()
        candidate = episode_package()
        real_replace = os.replace

        def fail_page_publish(source, destination):
            if Path(destination).name == "study.html":
                raise OSError("simulated page write failure")
            return real_replace(source, destination)

        with patch.object(study_page, "ROOT", self.runtime), patch.object(renderer, "audit_layout", return_value={"status": "passed", "cases": 1, "checks": 1}), patch.object(study_page.os, "replace", side_effect=fail_page_publish):
            with self.assertRaisesRegex(OSError, "page write failure"):
                general._write_bundle(candidate, self.folder, lambda *args: None)
        for name, content in previous.items():
            self.assertEqual((self.folder / name).read_bytes(), content, name)
        published = json.loads((self.folder / "episode.json").read_text(encoding="utf-8"))
        self.assertEqual(published["episode"]["title"], "Previous published edition")
        self.assertNotEqual(candidate["artifacts"]["study"], "ready")
        self.assertFalse(any(self.folder.glob(".bundle-*")))

    def test_render_metadata_save_failure_rolls_back_video_page_and_package(self):
        self.seed_previous()
        candidate = episode_package()
        storage.save_episode(candidate, self.folder)
        package_path = self.folder / "episode.json"
        previous = {name: (self.folder / name).read_bytes() for name in ("lesson.mp4", "lesson.mp4.manifest.json", "study.html", "episode.json")}
        cli = self.runtime / "node_modules" / "@remotion" / "cli" / "remotion-cli.js"
        cli.parent.mkdir(parents=True)
        cli.write_text("fixture", encoding="utf-8")

        class FinishedRender:
            returncode = 0

            def __init__(self, command, **kwargs):
                output = Path(command[command.index("Lesson") + 1])
                output.write_bytes(b"new mocked media output" * 6000)

            def poll(self):
                return 0

            def wait(self):
                return 0

        probe = {"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1"},
                              {"codec_type": "audio", "codec_name": "aac"}], "format": {"duration": "2.000"}}
        real_save = storage.save_episode
        attempted_states = []

        def save_then_fail(value, folder, *args, **kwargs):
            # The failure occurs after a real metadata replacement. The
            # publisher must roll back that file as well as video and page.
            path = real_save(value, folder, *args, **kwargs)
            attempted_states.append(json.loads(path.read_text(encoding="utf-8"))["artifacts"]["video"])
            raise OSError("simulated metadata disk failure")

        with patch.object(renderer, "ROOT", self.runtime), patch.object(study_page, "ROOT", self.runtime), patch.object(renderer.shutil, "which", return_value="node"), patch.object(renderer, "_browser_path", return_value=""), patch.object(renderer, "audit_layout", return_value={"status": "passed", "cases": 1, "checks": 1}), patch.object(renderer.subprocess, "Popen", FinishedRender), patch.object(renderer, "_probe", return_value=probe), patch.object(storage, "save_episode", side_effect=save_then_fail):
            with self.assertRaisesRegex(OSError, "metadata disk failure"):
                renderer.export_video(package_path, self.folder, lambda *args: None)
        self.assertEqual(attempted_states, ["ready"])
        for name, content in previous.items():
            self.assertEqual((self.folder / name).read_bytes(), content, name)
        self.assertFalse((self.folder / "lesson.rendering.mp4").exists())
        self.assertFalse(any(self.folder.glob(".publish-*")))
        self.assertNotEqual(json.loads(package_path.read_text(encoding="utf-8"))["artifacts"]["video"], "ready")

    def test_cross_chapter_overlap_is_blocked_even_when_each_chapter_is_valid(self):
        package = episode_package()
        package["words"][0]["end_ms"] = 1100
        package["sentences"][0]["end_ms"] = 1100
        package["layout"]["pages"][0]["end_ms"] = 1100
        package["words"].append({"id": "w2", "index": 1, "sentence_id": "s2", "text": "Again.", "start_ms": 1000,
                                 "end_ms": 1500, "probability": 1, "focus": True, "gloss_zh": "再次"})
        package["sentences"].append({"id": "s2", "start_ms": 1000, "end_ms": 1500, "text": "Again.", "translation": "再次。",
                                     "speaker": "", "word_ids": ["w2"], "card_ids": []})
        package["layout"]["pages"].append({"id": "p2", "sentence_id": "s2", "start_ms": 1000, "end_ms": 1500,
                                            "lines": [["w2"]], "translation": "再次。"})
        package["chapters"] = [{"id": "ch1", "title": "学习单元 1", "start_ms": 0, "end_ms": 1000, "status": "ready", "flags": []},
                                {"id": "ch2", "title": "学习单元 2", "start_ms": 1000, "end_ms": 2000, "status": "ready", "flags": []}]
        for chapter in package["chapters"]:
            local = pipeline._quality(general._chapter_package(package, chapter), "")
            self.assertEqual(local["structural_flags"], [])
        with patch.object(study_page, "ROOT", self.runtime), patch.object(renderer, "audit_layout", return_value={"status": "passed", "cases": 2, "checks": 2}), patch.object(pipeline, "_annotate", side_effect=AssertionError("No model request is allowed")):
            with self.assertRaises(pipeline.ReviewRequired):
                general._process_chapters(package, self.folder, "", [], lambda *args: None)
        published = json.loads((self.folder / "episode.json").read_text(encoding="utf-8"))
        self.assertTrue(published["quality"]["needs_review"])
        self.assertTrue(published["quality"]["structural_flags"])
        self.assertTrue(published["quality"]["structural_issues"])
        self.assertEqual(published["chapters"][1]["status"], "review")
        with patch.object(renderer, "_render_video") as render:
            with self.assertRaises(ValueError):
                renderer.export_video(self.folder / "episode.json", self.folder, lambda *args: None)
            render.assert_not_called()


if __name__ == "__main__":
    unittest.main()
