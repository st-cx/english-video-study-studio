from __future__ import annotations

import array
import json
import math
import shutil
import socket
import subprocess
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from app import media, sources
from app.subtitle_input import parse_subtitles

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)


class FakeResponse:
    def __init__(self, data=b"", *, status=200, headers=None, fail_after=False, url="https://public.example/file.mp3"):
        self.data, self.position, self.status, self.fail_after, self.url = data, 0, status, fail_after, url
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = str(value)

    def read(self, count):
        if self.position >= len(self.data):
            if self.fail_after:
                raise OSError("interrupted")
            return b""
        block = self.data[self.position:self.position + count]
        self.position += len(block)
        return block

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class SubtitleInputTests(unittest.TestCase):
    def parse(self, text, suffix):
        with tempfile.TemporaryDirectory(dir=WORK) as temporary:
            path = Path(temporary) / ("captions" + suffix)
            path.write_text(text, encoding="utf-8")
            return parse_subtitles(path)

    def test_srt_keeps_repeated_spoken_words_and_separate_dialogue(self):
        value = self.parse("1\n00:00:00,125 --> 00:00:01,250\nI, I think so.\n\n2\n00:00:02,000 --> 00:00:02,500\nYes.\n\n3\n00:00:02,600 --> 00:00:03,000\nYes.\n", ".srt")
        self.assertEqual(value["cues"][0], {"start_ms": 125, "end_ms": 1250, "text": "I, I think so."})
        self.assertEqual(value["text"].count("Yes."), 2)

    def test_vtt_deduplicates_only_overlapping_rolling_context(self):
        value = self.parse("WEBVTT\n\n00:00.000 --> 00:02.000\nHello there\n\n00:01.500 --> 00:03.000\nHello there <00:02.000><c>everyone.</c>\n\n00:04.000 --> 00:05.000\nHello there\n", ".vtt")
        self.assertEqual(value["text"], "Hello there\neveryone.\nHello there")

    def test_overlapping_single_word_answers_are_not_deleted(self):
        value = self.parse("WEBVTT\n\n00:00.000 --> 00:01.000\nNo.\n\n00:00.800 --> 00:01.500\nNo.\n", ".vtt")
        self.assertEqual(value["text"], "No.\nNo.")

    def test_line_wrapping_does_not_remove_oral_repetition(self):
        value = self.parse("1\n00:00:00,000 --> 00:00:02,000\nReally?\nReally?\n", ".srt")
        self.assertEqual(value["text"], "Really? Really?")

    def test_explicit_sound_notes_are_removed_but_real_parentheses_remain(self):
        value = self.parse("WEBVTT\n\n00:00.000 --> 00:01.000\n(Laughter)\n\n00:01.000 --> 00:03.000\nI think (we can do this) [Applause].\n\n00:03.000 --> 00:04.000\n[inaudible]\n", ".vtt")
        self.assertEqual(value["text"], "I think (we can do this) .")
        self.assertEqual(len(value["cues"]), 1)

    def test_long_single_animal_sound_removed_without_deleting_english(self):
        value = self.parse("WEBVTT\n\n00:00.000 --> 00:01.000\n(baaaaaaaaaaahhh!!)\n\n00:01.000 --> 00:02.000\n(and that's cool) (sooooo good)\n", ".vtt")
        self.assertEqual(value["text"], "(and that's cool) (sooooo good)")

    def test_ass_respects_format_comma_text_and_karaoke_duplicates(self):
        value = self.parse("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.25,0:00:02.00,Default,,0,0,0,,{\\1c&H00FF00&}Hello, everyone.\nDialogue: 0,0:00:02.00,0:00:02.50,Default,,0,0,0,,{\\1c&H00AA00&}Hello, everyone.\n", ".ass")
        self.assertEqual(value["text"], "Hello, everyone.")
        self.assertEqual(value["cues"][0]["end_ms"], 2500)

    def test_txt_missing_and_empty_subtitle_are_normal(self):
        self.assertEqual(parse_subtitles("")["kind"], "none")
        value = self.parse("No.\nNo.\nI I don't know.", ".txt")
        self.assertEqual(value["cues"], [])
        self.assertEqual(value["text"].count("No."), 2)
        self.assertEqual(self.parse("", ".vtt")["cues"], [])

    def test_pdf_removes_repeated_page_furniture_preserves_dialogue(self):
        import fitz
        with tempfile.TemporaryDirectory(dir=WORK) as temporary:
            path = Path(temporary) / "transcript.pdf"
            document = fitz.open()
            for number in range(2):
                page = document.new_page()
                page.insert_text((30, 30), "BBC LEARNING ENGLISH")
                page.insert_text((30, 100), "I, I think the answer is yes.")
                page.insert_text((30, 750), "Copyright BBC")
            document.save(path)
            document.close()
            value = parse_subtitles(path)
            self.assertNotIn("BBC LEARNING ENGLISH", value["text"])
            self.assertEqual(value["text"].count("I, I think"), 2)
            self.assertEqual(value["cues"], [])

    def test_invalid_timing_does_not_become_a_plain_transcript(self):
        with self.assertRaises(ValueError):
            self.parse("1\n00:00:02,000 --> 00:00:01,000\nHello\n", ".srt")


class SourceSafetyTests(unittest.TestCase):
    def test_private_loopback_and_userinfo_are_rejected(self):
        for url in ("file:///C:/secret.mp3", "http://user:pass@public.example/a.mp3", "http://localhost/a.mp3"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                sources.checked_url(url)
        for address in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1"):
            with self.subTest(address=address), patch.object(socket, "getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))]):
                with self.assertRaises(ValueError):
                    sources.checked_url("http://public.example/media.mp4")

    def test_mixed_public_private_dns_answers_are_rejected(self):
        records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, 443)) for host in ("93.184.216.34", "192.168.1.1")]
        with patch.object(socket, "getaddrinfo", return_value=records), self.assertRaises(ValueError):
            sources.checked_url("https://public.example/a.mp3")

    def test_youtube_redirect_url_cannot_reach_generic_extractor(self):
        with self.assertRaises(ValueError):
            sources._youtube_target("https://www.youtube.com/redirect?q=http://127.0.0.1/private")
        self.assertEqual(sources._youtube_target("https://youtu.be/jNQXAC9IVRw?si=ignored"), "https://www.youtube.com/watch?v=jNQXAC9IVRw")

    def test_youtube_resolution_cap_preserves_best_english_audio(self):
        metadata = {"formats": [
            {"url": "https://public.example/hd", "protocol": "https", "ext": "mp4", "vcodec": "h264", "acodec": "none", "height": 1080, "width": 1920},
            {"url": "https://public.example/low", "protocol": "https", "ext": "mp4", "vcodec": "h264", "acodec": "none", "height": 360, "width": 640},
            {"url": "https://public.example/audio", "protocol": "https", "ext": "m4a", "vcodec": "none", "acodec": "aac", "language": "en", "abr": 160},
        ]}
        video, audio = sources._youtube_streams(metadata, max_video_height=480)
        self.assertEqual(video["height"], 360)
        self.assertEqual(audio["abr"], 160)

    def test_redirect_private_ip_is_checked_before_connection(self):
        class FakeConnection:
            def __init__(self, *args, **kwargs):
                pass
            def request(self, *args, **kwargs):
                pass
            def getresponse(self):
                return FakeResponse(status=302, headers={"Location": "http://127.0.0.1/private"})
            def close(self):
                pass
        with patch.object(sources, "_PinnedHTTPSConnection", FakeConnection), patch.object(sources, "_public_addresses", side_effect=[["93.184.216.34"], ValueError("private")]) as resolve:
            with self.assertRaises(ValueError):
                sources._open_url("https://public.example/file")
            self.assertEqual(resolve.call_count, 2)

    def test_html_is_not_published_as_media(self):
        with tempfile.TemporaryDirectory(dir=WORK) as temporary:
            target = Path(temporary) / "original.mp4"
            target.write_bytes(b"previous valid file")
            response = FakeResponse(b"<!doctype html><title>Sign in</title>", headers={"Content-Type": "text/html"})
            with patch.object(sources, "_open_url", return_value=response), self.assertRaises(ValueError):
                sources._download("https://public.example/video.mp4", target)
            self.assertEqual(target.read_bytes(), b"previous valid file")

    def test_interrupted_download_resumes_with_validator_and_exact_range(self):
        with tempfile.TemporaryDirectory(dir=WORK) as temporary:
            target = Path(temporary) / "audio.mp3"
            replies = [FakeResponse(b"01234", headers={"Content-Type": "audio/mpeg", "Content-Length": 10, "ETag": '"v1"'}, fail_after=True),
                       FakeResponse(b"56789", status=206, headers={"Content-Type": "audio/mpeg", "Content-Length": 5, "Content-Range": "bytes 5-9/10", "ETag": '"v1"'})]
            with patch.object(sources, "_open_url", side_effect=replies) as opened, patch.object(sources, "probe_media", return_value={"streams": [{"codec_type": "audio"}]}):
                sources._download("https://public.example/audio.mp3", target)
                self.assertEqual(opened.call_args_list[1].kwargs["headers"]["Range"], "bytes=5-")
            self.assertEqual(target.read_bytes(), b"0123456789")
            self.assertFalse(target.with_name("audio.mp3.part").exists())

    def test_optional_download_never_swallows_cooperative_pause(self):
        class Paused(RuntimeError):
            pass
        def pause(*args):
            raise Paused("pause")
        with tempfile.TemporaryDirectory(dir=WORK) as temporary:
            with self.assertRaises(Paused):
                sources._optional_download("https://public.example/cover.jpg", Path(temporary) / "cover.jpg", pause, "image")

    def test_bilibili_dispatch_does_not_call_youtube(self):
        with tempfile.TemporaryDirectory(dir=WORK) as temporary:
            folder = Path(temporary)
            media_path = folder / "video.mp4"
            media_path.write_bytes(b"public video fixture")
            value = {"media_path": str(media_path), "title": "Bili", "source_kind": "bilibili", "source_url": "https://www.bilibili.com/video/BV1XrL56vEgu/", "subtitle_path": "", "cover_path": ""}
            probe = {"streams": [{"codec_type": "audio"}], "format": {"duration": "1"}}
            with patch.object(sources, "checked_url", side_effect=lambda value, **kwargs: value), patch.object(sources, "_acquire_bili", return_value=value) as bili, patch.object(sources, "_youtube_metadata") as youtube, patch.object(sources, "probe_media", return_value=probe):
                result = sources.acquire_source(value["source_url"], folder)
                self.assertEqual(result["source_kind"], "bilibili")
                bili.assert_called_once()
                youtube.assert_not_called()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required for real media integration")
class MediaIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=WORK)
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def ffmpeg(self, *arguments):
        subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, arguments)],
                       check=True, capture_output=True, timeout=60)

    def samples(self, path):
        result = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "8000", "pipe:1"],
                                check=True, capture_output=True, timeout=30)
        samples = array.array("h")
        samples.frombytes(result.stdout)
        return samples

    def test_audio_crop_real_mp3_lazy_wav_and_content_fingerprint(self):
        source = self.folder / "first-name.wav"
        self.ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3", source)
        out = self.folder / "prepared"
        value = media.prepare_media(source, out, start_ms=500, end_ms=2500)
        self.assertAlmostEqual(value["duration_s"], 2, places=2)
        self.assertEqual(value["source_offset_ms"], 500)
        self.assertEqual(value["video_path"], "")
        self.assertEqual(media.probe_media(value["audio_path"])["streams"][0]["codec_name"], "mp3")
        self.assertFalse((out / "asr-16k.wav").exists())
        wav = media.ensure_asr_wav(value["audio_path"], out)
        stream = media.probe_media(wav)["streams"][0]
        self.assertEqual((stream["sample_rate"], stream["channels"], stream["codec_name"]), ("16000", 1, "pcm_s16le"))
        moved = self.folder / "different-name.wav"
        shutil.copyfile(source, moved)
        reused = media.prepare_media(moved, out, start_ms=500, end_ms=2500)
        self.assertEqual(value["fingerprint"], reused["fingerprint"])
        self.assertNotIn(str(source), json.dumps(value["media_info"]))

    def test_portrait_multiple_audio_tracks_and_selected_track_content(self):
        source = self.folder / "portrait.mkv"
        self.ffmpeg("-f", "lavfi", "-i", "testsrc=size=90x160:rate=15:duration=2", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=880:duration=2", "-map", "0:v", "-map", "1:a", "-map", "2:a",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=jpn", source)
        value = media.prepare_media(source, self.folder / "normalized", audio_stream_index=2)
        info = media.probe_media(value["video_path"])
        video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
        self.assertEqual((video["width"], video["height"], video["codec_name"], video["avg_frame_rate"]), (90, 160, "h264", "30/1"))
        self.assertEqual(value["audio_stream_index"], 2)
        samples = self.samples(value["audio_path"])[2000:8000]
        def energy(hz):
            return abs(sum(sample * complex(math.cos(2 * math.pi * hz * index / 8000), math.sin(2 * math.pi * hz * index / 8000)) for index, sample in enumerate(samples)))
        self.assertGreater(energy(880), energy(440) * 20)
        with self.assertRaises(ValueError):
            media.prepare_media(source, self.folder / "invalid", audio_stream_index=0)

    def test_delayed_audio_keeps_common_pts_after_crop(self):
        source = self.folder / "offset.mkv"
        self.ffmpeg("-f", "lavfi", "-i", "color=c=blue:size=160x90:rate=10:duration=3", "-itsoffset", "0.8", "-f", "lavfi", "-i", "sine=frequency=600:sample_rate=48000:duration=2.2",
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-c:a", "aac", source)
        value = media.prepare_media(source, self.folder / "normalized", start_ms=500, end_ms=2500)
        samples = self.samples(value["audio_path"])
        rms = lambda values: math.sqrt(sum(sample * sample for sample in values) / max(1, len(values)))
        self.assertLess(rms(samples[0:1200]), 40)
        self.assertGreater(rms(samples[4000:5600]), 1000)
        self.assertAlmostEqual(media.media_duration(media.probe_media(value["video_path"])), 2, delta=0.08)

    def test_vfr_and_landscape_become_cfr_without_changing_speed(self):
        source = self.folder / "vfr.mkv"
        self.ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=3", "-f", "lavfi", "-i", "sine=frequency=500:duration=5",
                    "-vf", "setpts=if(lt(N\\,10)\\,N/(10*TB)\\,(N-10)/(5*TB)+1/TB)", "-map", "0:v", "-map", "1:a", "-fps_mode", "vfr",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", source)
        value = media.prepare_media(source, self.folder / "normalized")
        original = media.media_duration(media.probe_media(source))
        info = media.probe_media(value["video_path"])
        video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
        self.assertEqual(video["avg_frame_rate"], "30/1")
        self.assertAlmostEqual(media.media_duration(info), original, delta=0.08)

    def test_pause_terminates_conversion_and_preserves_existing_artifact(self):
        source = self.folder / "source.mp4"
        self.ffmpeg("-f", "lavfi", "-i", "color=size=160x90:rate=10:duration=1", "-f", "lavfi", "-i", "sine=duration=1", "-c:v", "libx264", "-c:a", "aac", source)
        out = self.folder / "paused"
        out.mkdir()
        (out / "audio.mp3").write_bytes(b"previous artifact")
        class Paused(RuntimeError):
            pass
        def progress(stage, percent, message):
            if percent == 4:
                raise Paused("pause")
        with self.assertRaises(Paused):
            media.prepare_media(source, out, progress=progress)
        self.assertEqual((out / "audio.mp3").read_bytes(), b"previous artifact")
        self.assertFalse((out / "media-prepared.json").exists())

    def test_fake_html_rejected_by_real_probe(self):
        source = self.folder / "fake.mp4"
        source.write_text("<!doctype html><title>Login required</title>")
        with self.assertRaises(ValueError):
            media.probe_media(source)

    def test_disguised_playlist_is_rejected_before_following_references(self):
        source = self.folder / "disguised.mp4"
        source.write_text("#EXTM3U\n#EXTINF:10,\nfile:///C:/private/movie.mp4\n")
        with patch.object(media.subprocess, "run") as runner, self.assertRaises(ValueError):
            media.probe_media(source)
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
