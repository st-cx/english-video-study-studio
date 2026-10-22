"""Local media normalization with a shared audio/video presentation clock.

All external URLs must first be acquired by sources.py. FFmpeg is restricted
to local file/pipe protocols, so a disguised playlist cannot fetch the LAN.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

Progress = Callable[[str, int, str], None]
MEDIA_VERSION = 1


def _emit(progress: Progress | None, stage: str, percent: int, message: str) -> None:
    if progress:
        progress(stage, percent, message)


def _executable(name: str) -> str:
    result = shutil.which(name)
    if not result:
        raise RuntimeError(f"未找到 {name}，请安装 FFmpeg 并加入 PATH。")
    return result


def _local_file(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve(strict=True)
    if not result.is_file():
        raise ValueError("请选择本地音频或视频文件。")
    return result


def probe_media(path: str | Path) -> dict:
    """Return the raw ffprobe object; reject remote/playlist and non-media input."""
    source = _local_file(path)
    with source.open("rb") as stream:
        prefix = stream.read(512).lstrip().lower()
    if source.suffix.lower() in {".m3u", ".m3u8", ".ffconcat", ".sdp", ".mpd"} or prefix.startswith((b"#extm3u", b"ffconcat version", b"<?xml", b"<mpd", b"v=0\r\n", b"v=0\n")):
        raise ValueError("请选择独立的音视频文件，不支持会引用其他文件或网址的播放列表。")
    try:
        result = subprocess.run(
            [_executable("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe",
             "-show_streams", "-show_format", "-show_chapters", "-of", "json", str(source)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("读取媒体信息超时，请检查文件是否损坏。") from exc
    if result.returncode:
        raise ValueError("文件无法解码为音频或视频；请检查是否误下载了网页、登录提示或不完整文件。")
    try:
        value = json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError("媒体信息不是有效 JSON。") from exc
    if not any(stream.get("codec_type") in {"audio", "video"} for stream in value.get("streams", [])):
        raise ValueError("文件没有音频或视频轨道。")
    return value


def _number(value, default: float = 0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def media_duration(info: dict) -> float:
    duration = _number(info.get("format", {}).get("duration"))
    if duration <= 0:
        duration = max((_number(stream.get("duration")) for stream in info.get("streams", [])), default=0)
    if duration <= 0:
        raise ValueError("媒体缺少有限时长，暂不支持直播或未完成的媒体流。")
    return duration


def file_sha256(path: str | Path, progress: Progress | None = None) -> str:
    source = _local_file(path)
    size = source.stat().st_size
    value = hashlib.sha256()
    with source.open("rb") as stream:
        read = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
            read += len(block)
            _emit(progress, "media", 2, f"正在核对素材指纹 {read // (1024 * 1024)} / {max(1, size // (1024 * 1024))} MB…")
    return value.hexdigest()


def run_ffmpeg(arguments: list[str], folder: Path, *, progress: Progress | None = None,
               duration_s: float = 0, message: str = "正在处理媒体…", percent: int = 5) -> None:
    """Run a local-only FFmpeg job and propagate cooperative pause exceptions."""
    folder.mkdir(parents=True, exist_ok=True)
    command = [_executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *arguments]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        started = time.monotonic()
        timeout = max(300, duration_s * 15 + 180)
        try:
            while process.poll() is None:
                _emit(progress, "media", percent, message)
                if time.monotonic() - started > timeout:
                    raise TimeoutError("媒体转换超时，已保留原素材。")
                time.sleep(0.25)
            if process.returncode:
                stderr.seek(0)
                detail = stderr.read().decode("utf-8", errors="replace")[-1800:]
                raise RuntimeError("媒体转换失败：" + detail)
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=20)
            raise
    _emit(progress, "media", percent, message)


def _audio_stream(info: dict, audio_stream_index: int | None) -> dict:
    audios = [stream for stream in info.get("streams", []) if stream.get("codec_type") == "audio"]
    if not audios:
        raise ValueError("媒体没有音轨，无法制作英语听力学习包。")
    if audio_stream_index is not None:
        if isinstance(audio_stream_index, bool) or not isinstance(audio_stream_index, int):
            raise ValueError("音轨编号必须是 ffprobe 返回的整数 stream.index。")
        chosen = next((stream for stream in audios if stream.get("index") == audio_stream_index), None)
        if not chosen:
            raise ValueError("所选音轨不存在，请按媒体信息中的音轨编号重新选择。")
        return chosen
    return next((stream for stream in audios if stream.get("disposition", {}).get("default")), audios[0])


def _cache_valid(folder: Path, fingerprint: str) -> dict | None:
    try:
        cached = json.loads((folder / "media-prepared.json").read_text(encoding="utf-8"))
        if cached["fingerprint"] != fingerprint:
            return None
        for name, expected in cached["output_hashes"].items():
            if Path(name).name != name or file_sha256(folder / name) != expected:
                return None
        result = cached["result"]
        result["audio_path"] = str(folder / "audio.mp3")
        result["video_path"] = str(folder / "source.mp4") if result.get("video_path") else ""
        result["cover_path"] = str(folder / "cover.jpg") if result.get("cover_path") else ""
        return result
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _portable_probe(info: dict | None) -> dict | None:
    if info is None:
        return None
    value = json.loads(json.dumps(info))
    value.get("format", {}).pop("filename", None)
    value.get("format", {}).pop("tags", None)
    for stream in value.get("streams", []):
        tags = stream.pop("tags", {})
        if tags.get("language"):
            stream["language"] = tags["language"]
    return value


def _publish(staged: Path, folder: Path, names: list[str]) -> None:
    """Keep a rollback copy until every normalized artifact has been published."""
    replaced, previous = [], []
    try:
        for name in names:
            destination = folder / name
            backup = staged / ("previous-" + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                os.replace(destination, backup)
                previous.append((destination, backup))
            os.replace(staged / name, destination)
            replaced.append(destination)
    except BaseException:
        for path in reversed(replaced):
            path.unlink(missing_ok=True)
        for destination, backup in reversed(previous):
            if backup.is_file():
                os.replace(backup, destination)
        raise


def prepare_media(path: str | Path, folder: str | Path, audio_stream_index: int | None = None,
                  start_ms: int = 0, end_ms: int | None = None, progress: Progress | None = None) -> dict:
    """Normalize a local source, preserving its aspect ratio and A/V offset.

    audio_stream_index is a global FFprobe stream.index, not an audio ordinal.
    Cropping is relative to the common source presentation origin. Both
    streams subtract that same origin and crop offset; audio is padded with
    measured silence instead of independently moving each stream to zero.
    """
    source, folder = _local_file(path), Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    info = probe_media(source)
    full_duration = media_duration(info)
    for value in (start_ms, end_ms):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("裁剪时间必须是整数毫秒。")
    end_ms = round(full_duration * 1000) if end_ms is None else end_ms
    if start_ms < 0 or end_ms <= start_ms or end_ms > round(full_duration * 1000) + 100:
        raise ValueError("裁剪范围应位于素材内，结束位置必须晚于起始位置。")
    end_ms = min(end_ms, round(full_duration * 1000))
    audio = _audio_stream(info, audio_stream_index)
    video = next((stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"
                  and not stream.get("disposition", {}).get("attached_pic")), None)
    original_hash = file_sha256(source, progress)
    spec = {"version": MEDIA_VERSION, "source_sha256": original_hash, "audio_stream_index": audio["index"],
            "start_ms": start_ms, "end_ms": end_ms}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    cached = _cache_valid(folder, fingerprint)
    if cached:
        origin=_number(info.get('format',{}).get('start_time'))
        start,end=start_ms/1000,end_ms/1000
        cached['chapters']=[{'start_time':max(0,_number(ch.get('start_time'))-origin-start),
                            'end_time':min(end-start,_number(ch.get('end_time'))-origin-start),
                            'title':str(ch.get('tags',{}).get('title',''))[:200]}
                           for ch in info.get('chapters',[]) if _number(ch.get('end_time'))>origin+start and _number(ch.get('start_time'))<origin+end]
        _emit(progress, "media", 9, "复用已完成的媒体归一结果。")
        return cached
    origin = _number(info.get("format", {}).get("start_time"), min((_number(s.get("start_time")) for s in info["streams"]), default=0))
    start, end, duration = start_ms / 1000, end_ms / 1000, (end_ms - start_ms) / 1000
    # With -copyts, the two filters subtract an identical origin and offset.
    # first_pts=0 inserts real silence before a delayed audio stream.
    af = (f"asetpts=PTS-({origin:.9f})/TB,atrim=start={start:.9f}:end={end:.9f},"
          f"asetpts=PTS-({start:.9f})/TB,aresample=48000:async=1:first_pts=0,apad=whole_dur={duration:.9f}")
    with tempfile.TemporaryDirectory(prefix=".media-stage-", dir=folder) as temporary:
        staged = Path(temporary)
        names = []
        if video:
            vf = (f"setpts=PTS-({origin:.9f})/TB,trim=start={start:.9f}:end={end:.9f},"
                  f"setpts=PTS-({start:.9f})/TB,fps=fps=30:start_time=0,"
                  f"scale=trunc(iw*sar/2)*2:trunc(ih/2)*2,setsar=1,tpad=stop_mode=clone:stop_duration={duration:.9f}")
            arguments = ["-protocol_whitelist", "file,pipe", "-copyts", "-i", str(source),
                         "-filter_complex", f"[0:{video['index']}]{vf}[v];[0:{audio['index']}]{af}[a]",
                         "-map", "[v]", "-map", "[a]", "-t", f"{duration:.9f}",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-map_metadata", "-1",
                         str(staged / "source.mp4")]
            run_ffmpeg(arguments, staged, progress=progress, duration_s=duration, message="正在保留原画面并统一音画时间…", percent=4)
            normalized = probe_media(staged / "source.mp4")
            audio_input = staged / "source.mp4"
            mp3_args = ["-protocol_whitelist", "file,pipe", "-i", str(audio_input), "-map", "0:a:0", "-vn"]
            names.append("source.mp4")
        else:
            normalized = None
            mp3_args = ["-protocol_whitelist", "file,pipe", "-copyts", "-i", str(source),
                        "-map", f"0:{audio['index']}", "-vn", "-af", af]
        mp3_args += ["-t", f"{duration:.9f}", "-c:a", "libmp3lame", "-b:a", "160k", "-ar", "48000",
                     "-map_metadata", "-1", str(staged / "audio.mp3")]
        run_ffmpeg(mp3_args, staged, progress=progress, duration_s=duration, message="正在生成学习原声 MP3…", percent=6)
        audio_info = probe_media(staged / "audio.mp3")
        if not any(stream.get("codec_name") == "mp3" for stream in audio_info["streams"]):
            raise RuntimeError("原声转换未生成有效 MP3。")
        if abs(media_duration(audio_info) - duration) > 0.15:
            raise RuntimeError("归一音频时长与裁剪范围不符，已保留原素材。")
        names.append("audio.mp3")
        cover = False
        attached = next((stream for stream in info["streams"] if stream.get("disposition", {}).get("attached_pic")), None)
        if video or attached:
            cover_input = staged / "source.mp4" if video else source
            cover_args = ["-protocol_whitelist", "file,pipe", "-i", str(cover_input)]
            cover_args += ["-ss", str(min(3.0, duration / 4)), "-map", "0:v:0"] if video else ["-map", f"0:{attached['index']}"]
            cover_args += ["-frames:v", "1", "-q:v", "3", str(staged / "cover.jpg")]
            run_ffmpeg(cover_args, staged, progress=progress, duration_s=duration, message="正在提取节目封面…", percent=8)
            cover = (staged / "cover.jpg").is_file()
            if cover:
                names.append("cover.jpg")
        result = {
            "audio_path": str(folder / "audio.mp3"), "video_path": str(folder / "source.mp4") if video else "",
            "cover_path": str(folder / "cover.jpg") if cover else "", "duration_s": duration,
            "source_offset_ms": start_ms, "fingerprint": fingerprint, "source_sha256": original_hash,
            "audio_stream_index": audio["index"],
            "chapters": [{"start_time":max(0,_number(ch.get('start_time'))-origin-start),
                          "end_time":min(duration,_number(ch.get('end_time'))-origin-start),
                          "title":str(ch.get('tags',{}).get('title',''))[:200]}
                         for ch in info.get('chapters',[]) if _number(ch.get('end_time'))>origin+start and _number(ch.get('start_time'))<origin+end],
            "media_info": {"original": _portable_probe(info), "normalized": _portable_probe(normalized), "audio_streams": [
                {"index": stream["index"], "codec": stream.get("codec_name"), "language": stream.get("tags", {}).get("language", ""),
                 "channels": stream.get("channels")}
                for stream in info["streams"] if stream.get("codec_type") == "audio"]},
        }
        manifest = {"fingerprint": fingerprint, "spec": spec, "result": result,
                    "output_hashes": {name: file_sha256(staged / name) for name in names}}
        (staged / "media-prepared.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _emit(progress, "media", 9, "媒体归一检查通过，正在保存…")
        _publish(staged, folder, [*names, "media-prepared.json"])
    return result


def ensure_asr_wav(audio_path: str | Path, folder: str | Path, progress: Progress | None = None) -> Path:
    """Materialize 16 kHz mono PCM only when an ASR consumer needs it."""
    source, folder = _local_file(audio_path), Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "asr-16k.wav"
    fingerprint = file_sha256(source)
    marker = folder / "asr-16k.sha256"
    if target.is_file() and marker.is_file() and marker.read_text().strip() == fingerprint:
        info = probe_media(target)
        if any(stream.get("sample_rate") == "16000" and stream.get("channels") == 1
               and stream.get("codec_name") == "pcm_s16le" for stream in info["streams"]):
            return target
    with tempfile.TemporaryDirectory(prefix=".wav-stage-", dir=folder) as temporary:
        staged = Path(temporary)
        run_ffmpeg(["-protocol_whitelist", "file,pipe", "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(staged / target.name)], staged, progress=progress,
                   duration_s=media_duration(probe_media(source)), message="正在准备 16kHz 单声道识别音频…", percent=10)
        os.replace(staged / target.name, target)
        marker.write_text(fingerprint, encoding="ascii")
    return target


prepare_asr_wav = ensure_asr_wav
