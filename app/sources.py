"""Acquire public sources without browser credentials or remote FFmpeg input.

BBC uses its official assets; YouTube metadata uses yt-dlp with configuration
and cookies disabled; Bilibili uses bili-cli's underlying bilibili_api Video
API with an empty Credential. Every downloaded URL and redirect is resolved
to public IPs and connected through that validated IP, preventing DNS rebinding
between URL validation and the actual media request.
"""
from __future__ import annotations

import asyncio
import hashlib
import http.client
import ipaddress
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlsplit, urlunsplit

from .media import file_sha256, media_duration, probe_media, run_ffmpeg

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EnglishVideoStudy/0.5"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
MAX_MEDIA_BYTES = 12 * 1024 * 1024 * 1024
BBC_PAGE_HOSTS = {"www.bbc.co.uk", "bbc.co.uk", "www.bbc.com", "bbc.com"}
BBC_ASSET_HOSTS = BBC_PAGE_HOSTS | {"downloads.bbc.co.uk", "ichef.bbci.co.uk", "ichef.bbc.co.uk"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtube-nocookie.com"}
BILI_HOSTS = {"bilibili.com", "www.bilibili.com", "m.bilibili.com", "b23.tv"}


def _emit(progress, percent: int, message: str) -> None:
    if progress:
        try:
            progress("download", percent, message)
        except BaseException as exc:
            # Optional cover/subtitle handling must never swallow a caller's
            # cooperative pause, even if its class derives from RuntimeError.
            setattr(exc, "_cooperative_progress", True)
            raise


def _public_addresses(hostname: str, port: int) -> list[str]:
    if hostname.lower().rstrip(".") in {"localhost", "localhost.localdomain"} or hostname.lower().endswith((".localhost", ".local", ".internal")):
        raise ValueError("来源地址不能指向本机或内网。")
    try:
        records = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("来源域名无法解析，请检查网络或改用本地导入。") from exc
    addresses = []
    for record in records:
        address = record[4][0].split("%", 1)[0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError("来源返回了无效网络地址。") from exc
        if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
            parsed = parsed.ipv4_mapped
        if not parsed.is_global:
            raise ValueError("来源地址不能指向本机、内网或保留网络。")
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise ValueError("来源没有可用的公网地址。")
    return addresses


def checked_url(url: str, *, resolve: bool = True) -> str:
    value = str(url).strip()
    if not value or any(ord(char) < 32 for char in value) or "\\" in value:
        raise ValueError("来源链接格式不正确。")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("来源链接端口或地址无效。") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("只接受不含账号密码的公开 HTTP／HTTPS 来源链接。")
    hostname = parsed.hostname.rstrip(".").lower()
    if "%" in hostname:
        raise ValueError("来源主机名不能包含本地网络区域标识。")
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("来源主机名无效。") from exc
    effective_port = port or (443 if parsed.scheme.lower() == "https" else 80)
    if not 1 <= effective_port <= 65535:
        raise ValueError("来源端口无效。")
    if resolve:
        _public_addresses(hostname, effective_port)
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host + (f":{port}" if port else "")
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host, port, address, timeout=30):
        self._address = address
        super().__init__(host, port=port, timeout=timeout)

    def connect(self):
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, port, address, timeout=30):
        self._address = address
        super().__init__(host, port=port, timeout=timeout, context=ssl.create_default_context())

    def connect(self):
        raw = socket.create_connection((self._address, self.port), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


class _Response:
    def __init__(self, connection, response, url):
        self.connection, self.response, self.url = connection, response, url
        self.status, self.headers = response.status, response.headers

    def read(self, amount: int) -> bytes:
        # read1 returns currently available data instead of waiting to fill a
        # 256 KiB buffer on a slow CDN, so progress/pause remain responsive.
        return self.response.read1(amount)

    def close(self):
        self.response.close()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _open_url(url: str, *, method: str = "GET", headers: dict | None = None, timeout: int = 30) -> _Response:
    current = checked_url(url, resolve=False)
    for _ in range(6):
        parsed = urlsplit(current)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = _public_addresses(parsed.hostname, port)
        connection, response, last_error = None, None, None
        for address in addresses:
            connection_cls = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
            connection = connection_cls(parsed.hostname, port, address, timeout=timeout)
            try:
                target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                connection.request(method, target, headers={**HEADERS, **(headers or {})})
                response = connection.getresponse()
                break
            except (OSError, http.client.HTTPException) as exc:
                last_error = exc
                connection.close()
        if response is None:
            raise ConnectionError("无法连接公开来源，请检查网络或改用本地导入。") from last_error
        if response.status in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            connection.close()
            if not location:
                raise ValueError("来源返回了没有目标地址的重定向。")
            current = checked_url(urljoin(current, location), resolve=False)
            # The next iteration revalidates every resolved IP before sending.
            if response.status == 303 and method != "HEAD":
                method = "GET"
            continue
        return _Response(connection, response, current)
    raise ValueError("来源重定向次数过多，请使用最终公开媒体链接或本地导入。")


def _read_url(url: str, limit: int = 8 * 1024 * 1024, *, headers: dict | None = None) -> tuple[bytes, str, dict]:
    with _open_url(url, headers=headers) as response:
        _require_status(response)
        data = bytearray()
        while True:
            block = response.read(min(256 * 1024, limit + 1 - len(data)))
            if not block:
                break
            data.extend(block)
            if len(data) > limit:
                raise ValueError("来源页面／字幕超过大小限制。")
        return bytes(data), response.url, dict(response.headers)


def _require_status(response) -> None:
    if response.status in {401, 403, 410, 412}:
        raise ValueError(f"来源暂不可公开获取（HTTP {response.status}），可能需要登录、已失效或受限；可改用本地导入。")
    if response.status not in {200, 206}:
        raise ValueError(f"来源请求失败（HTTP {response.status}），请稍后重试或本地导入。")


def _sniff(data: bytes, content_type: str, kind: str) -> None:
    prefix = data.lstrip()[:100].lower()
    content_type = content_type.split(";", 1)[0].strip().lower()
    if content_type == "text/html" or prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        raise ValueError("来源返回的是网页／登录提示，不是媒体；请使用公开媒体链接或本地导入。")
    if kind == "media" and (content_type.startswith("text/") or content_type in {"application/json", "application/xml", "application/vnd.apple.mpegurl", "application/x-mpegurl"}
                            or prefix.startswith((b"#extm3u", b"<?xml", b"{\"error", b"{\"code"))):
        raise ValueError("来源返回文本或播放列表，不是可独立导入的媒体文件。")
    if kind == "pdf" and not data.startswith(b"%PDF-"):
        raise ValueError("文稿链接没有返回有效 PDF。")
    if kind == "image" and not (data.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")) or (data.startswith(b"RIFF") and data[8:12] == b"WEBP")):
        raise ValueError("封面链接没有返回有效图片。")


def _download(url: str, destination: Path, progress=None, *, kind: str = "media", headers: dict | None = None,
              limit: int = MAX_MEDIA_BYTES) -> Path:
    """Stream atomically and resume only when a server validator is available."""
    url = checked_url(url, resolve=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    marker = destination.with_name(destination.name + ".part.json")
    completed_marker = destination.with_name(destination.name + ".download.json")
    request_id = hashlib.sha256(url.encode()).hexdigest()
    reusable = False
    try:
        completed = json.loads(completed_marker.read_text(encoding="utf-8"))
        if completed.get("url_sha256") == request_id and completed.get("sha256") == file_sha256(destination):
            reusable = True
    except (OSError, ValueError, TypeError):
        pass
    if reusable:
        _emit(progress, 8, "复用已完整下载的素材文件。")
        return destination
    for attempt in range(3):
        state = {}
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        offset = partial.stat().st_size if partial.is_file() and state.get("url_sha256") == request_id and (state.get("etag") or state.get("last_modified")) else 0
        request_headers = dict(headers or {})
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
            request_headers["If-Range"] = state.get("etag") or state["last_modified"]
        _emit(progress, 1, f"正在{'继续' if offset else ''}下载素材…")
        try:
            with _open_url(url, headers=request_headers, timeout=40) as response:
                if response.status in {429, 500, 502, 503, 504}:
                    raise OSError("来源暂时繁忙。")
                if response.status == 416 and offset:
                    marker.unlink(missing_ok=True)
                    if attempt < 2:
                        continue
                _require_status(response)
                append = bool(offset and response.status == 206)
                if append:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset:
                        raise ValueError("来源返回了错误的断点范围，旧素材已保留。")
                    current_validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                    old_validator = state.get("etag") or state.get("last_modified")
                    if current_validator and current_validator != old_validator:
                        marker.unlink(missing_ok=True)
                        continue
                else:
                    offset = 0
                length_text = response.headers.get("Content-Length", "")
                content_length = int(length_text) if length_text.isdigit() else None
                if content_length is not None and offset + content_length > limit:
                    raise ValueError("素材超过下载大小限制，请裁剪后本地导入。")
                if response.headers.get("Content-Encoding", "identity").lower() not in {"identity", ""}:
                    raise ValueError("来源忽略了二进制下载要求，请使用本地导入。")
                state = {"url_sha256": request_id, "etag": response.headers.get("ETag", ""),
                         "last_modified": response.headers.get("Last-Modified", "")}
                marker.write_text(json.dumps(state), encoding="utf-8")
                received = 0
                with partial.open("ab" if append else "wb") as stream:
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        if offset + received == 0:
                            _sniff(block, response.headers.get("Content-Type", ""), kind)
                        received += len(block)
                        if offset + received > limit:
                            raise ValueError("素材超过下载大小限制，请裁剪后本地导入。")
                        stream.write(block)
                        stream.flush()
                        total = offset + content_length if content_length else 0
                        pct = min(9, 1 + int((offset + received) / total * 8)) if total else 2
                        _emit(progress, pct, f"素材已下载 {(offset + received) / 1024 / 1024:.1f} MB…")
                if content_length is not None and received != content_length:
                    raise OSError("下载连接提前结束。")
                if offset + received == 0:
                    raise ValueError("来源返回空文件。")
            if kind == "media":
                probe_media(partial)
            os.replace(partial, destination)
            marker.unlink(missing_ok=True)
            completed_marker.write_text(json.dumps({"url_sha256": request_id, "sha256": file_sha256(destination)}), encoding="utf-8")
            return destination
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            if getattr(exc, "_cooperative_progress", False):
                raise
            if attempt == 2:
                raise RuntimeError("下载未完成，断点与旧成品已保留；请重试或使用本地导入。") from exc
            _emit(progress, 2, "下载连接中断，正在尝试恢复…")
    raise RuntimeError("下载未完成，请重试或本地导入。")


def _kind(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname.lower()
    if host in BBC_PAGE_HOSTS and "/learningenglish/" in parsed.path:
        return "bbc"
    if host in YOUTUBE_HOSTS:
        return "youtube"
    if host in BILI_HOSTS:
        return "bilibili"
    return "media"


def _bbc_metadata(url: str) -> dict:
    from bs4 import BeautifulSoup
    from .bbc import extract_title
    data, final_url, _ = _read_url(url)
    if urlsplit(final_url).hostname not in BBC_PAGE_HOSTS:
        raise ValueError("BBC 页面跳转到了非官方来源。")
    soup = BeautifulSoup(data, "html.parser")
    audio, subtitle, cover = "", "", ""
    for anchor in soup.find_all("a", href=True):
        candidate = urljoin(final_url, anchor["href"])
        parsed = urlsplit(candidate)
        if parsed.hostname not in BBC_ASSET_HOSTS:
            continue
        if parsed.path.lower().endswith(".mp3"):
            audio = candidate
        elif parsed.path.lower().endswith(".pdf") and "transcript" in parsed.path.lower():
            subtitle = candidate
    if not audio:
        for source in soup.find_all(["audio", "source"]):
            candidate = urljoin(final_url, source.get("src") or source.get("data-src") or "")
            if urlsplit(candidate).hostname in BBC_ASSET_HOSTS and urlsplit(candidate).path.lower().endswith(".mp3"):
                audio = candidate
                break
    meta = soup.select_one("meta[property='og:image']")
    if meta:
        candidate = urljoin(final_url, meta.get("content", ""))
        if urlsplit(candidate).hostname in BBC_ASSET_HOSTS:
            cover = candidate
    if not audio:
        raise ValueError("BBC 页面没有公开 MP3；请换节目页或本地导入。")
    return {"title": extract_title(soup), "duration_s": None, "source_kind": "bbc", "source_url": final_url,
            "audio_url": audio, "subtitle_url": subtitle, "cover_url": cover}


def _youtube_target(url: str) -> str:
    """Reject redirect/generic-extractor URLs even on a trusted site host."""
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    identifier = ""
    if parsed.hostname == "youtu.be":
        identifier = parsed.path.strip("/")
    elif parsed.path.rstrip("/") == "/watch":
        identifier = query.get("v", [""])[0]
    else:
        match = re.fullmatch(r"/(?:shorts|live|embed)/([A-Za-z0-9_-]{11})/?", parsed.path)
        if match:
            identifier = match[1]
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", identifier):
        return "https://www.youtube.com/watch?" + urlencode({"v": identifier})
    if parsed.path.rstrip("/") == "/playlist" and re.fullmatch(r"[A-Za-z0-9_-]{10,100}", query.get("list", [""])[0]):
        return "https://www.youtube.com/playlist?" + urlencode({"list": query["list"][0]})
    raise ValueError("请提供具体 YouTube 视频或播放列表链接，不支持跳转／搜索网址。")


def _youtube_metadata(url: str, progress=None, *, flat: bool = False) -> dict:
    # Execute only the isolated project's installed module; never use a
    # neighbouring project's Python, browser cookies or yt-dlp configuration.
    command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-cache-dir", "--no-warnings",
               "--dump-single-json", "--skip-download", "--socket-timeout", "20", "--retries", "1",
               "--extractor-retries", "1", "--no-playlist", "--use-extractors", "youtube.*"]
    if shutil.which('node'):
        command += ['--js-runtimes', 'node']
    if flat:
        command += ["--flat-playlist", "--playlist-end", "50"]
    command += [_youtube_target(url)]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        started = time.monotonic()
        try:
            while process.poll() is None:
                _emit(progress, 1, "正在通过 yt-dlp 读取 YouTube 公开信息…")
                if time.monotonic() - started > 100:
                    raise TimeoutError("YouTube 信息获取超时，请重试或本地导入。")
                time.sleep(0.25)
            if process.returncode:
                raise ValueError("YouTube 暂无法公开获取，可能需要登录、验证或已失效；可使用本地文件导入。")
            stdout.seek(0)
            value = json.load(stdout)
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=20)
            raise
    if not isinstance(value, dict):
        raise ValueError("YouTube 返回了无效元数据。")
    return value


def _bili_id(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    match = re.search(r"/(BV[A-Za-z0-9]{10})(?:/|$)", parsed.path)
    if not match:
        raise ValueError("请使用 B站视频 BV 链接；合集／空间页请先选具体视频，或使用本地导入。")
    try:
        page = int(parse_qs(parsed.query).get("p", ["1"])[0]) - 1
    except ValueError as exc:
        raise ValueError("B站分 P 编号无效。") from exc
    if page < 0 or page > 10000:
        raise ValueError("B站分 P 编号超出范围。")
    return match[1], page


def _resolve_bili_short(url: str) -> str:
    if urlsplit(url).hostname == "b23.tv":
        with _open_url(url, method="HEAD") as response:
            _require_status(response)
            result = response.url
        if urlsplit(result).hostname not in BILI_HOSTS - {"b23.tv"}:
            raise ValueError("B站短链接没有指向公开 BV 视频。")
        return result
    return url


async def _bili_fetch(url: str, include_streams: bool = False, progress=None) -> dict:
    try:
        from bilibili_api import Credential, video
    except ImportError as exc:
        raise RuntimeError("B站支持需要 bilibili-api-python（bili-cli 的底层库），请运行项目安装程序。") from exc
    bvid, page = _bili_id(url)
    item = video.Video(bvid=bvid, credential=Credential())
    try:
        info = await asyncio.wait_for(item.get_info(), timeout=40)
        _emit(progress, 1, "已读取 B站公开视频信息…")
        pages = info.get("pages") or []
        if pages and page >= len(pages):
            raise ValueError("B站视频没有所选分 P。")
        selected = pages[page] if pages else {"duration": info.get("duration", 0), "cid": info.get("cid")}
        result = {"title": info.get("title") or bvid, "duration_s": selected.get("duration") or info.get("duration"),
                  "source_kind": "bilibili", "source_url": f"https://www.bilibili.com/video/{bvid}/" + (f"?p={page + 1}" if page else ""),
                  "cover_url": info.get("pic", ""), "page_index": page,
                  "entries": [{"title": entry.get("part") or info.get("title"), "duration_s": entry.get("duration"),
                               "source_url": f"https://www.bilibili.com/video/{bvid}/?p={index + 1}"}
                              for index, entry in enumerate(pages)] if len(pages) > 1 else []}
        if include_streams:
            result["download_data"] = await asyncio.wait_for(item.get_download_url(page_index=page), timeout=40)
            _emit(progress, 1, "已读取 B站公开播放流…")
            try:
                result["subtitle_info"] = await asyncio.wait_for(item.get_subtitle(cid=selected.get("cid")), timeout=20)
            except Exception:
                result["subtitle_info"] = None  # Missing optional subtitles is normal.
        return result
    except (ValueError, asyncio.TimeoutError):
        raise
    except Exception as exc:
        if getattr(exc, "_cooperative_progress", False):
            raise
        if "尚未安装第三方请求库" in str(exc):
            raise RuntimeError("B站接口缺少 HTTP 后端，请运行项目安装程序安装 httpx；这不是平台限制响应。") from exc
        raise ValueError("B站视频暂不可公开获取，可能已失效、需要登录或受限；不会读取浏览器凭据，可改用本地导入。") from exc


def _run_bili(url: str, *, include_streams=False, progress=None) -> dict:
    try:
        return asyncio.run(_bili_fetch(url, include_streams=include_streams, progress=progress))
    except asyncio.TimeoutError as exc:
        raise RuntimeError("B站公开接口超时，请重试或本地导入。") from exc


def inspect_source(url: str) -> dict:
    canonical = checked_url(url)
    kind = _kind(canonical)
    if kind == "bbc":
        value = _bbc_metadata(canonical)
        return {key: value[key] for key in ("title", "duration_s", "source_kind", "source_url")}
    if kind == "bilibili":
        value = _run_bili(_resolve_bili_short(canonical))
        return {key: value[key] for key in ("title", "duration_s", "source_kind", "source_url", "entries")}
    if kind == "youtube":
        value = _youtube_metadata(canonical, flat=True)
        result = {"title": value.get("title") or "YouTube 视频", "duration_s": value.get("duration"),
                  "source_kind": "youtube", "source_url": canonical}
        if value.get("entries"):
            result["entries"] = [{"title": entry.get("title") or "YouTube 视频", "duration_s": entry.get("duration"),
                                  "source_url": entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry.get('id', '')}"}
                                 for entry in value["entries"] if isinstance(entry, dict)]
        return result
    with _open_url(canonical, headers={"Range": "bytes=0-1023"}) as response:
        _require_status(response)
        data = response.read(1024)
        if not data:
            raise ValueError("来源返回了空媒体。")
        _sniff(data, response.headers.get("Content-Type", ""), "media")
        title = Path(unquote(urlsplit(response.url).path)).stem or "公开媒体"
        total = response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
        size = int(total) if total.isdigit() else None
        return {"title": title[:200], "duration_s": None, "source_kind": "media", "source_url": canonical, "size_bytes": size}


def _merge_streams(video_path: Path, audio_path: Path | None, destination: Path, progress=None) -> Path:
    arguments = ["-protocol_whitelist", "file,pipe", "-i", str(video_path)]
    if audio_path:
        arguments += ["-protocol_whitelist", "file,pipe", "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"]
    else:
        arguments += ["-map", "0"]
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        run_ffmpeg([*arguments, "-c", "copy", str(staged)], Path(temporary), progress=progress,
                   duration_s=media_duration(probe_media(video_path)), message="正在合并公开音视频流…", percent=9)
        info = probe_media(staged)
        if not any(stream.get("codec_type") == "audio" for stream in info["streams"]):
            raise ValueError("来源未提供可获取的音轨，请使用本地导入。")
        os.replace(staged, destination)
    return destination


def _optional_download(url: str, path: Path, progress, kind: str, headers: dict | None = None) -> str:
    if not url:
        return ""
    try:
        return str(_download(url, path, progress, kind=kind, headers=headers, limit=20 * 1024 * 1024))
    except (ValueError, RuntimeError, OSError) as exc:
        if getattr(exc, "_cooperative_progress", False):
            raise
        return ""


def _acquire_bbc(url: str, folder: Path, progress) -> dict:
    value = _bbc_metadata(url)
    media = _download(value["audio_url"], folder / "original.mp3", progress)
    subtitle = _optional_download(value["subtitle_url"], folder / "transcript.pdf", progress, "pdf")
    cover = _optional_download(value["cover_url"], folder / "source-cover.jpg", progress, "image")
    result = {key: value[key] for key in ("title", "source_kind", "source_url")}
    result.update({"media_path": str(media), "subtitle_path": subtitle, "cover_path": cover,
                   "transcript_kind": "official_pdf" if subtitle else "none"})
    if subtitle:
        import fitz
        with fitz.open(subtitle) as document:
            result["transcript"] = "\n".join(page.get_text("text") for page in document)
    return result


def _youtube_streams(metadata: dict, max_video_height: int | None = None) -> tuple[dict, dict | None]:
    formats = [item for item in metadata.get("formats", []) if item.get("url") and not item.get("has_drm")
               and item.get("protocol", "https") in {"http", "https"} and item.get("ext") not in {"mhtml", "json"}]
    audios = [item for item in formats if item.get("acodec") not in {None, "none"} and item.get("vcodec") in {None, "none"}]
    videos = [item for item in formats if item.get("vcodec") not in {None, "none"}
              and (max_video_height is None or 0 < (item.get("height") or 0) <= max_video_height)]
    if videos:
        video = max(videos, key=lambda item: (item.get("height") or 0, item.get("width") or 0,
                                              bool(audios) and item.get("acodec") in {None, "none"}, item.get("tbr") or 0))
        if video.get("acodec") not in {None, "none"} and not audios:
            return video, None
        if not audios:
            raise ValueError("YouTube 没有可公开获取的音轨，请本地导入。")
        audio = max(audios, key=lambda item: (str(item.get("language", "")).startswith("en"), item.get("abr") or item.get("tbr") or 0))
        return video, audio
    if max_video_height is not None and any(item.get("vcodec") not in {None, "none"} for item in formats):
        raise ValueError("公开来源没有符合所选清晰度的画面流，请换清晰度或本地导入。")
    audios = [item for item in formats if item.get("acodec") not in {None, "none"}]
    if audios:
        return max(audios, key=lambda item: item.get("abr") or item.get("tbr") or 0), None
    raise ValueError("YouTube 没有可直接下载的公开媒体流，可能为受限／直播素材；请本地导入。")


def _acquire_youtube(url: str, folder: Path, progress, max_video_height: int | None = None) -> dict:
    value = _youtube_metadata(url, progress)
    if value.get("is_live") or value.get("_type") in {"playlist", "multi_video"}:
        raise ValueError("请先选择具体、已结束的 YouTube 视频；合集可在检查来源后逐个添加。")
    first, second = _youtube_streams(value, max_video_height=max_video_height)
    first_path = _download(first["url"], folder / ("youtube-video." + re.sub(r"[^a-z0-9]", "", first.get("ext") or "bin")), progress)
    if second:
        audio_path = _download(second["url"], folder / ("youtube-audio." + re.sub(r"[^a-z0-9]", "", second.get("ext") or "bin")), progress)
        media = _merge_streams(first_path, audio_path, folder / "original.mkv", progress)
    else:
        media = first_path
    subtitle, transcript_kind = "", "none"
    for category, kind in (("subtitles", "provided_subtitle"), ("automatic_captions", "automatic_subtitle")):
        candidates = value.get(category) or {}
        languages = sorted((language for language in candidates if language.lower().startswith("en")), key=lambda language: (language != "en", language))
        if languages:
            options = [item for item in candidates[languages[0]] if item.get("ext") in {"vtt", "srt"} and item.get("url")]
            if options:
                chosen = next((item for item in options if item["ext"] == "vtt"), options[0])
                subtitle = _optional_download(chosen["url"], folder / ("source-subtitles." + chosen["ext"]), progress, "subtitle")
                if subtitle:
                    transcript_kind = kind
                    break
    cover = _optional_download(value.get("thumbnail", ""), folder / "source-cover.jpg", progress, "image")
    return {"media_path": str(media), "title": value.get("title") or "YouTube 视频", "source_kind": "youtube", "source_url": url,
            "subtitle_path": subtitle, "cover_path": cover, "transcript_kind": transcript_kind,
            "source_metadata": {key: str(value.get(key) or '')[:500] for key in ('title', 'uploader', 'channel', 'license', 'upload_date')}}


def _acquire_bili(url: str, folder: Path, progress) -> dict:
    value = _run_bili(_resolve_bili_short(url), include_streams=True, progress=progress)
    from bilibili_api import video
    data = value["download_data"]
    headers = {"Referer": value["source_url"], "User-Agent": USER_AGENT}
    detector = video.VideoDownloadURLDataDetecter(data)
    streams = detector.detect_best_streams(video_max_quality=video.VideoQuality._1080P, no_dolby_audio=True,
                                          no_dolby_video=True, no_hdr=True, no_hires=True)
    if detector.check_flv_mp4_stream():
        parts = data.get("durl") or []
        if len(parts) > 1:
            paths = [_download(part["url"], folder / f"bili-part-{index:04d}.flv", progress, headers=headers) for index, part in enumerate(parts)]
            with tempfile.TemporaryDirectory(prefix=".concat-", dir=folder) as temporary:
                concat = Path(temporary) / "streams.txt"
                concat.write_text("\n".join("file '" + str(path.resolve()).replace("\\", "/").replace("'", "'\\''") + "'" for path in paths), encoding="utf-8")
                staged = Path(temporary) / "original.mkv"
                run_ffmpeg(["-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(staged)], Path(temporary),
                           progress=progress, duration_s=float(value.get("duration_s") or 0), message="正在合并 B站公开分段…", percent=9)
                probe_media(staged)
                media = folder / "original.mkv"
                os.replace(staged, media)
        elif streams and streams[0] is not None:
            media = _download(streams[0].url, folder / "original.mp4", progress, headers=headers)
        else:
            raise ValueError("B站没有可公开下载的媒体，请本地导入。")
    else:
        visual = next((item for item in streams if item is not None and hasattr(item, "video_quality")), None)
        audible = next((item for item in streams if item is not None and hasattr(item, "audio_quality")), None)
        if visual is None or audible is None:
            raise ValueError("B站没有同时提供公开画面和音轨，可能需要登录或会员；请本地导入。")
        video_path = _download(visual.url, folder / "bili-video.m4s", progress, headers=headers)
        audio_path = _download(audible.url, folder / "bili-audio.m4s", progress, headers=headers)
        media = _merge_streams(video_path, audio_path, folder / "original.mkv", progress)
    subtitle = ""
    subtitles = (value.get("subtitle_info") or {}).get("subtitles") or []
    english = next((item for item in subtitles if str(item.get("lan", "")).startswith("en")), None)
    if english and english.get("subtitle_url"):
        candidate = english["subtitle_url"]
        candidate = "https:" + candidate if candidate.startswith("//") else candidate
        try:
            raw, _, _ = _read_url(candidate, headers=headers)
            body = json.loads(raw).get("body", [])
            def stamp(seconds):
                milliseconds = round(float(seconds) * 1000)
                hours, milliseconds = divmod(milliseconds, 3600000)
                minutes, milliseconds = divmod(milliseconds, 60000)
                seconds, milliseconds = divmod(milliseconds, 1000)
                return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
            lines = ["WEBVTT", ""]
            for cue in body:
                if cue.get("content") and float(cue["to"]) > float(cue["from"]):
                    lines.extend([f"{stamp(cue['from'])} --> {stamp(cue['to'])}", str(cue["content"]), ""])
            if len(lines) > 2:
                target = folder / "source-subtitles.vtt"
                target.write_text("\n".join(lines), encoding="utf-8")
                subtitle = str(target)
        except (ValueError, OSError, RuntimeError, KeyError):
            subtitle = ""
    cover = _optional_download(value.get("cover_url", ""), folder / "source-cover.jpg", progress, "image", headers)
    return {"media_path": str(media), "title": value["title"], "source_kind": "bilibili", "source_url": value["source_url"],
            "subtitle_path": subtitle, "cover_path": cover, "transcript_kind": "provided_subtitle" if subtitle else "none"}


def _cached_source(folder: Path, request_id: str) -> dict | None:
    try:
        cached = json.loads((folder / "source-acquired.json").read_text(encoding="utf-8"))
        if cached["request_id"] != request_id:
            return None
        result = cached["result"]
        for key, expected in cached["hashes"].items():
            filename = result[key]
            if Path(filename).name != filename or file_sha256(folder / filename) != expected:
                return None
            result[key] = str(folder / filename)
        return result
    except (OSError, ValueError, TypeError, KeyError):
        return None


def acquire_source(url: str, folder: str | Path, progress=None, *, max_video_height: int | None = None) -> dict:
    canonical, folder = checked_url(url, resolve=False), Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if max_video_height is not None and (isinstance(max_video_height, bool) or not isinstance(max_video_height, int) or max_video_height < 144):
        raise ValueError("下载清晰度上限必须是至少 144 的整数高度。")
    request_identity = canonical if max_video_height is None else json.dumps([canonical, max_video_height])
    request_id = hashlib.sha256(request_identity.encode()).hexdigest()
    cached = _cached_source(folder, request_id)
    if cached:
        _emit(progress, 9, "复用已完整下载并核对的来源素材。")
        return cached
    kind = _kind(canonical)
    _emit(progress, 1, "正在读取公开来源；不读取浏览器登录凭据…")
    if kind == "bbc":
        result = _acquire_bbc(canonical, folder, progress)
    elif kind == "youtube":
        result = _acquire_youtube(canonical, folder, progress, max_video_height=max_video_height)
    elif kind == "bilibili":
        result = _acquire_bili(canonical, folder, progress)
    else:
        suffix = Path(urlsplit(canonical).path).suffix.lower()
        if suffix not in {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".webm", ".mkv", ".mov", ".ogg", ".opus", ".aac", ".avi", ".ts"}:
            suffix = ".media"
        media = _download(canonical, folder / ("original" + suffix), progress)
        result = {"media_path": str(media), "title": Path(unquote(urlsplit(canonical).path)).stem[:200] or "公开媒体",
                  "source_kind": "media", "source_url": canonical, "subtitle_path": "", "cover_path": "", "transcript_kind": "none"}
    info = probe_media(result["media_path"])
    if not any(stream.get("codec_type") == "audio" for stream in info["streams"]):
        raise ValueError("公开素材没有音轨，无法制作英语听力学习包。")
    result["duration_s"] = media_duration(info)
    serializable = dict(result)
    hashes = {}
    for key in ("media_path", "subtitle_path", "cover_path"):
        if result.get(key):
            item = Path(result[key]).resolve()
            if item.parent != folder:
                raise ValueError("下载产物不在当前任务目录。")
            hashes[key] = file_sha256(item)
            serializable[key] = item.name
    target = folder / "source-acquired.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"request_id": request_id, "result": serializable, "hashes": hashes}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return result
