from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import fitz
import requests
from bs4 import BeautifulSoup


BBC_HOSTS = {"www.bbc.co.uk", "bbc.co.uk", "www.bbc.com", "bbc.com", "downloads.bbc.co.uk"}
BBC_MEDIA_HOSTS = BBC_HOSTS | {"ichef.bbci.co.uk", "ichef.bbc.co.uk"}
LATEST_URL = "https://www.bbc.co.uk/learningenglish/features/6-minute-english"
HEADERS = {"User-Agent": "BBCEnglishStudyStudio/1.0 (personal local learning tool)"}


def checked_url(url: str, *, media: bool = False) -> str:
    value = url.strip()
    if not value.startswith(("https://", "http://")):
        value = "https://" + value
    parsed = urlparse(value)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in (BBC_MEDIA_HOSTS if media else BBC_HOSTS):
        raise ValueError("只接受 BBC 官方 HTTPS 页面或素材链接。")
    return value


def _get(url: str, **kwargs):
    result = requests.get(checked_url(url), headers=HEADERS, timeout=40, **kwargs)
    result.raise_for_status()
    return result


def list_recent_episodes() -> list[dict]:
    soup = BeautifulSoup(_get(LATEST_URL).text, "html.parser")
    seen: set[str] = set()
    result = []
    for anchor in soup.select('a[href*="6-minute-english"]'):
        href = urljoin(LATEST_URL, anchor.get("href", ""))
        if not re.search(r"/ep-\d+", href):
            continue
        href = checked_url(href)
        if href in seen:
            continue
        seen.add(href)
        title = anchor.get_text(" ", strip=True)
        result.append({"url": href, "title": title})
    return sorted(result, key=lambda item: re.search(r'ep-(\d+)', item['url']).group(1), reverse=True)


def fetch_episode(url: str, folder: Path) -> dict:
    canonical = checked_url(url)
    response = _get(canonical)
    soup = BeautifulSoup(response.text, "html.parser")
    title = extract_title(soup)
    title = re.sub(r"\s*[-|]\s*BBC Learning English.*$", "", title, flags=re.I).strip() or "BBC 6 Minute English"
    audio_url = ""
    transcript_url = ""
    for anchor in soup.find_all("a", href=True):
        href = urljoin(canonical, anchor["href"])
        label = anchor.get_text(" ", strip=True).lower()
        path = urlparse(href).path.lower()
        if path.endswith(".mp3") and ("download" in path or "audio" in label):
            audio_url = href
        if path.endswith(".pdf") and "transcript" in path:
            transcript_url = href
    if not audio_url:
        for source in soup.find_all(["audio", "source"]):
            candidate = source.get("src") or source.get("data-src") or ""
            if candidate.lower().endswith(".mp3"):
                audio_url = urljoin(canonical, candidate)
                break
    if not audio_url or not transcript_url:
        raise ValueError("BBC 页面没有同时提供音频和文稿 PDF。请改用有官方素材的节目页面。")
    audio_url = checked_url(audio_url)
    transcript_url = checked_url(transcript_url)
    folder.mkdir(parents=True, exist_ok=True)
    audio_path = folder / "audio.mp3"
    transcript_path = folder / "transcript.pdf"
    _download(audio_url, audio_path)
    _download(transcript_url, transcript_path)
    with fitz.open(transcript_path) as transcript_doc:
        transcript = "\n".join(page.get_text("text") for page in transcript_doc).strip()
    image = soup.select_one('meta[property="og:image"]')
    cover_url = checked_url(urljoin(canonical, image.get("content", "")), media=True) if image and image.get("content", "").startswith("https://") and urlparse(urljoin(canonical, image["content"])).hostname in BBC_MEDIA_HOSTS else ""
    cover_path = folder / "cover.jpg"
    if cover_url:
        try:
            _download(cover_url, cover_path, limit=8_000_000)
        except Exception:
            cover_path = None
    else:
        cover_path = None
    episode_id = re.search(r"ep-(\d+)", canonical)
    slug = episode_id.group(1) if episode_id else hashlib.sha1(canonical.encode()).hexdigest()[:12]
    date = re.search(r"Episode\s+\S+\s*/\s*([^\n]+)", transcript)
    source_date = date.group(1).strip() if date else ''
    if not source_date and re.fullmatch(r'\d{6}', slug):
        source_date = f'20{slug[:2]}-{slug[2:4]}-{slug[4:]}'
    return {
        "id": slug,
        "title": title,
        "source_url": canonical,
        "audio_url": audio_url,
        "transcript_url": transcript_url,
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path),
        "transcript": transcript,
        "cover_path": str(cover_path) if cover_path else "",
        "duration_s": 0,
        "source_date": source_date,
    }


def extract_title(soup: BeautifulSoup) -> str:
    meta=soup.select_one("meta[property='og:title']")
    title=str(meta.get('content','')).strip() if meta else ''
    title=re.sub(r'^BBC Learning English\s*[-|:]\s*6 Minute English\s*[/|:-]\s*','',title,flags=re.I)
    if title and title.lower() not in {'learning english','bbc learning english','6 minute english'}:
        return title
    for heading in soup.select('h1,h2,h3'):
        text=heading.get_text(' ',strip=True)
        if text and text.lower() not in {'learning english','bbc learning english','6 minute english','accessibility links'}:
            return text
    return 'BBC 6 Minute English'


def _download(url: str, destination: Path, limit: int = 80_000_000) -> None:
    parsed = urlparse(checked_url(url, media=True))
    partial = destination.with_name(destination.name + '.part')
    with requests.get(parsed.geturl(), headers=HEADERS, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        checked_url(response.url, media=True)
        size = 0
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1024 * 512):
                if not chunk:
                    continue
                size += len(chunk)
                if size > limit:
                    raise ValueError("BBC 素材文件超过大小限制。")
                stream.write(chunk)
        expected = response.headers.get('Content-Length', '')
        if expected.isdigit() and not response.headers.get('Content-Encoding') and size != int(expected):
            raise ValueError('BBC 素材下载不完整，已保留旧文件，请重试。')
    partial.replace(destination)


def episode_fingerprint(episode: dict, audio_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(episode["source_url"].encode())
    with audio_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
