from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _browser_path() -> str:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    for name in ("chrome", "msedge", "chrome.exe", "msedge.exe"):
        found = shutil.which(name)
        if found:
            return found
    return ""


def _probe(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("找不到 ffprobe。请安装 FFmpeg 并将其加入 PATH。")
    result = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, timeout=45, check=True)
    return json.loads(result.stdout)


def audit_layout(package_path:Path,folder:Path,progress) -> dict:
    node=shutil.which('node')
    if not node:raise RuntimeError('未安装Node.js，请先运行install.bat。')
    progress('layout',93,'正在逐页检查字幕裁切和讲解卡重叠…')
    audit=subprocess.run([node,str(ROOT/'scripts/check-layout.mjs'),str(package_path.resolve()),str(folder.resolve())],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
    if audit.returncode:
        from .pipeline import ReviewRequired
        from .storage import save_episode
        package=json.loads(package_path.read_text(encoding='utf-8'))
        message='画面检查未通过，已暂停导出。请缩短过密讲解后重试。'
        package.setdefault('quality',{})['needs_review']=True
        package['quality'].setdefault('flags',[]).append(message)
        save_episode(package,folder)
        raise ReviewRequired(message,package)
    result=json.loads(audit.stdout.strip().splitlines()[-1])
    record=json.loads(Path(result['report']).read_text(encoding='utf-8'))
    record.pop('package_path',None)
    (folder/'layout-audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:result[k] for k in ('status','cases','checks')}


def export_video(package_path:Path,folder:Path,progress,start_ms:int=0,end_ms:int|None=None)->Path:
    from .storage import save_episode
    package=json.loads(package_path.read_text(encoding='utf-8'))
    duration=round(package['episode']['duration_s']*1000)
    end_ms=duration if end_ms is None else round(end_ms)
    start_ms=round(start_ms)
    if not 0<=start_ms<end_ms<=duration:raise ValueError('导出范围必须位于原视频时长内。')
    if any(c.get('status')!='ready' and c['start_ms']<end_ms and c['end_ms']>start_ms for c in package.get('chapters',[])):
        raise ValueError('所选范围包含待检查或未完成章节，请先检查。')
    quality=package.get('quality',{})
    structural=quality.get('structural_issues',[])
    if quality.get('structural_flags') and (not structural or any(x['start_ms']<end_ms and x['end_ms']>start_ms for x in structural)):
        raise ValueError('时间轴结构检查未通过。')
    return _render_video(package_path,folder,progress,start_ms,end_ms)


def render_video(package_path:Path,folder:Path,progress)->Path:
    return export_video(package_path,folder,progress)


def _render_video(package_path: Path, folder: Path, progress,start_ms:int,end_ms:int) -> Path:
    node = shutil.which("node")
    cli = ROOT / "node_modules" / "@remotion" / "cli" / "remotion-cli.js"
    if not node or not cli.is_file():
        raise RuntimeError("未安装 Node.js 或 Remotion。请先双击 install.bat。")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package.setdefault('processing',{})['layout_audit']=audit_layout(package_path,folder,progress)
    props_path = folder / "render-props.json"
    props_path.write_text(json.dumps({"episode": package,'startMs':start_ms,'endMs':end_ms}, ensure_ascii=False), encoding="utf-8")
    full=start_ms==0 and end_ms==round(package['episode']['duration_s']*1000)
    final_output=folder/'lesson.mp4' if full else folder/'exports'/f'clip-{start_ms}-{end_ms}.mp4'
    final_output.parent.mkdir(exist_ok=True)
    temp_output=final_output.with_name(final_output.stem+'.rendering.mp4')
    command = [node, str(cli), "render", str(ROOT / "src" / "remotion-entry.tsx"), "Lesson", str(temp_output),
               "--props", str(props_path), "--public-dir", str(folder), "--codec", "h264", "--crf", "20",
               "--x264-preset", "veryfast", "--pixel-format", "yuv420p", "--concurrency", "4" if (os.cpu_count() or 2)>=8 else "2", "--overwrite", "--log", "info"]
    browser = _browser_path()
    if browser:
        command.extend(["--browser-executable", browser])
    progress("render", 95, "正在渲染 1920×1080 逐词高亮视频…")
    try:
        log_folder = ROOT / 'work' / 'render-logs'
        log_folder.mkdir(parents=True, exist_ok=True)
        log_path = log_folder / (folder.name + '.log')
        started = time.monotonic()
        last_pct = -1
        timeout = max(600, int((end_ms-start_ms)/1000 * 45))
        with log_path.open('wb') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            while process.poll() is None:
                if time.monotonic()-started > timeout:
                    process.kill();process.wait()
                    raise TimeoutError('视频渲染超时；教学数据与原有成品已保留。')
                time.sleep(3)
                with log_path.open('rb') as tail:
                    tail.seek(max(0,log_path.stat().st_size-12000))
                    recent=tail.read().decode('utf-8',errors='replace')
                matches=re.findall(r'Rendered\s+(\d+)[ /]+(\d+)',recent)
                if matches:
                    done,total=map(int,matches[-1]);pct=round(done/max(1,total)*100)
                    if pct != last_pct:
                        progress('render',95+int(pct*.03),f'正在导出画面：{pct}%（{done} / {total} 帧）')
                        last_pct = pct
        if process.returncode:
            detail = log_path.read_text(encoding='utf-8',errors='replace')[-1800:] or 'Remotion 渲染失败'
            raise RuntimeError(detail)
        if not temp_output.is_file() or temp_output.stat().st_size < 100_000:
            raise RuntimeError("渲染器没有生成有效的 MP4 文件。")
        metadata = _probe(temp_output)
        video = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), None)
        audio = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "audio"), None)
        if not video or video.get("width") != 1920 or video.get("height") != 1080 or not audio:
            raise RuntimeError("MP4 验收失败：需要 1920×1080 画面和原声音轨。")
        if video.get('r_frame_rate') != '30/1' or abs(float(metadata['format']['duration'])-(end_ms-start_ms)/1000)>.15:
            raise RuntimeError('MP4 验收失败：帧率或时长与节目数据不一致。')
        # Prepare the companion page before publishing the new video. Keep a
        # rollback copy until both files have been replaced successfully.
        from .study_page import write_study_page
        with tempfile.TemporaryDirectory(prefix='.publish-', dir=folder) as staging:
            staged = Path(staging)
            ready_package = {**package, 'render': {**package['render'], 'status': 'ready' if full else package['render'].get('status','prepared')},'artifacts':{**package.get('artifacts',{}),'video':'ready' if full else package.get('artifacts',{}).get('video','not_requested')}}
            staged_page = write_study_page(ready_package, staged)
            old_video = staged / 'previous.mp4'
            old_page=staged/'previous-study.html'
            old_package=staged/'previous-episode.json'
            page_file=folder/'study.html'
            had_page=page_file.is_file()
            had_package=package_path.is_file()
            if had_page:shutil.copy2(page_file,old_page)
            if had_package:shutil.copy2(package_path,old_package)
            had_video = final_output.is_file()
            if had_video: os.replace(final_output, old_video)
            try:
                os.replace(temp_output, final_output)
                os.replace(staged_page, folder / 'study.html')
                from .storage import save_episode
                if full:ready_package['render']['video_file']='lesson.mp4'
                save_episode(ready_package,folder)
            except Exception:
                if had_video: os.replace(old_video, final_output)
                else: final_output.unlink(missing_ok=True)
                if had_page:os.replace(old_page,page_file)
                else:page_file.unlink(missing_ok=True)
                if had_package:os.replace(old_package,package_path)
                raise
    except Exception:
        if 'process' in locals() and process.poll() is None:
            process.kill();process.wait()
        temp_output.unlink(missing_ok=True)
        raise
    progress("render", 99, "视频已生成，正在整理离线学习页面…")
    return final_output
