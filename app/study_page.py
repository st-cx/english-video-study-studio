from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_study_page(package: dict, folder: Path) -> Path:
    """Embed the same React scene and timeline used by the MP4 renderer."""
    script_path = ROOT / 'dist/offline/lesson.js'
    css_path = ROOT / 'dist/offline/lesson.css'
    if not script_path.is_file() or not css_path.is_file():
        raise RuntimeError('学习页界面尚未构建，请先运行 npm run build 或 install.bat。')
    from .privacy import public_package
    raw = json.dumps(public_package(package), ensure_ascii=False, separators=(',', ':'))
    raw = raw.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    script = script_path.read_text(encoding='utf-8').replace('</script', '<\\/script')
    css = css_path.read_text(encoding='utf-8').replace('</style', '<\\/style')
    title = html.escape(package['episode'].get('title_zh') or package['episode']['title'])
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>{title} · 英语精读精听</title><style>html,body{{margin:0}}{css}</style></head><body><div id="study-root"></div><script>window.EPISODE_DATA={raw};</script><script>{script}</script></body></html>'''
    path = folder / 'study.html'
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=folder,prefix='.study-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);stream.write(page)
        os.replace(temporary,path)
    finally:
        if temporary:temporary.unlink(missing_ok=True)
    return path
