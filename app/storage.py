from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def data_root() -> Path:
    explicit = os.environ.get('ENGLISH_VIDEO_DATA_DIR', '').strip()
    if explicit:
        path = Path(explicit).expanduser()
    else:
        # Project-local private data avoids Store-app filesystem virtualization
        # and prevents this edition from sharing another application's queue.
        path = Path(__file__).resolve().parents[1] / 'work' / 'profile'
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_root() -> Path:
    path = Path(__file__).resolve().parent.parent / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


DB_PATH = data_root() / "studio.sqlite3"
CONFIG_PATH = data_root() / "settings.json"
SECRET_PATH = data_root() / "deepseek.key.dpapi"
_CONNECT_LOCK=threading.RLock()


def connect() -> sqlite3.Connection:
    # Serialize the short schema/WAL initialization, not task transactions.
    # The OS lock also protects two independently launched studio processes.
    with _CONNECT_LOCK, _file_lock(DB_PATH.with_suffix('.init.lock'),30,'初始化任务数据库超时，请稍后重试。',interval=.02):
        return _connect_initialized()


def _connect_initialized() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        return _initialize_schema(db)
    except BaseException:
        db.close()
        raise


def _initialize_schema(db:sqlite3.Connection) -> sqlite3.Connection:
    db.execute('PRAGMA busy_timeout=30000')
    if db.execute('PRAGMA journal_mode').fetchone()[0].lower()!='wal':
        db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, episode_id TEXT, url TEXT NOT NULL, title TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL, stage TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
        message TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
        package_path TEXT NOT NULL DEFAULT '', input_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
    if "input_json" not in columns:
        db.execute("ALTER TABLE jobs ADD COLUMN input_json TEXT NOT NULL DEFAULT '{}'")
    db.execute("""CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY, value TEXT NOT NULL
    )""")
    db.commit()
    return db


def get_settings() -> dict[str, Any]:
    defaults = {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "daily_check_enabled": False,
        "last_checked_at": "",
        "last_episode_id": "",
    }
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        defaults.update({k: raw[k] for k in defaults if k in raw})
    except (OSError, ValueError):
        pass
    return defaults


def save_settings(settings: dict[str, Any]) -> None:
    allow = {"model", "base_url", "daily_check_enabled", "last_checked_at", "last_episode_id"}
    safe = {k: v for k, v in settings.items() if k in allow}
    # The scheduled producer and the UI write different fields. Merge while
    # holding the same OS lock so neither can restore a stale configuration.
    with _file_lock(CONFIG_PATH.with_suffix('.lock'), 30, '保存设置超时，请稍后重试。', interval=0.02):
        current = get_settings()
        current.update(safe)
        _write_json_atomic(CONFIG_PATH, current)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_episode(package: dict[str, Any], folder: Path, filename: str = "episode.json") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    _write_json_atomic(path, package)
    return path


def load_episode(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def _file_lock(path: Path, timeout: float, message: str, waiting=None, interval: float = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        acquired=False;started=time.monotonic()
        while not acquired:
            try:
                stream.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                acquired=True
            except (OSError,BlockingIOError):
                if time.monotonic()-started>timeout:raise TimeoutError(message)
                if waiting:waiting()
                time.sleep(interval)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_UN)


@contextmanager
def production_lock(waiting=None):
    """One producer across server threads and the Windows scheduled process."""
    with _file_lock(data_root()/'production.lock', 6*3600, '等待前一项制作超时，请检查运行记录。', waiting):
        yield
