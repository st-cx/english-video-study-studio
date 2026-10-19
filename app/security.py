from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path

from .storage import SECRET_PATH


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _protect(data: bytes, decrypt: bool = False) -> bytes:
    if not data:
        return b""
    source = ctypes.create_string_buffer(data)
    source_blob = _Blob(len(data), ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte)))
    target_blob = _Blob()
    crypt = ctypes.windll.crypt32
    if decrypt:
        ok = crypt.CryptUnprotectData(ctypes.byref(source_blob), None, None, None, None, 0, ctypes.byref(target_blob))
    else:
        ok = crypt.CryptProtectData(ctypes.byref(source_blob), "English Video Study Studio", None, None, None, 0, ctypes.byref(target_blob))
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(target_blob.pbData, target_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(target_blob.pbData)


def has_api_key() -> bool:
    return SECRET_PATH.is_file() and SECRET_PATH.stat().st_size > 0


def save_api_key(value: str) -> None:
    text = value.strip()
    if not text:
        raise ValueError("请输入 DeepSeek API 密钥。")
    encrypted = _protect(text.encode("utf-8"))
    tmp = SECRET_PATH.with_suffix(".tmp")
    tmp.write_bytes(encrypted)
    tmp.replace(SECRET_PATH)


def read_api_key() -> str:
    if not has_api_key():
        return ""
    return _protect(SECRET_PATH.read_bytes(), decrypt=True).decode("utf-8")


def delete_api_key() -> None:
    Path(SECRET_PATH).unlink(missing_ok=True)
