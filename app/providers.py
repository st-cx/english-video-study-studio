from __future__ import annotations

import json
import hashlib
import re
import time
from pathlib import Path
import urllib.error
import urllib.request

from .security import read_api_key
from .storage import get_settings


class ProviderError(RuntimeError):
    pass


def ask_deepseek(system: str, user: str, max_tokens: int = 8192, cache_dir: Path | None = None, settings_override: dict | None = None, api_key_override: str | None = None) -> dict:
    settings = settings_override if settings_override is not None else get_settings()
    key = api_key_override if api_key_override is not None else read_api_key()
    if not key:
        raise ProviderError("请先在设置中保存 DeepSeek API 密钥。")
    base = settings.get("base_url", "https://api.deepseek.com").rstrip("/")
    body = {
        "model": settings.get("model", "deepseek-chat"),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    cache = None
    if cache_dir is not None:
        signature = hashlib.sha256(json.dumps([base, body], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache = Path(cache_dir) / (signature + '.json')
        try:
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if isinstance(saved, dict):
                return saved
        except (OSError, ValueError):
            pass
    result = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            if result['choices'][0].get('finish_reason') == 'length':
                # Some compatible reasoning models spend the initial budget on
                # internal reasoning before emitting the required JSON.
                body['max_tokens'] = min(32768, body['max_tokens'] * 2)
                request.data = json.dumps(body,ensure_ascii=False).encode('utf-8')
                raise ValueError('模型输出被长度上限截断')
            content = result['choices'][0]['message']['content']
            if not content or not str(content).strip():
                raise ValueError('模型返回了空内容')
            content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip(), flags=re.I)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError('模型结果不是 JSON 对象')
            break
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                delay = min(20, max(1, int(exc.headers.get('Retry-After', '0'))) if exc.headers.get('Retry-After', '').isdigit() else 2 ** (attempt + 1))
                time.sleep(delay)
                continue
            # Gateway error bodies may echo request credentials; do not put them in logs.
            raise ProviderError(f"模型接口返回 HTTP {exc.code}，请检查网关、模型名称或配额。") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise ProviderError("模型接口连接失败或超时；已保留完成的步骤，可稍后重试。") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise ProviderError('模型连续返回空内容、截断内容或无效格式；已保留完成的步骤，可稍后重试。') from exc
    try:
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix('.tmp')
            temp.write_text(json.dumps(parsed, ensure_ascii=False), encoding='utf-8')
            temp.replace(cache)
        return parsed
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderError(f"DeepSeek 返回内容无法解析：{exc}") from exc
