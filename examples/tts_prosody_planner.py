"""Conservative TTS prosody planning with an OpenAI-compatible small model."""

from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib import request

DEFAULT_BASE_URL = "http://agx.taild500c8.ts.net:11434/v1"
DEFAULT_MODEL = "caps-voice-edit-qwen3-4b:latest"
_failure_lock = threading.Lock()
_failure_until = 0.0
_no_proxy_opener = request.build_opener(request.ProxyHandler({}))


@dataclass(frozen=True)
class ProsodyPlan:
    text: str
    changed: bool
    source: str
    latency_ms: int = 0
    reason: str = ""
    error: str = ""

    def to_trace(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "source": self.source,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
            "error": self.error,
        }


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def prosody_enabled() -> bool:
    return _env_bool("QWEN_TTS_PROSODY_OPTIMIZER_ENABLED", True)


def _base_url() -> str:
    return os.getenv("QWEN_TTS_PROSODY_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _model() -> str:
    return os.getenv("QWEN_TTS_PROSODY_MODEL", DEFAULT_MODEL).strip()


def _timeout_s() -> float:
    return max(0.2, min(30.0, float(os.getenv("QWEN_TTS_PROSODY_TIMEOUT_S", "2.5"))))


def _max_tokens() -> int:
    return max(32, min(512, int(os.getenv("QWEN_TTS_PROSODY_MAX_TOKENS", "160"))))


def _max_input_chars() -> int:
    return max(20, int(os.getenv("QWEN_TTS_PROSODY_MAX_INPUT_CHARS", "1200")))


def _cooldown_s() -> float:
    return max(0.0, float(os.getenv("QWEN_TTS_PROSODY_FAILURE_COOLDOWN_S", "30")))


def prosody_config() -> dict[str, Any]:
    return {
        "enabled": prosody_enabled(),
        "base_url": _base_url(),
        "model": _model(),
        "timeout_s": _timeout_s(),
        "max_tokens": _max_tokens(),
        "max_input_chars": _max_input_chars(),
    }


def _source_name(base_url: str) -> str:
    lowered = (base_url or "").lower()
    if "agx" in lowered:
        return "agx_lm"
    if "127.0.0.1" in lowered or "localhost" in lowered:
        return "local_lm"
    return "openai_compatible_lm"


def _cooldown_active(now: float) -> bool:
    with _failure_lock:
        return now < _failure_until


def _mark_failure() -> None:
    global _failure_until
    cooldown = _cooldown_s()
    if cooldown <= 0:
        return
    with _failure_lock:
        _failure_until = time.monotonic() + cooldown


def _normalize_compare(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", (text or "").lower(), flags=re.UNICODE)


def _semantic_similarity(left: str, right: str) -> float:
    a = _normalize_compare(left)
    b = _normalize_compare(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def _protected_tokens(text: str) -> list[str]:
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9_./:-]*\d[A-Za-z0-9_./:-]*|[A-Z]{2,}|[A-Za-z]+[-_/][A-Za-z0-9_.:-]+)(?![A-Za-z0-9_])"
    )
    return sorted(set(match.group(0) for match in pattern.finditer(text or "")))


def _is_safe_rewrite(original: str, rewritten: str) -> tuple[bool, str]:
    candidate = (rewritten or "").strip()
    if not candidate:
        return False, "empty_output"
    if len(candidate) > max(len(original) * 2 + 20, 80):
        return False, "output_too_long"

    digit_runs = re.findall(r"\d+", original or "")
    for run in digit_runs:
        if run not in candidate:
            return False, "digit_changed"

    lowered = candidate.lower()
    for token in _protected_tokens(original):
        if token.lower() not in lowered:
            return False, "protected_token_changed"

    if _semantic_similarity(original, candidate) < float(os.getenv("QWEN_TTS_PROSODY_MIN_SIMILARITY", "0.82")):
        return False, "semantic_drift"
    return True, ""


def _extract_json_object(content: str) -> dict[str, Any] | None:
    value = (content or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE | re.DOTALL).strip()
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _prompt(text: str, lang_hint: str | None) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是中文TTS韵律助手。只输出JSON对象："
                '{"text":"...","changed":true,"reason":"..."}。'
                "只加标点、停顿或少量连接词；不改事实、数字、专名、英文、路径。"
            ),
        },
        {
            "role": "user",
            "content": f"语言：{lang_hint or 'Auto'}。改成稳定TTS朗读稿，保持含义不变：\n{text}",
        },
    ]


def _request_prosody(
    text: str,
    lang_hint: str | None,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
) -> ProsodyPlan:
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": _prompt(text, lang_hint),
    }
    req = request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with _no_proxy_opener.open(req, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    latency_ms = int((time.monotonic() - started) * 1000)
    content = str(payload["choices"][0]["message"].get("content") or "")
    decoded = _extract_json_object(content)
    if decoded is None:
        return ProsodyPlan(text, False, "fallback", latency_ms, error="invalid_json")

    rewritten = str(decoded.get("text") or "").strip()
    ok, error = _is_safe_rewrite(text, rewritten)
    if not ok:
        return ProsodyPlan(text, False, "fallback", latency_ms, reason=str(decoded.get("reason") or ""), error=error)
    return ProsodyPlan(rewritten, rewritten != text, _source_name(base_url), latency_ms, str(decoded.get("reason") or "").strip())


@lru_cache(maxsize=512)
def _optimize_cached(
    text: str,
    lang_hint: str,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    max_input_chars: int,
) -> ProsodyPlan:
    if len(text) > max_input_chars:
        return ProsodyPlan(text, False, "skipped", error="input_too_long")
    try:
        plan = _request_prosody(text, lang_hint or None, base_url, model, timeout_s, max_tokens)
    except Exception as exc:
        _mark_failure()
        return ProsodyPlan(text, False, "fallback", error=f"{type(exc).__name__}: {exc}")
    return plan


def optimize_prosody(text: str, lang_hint: str | None = None, trace_id: str | None = None) -> ProsodyPlan:
    content = (text or "").strip()
    if not content:
        return ProsodyPlan("", False, "skipped", error="empty_input")
    if not prosody_enabled():
        return ProsodyPlan(content, False, "disabled")
    if _cooldown_active(time.monotonic()):
        return ProsodyPlan(content, False, "fallback", error="failure_cooldown")
    return _optimize_cached(content, lang_hint or "", _base_url(), _model(), _timeout_s(), _max_tokens(), _max_input_chars())
