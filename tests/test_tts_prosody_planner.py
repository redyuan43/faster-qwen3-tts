from examples import tts_prosody_planner as prosody


def setup_function():
    prosody._optimize_cached.cache_clear()
    prosody._failure_until = 0.0


def test_prosody_can_be_disabled(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_PROSODY_OPTIMIZER_ENABLED", "0")

    result = prosody.optimize_prosody("好的按照你说的来吧", lang_hint="Chinese")

    assert result.text == "好的按照你说的来吧"
    assert result.changed is False
    assert result.source == "disabled"


def test_safe_rewrite_allows_punctuation_only_change():
    ok, error = prosody._is_safe_rewrite("好的按照你说的来吧", "好的。按照你说的来吧。")

    assert ok is True
    assert error == ""


def test_safe_rewrite_rejects_digit_changes():
    ok, error = prosody._is_safe_rewrite("端口 8091 正常", "端口 8092 正常。")

    assert ok is False
    assert error == "digit_changed"


def test_safe_rewrite_rejects_protected_token_changes():
    ok, error = prosody._is_safe_rewrite("QWEN_TTS_PORT=8092 正常", "端口正常。")

    assert ok is False
    assert error == "digit_changed"


def test_extract_json_object_accepts_markdown_fence():
    decoded = prosody._extract_json_object('```json\n{"text":"你好。","changed":true,"reason":"pause"}\n```')

    assert decoded == {"text": "你好。", "changed": True, "reason": "pause"}


def test_source_name_marks_agx_endpoint():
    assert prosody._source_name("http://agx.taild500c8.ts.net:11434/v1") == "agx_lm"
    assert prosody._source_name("http://127.0.0.1:1234/v1") == "local_lm"


def test_optimize_prosody_uses_cache(monkeypatch):
    calls = []

    def fake_request(text, lang_hint, base_url, model, timeout_s, max_tokens):
        calls.append((text, lang_hint, base_url, model, timeout_s, max_tokens))
        return prosody.ProsodyPlan("好的。按照你说的来吧。", True, "local_lm", 12, "pause")

    monkeypatch.setattr(prosody, "_request_prosody", fake_request)

    first = prosody.optimize_prosody("好的按照你说的来吧", lang_hint="Chinese")
    second = prosody.optimize_prosody("好的按照你说的来吧", lang_hint="Chinese")

    assert first.text == "好的。按照你说的来吧。"
    assert second.text == first.text
    assert len(calls) == 1


def test_optimize_prosody_falls_back_and_sets_cooldown(monkeypatch):
    def raise_request(*_args):
        raise OSError("down")

    monkeypatch.setenv("QWEN_TTS_PROSODY_FAILURE_COOLDOWN_S", "30")
    monkeypatch.setattr(prosody, "_request_prosody", raise_request)

    first = prosody.optimize_prosody("第一句。", lang_hint="Chinese")
    second = prosody.optimize_prosody("第二句。", lang_hint="Chinese")

    assert first.source == "fallback"
    assert "OSError" in first.error
    assert second.source == "fallback"
    assert second.error == "failure_cooldown"


def test_optimize_prosody_skips_long_input(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_PROSODY_MAX_INPUT_CHARS", "20")

    result = prosody.optimize_prosody("这是一段超过二十个字符的长文本，用于跳过。", lang_hint="Chinese")

    assert result.source == "skipped"
    assert result.error == "input_too_long"
