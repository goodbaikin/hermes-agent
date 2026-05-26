"""Tests for the non-stream stale-call detector context estimator.

Covers:
- ``estimate_request_context_tokens`` for Chat Completions, Responses API,
  bare lists, and mixed-shape dicts.
- ``AIAgent._compute_non_stream_stale_timeout`` with both legacy ``messages``
  list and full ``api_kwargs`` dicts.
- The May 2026 default-base change (300s -> 90s), the lowered generic
  context-tier ceilings, and the openai-codex first-event grace for normal
  GPT-5.x agent payloads that gives ChatGPT's Codex backend extra first-event
  grace without falling all the way back to the 1800s SDK default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _write_config(tmp_path: Path, body: str) -> None:
    hermes_home = tmp_path
    (hermes_home / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


# ── estimator ──────────────────────────────────────────────────────────────


def test_estimator_chat_completions_messages():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "user", "content": "x" * 400},
            {"role": "assistant", "content": "y" * 400},
        ],
    }
    # 800+ chars from messages -> ~200 tokens (char/4 estimate)
    assert estimate_request_context_tokens(payload) >= 200


def test_estimator_responses_api_input():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {
        "model": "gpt-5.5",
        "instructions": "i" * 1000,
        "input": "x" * 4000,
        "tools": [{"name": "t", "description": "d" * 200}],
    }
    # input(4000) + instructions(1000) + tools (~stringified) -> well over 1000 tokens
    tokens = estimate_request_context_tokens(payload)
    assert tokens >= 1200, f"Responses API estimator returned {tokens}"


def test_estimator_responses_api_long_session_triggers_tier():
    """A real long Codex session (large ``input``) should clear the 50k boundary."""
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {
        "model": "gpt-5.5",
        "input": "x" * 240_000,  # ~60k tokens (240k chars / 4)
        "instructions": "s" * 4000,
    }
    assert estimate_request_context_tokens(payload) > 50_000


def test_estimator_bare_list_back_compat():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    messages = [
        {"role": "user", "content": "x" * 800},
    ]
    assert estimate_request_context_tokens(messages) >= 200


def test_estimator_empty_inputs():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    assert estimate_request_context_tokens({}) == 0
    assert estimate_request_context_tokens([]) == 0
    assert estimate_request_context_tokens(None) == 0


def test_estimator_unknown_dict_fallback():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {"random_field": "z" * 400}
    assert estimate_request_context_tokens(payload) > 50


# ── Codex TTFB cutoff ──────────────────────────────────────────────────────


def test_codex_stream_ttfb_default_is_20s(monkeypatch):
    from types import SimpleNamespace

    from agent.chat_completion_helpers import _codex_stream_ttfb_timeout

    monkeypatch.delenv("HERMES_CODEX_STREAM_TTFB_TIMEOUT", raising=False)
    agent = SimpleNamespace(api_mode="codex_responses")
    assert _codex_stream_ttfb_timeout(agent, 90.0) == 20.0


def test_codex_stream_ttfb_honors_env_override(monkeypatch):
    from types import SimpleNamespace

    from agent.chat_completion_helpers import _codex_stream_ttfb_timeout

    monkeypatch.setenv("HERMES_CODEX_STREAM_TTFB_TIMEOUT", "45")
    agent = SimpleNamespace(api_mode="codex_responses")
    assert _codex_stream_ttfb_timeout(agent, 90.0) == 45.0


def test_codex_stream_ttfb_never_exceeds_stale_timeout(monkeypatch):
    from types import SimpleNamespace

    from agent.chat_completion_helpers import _codex_stream_ttfb_timeout

    monkeypatch.setenv("HERMES_CODEX_STREAM_TTFB_TIMEOUT", "45")
    agent = SimpleNamespace(api_mode="codex_responses")
    assert _codex_stream_ttfb_timeout(agent, 12.0) == 12.0


# ── default base + tier scaling ────────────────────────────────────────────


def test_default_base_is_90s(monkeypatch, tmp_path):
    """Default base stale timeout dropped from 300s to 90s (May 2026)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 90.0
    assert implicit is True


def test_short_codex_request_uses_base_only(monkeypatch, tmp_path):
    """Codex payload below 50k tokens -> default 90s base."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    payload = {"model": "gpt-5.5", "input": "hi", "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0


def test_normal_gpt5_codex_agent_payload_uses_bounded_first_event_grace(monkeypatch, tmp_path):
    """openai-codex GPT-5.x payload around 20k tokens gets bounded grace."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    # Matches the real API Server deployment's env override from the reported
    # log: the fix must not let this 120s base kill normal GPT-5.5 turns,
    # but also must not stretch every stall to the 1800s SDK default.
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "120")
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {"model": "gpt-5.5", "input": "x" * 76_800, "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) == 240.0


def test_long_codex_request_uses_bounded_first_event_grace(monkeypatch, tmp_path):
    """openai-codex payload >50k tokens gets bounded first-event grace."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    payload = {"model": "gpt-5.5", "input": "x" * 240_000, "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) == 480.0


def test_very_long_codex_request_uses_bounded_first_event_grace(monkeypatch, tmp_path):
    """openai-codex payload >100k tokens gets bounded first-event grace."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    payload = {"model": "gpt-5.5", "input": "x" * 500_000, "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) == 720.0


def test_codex_first_event_grace_caps_explicit_request_timeout(monkeypatch, tmp_path):
    """An explicit long request timeout does not stretch Codex stalls forever."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_API_TIMEOUT", "900")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {"model": "gpt-5.5", "input": "x" * 287_368, "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) == 480.0


def test_generic_very_long_request_uses_lowered_100k_tier(monkeypatch, tmp_path):
    """Non-ChatGPT providers keep the #31967 240s large-context ceiling."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(
        tmp_path,
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.5",
    )
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    payload = {"model": "gpt-5.5", "input": "x" * 500_000, "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) == 240.0


def test_generic_long_request_uses_lowered_50k_tier(monkeypatch, tmp_path):
    """Non-ChatGPT providers keep the #31967 150s mid-context ceiling."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(
        tmp_path,
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.5",
    )
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    payload = {"model": "gpt-5.5", "input": "x" * 240_000, "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) == 150.0


def test_chat_completions_long_messages_bumps_tier(monkeypatch, tmp_path):
    """Chat Completions estimator still works for the legacy messages path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(
        tmp_path,
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.4",
    )
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    payload = {
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "x" * 240_000}],
    }
    assert agent._compute_non_stream_stale_timeout(payload) >= 150.0


def test_explicit_user_config_overrides_default(monkeypatch, tmp_path):
    """If the user explicitly sets a stale_timeout, the new defaults don't apply."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, """\
providers:
  openai-codex:
    stale_timeout_seconds: 1800
""")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    import importlib
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)

    agent = _make_agent(tmp_path)
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 1800.0
