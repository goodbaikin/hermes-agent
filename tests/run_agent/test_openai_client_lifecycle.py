import sys
import threading
import time
import types
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


class FakeRequestClient:
    def __init__(self, responder, stream_factory=None):
        self._responder = responder
        self._stream_factory = stream_factory
        self._client = SimpleNamespace(is_closed=False)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.responses = SimpleNamespace(stream=self._stream)
        self.close_calls = 0

    def _create(self, **kwargs):
        return self._responder(**kwargs)

    def _stream(self, **kwargs):
        if self._stream_factory is None:
            raise AssertionError("responses.stream not configured")
        return self._stream_factory(**kwargs)

    def close(self):
        self.close_calls += 1
        self._client.is_closed = True


class FakeSharedClient(FakeRequestClient):
    pass


class SlowCodexStream:
    def __init__(self, *, event_count=28, delay=0.1, final_response=None):
        self.event_count = event_count
        self.delay = delay
        self.final_response = final_response or {"ok": "codex-stream"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        for _ in range(self.event_count):
            time.sleep(self.delay)
            yield SimpleNamespace(type="response.in_progress")

    def get_final_response(self):
        return self.final_response


class OpenAIFactory:
    def __init__(self, clients):
        self._clients = list(clients)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self._clients:
            raise AssertionError("OpenAI factory exhausted")
        return self._clients.pop(0)


def _build_agent(shared_client=None):
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.api_mode = "chat_completions"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.model = "gpt-5-codex"
    agent.log_prefix = ""
    agent.quiet_mode = True
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._client_lock = threading.RLock()
    agent._client_kwargs = {"api_key": "***", "base_url": agent.base_url}
    agent.client = shared_client or FakeSharedClient(lambda **kwargs: {"shared": True})
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent.reasoning_callback = None
    agent.status_callback = None
    return agent


def _connection_error():
    return APIConnectionError(
        message="Connection error.",
        request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
    )


def test_retry_after_api_connection_error_recreates_request_client(monkeypatch):
    first_request = FakeRequestClient(lambda **kwargs: (_ for _ in ()).throw(_connection_error()))
    second_request = FakeRequestClient(lambda **kwargs: {"ok": True})
    factory = OpenAIFactory([first_request, second_request])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent()

    with pytest.raises(APIConnectionError):
        agent._interruptible_api_call({"model": agent.model, "messages": []})

    result = agent._interruptible_api_call({"model": agent.model, "messages": []})

    assert result == {"ok": True}
    assert len(factory.calls) == 2
    assert first_request.close_calls >= 1
    assert second_request.close_calls >= 1


def test_stale_non_stream_close_is_single_owner(monkeypatch):
    def slow_responder(**kwargs):
        time.sleep(0.1)
        raise _connection_error()

    request_client = FakeRequestClient(slow_responder)
    factory = OpenAIFactory([request_client])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent()
    agent._compute_non_stream_stale_timeout = lambda api_payload: 0.01

    with pytest.raises(APIConnectionError):
        agent._interruptible_api_call({"model": agent.model, "messages": []})

    assert request_client.close_calls == 1


def test_codex_stream_activity_prevents_outer_stale_timeout(monkeypatch):
    final_response = {"ok": "codex-stream-completed"}
    request_client = FakeRequestClient(
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("chat path used")),
        stream_factory=lambda **kwargs: SlowCodexStream(final_response=final_response),
    )
    factory = OpenAIFactory([request_client])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent()
    agent.api_mode = "codex_responses"
    agent._compute_non_stream_stale_timeout = lambda api_payload: 0.12

    result = agent._interruptible_api_call({"model": agent.model, "input": [], "store": False})

    assert result == final_response
    assert request_client.close_calls == 1


def test_closed_shared_client_is_recreated_before_request(monkeypatch):
    stale_shared = FakeSharedClient(lambda **kwargs: (_ for _ in ()).throw(AssertionError("stale shared client used")))
    stale_shared._client.is_closed = True

    replacement_shared = FakeSharedClient(lambda **kwargs: {"replacement": True})
    request_client = FakeRequestClient(lambda **kwargs: {"ok": "fresh-request-client"})
    factory = OpenAIFactory([replacement_shared, request_client])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent(shared_client=stale_shared)
    result = agent._interruptible_api_call({"model": agent.model, "messages": []})

    assert result == {"ok": "fresh-request-client"}
    assert agent.client is replacement_shared
    assert stale_shared.close_calls >= 1
    assert replacement_shared.close_calls == 0
    assert len(factory.calls) == 2


def test_concurrent_requests_do_not_break_each_other_when_one_client_closes(monkeypatch):
    first_started = threading.Event()
    first_closed = threading.Event()

    def first_responder(**kwargs):
        first_started.set()
        first_client.close()
        first_closed.set()
        raise _connection_error()

    def second_responder(**kwargs):
        assert first_started.wait(timeout=2)
        assert first_closed.wait(timeout=2)
        return {"ok": "second"}

    first_client = FakeRequestClient(first_responder)
    second_client = FakeRequestClient(second_responder)
    factory = OpenAIFactory([first_client, second_client])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent()
    results = {}

    def run_call(name):
        try:
            results[name] = agent._interruptible_api_call({"model": agent.model, "messages": []})
        except Exception as exc:  # noqa: BLE001 - asserting exact type below
            results[name] = exc

    thread_one = threading.Thread(target=run_call, args=("first",), daemon=True)
    thread_two = threading.Thread(target=run_call, args=("second",), daemon=True)
    thread_one.start()
    thread_two.start()
    thread_one.join(timeout=5)
    thread_two.join(timeout=5)

    values = list(results.values())
    assert sum(isinstance(value, APIConnectionError) for value in values) == 1
    assert values.count({"ok": "second"}) == 1
    assert len(factory.calls) == 2



def test_streaming_call_recreates_closed_shared_client_before_request(monkeypatch):
    chunks = iter([
        SimpleNamespace(
            model="gpt-5-codex",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello", tool_calls=None), finish_reason=None)],
        ),
        SimpleNamespace(
            model="gpt-5-codex",
            choices=[SimpleNamespace(delta=SimpleNamespace(content=" world", tool_calls=None), finish_reason="stop")],
        ),
    ])

    stale_shared = FakeSharedClient(lambda **kwargs: (_ for _ in ()).throw(AssertionError("stale shared client used")))
    stale_shared._client.is_closed = True

    replacement_shared = FakeSharedClient(lambda **kwargs: {"replacement": True})
    request_client = FakeRequestClient(lambda **kwargs: chunks)
    factory = OpenAIFactory([replacement_shared, request_client])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent(shared_client=stale_shared)
    agent.stream_delta_callback = lambda _delta: None
    # Force chat_completions mode so the streaming path uses
    # chat.completions.create(stream=True) instead of Codex responses.stream()
    agent.api_mode = "chat_completions"
    response = agent._interruptible_streaming_api_call({"model": agent.model, "messages": []})

    assert response.choices[0].message.content == "Hello world"
    assert agent.client is replacement_shared
    assert stale_shared.close_calls >= 1
    assert request_client.close_calls >= 1
    assert len(factory.calls) == 2
