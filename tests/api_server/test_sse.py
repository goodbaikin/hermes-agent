"""Tests for api_server.sse SSEWriter."""

import pytest
from api_server.sse import build_cors_headers, SSEWriter


def test_build_cors_headers_no_origin():
    assert build_cors_headers("", ()) is None


def test_build_cors_headers_wildcard():
    headers = build_cors_headers("http://localhost", ("*",))
    assert headers is not None
    assert headers["Access-Control-Allow-Origin"] == "*"


def test_build_cors_headers_allowed():
    headers = build_cors_headers("http://localhost", ("http://localhost", "http://example.com"))
    assert headers is not None
    assert headers["Access-Control-Allow-Origin"] == "http://localhost"


def test_build_cors_headers_denied():
    headers = build_cors_headers("http://evil.com", ("http://localhost",))
    assert headers is None


@pytest.mark.asyncio
async def test_sse_writer_prepare():
    # Minimal mock request
    class MockRequest:
        headers = {}

    writer = SSEWriter(MockRequest())
    # prepare() requires a real aiohttp response; skip in unit test
    # but verify object state
    assert writer._headers["Content-Type"] == "text/event-stream"
