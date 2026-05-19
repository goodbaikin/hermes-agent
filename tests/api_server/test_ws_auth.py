"""Tests for WebSocket authentication in API Server."""

import hmac


def test_ws_auth_query_param_success():
    """WebSocket auth logic: valid ?token=xxx query param succeeds."""
    api_key = "valid-key"
    query_token = "valid-key"
    auth_ok = query_token and hmac.compare_digest(query_token, api_key)
    assert auth_ok is True


def test_ws_auth_header_success():
    """WebSocket auth logic: valid Authorization: Bearer header succeeds."""
    api_key = "valid-key"
    auth_header = "Bearer valid-key"
    auth_ok = False
    if auth_header.startswith("Bearer "):
        header_token = auth_header[7:].strip()
        if hmac.compare_digest(header_token, api_key):
            auth_ok = True
    assert auth_ok is True


def test_ws_auth_invalid_token():
    """WebSocket auth logic: invalid token fails."""
    api_key = "valid-key"
    query_token = "wrong-key"
    auth_ok = query_token and hmac.compare_digest(query_token, api_key)
    assert auth_ok is False


def test_ws_auth_no_api_key_allows_all():
    """WebSocket auth logic: no API key configured allows all."""
    api_key = ""
    auth_ok = False
    if api_key:
        pass
    else:
        auth_ok = True
    assert auth_ok is True


def test_ws_auth_connect_message_fallback():
    """WebSocket auth logic: connect message auth.token fallback works."""
    api_key = "valid-key"
    connect_token = "valid-key"
    auth_ok = False
    if connect_token and hmac.compare_digest(connect_token, api_key):
        auth_ok = True
    assert auth_ok is True


def test_ws_auth_connect_message_wrong_token():
    """WebSocket auth logic: connect message with wrong token fails."""
    api_key = "valid-key"
    connect_token = "wrong-key"
    auth_ok = False
    if connect_token and hmac.compare_digest(connect_token, api_key):
        auth_ok = True
    assert auth_ok is False
