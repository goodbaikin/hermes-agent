import asyncio
import base64
import json
import os
import sys

import pytest

from node_client import hermes_node_client as client


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Linux PTY behavior")


async def _read_pty(fd: int, timeout: float = 3.0) -> str:
    os.set_blocking(fd, False)
    output = bytearray()
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            chunk = os.read(fd, 4096)
        except BlockingIOError:
            await asyncio.sleep(0.05)
            continue
        except OSError:
            break
        if not chunk:
            break
        output.extend(chunk)
    return output.decode("utf-8", errors="replace")


@pytest.mark.asyncio
async def test_linux_interactive_bash_sources_bashrc(tmp_path, monkeypatch):
    bashrc = tmp_path / ".bashrc"
    bashrc.write_text("export HERMES_NODE_RC_MARKER=from_bashrc\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    proxy_id = "pytest-bashrc"
    result = await client._terminal_open(proxy_id, {"shell": "bash", "cols": 80, "rows": 24})
    assert result["ok"] is True
    try:
        payload = base64.b64encode(b"echo MARKER=$HERMES_NODE_RC_MARKER\nexit\n").decode("ascii")
        assert (await client._terminal_write(proxy_id, {"data": payload}))["ok"] is True
        output = await _read_pty(client.TERMINAL_SESSIONS[proxy_id]["master_fd"])
    finally:
        await client._terminal_close(proxy_id)

    assert "MARKER=from_bashrc" in output


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send(self, message: str):
        self.messages.append(json.loads(message))


@pytest.mark.asyncio
async def test_linux_pty_reader_drains_fast_command_output_after_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    proxy_id = "pytest-reader-drain"
    result = await client._terminal_open(proxy_id, {"shell": "bash", "cols": 80, "rows": 24})
    assert result["ok"] is True
    session = client.TERMINAL_SESSIONS[proxy_id]
    master_fd = session["master_fd"]
    ws = FakeWebSocket()
    reader = asyncio.create_task(
        client._read_terminal_output_pty(ws, proxy_id, master_fd, session["proc"])
    )
    try:
        payload = base64.b64encode(b"echo FAST_OUTPUT_MARKER\nexit\n").decode("ascii")
        assert (await client._terminal_write(proxy_id, {"data": payload}))["ok"] is True
        await asyncio.wait_for(reader, timeout=3)
    finally:
        if not reader.done():
            reader.cancel()
        try:
            os.close(master_fd)
        except OSError:
            pass
        await client._terminal_close(proxy_id)

    output = ""
    for message in ws.messages:
        if message.get("event") == "node.terminal.output":
            data = message["payload"]["data"]
            output += base64.b64decode(data).decode("utf-8", errors="replace")
    assert "FAST_OUTPUT_MARKER" in output


def test_linux_pty_reader_does_not_use_cancellable_blocking_executor():
    source = client._read_terminal_output_pty.__code__.co_names
    assert "run_in_executor" not in source
    assert "wait_for" not in source


def test_linux_pty_bash_is_not_started_with_norc():
    constants = client._terminal_open.__code__.co_consts
    assert "--norc" not in constants
