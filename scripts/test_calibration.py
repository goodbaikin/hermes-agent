#!/usr/bin/env python3
"""Test script for the calibration plugin.

Validates end-to-end:
  1. Domain inference from tool names + args
  2. Success/failure detection from result strings
  3. Bias pattern detection over multiple recordings
  4. Nudge cooldown mechanism
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure hermes-agent is importable
HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))

from plugins.observability.calibration import (
    CalibrationDB,
    infer_domain,
    _is_success,
    post_tool_call,
    pre_tool_call,
    transform_tool_result,
)


def test_infer_domain():
    """Test domain inference from tool names and arguments."""
    cases = [
        (("terminal", {"command": "powershell -c 'Get-Process'"}), "powershell"),
        (("terminal", {"command": "dotnet build"}), "csharp"),
        (("terminal", {"command": "npm install"}), "javascript"),
        (("terminal", {"command": "cargo build"}), "rust"),
        (("terminal", {"command": "docker ps"}), "docker"),
        (("terminal", {"command": "kubectl get pods"}), "kubernetes"),
        (("browser_navigate", {"url": "https://example.com"}), "web_research"),
        (("patch", {"path": "main.bicep"}), "azure_deploy"),
        (("write_file", {"path": "test.py"}), "python"),
        (("node_invoke", {"node_id": "lxc-204"}), "node_operations"),
        (("read_file", {"path": "/etc/passwd"}), "file_ops"),
        (("terminal", {"command": "git status"}), "git"),
        (("cronjob", {"action": "list"}), "scheduling"),
        (("hindsight_retain", {"content": "test"}), "memory"),
        (("unknown_tool", {}), "general"),
    ]
    for (tool, args), expected in cases:
        result = infer_domain(tool, args)
        assert result == expected, f"infer_domain({tool}, {args}) = {result}, expected {expected}"
    print("  infer_domain: OK")


def test_is_success():
    """Test success/failure detection from result strings."""
    cases = [
        ('{"result": "ok"}', (True, None)),
        ('{"error": "Something broke"}', (False, "Something broke")),
        ('{"exit_code": 1}', (False, "exit_code:1")),
        ('Error: command not found', (False, "error")),
        ('Traceback (most recent call last):\n  File "x.py"\nerror', (False, "exception")),
        ('some plain text output', (True, None)),
        ('', (True, None)),
    ]
    for result_str, expected in cases:
        success, err = _is_success(result_str)
        assert (success, err) == expected, f"_is_success({result_str!r}) = {(success, err)}, expected {expected}"
    print("  _is_success: OK")


def test_db_schema():
    """Test database creation and basic operations."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        db = CalibrationDB(db_path)

        # Record some judgments + outcomes
        jid1 = db.record_judgment("terminal", "powershell", '{"command": "ls"}')
        db.record_outcome(jid1, True)

        jid2 = db.record_judgment("terminal", "powershell", '{"command": "bad-cmd"}')
        db.record_outcome(jid2, False, "command not found")

        # Check domain stats
        stats = db.get_domain_stats("powershell")
        assert stats["total"] == 2
        assert stats["success"] == 1
        assert stats["failure"] == 1
        assert stats["accuracy"] == 0.5
        print("  db_schema + basic ops: OK")

        # Bias detection (need 5+ samples, so add more)
        for i in range(4):
            jid = db.record_judgment("terminal", "powershell", '{"command": "fail"}')
            db.record_outcome(jid, False, "error")

        patterns = db.detect_bias_patterns()
        assert len(patterns) == 1
        assert patterns[0]["domain"] == "powershell"
        assert patterns[0]["accuracy"] < 0.5
        print("  bias_detection: OK")

        # Upsert pattern
        db.upsert_bias_pattern("powershell", patterns[0]["accuracy"], patterns[0]["total"])
        active = db.get_active_bias_patterns()
        assert len(active) == 1
        print("  upsert + get_active: OK")

        # Nudge cooldown
        assert not db.check_nudge_cooldown("powershell")  # no nudge yet
        db.record_nudge("powershell", "terminal", "test nudge")
        assert db.check_nudge_cooldown("powershell")  # now on cooldown
        print("  nudge_cooldown: OK")

    finally:
        db_path.unlink(missing_ok=True)


def test_hooks():
    """Test pre/post tool call hooks end-to-end."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Monkey-patch _get_db to use temp DB
    import plugins.observability.calibration as calib
    orig_db = calib._db
    # Clear loop history for clean test
    orig_loop_history = list(calib._loop_history)
    try:
        calib._db = CalibrationDB(db_path)
        calib._bias_counter = 0  # reset counter for clean test
        calib._loop_history = []  # clear loop history

        # Simulate successful tool calls (all in powershell domain)
        for i in range(3):
            post_tool_call(
                tool_name="terminal",
                args={"command": "powershell -c 'echo ok'"},
                result='{"output": "ok"}',
            )

        # Simulate failures
        for i in range(7):
            post_tool_call(
                tool_name="terminal",
                args={"command": "powershell -c 'bad-cmd'"},
                result='{"error": "command not found"}',
            )

        # Bias should be detected (7 failures / 10 total = 30% accuracy)
        patterns = calib._db.detect_bias_patterns()
        assert len(patterns) == 1
        assert patterns[0]["domain"] == "powershell"
        assert abs(patterns[0]["accuracy"] - 0.3) < 0.01
        print("  post_tool_call hook + bias detection: OK")

        # Pre-tool call should stage nudge (non-blocking)
        result = pre_tool_call(
            tool_name="terminal",
            args={"command": "powershell -c 'something'"},
        )
        assert result is None  # non-blocking nudge
        print("  pre_tool_call hook: OK")

        # Transform tool result should inject nudge into JSON result
        nudge_result = transform_tool_result(
            tool_name="terminal",
            args={"command": "powershell -c 'something'"},
            result='{"output": "ok"}',
        )
        assert nudge_result is not None
        parsed = json.loads(nudge_result)
        assert "_calibration_warning" in parsed
        assert "powershell" in parsed["_calibration_warning"]
        print("  transform_tool_result JSON injection: OK")

        # Transform tool result should inject nudge into plain text result
        nudge_result2 = transform_tool_result(
            tool_name="terminal",
            args={"command": "powershell -c 'something'"},
            result="plain text output",
        )
        assert nudge_result2 is not None
        assert nudge_result2.startswith("[calibration]")
        assert "plain text output" in nudge_result2
        print("  transform_tool_result plain text injection: OK")

        # Clear nudge state
        post_tool_call(
            tool_name="terminal",
            args={"command": "powershell -c 'cleanup'"},
            result='{"output": "ok"}',
        )

        # After post_tool_call, nudge should be cleared
        nudge_result3 = transform_tool_result(
            tool_name="terminal",
            args={"command": "powershell -c 'something'"},
            result='{"output": "ok"}',
        )
        assert nudge_result3 is None  # no nudge staged
        print("  nudge cleanup after post_tool_call: OK")

        # --- Loop detection tests -------------------------------------------
        calib._loop_history = []  # clear for loop tests

        # First 2 calls with same args — no loop yet
        for i in range(2):
            pre_tool_call(tool_name="read_file", args={"path": "/etc/passwd"})
            post_tool_call(
                tool_name="read_file",
                args={"path": "/etc/passwd"},
                result='{"content": "root:x:0:0"}',
            )
        loop_nudge = transform_tool_result(
            tool_name="read_file",
            args={"path": "/etc/passwd"},
            result='{"content": "root:x:0:0"}',
        )
        assert loop_nudge is None  # only 2 calls, no loop
        print("  loop detection (2 calls, no alert): OK")

        # 3rd call with same args — loop detected
        pre_tool_call(tool_name="read_file", args={"path": "/etc/passwd"})
        loop_nudge = transform_tool_result(
            tool_name="read_file",
            args={"path": "/etc/passwd"},
            result='{"content": "root:x:0:0"}',
        )
        assert loop_nudge is not None
        assert "Loop detected" in loop_nudge
        assert "read_file" in loop_nudge
        print("  loop detection (3 calls, alert fired): OK")

        # Different args — should not trigger loop
        calib._loop_history = []  # clear
        pre_tool_call(tool_name="read_file", args={"path": "/etc/hosts"})
        post_tool_call(
            tool_name="read_file",
            args={"path": "/etc/hosts"},
            result='{"content": "127.0.0.1 localhost"}',
        )
        no_loop = transform_tool_result(
            tool_name="read_file",
            args={"path": "/etc/hosts"},
            result='{"content": "127.0.0.1 localhost"}',
        )
        assert no_loop is None
        print("  loop detection (different args, no alert): OK")

    finally:
        calib._db = orig_db
        calib._loop_history = orig_loop_history
        db_path.unlink(missing_ok=True)


def main():
    print("Running calibration plugin tests...")
    test_infer_domain()
    test_is_success()
    test_db_schema()
    test_hooks()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
