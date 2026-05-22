"""Calibration plugin — LLM bias detection and self-correction for Hermes.

Inspired by gbrain's calibration system (v0.36+). Tracks tool-call outcomes
per domain, detects recurring bias patterns, and nudges the agent before
repeating known failure modes.

Schema: ~/.hermes/calibration.db (SQLite)
  judgments      — prediction + domain + confidence
  outcomes       — actual result (success/failure/partial)
  bias_patterns  — detected recurring biases with accuracy rates
  nudge_log      — cooldown tracking per pattern

Hooks:
  post_tool_call  — record outcome, update patterns
  pre_tool_call   — warn if current domain has active bias pattern
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_BUCKET_N = 5          # minimum samples before declaring a bias pattern
NUDGE_COOLDOWN_DAYS = 14
ACCURACY_THRESHOLD = 0.5  # below this = bias detected

# Domain inference rules: (tool_name_predicate, arg_predicate, domain)
_DOMAIN_RULES: List[Tuple[str, Optional[str], str]] = [
    # node operations
    ("node_invoke", None, "node_operations"),
    ("node_lib", None, "node_operations"),
    # PowerShell / Azure
    ("terminal", "powershell", "powershell"),
    ("terminal", "pwsh", "powershell"),
    ("terminal", ".ps1", "powershell"),
    ("terminal", "get-az", "powershell"),
    ("terminal", "new-az", "powershell"),
    ("terminal", "set-az", "powershell"),
    # C# / .NET
    ("terminal", "dotnet", "csharp"),
    ("terminal", "msbuild", "csharp"),
    ("terminal", ".csproj", "csharp"),
    ("terminal", ".sln", "csharp"),
    # Web research
    ("browser_", None, "web_research"),
    ("web_search", None, "web_research"),
    ("web_extract", None, "web_research"),
    # Azure deploy (Bicep)
    ("patch", ".bicep", "azure_deploy"),
    ("write_file", ".bicep", "azure_deploy"),
    ("terminal", "az ", "azure_deploy"),
    ("terminal", "azdeploy", "azure_deploy"),
    ("terminal", "new-azresourcegroupdeployment", "azure_deploy"),
    # Python
    ("patch", ".py", "python"),
    ("write_file", ".py", "python"),
    ("execute_code", None, "python"),
    # JavaScript / TypeScript
    ("patch", ".js", "javascript"),
    ("patch", ".ts", "javascript"),
    ("write_file", ".js", "javascript"),
    ("write_file", ".ts", "javascript"),
    ("terminal", "npm", "javascript"),
    ("terminal", "node", "javascript"),
    ("terminal", "pnpm", "javascript"),
    ("terminal", "yarn", "javascript"),
    # Rust
    ("patch", ".rs", "rust"),
    ("write_file", ".rs", "rust"),
    ("terminal", "cargo", "rust"),
    # Go
    ("patch", ".go", "go"),
    ("write_file", ".go", "go"),
    # Docker
    ("terminal", "docker", "docker"),
    ("terminal", "docker-compose", "docker"),
    ("patch", "dockerfile", "docker"),
    ("write_file", "dockerfile", "docker"),
    # Kubernetes
    ("terminal", "kubectl", "kubernetes"),
    ("patch", ".yaml", "kubernetes"),
    ("write_file", ".yaml", "kubernetes"),
    ("patch", ".yml", "kubernetes"),
    ("write_file", ".yml", "kubernetes"),
    # File operations (generic)
    ("read_file", None, "file_ops"),
    ("search_files", None, "file_ops"),
    # Git
    ("terminal", "git", "git"),
    # Cron / scheduling
    ("cronjob", None, "scheduling"),
    # Memory
    ("hindsight_retain", None, "memory"),
    ("hindsight_recall", None, "memory"),
    ("memory", None, "memory"),
    # Database
    ("terminal", "psql", "database"),
    ("terminal", "mysql", "database"),
    ("terminal", "sqlite3", "database"),
    # SSH / Remote
    ("terminal", "ssh", "remote"),
    ("terminal", "scp", "remote"),
    # Build tools
    ("terminal", "make", "build"),
    ("terminal", "cmake", "build"),
    ("terminal", "ninja", "build"),
]


# ---------------------------------------------------------------------------
# Calibration database
# ---------------------------------------------------------------------------

class CalibrationDB:
    """SQLite-backed store for judgment → outcome tracking."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.executescript(_SCHEMA)
                conn.commit()

    # -- Judgments ----------------------------------------------------------

    def record_judgment(
        self,
        tool_name: str,
        domain: str,
        args_json: str,
        confidence: float = 0.5,
    ) -> int:
        """Record a prediction (before tool execution). Returns judgment_id."""
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO judgments (tool_name, domain, args_json, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (tool_name, domain, args_json, confidence, _now_iso()),
                )
                conn.commit()
                return cur.lastrowid

    def record_outcome(
        self,
        judgment_id: int,
        success: bool,
        error_type: Optional[str] = None,
        result_summary: Optional[str] = None,
    ) -> None:
        """Record the actual outcome for a judgment."""
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO outcomes (judgment_id, success, error_type, result_summary, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (judgment_id, int(success), error_type, result_summary, _now_iso()),
                )
                conn.commit()

    # -- Aggregation / bias detection ---------------------------------------

    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Return {total, success_count, failure_count, accuracy} for a domain."""
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        SUM(o.success) AS success_count,
                        SUM(1 - o.success) AS failure_count
                    FROM judgments j
                    JOIN outcomes o ON o.judgment_id = j.id
                    WHERE j.domain = ?
                    """,
                    (domain,),
                ).fetchone()
        total = row["total"] or 0
        success = row["success_count"] or 0
        failure = row["failure_count"] or 0
        accuracy = success / total if total > 0 else 0.0
        return {
            "total": total,
            "success": success,
            "failure": failure,
            "accuracy": accuracy,
        }

    def detect_bias_patterns(self) -> List[Dict[str, Any]]:
        """Scan all domains and return those with accuracy below threshold."""
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        j.domain,
                        COUNT(*) AS total,
                        SUM(o.success) AS success_count,
                        SUM(1 - o.success) AS failure_count
                    FROM judgments j
                    JOIN outcomes o ON o.judgment_id = j.id
                    GROUP BY j.domain
                    HAVING COUNT(*) >= ?
                    """,
                    (MIN_BUCKET_N,),
                ).fetchall()

        patterns = []
        for row in rows:
            total = row["total"]
            success = row["success_count"] or 0
            accuracy = success / total
            if accuracy < ACCURACY_THRESHOLD:
                patterns.append({
                    "domain": row["domain"],
                    "total": total,
                    "success": success,
                    "failure": total - success,
                    "accuracy": accuracy,
                })
        return patterns

    def upsert_bias_pattern(self, domain: str, accuracy: float, total: int) -> None:
        """Store or update a detected bias pattern."""
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO bias_patterns (domain, accuracy_rate, occurrence_count, first_detected, last_seen, active)
                    VALUES (?, ?, ?, ?, ?, 1)
                    ON CONFLICT(domain) DO UPDATE SET
                        accuracy_rate = excluded.accuracy_rate,
                        occurrence_count = occurrence_count + excluded.occurrence_count,
                        last_seen = excluded.last_seen,
                        active = CASE WHEN excluded.accuracy_rate < ? THEN 1 ELSE 0 END
                    """,
                    (domain, accuracy, total, _now_iso(), _now_iso(), ACCURACY_THRESHOLD),
                )
                conn.commit()

    def get_active_bias_patterns(self) -> List[Dict[str, Any]]:
        """Return all currently active bias patterns."""
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT domain, accuracy_rate, occurrence_count, first_detected, last_seen
                    FROM bias_patterns
                    WHERE active = 1
                    ORDER BY accuracy_rate ASC
                    """
                ).fetchall()
        return [dict(r) for r in rows]

    def get_bias_pattern(self, domain: str) -> Optional[Dict[str, Any]]:
        """Return a specific bias pattern, or None."""
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM bias_patterns WHERE domain = ? AND active = 1",
                    (domain,),
                ).fetchone()
        return dict(row) if row else None

    # -- Nudge / cooldown ---------------------------------------------------

    def check_nudge_cooldown(self, domain: str) -> bool:
        """Return True if nudge is ON cooldown (should NOT fire)."""
        cutoff = (datetime.now() - timedelta(days=NUDGE_COOLDOWN_DAYS)).isoformat()
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    """
                    SELECT 1 FROM nudge_log
                    WHERE domain = ? AND fired_at >= ?
                    LIMIT 1
                    """,
                    (domain, cutoff),
                ).fetchone()
        return row is not None

    def record_nudge(self, domain: str, tool_name: str, message: str) -> None:
        """Log that a nudge was fired."""
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO nudge_log (domain, tool_name, message, fired_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (domain, tool_name, message, _now_iso()),
                )
                conn.commit()

    def reset_nudge_cooldown(self, domain: str) -> int:
        """Clear cooldown for a domain. Returns number of rows deleted."""
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "DELETE FROM nudge_log WHERE domain = ?",
                    (domain,),
                )
                conn.commit()
                return cur.rowcount


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS judgments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name   TEXT NOT NULL,
    domain      TEXT NOT NULL,
    args_json   TEXT,
    confidence  REAL DEFAULT 0.5,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    judgment_id     INTEGER NOT NULL UNIQUE REFERENCES judgments(id) ON DELETE CASCADE,
    success         INTEGER NOT NULL,  -- 0 or 1
    error_type      TEXT,
    result_summary  TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bias_patterns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    domain            TEXT NOT NULL UNIQUE,
    accuracy_rate     REAL NOT NULL,
    occurrence_count  INTEGER NOT NULL DEFAULT 1,
    first_detected    TEXT NOT NULL,
    last_seen         TEXT NOT NULL,
    active            INTEGER NOT NULL DEFAULT 1  -- 0 or 1
);

CREATE TABLE IF NOT EXISTS nudge_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    domain      TEXT NOT NULL,
    tool_name   TEXT NOT NULL,
    message     TEXT NOT NULL,
    fired_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_judgments_domain ON judgments(domain);
CREATE INDEX IF NOT EXISTS idx_judgments_created ON judgments(created_at);
CREATE INDEX IF NOT EXISTS idx_outcomes_judgment ON outcomes(judgment_id);
CREATE INDEX IF NOT EXISTS idx_nudge_domain_fired ON nudge_log(domain, fired_at);
"""


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

def infer_domain(tool_name: str, args: Dict[str, Any]) -> str:
    """Infer the domain from tool name and arguments."""
    args_json = json.dumps(args, ensure_ascii=False, default=str)

    for rule_tool, rule_arg, domain in _DOMAIN_RULES:
        if not tool_name.startswith(rule_tool):
            continue
        if rule_arg is None:
            return domain
        # rule_arg is a substring to look for in args JSON
        if rule_arg in args_json.lower():
            return domain

    return "general"


def _is_success(result: str) -> Tuple[bool, Optional[str]]:
    """Parse a tool result string and determine success/failure.

    Returns (success: bool, error_type: str|None).
    """
    if not result:
        return True, None

    # Try to parse as JSON
    try:
        data = json.loads(result)
    except Exception:
        # Non-JSON: heuristic — look for error markers
        lower = result.lower()
        if "error" in lower and "traceback" in lower:
            return False, "exception"
        if result.strip().startswith("Error:"):
            return False, "error"
        return True, None

    # JSON result
    if isinstance(data, dict):
        if data.get("error"):
            err = data["error"]
            if isinstance(err, str):
                return False, err[:100]
            return False, "error"
        if data.get("exit_code", 0) != 0:
            return False, f"exit_code:{data['exit_code']}"

    return True, None


# ---------------------------------------------------------------------------
# Plugin lifecycle
# ---------------------------------------------------------------------------

_db: Optional[CalibrationDB] = None
_db_lock = threading.Lock()

# Thread-local storage for nudge messages that need to be injected into
# tool results via transform_tool_result.
_nudge_tls = threading.local()


def _get_db() -> CalibrationDB:
    global _db
    if _db is not None:
        return _db
    with _db_lock:
        if _db is not None:
            return _db
        home = Path(get_hermes_home())
        db_path = home / "calibration.db"
        _db = CalibrationDB(db_path)
        return _db


def _now_iso() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------

def post_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    result: str,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
    **kwargs: Any,
) -> None:
    """Record tool execution outcome and update bias patterns."""
    try:
        db = _get_db()
        domain = infer_domain(tool_name, args)
        success, error_type = _is_success(result)

        # Record judgment + outcome
        args_json = json.dumps(args, ensure_ascii=False, default=str)
        judgment_id = db.record_judgment(tool_name, domain, args_json)
        db.record_outcome(judgment_id, success, error_type)

        # Periodically re-run bias detection (every 10 recordings)
        # Use a simple counter in memory to avoid DB hits
        _maybe_detect_bias(db)

        logger.debug(
            "Calibration recorded: %s/%s success=%s error=%s",
            tool_name, domain, success, error_type,
        )
    except Exception as exc:
        logger.debug("Calibration post_tool_call error: %s", exc)
    finally:
        # Clear any pending nudge for this thread
        try:
            delattr(_nudge_tls, "message")
        except AttributeError:
            pass


def pre_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Check for active bias patterns and stage a nudge if needed.

    Also detects repetitive tool-call loops.  The nudge is stored thread-locally
    and injected into the tool result by transform_tool_result.
    Returns None (never blocks).
    """
    try:
        # --- Loop detection (runs first, lightweight) -------------------------
        loop_msg = _detect_loop(tool_name, args)
        if loop_msg:
            _nudge_tls.message = loop_msg
            _nudge_tls.domain = "loop_detection"
            _nudge_tls.tool_name = tool_name
            logger.warning(loop_msg)
            return None

        # --- Bias-pattern nudge -----------------------------------------------
        db = _get_db()
        domain = infer_domain(tool_name, args)

        # Check if this domain has an active bias pattern
        pattern = db.get_bias_pattern(domain)
        if not pattern:
            return None

        # Check cooldown
        if db.check_nudge_cooldown(domain):
            return None

        # Build nudge message
        accuracy = pattern["accuracy_rate"]
        total = pattern["occurrence_count"]
        msg = (
            f"[calibration] Bias detected in '{domain}': "
            f"{accuracy:.0%} accuracy over {total} calls. "
            f"Consider reviewing your approach for this tool/domain."
        )

        # Log the nudge
        db.record_nudge(domain, tool_name, msg)

        # Stage for injection into tool result
        _nudge_tls.message = msg
        _nudge_tls.domain = domain
        _nudge_tls.tool_name = tool_name

        logger.warning(msg)
        return None

    except Exception as exc:
        logger.debug("Calibration pre_tool_call error: %s", exc)
        return None


def transform_tool_result(
    tool_name: str,
    args: Dict[str, Any],
    result: str,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
    **kwargs: Any,
) -> Optional[str]:
    """Inject pending nudge messages into tool results.

    Returns the modified result string if a nudge is pending, else None.
    """
    try:
        msg = getattr(_nudge_tls, "message", None)
        if not msg:
            return None

        # Inject nudge at the top of the result
        # Try to preserve JSON structure if the result is JSON
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                # Add calibration warning as a special field
                data["_calibration_warning"] = msg
                return json.dumps(data, ensure_ascii=False, default=str)
        except Exception:
            pass

        # Non-JSON or parse failed: prepend as plain text
        return f"{msg}\n\n{result}"

    except Exception as exc:
        logger.debug("Calibration transform_tool_result error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

_LOOP_WINDOW_SIZE = 20       # keep last N calls in memory
_LOOP_REPEAT_THRESHOLD = 3   # same (tool, args) this many times = loop
_LOOP_TIME_WINDOW_SEC = 120  # within this many seconds

_loop_history: List[Dict[str, Any]] = []
_loop_history_lock = threading.Lock()


def _detect_loop(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Check if the same tool+args has been called repeatedly in a short window.

    Returns a warning message if a loop is detected, else None.
    """
    global _loop_history
    args_json = json.dumps(args, ensure_ascii=False, default=str, sort_keys=True)
    key = f"{tool_name}:{args_json}"
    now = time.time()

    with _loop_history_lock:
        # Prune old entries outside the time window
        cutoff = now - _LOOP_TIME_WINDOW_SEC
        _loop_history = [h for h in _loop_history if h["ts"] > cutoff]

        # Count occurrences of this exact call
        count = sum(1 for h in _loop_history if h["key"] == key)

        # Record this call
        _loop_history.append({"key": key, "ts": now})

        # Trim to window size
        if len(_loop_history) > _LOOP_WINDOW_SIZE:
            _loop_history = _loop_history[-_LOOP_WINDOW_SIZE:]

        if count >= _LOOP_REPEAT_THRESHOLD - 1:  # -1 because we just added this one
            return (
                f"[calibration] Loop detected: '{tool_name}' with same arguments "
                f"has been called {count + 1} times in the last {_LOOP_TIME_WINDOW_SEC}s. "
                f"Consider stepping back and reviewing your approach."
            )

    return None
# ---------------------------------------------------------------------------

_bias_counter = 0
_bias_counter_lock = threading.Lock()


def _maybe_detect_bias(db: CalibrationDB) -> None:
    global _bias_counter
    with _bias_counter_lock:
        _bias_counter += 1
        if _bias_counter < 10:
            return
        _bias_counter = 0

    try:
        patterns = db.detect_bias_patterns()
        for p in patterns:
            db.upsert_bias_pattern(p["domain"], p["accuracy"], p["total"])
            logger.info(
                "Calibration bias detected: %s accuracy=%.2f%% (n=%d)",
                p["domain"], p["accuracy"] * 100, p["total"],
            )
    except Exception as exc:
        logger.debug("Calibration bias detection error: %s", exc)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register calibration hooks with the plugin manager."""
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("transform_tool_result", transform_tool_result)
    logger.info("Calibration plugin registered")
