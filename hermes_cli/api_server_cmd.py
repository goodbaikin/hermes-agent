"""
API Server CLI commands — `hermes api-server run|status|stop`.

Allows starting the API server independently of the full gateway.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

from hermes_logging import setup_logging

logger = logging.getLogger(__name__)


def api_server_command(args):
    """Dispatch api-server subcommands."""
    sub = getattr(args, "api_server_command", "run")
    if sub == "run" or sub is None:
        return _cmd_run(args)
    if sub == "status":
        return _cmd_status(args)
    if sub == "stop":
        return _cmd_stop(args)
    print(f"Unknown api-server command: {sub}", file=sys.stderr)
    sys.exit(1)


def _cmd_run(args):
    """Start the API server in the foreground."""
    import asyncio
    from api_server.config import load_api_server_config
    from api_server.server import StandaloneAPIServer
    from gateway.config import PlatformConfig

    verbose = getattr(args, "verbose", 0) or 0
    quiet = getattr(args, "quiet", False) or False

    log_level = "WARNING"
    if quiet:
        log_level = "ERROR"
    elif verbose >= 2:
        log_level = "DEBUG"
    elif verbose == 1:
        log_level = "INFO"
    setup_logging(log_level=log_level)

    config = load_api_server_config()
    platform_config = PlatformConfig(
        name="api_server",
        platform="api_server",
        enabled=True,
        extra={
            "host": config.host,
            "port": config.port,
            "key": config.api_key,
            "cors_origins": config.cors_origins,
            "model_name": config.model_name,
        },
    )
    server = StandaloneAPIServer(platform_config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        ok = await server.connect()
        if not ok:
            logger.error("API server failed to start")
            sys.exit(1)
        logger.info(
            "API server listening on http://%s:%d (model: %s)",
            config.host, config.port, config.model_name,
        )
        # Block forever
        while server._running:
            await asyncio.sleep(1)

    def _shutdown(sig):
        logger.info("Received signal %s, shutting down API server...", sig.name)
        server._running = False
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _shutdown(s))

    try:
        loop.run_until_complete(_run())
    finally:
        try:
            loop.run_until_complete(server.disconnect())
        except Exception:
            pass
        loop.close()


def _cmd_status(args):
    """Check whether the API server is running."""
    import socket
    from api_server.config import load_api_server_config

    config = load_api_server_config()
    try:
        with socket.create_connection((config.host, config.port), timeout=2):
            print(f"API server is running on http://{config.host}:{config.port}")
    except (ConnectionRefusedError, OSError):
        print(f"API server is NOT running on http://{config.host}:{config.port}")
        sys.exit(1)


def _cmd_stop(args):
    """Stop a running API server process."""
    import subprocess
    # Find hermes api-server processes
    try:
        result = subprocess.run(
            ["pgrep", "-f", "hermes api-server run"],
            capture_output=True,
            text=True,
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        if not pids:
            print("No API server process found.")
            return
        for pid in pids:
            print(f"Stopping API server process {pid}...")
            os.kill(int(pid), signal.SIGTERM)
            # Wait briefly for graceful shutdown
            for _ in range(10):
                try:
                    os.kill(int(pid), 0)
                    time.sleep(0.2)
                except ProcessLookupError:
                    break
        print("API server stopped.")
    except Exception as exc:
        logger.error("Failed to stop API server: %s", exc)
        sys.exit(1)
