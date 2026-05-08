"""
Standalone API Server entry point.

Usage:
    python -m api_server
    python -m api_server --host 0.0.0.0 --port 8642

Environment variables:
    API_SERVER_HOST     Bind address (default: 127.0.0.1)
    API_SERVER_PORT     Listen port (default: 8642)
    API_SERVER_KEY      API key for authentication
    API_SERVER_CORS_ORIGINS  Comma-separated allowed origins
    API_SERVER_MODEL_NAME    Default model name
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from api_server.server import StandaloneAPIServer, check_api_server_requirements

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent API Server")
    parser.add_argument("--host", default=os.getenv("API_SERVER_HOST", "127.0.0.1"), help="Bind address")
    parser.add_argument("--port", type=int, default=int(os.getenv("API_SERVER_PORT", "8642")), help="Listen port")
    parser.add_argument("--key", default=os.getenv("API_SERVER_KEY", ""), help="API key")
    parser.add_argument("--cors-origins", default=os.getenv("API_SERVER_CORS_ORIGINS", ""), help="Allowed CORS origins")
    parser.add_argument("--model", default=os.getenv("API_SERVER_MODEL_NAME", ""), help="Default model name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not check_api_server_requirements():
        logger.error("aiohttp is required. Install: pip install aiohttp")
        sys.exit(1)

    # Build a minimal PlatformConfig-like object from CLI args
    from dataclasses import dataclass, field

    @dataclass
    class _CLIConfig:
        extra: dict = field(default_factory=dict)

    cfg = _CLIConfig(extra={
        "host": args.host,
        "port": str(args.port),
        "key": args.key,
        "cors_origins": args.cors_origins,
        "model_name": args.model,
    })

    server = StandaloneAPIServer(config=cfg)

    async def _run():
        ok = await server.connect()
        if not ok:
            logger.error("Failed to start API server")
            sys.exit(1)
        logger.info("API server running. Press Ctrl+C to stop.")
        try:
            while server._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await server.disconnect()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")


if __name__ == "__main__":
    main()
