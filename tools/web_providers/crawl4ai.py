"""Crawl4AI web extract provider.

Crawl4AI is an open-source LLM-friendly web crawler and scraper.
It runs as a self-hosted REST API service.

Configuration::

    # ~/.hermes/config.yaml
    CRAWL4AI_URL: http://192.168.13.10:11235

    web:
      extract_backend: "crawl4ai"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from tools.web_providers.base import WebExtractProvider

logger = logging.getLogger(__name__)


def _get_crawl4ai_url() -> str:
    """Resolve Crawl4AI URL from env or config.yaml."""
    env_url = os.getenv("CRAWL4AI_URL", "").strip()
    if env_url:
        return env_url
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        url = cfg.get("CRAWL4AI_URL", "").strip()
        if url:
            return url
        # Also check web: section
        web_cfg = cfg.get("web", {})
        url = web_cfg.get("crawl4ai_url", "").strip()
        if url:
            return url
    except Exception:
        pass
    return ""


class Crawl4AIExtractProvider(WebExtractProvider):
    """Extract via a self-hosted Crawl4AI REST API instance.

    Requires ``CRAWL4AI_URL`` to be set (e.g. ``http://192.168.13.10:11235``).
    No API key needed — Crawl4AI is open-source and self-hosted.
    """

    def provider_name(self) -> str:
        return "crawl4ai"

    def is_configured(self) -> bool:
        """Return True when ``CRAWL4AI_URL`` is set to a non-empty value."""
        return bool(_get_crawl4ai_url())

    def extract(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        """Extract content from the given URLs via Crawl4AI API.

        Returns normalized results::

            {
                "success": True,
                "data": [
                    {"url": str, "title": str, "content": str,
                     "raw_content": str, "metadata": dict},
                    ...
                ]
            }

        On failure returns ``{"success": False, "error": str}``.
        """
        import httpx

        base_url = _get_crawl4ai_url().rstrip("/")
        if not base_url:
            return {"success": False, "error": "CRAWL4AI_URL is not set"}

        fmt = kwargs.get("format", "markdown")
        results: List[Dict[str, Any]] = []

        for url in urls:
            try:
                resp = httpx.post(
                    f"{base_url}/extract",
                    json={"url": url, "format": fmt},
                    timeout=60,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("success"):
                    results.append({
                        "url": data.get("url", url),
                        "title": data.get("title", ""),
                        "content": data.get("content", ""),
                        "raw_content": data.get("raw_content", ""),
                        "metadata": data.get("metadata", {}),
                    })
                else:
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": data.get("error", "Unknown error"),
                    })
            except httpx.HTTPStatusError as exc:
                logger.warning("Crawl4AI HTTP error: %s", exc)
                results.append({
                    "url": url, "title": "", "content": "", "raw_content": "",
                    "error": f"Crawl4AI returned HTTP {exc.response.status_code}",
                })
            except httpx.RequestError as exc:
                logger.warning("Crawl4AI request error: %s", exc)
                results.append({
                    "url": url, "title": "", "content": "", "raw_content": "",
                    "error": f"Could not reach Crawl4AI at {base_url}: {exc}",
                })

        logger.info("Crawl4AI extract: %d URL(s), %d results", len(urls), len(results))
        return {"success": True, "data": results}
