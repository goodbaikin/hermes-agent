import hashlib
import ipaddress
import json
import logging
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

MAX_NORMALIZED_TEXT_LENGTH = 65_536
MAX_CONTENT_LIST_SIZE = 1_000

_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_FILE_PART_TYPES = frozenset({"file", "input_file"})


def _parse_int(value: Any, default: int, minimum: int = 0) -> int:
    """Parse an integer query parameter with bounds."""
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"Value must be >= {minimum}")
    return parsed


def is_network_accessible(host: str) -> bool:
    """Return True if *host* would expose the server beyond loopback.

    Loopback addresses (127.0.0.1, ::1, IPv4-mapped ::ffff:127.0.0.1)
    are local-only.  Unspecified addresses (0.0.0.0, ::) bind all
    interfaces.  Hostnames are resolved; DNS failure fails closed.
    """
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_loopback:
            return False
        # ::ffff:127.0.0.1 — Python reports is_loopback=False for mapped
        # addresses, so check the underlying IPv4 explicitly.
        if getattr(addr, "ipv4_mapped", None) and addr.ipv4_mapped.is_loopback:
            return False
        return True
    except ValueError:
        # when host variable is a hostname, we should try to resolve below
        pass

    try:
        import socket
        resolved = socket.getaddrinfo(host, None, family=socket.AF_UNSPEC, proto=socket.IPPROTO_TCP)
        for _, _, _, _, sockaddr in resolved:
            addr = ipaddress.ip_address(sockaddr[0])
            if addr.is_loopback:
                continue
            if getattr(addr, "ipv4_mapped", None) and addr.ipv4_mapped.is_loopback:
                continue
            return True
        return False
    except Exception:
        return True


def _openai_error(
    message: str,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


def _normalize_chat_content(
    content: Any, *, _max_depth: int = 10, _depth: int = 0,
) -> str:
    """Normalize OpenAI chat message content into a plain text string."""
    if _depth > _max_depth:
        return ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content

    if isinstance(content, list):
        parts: List[str] = []
        items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
        total_len = 0
        for item in items:
            added = ""
            if isinstance(item, str):
                if item:
                    added = item[:MAX_NORMALIZED_TEXT_LENGTH]
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text", "")
                    if text:
                        try:
                            added = str(text)[:MAX_NORMALIZED_TEXT_LENGTH]
                        except Exception:
                            pass
            elif isinstance(item, list):
                nested = _normalize_chat_content(item, _max_depth=_max_depth, _depth=_depth + 1)
                if nested:
                    added = nested
            if added:
                parts.append(added)
                total_len += len(added)
            if total_len >= MAX_NORMALIZED_TEXT_LENGTH:
                break
        result = "\n".join(parts)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result

    try:
        result = str(content)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result
    except Exception:
        return ""


def _normalize_multimodal_content(content: Any) -> Any:
    """Validate and normalize multimodal content for the API server."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content
    if not isinstance(content, list):
        return _normalize_chat_content(content)

    items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
    normalized_parts: List[Dict[str, Any]] = []
    text_accum_len = 0

    for part in items:
        if isinstance(part, str):
            if part:
                trimmed = part[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
                text_accum_len += len(trimmed)
            continue

        if not isinstance(part, dict):
            continue

        raw_type = part.get("type")
        part_type = str(raw_type or "").strip().lower()

        if part_type in _TEXT_PART_TYPES:
            text = part.get("text")
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            if text:
                trimmed = text[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
                text_accum_len += len(trimmed)
            continue

        if part_type in _IMAGE_PART_TYPES:
            detail = part.get("detail")
            image_ref = part.get("image_url")
            if isinstance(image_ref, dict):
                url_value = image_ref.get("url")
                detail = image_ref.get("detail", detail)
            else:
                url_value = image_ref
            if not isinstance(url_value, str) or not url_value.strip():
                raise ValueError("invalid_image_url:Image parts must include a non-empty image URL.")
            url_value = url_value.strip()
            lowered = url_value.lower()
            if lowered.startswith("data:"):
                if not lowered.startswith("data:image/") or "," not in url_value:
                    raise ValueError(
                        "unsupported_content_type:Only image data URLs are supported. "
                        "Non-image data payloads are not supported."
                    )
            elif not (lowered.startswith("http://") or lowered.startswith("https://")):
                raise ValueError(
                    "invalid_image_url:Image inputs must use http(s) URLs or data:image/... URLs."
                )
            image_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url_value}}
            if detail is not None:
                if not isinstance(detail, str) or not detail.strip():
                    raise ValueError("invalid_content_part:Image detail must be a non-empty string when provided.")
                image_part["image_url"]["detail"] = detail.strip()
            normalized_parts.append(image_part)
            continue

        if part_type in _FILE_PART_TYPES:
            raise ValueError(
                "unsupported_content_type:Inline image inputs are supported, "
                "but uploaded files and document inputs are not supported on this endpoint."
            )

        raise ValueError(
            f"unsupported_content_type:Unsupported content part type {raw_type!r}. "
            "Only text and image_url/input_image parts are supported."
        )

    if not normalized_parts:
        return ""

    if all(p.get("type") == "text" for p in normalized_parts):
        return "\n".join(p["text"] for p in normalized_parts if p.get("text"))

    return normalized_parts


def _content_has_visible_payload(content: Any) -> bool:
    """True when content has any text or image attachment."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                ptype = str(part.get("type") or "").strip().lower()
                if ptype in _TEXT_PART_TYPES and str(part.get("text") or "").strip():
                    return True
                if ptype in _IMAGE_PART_TYPES:
                    return True
    return False


def _multimodal_validation_error(exc: ValueError, *, param: str) -> "web.Response":
    """Translate a ``_normalize_multimodal_content`` ValueError into a 400 response."""
    raw = str(exc)
    code, _, message = raw.partition(":")
    if not message:
        code, message = "invalid_content_part", raw
    return web.json_response(
        _openai_error(message, code=code, param=param),
        status=400,
    )


def _coerce_model(body: Dict[str, Any], model_name: str) -> str:
    """Coerce the model field from the request body."""
    requested = body.get("model")
    if isinstance(requested, str) and requested.strip():
        return requested.strip()
    return model_name


def _derive_chat_session_id(
    system_prompt: Optional[str],
    first_user_message: str,
) -> str:
    """Derive a stable session ID from the conversation's first user message."""
    seed = f"{system_prompt or ''}\n{first_user_message}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"api-{digest}"


def _make_request_fingerprint(body: Dict[str, Any], keys: List[str]) -> str:
    """Stable hash for idempotency cache lookup."""
    subset = {k: body.get(k) for k in keys}
    return hashlib.sha256(json.dumps(subset, sort_keys=True, default=str).encode()).hexdigest()


def _build_user_content(
    text: str, attachments: Optional[List[Dict[str, Any]]] = None
) -> tuple:
    """Build multimodal content from text + image attachments.

    Returns (user_content, persist_text) where user_content is either
    a plain string or a list of content parts for multimodal input.
    """
    if not attachments:
        return text, text

    image_parts: List[Dict[str, Any]] = []
    for att in attachments:
        if not isinstance(att, dict):
            continue
        mime = ""
        for key in ("contentType", "mimeType", "mediaType"):
            val = att.get(key)
            if isinstance(val, str) and val.strip():
                mime = val.strip()
                break
        if not mime.startswith("image/"):
            continue
        content = ""
        for key in ("content", "base64", "data"):
            val = att.get(key)
            if isinstance(val, str) and val.strip():
                content = val.strip()
                break
        if not content:
            # Try dataUrl format: data:image/png;base64,...
            data_url = att.get("dataUrl", "")
            if isinstance(data_url, str) and data_url.startswith("data:"):
                content = data_url.split(",", 1)[-1] if "," in data_url else ""
        if not content:
            continue
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{content}"},
        })

    if not image_parts:
        return text, text

    content_parts: List[Dict[str, Any]] = []
    if text.strip():
        content_parts.append({"type": "text", "text": text})
    content_parts.extend(image_parts)
    return content_parts, text
