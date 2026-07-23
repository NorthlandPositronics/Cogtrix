"""
HTTP request tool - Make HTTP GET and POST requests.
POST requests require user confirmation for safety.
"""

import json
import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field

# Try to import requests
try:
    import requests  # type: ignore[import-untyped]

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False


class HttpGetInput(BaseModel):
    """Input schema for HTTP GET requests."""

    url: str = Field(description="The URL to request")
    headers: str | None = Field(
        default=None,
        description='Optional headers as JSON string (e.g., \'{"Authorization": "Bearer token"}\')',
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")


class HttpPostInput(BaseModel):
    """Input schema for HTTP POST requests."""

    url: str = Field(description="The URL to request")
    data: str = Field(description='Request body as JSON string (e.g., \'{"key": "value"}\')')
    headers: str | None = Field(
        default=None,
        description="Optional headers as JSON string",
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL for safety."""
    try:
        parsed = urlparse(url)

        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format"

        # Only allow http and https
        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported scheme: {parsed.scheme}"

        # Block localhost/internal IPs (basic SSRF protection)
        hostname = parsed.hostname or ""
        blocked_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]  # nosec B104
        if hostname in blocked_hosts:
            return False, "Requests to localhost are not allowed"

        # Block private IP ranges (RFC 1918)
        if hostname.startswith("10.") or hostname.startswith("192.168."):
            return False, "Requests to private IP ranges are not allowed"
        # 172.16.0.0/12 covers 172.16.* through 172.31.*
        if hostname.startswith("172."):
            parts = hostname.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                second_octet = int(parts[1])
                if 16 <= second_octet <= 31:
                    return False, "Requests to private IP ranges are not allowed"

        return True, ""
    except Exception as e:
        return False, f"URL validation error: {e}"


def _parse_headers(headers_str: str | None) -> tuple[dict, str | None]:
    """Parse headers JSON string."""
    if not headers_str:
        return {}, None

    try:
        headers = json.loads(headers_str)
        if not isinstance(headers, dict):
            return {}, "Headers must be a JSON object"
        return headers, None
    except json.JSONDecodeError as e:
        return {}, f"Invalid headers JSON: {e}"


def _truncate_response(text: str, max_length: int = 10000) -> str:
    """Truncate long responses."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n\n... (truncated, {len(text)} total characters)"


def _extract_text_from_html(html: str) -> str:
    """
    Extract readable text from HTML, stripping tags and noise.

    Removes script/style blocks, HTML comments, and tags, then
    collapses whitespace into a clean text representation that is
    actually useful for an LLM (as opposed to raw markup).
    """
    # Remove script and style elements entirely
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # Remove SVG elements (often huge and useless)
    text = re.sub(r"<svg[^>]*>.*?</svg>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Insert newline before block-level elements for readability
    text = re.sub(
        r"<(?:p|div|br|h[1-6]|li|tr|section|article)[^>]*>", "\n", text, flags=re.IGNORECASE
    )
    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    for entity, char in (
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&nbsp;", " "),
        ("&#x27;", "'"),
        ("&ndash;", "\u2013"),
        ("&mdash;", "\u2014"),
    ):
        text = text.replace(entity, char)
    # Collapse runs of whitespace (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def http_get(url: str, headers: str | None = None, timeout: int = 30) -> str:
    """
    Make an HTTP GET request.

    Args:
        url: The URL to request
        headers: Optional headers as JSON string
        timeout: Request timeout in seconds

    Returns:
        Response body or error message
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Install it with: pip install requests"

    # Validate URL
    is_valid, error = _validate_url(url)
    if not is_valid:
        return f"Error: {error}"

    # Parse headers
    parsed_headers, header_error = _parse_headers(headers)
    if header_error:
        return f"Error: {header_error}"

    try:
        response = requests.get(
            url,
            headers=parsed_headers,
            timeout=timeout,
            allow_redirects=True,
        )

        # Build response info
        result = []
        result.append(f"Status: {response.status_code} {response.reason}")
        result.append(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        result.append(f"Content-Length: {len(response.content)} bytes")
        result.append("")

        # Try to parse as JSON for better formatting
        try:
            json_data = response.json()
            result.append("Response (JSON):")
            result.append(json.dumps(json_data, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, ValueError):
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                # Extract readable text from HTML (raw markup is useless for the LLM)
                extracted = _extract_text_from_html(response.text)
                result.append("Response (text extracted from HTML):")
                result.append(_truncate_response(extracted))
            else:
                result.append("Response:")
                result.append(_truncate_response(response.text))

        return "\n".join(result)

    except requests.exceptions.Timeout:
        return f"Error: Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError as e:
        return f"Error: Connection failed - {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {e}"
    except Exception as e:
        return f"Error: {e}"


def http_post(
    url: str,
    data: str,
    headers: str | None = None,
    timeout: int = 30,
) -> str:
    """
    Make an HTTP POST request.
    WARNING: This sends data to external servers. Use with caution.

    Args:
        url: The URL to request
        data: Request body as JSON string
        headers: Optional headers as JSON string
        timeout: Request timeout in seconds

    Returns:
        Response body or error message
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Install it with: pip install requests"

    # Validate URL
    is_valid, error = _validate_url(url)
    if not is_valid:
        return f"Error: {error}"

    # Parse headers
    parsed_headers, header_error = _parse_headers(headers)
    if header_error:
        return f"Error: {header_error}"

    # Parse request data
    try:
        json_data = json.loads(data)
    except json.JSONDecodeError as e:
        return f"Error: Invalid request data JSON: {e}"

    # Set Content-Type if not provided
    if "Content-Type" not in parsed_headers and "content-type" not in parsed_headers:
        parsed_headers["Content-Type"] = "application/json"

    try:
        response = requests.post(
            url,
            json=json_data,
            headers=parsed_headers,
            timeout=timeout,
            allow_redirects=True,
        )

        # Build response info
        result = []
        result.append(f"Status: {response.status_code} {response.reason}")
        result.append(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        result.append(f"Content-Length: {len(response.content)} bytes")
        result.append("")

        # Try to parse as JSON for better formatting
        try:
            json_response = response.json()
            result.append("Response (JSON):")
            result.append(json.dumps(json_response, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, ValueError):
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                extracted = _extract_text_from_html(response.text)
                result.append("Response (text extracted from HTML):")
                result.append(_truncate_response(extracted))
            else:
                result.append("Response:")
                result.append(_truncate_response(response.text))

        return "\n".join(result)

    except requests.exceptions.Timeout:
        return f"Error: Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError as e:
        return f"Error: Connection failed - {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {e}"
    except Exception as e:
        return f"Error: {e}"


# Tool configurations for registry
TOOL_CONFIGS = [
    {
        "name": "http_get",
        "description": (
            "Make an HTTP GET request to a URL. " "Returns status, headers, and response body."
        ),
        "input_schema": HttpGetInput,
        "requires_confirmation": False,
        "function": http_get,
    },
    {
        "name": "http_post",
        "description": (
            "Make an HTTP POST request with JSON data. "
            "WARNING: This sends data to external servers."
        ),
        "input_schema": HttpPostInput,
        "requires_confirmation": True,  # Requires confirmation for safety
        "function": http_post,
    },
]

# Default single tool config (for backwards compatibility)
TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "http_get",
    "http_post",
    "HttpGetInput",
    "HttpPostInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
