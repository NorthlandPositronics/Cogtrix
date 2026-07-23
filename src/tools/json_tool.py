"""
JSON operations tool - Parse, query, format, and manipulate JSON data.
"""

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class ParseJsonInput(BaseModel):
    """Input schema for parsing JSON."""

    json_str: str = Field(description="The JSON string to parse and validate")


class FormatJsonInput(BaseModel):
    """Input schema for formatting JSON."""

    json_str: str = Field(description="The JSON string to format")
    indent: int = Field(default=2, description="Number of spaces for indentation")


class QueryJsonInput(BaseModel):
    """Input schema for querying JSON."""

    json_str: str = Field(description="The JSON string to query")
    path: str = Field(description="Path to query (e.g., 'data.users[0].name' or 'items[*].price')")


class ExtractJsonInput(BaseModel):
    """Input schema for extracting JSON from text."""

    text: str = Field(description="Text that may contain JSON")


class JsonToTextInput(BaseModel):
    """Input schema for converting JSON to readable text."""

    json_str: str = Field(description="The JSON string to convert")


def parse_json(json_str: str) -> str:
    """
    Parse and validate a JSON string.

    Args:
        json_str: The JSON string to parse

    Returns:
        Confirmation message with structure info or error
    """
    try:
        data = json.loads(json_str)

        # Describe the structure
        def describe(obj, depth=0):
            if depth > 3:
                return "..."
            if isinstance(obj, dict):
                if not obj:
                    return "empty object"
                keys = list(obj.keys())[:5]
                more = f" (+{len(obj) - 5} more)" if len(obj) > 5 else ""
                return f"object with keys: {keys}{more}"
            elif isinstance(obj, list):
                if not obj:
                    return "empty array"
                return f"array of {len(obj)} items"
            elif isinstance(obj, str):
                return f'string: "{obj[:50]}..."' if len(obj) > 50 else f'string: "{obj}"'
            elif isinstance(obj, bool):
                return f"boolean: {obj}"
            elif isinstance(obj, (int, float)):
                return f"number: {obj}"
            elif obj is None:
                return "null"
            return str(type(obj).__name__)

        structure = describe(data)
        return f"Valid JSON: {structure}"

    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e.msg} at position {e.pos}"
    except Exception as e:
        return f"Error parsing JSON: {e}"


def format_json(json_str: str, indent: int = 2) -> str:
    """
    Format/pretty-print a JSON string.

    Args:
        json_str: The JSON string to format
        indent: Number of spaces for indentation

    Returns:
        Formatted JSON string or error
    """
    try:
        data = json.loads(json_str)
        return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=False)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e.msg} at position {e.pos}"
    except Exception as e:
        return f"Error formatting JSON: {e}"


def _get_by_path(data: Any, path: str) -> Any:
    """
    Get value from nested structure by path.
    Supports: dot notation, array indexing, wildcard [*]
    """
    if not path:
        return data

    parts = [p for p in re.split(r"\.(?![^\[]*\])", path) if p]
    if not parts and path.strip():
        raise KeyError(f"Invalid path: {path!r}")
    return _get_by_parts(data, parts)


def _get_by_parts(current: Any, parts: list[str]) -> Any:
    """Recursively traverse a parsed path parts list."""
    if not parts:
        return current

    part = parts[0]
    remaining = parts[1:]

    # Extract the key and all bracket expressions
    # e.g., "items[*][0]" -> key="items", indices=["*","0"]
    key_match = re.match(r"(\w+)((?:\[\d+\]|\[\*\])*)", part)
    if not key_match or key_match.end() != len(part):
        raise KeyError(f"Invalid path part: {part}")

    key = key_match.group(1)
    indices = re.findall(r"\[(\d+|\*)\]", key_match.group(2))

    if isinstance(current, dict):
        if key not in current:
            raise KeyError(f"Key not found: {key}")
        current = current[key]
    elif isinstance(current, list) and key.isdigit():
        idx = int(key)
        if idx >= len(current):
            raise IndexError(f"Index out of range: {idx}")
        current = current[idx]
    else:
        raise KeyError(f"Cannot access '{key}' on {type(current).__name__}")

    return _apply_bracket_indices(current, indices, remaining)


def _apply_bracket_indices(current: Any, indices: list[str], remaining: list[str]) -> Any:
    """Apply a sequence of bracket indices, then continue with remaining path parts."""
    if not indices:
        return _get_by_parts(current, remaining)

    index = indices[0]
    rest_indices = indices[1:]

    if not isinstance(current, list):
        raise TypeError(f"Cannot index non-array: {type(current).__name__}")

    if index == "*":
        return [_apply_bracket_indices(item, rest_indices, remaining) for item in current]
    else:
        idx = int(index)
        if idx >= len(current):
            raise IndexError(f"Index out of range: {idx}")
        return _apply_bracket_indices(current[idx], rest_indices, remaining)


def query_json(json_str: str, path: str) -> str:
    """
    Query a JSON structure using a path expression.

    Supports:
    - Dot notation: data.users.name
    - Array indexing: items[0], items[2]
    - Wildcard: items[*], items[*].price
    - Chained brackets: items[*][0], matrix[0][*]
    - Nested paths: data.users[0].name

    Args:
        json_str: The JSON string to query
        path: Path expression (e.g., 'data.users[0].name')

    Returns:
        Query result or error
    """
    try:
        data = json.loads(json_str)
        result = _get_by_path(data, path)

        # Format the result
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, ensure_ascii=False)
        elif result is None:
            return "null"
        else:
            return str(result)

    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e.msg}"
    except (KeyError, IndexError, TypeError) as e:
        return f"Query error: {e}"
    except Exception as e:
        return f"Error querying JSON: {e}"


def extract_json(text: str) -> str:
    """
    Extract JSON object(s) from text that may contain other content.

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted JSON or error message
    """
    try:
        # Try to find JSON objects or arrays in the text
        results = []

        # First try to parse the whole text
        try:
            data = json.loads(text.strip())
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code blocks
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                data = json.loads(match.strip())
                results.append(json.dumps(data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                pass

        if results:
            return "\n---\n".join(results)

        # Try to find JSON-like structures
        # Look for { ... } or [ ... ] patterns
        brace_start = text.find("{")
        bracket_start = text.find("[")

        if brace_start == -1 and bracket_start == -1:
            return "No JSON found in text"

        # Try from the first { or [
        start = min(
            brace_start if brace_start != -1 else len(text),
            bracket_start if bracket_start != -1 else len(text),
        )

        # Only try substrings ending at closing-bracket positions (O(k) vs O(n))
        closing = [i + 1 for i, ch in enumerate(text) if ch in ("}", "]") and i >= start]
        for end in closing:
            try:
                candidate = text[start:end]
                data = json.loads(candidate)
                results.append(json.dumps(data, indent=2, ensure_ascii=False))
                break
            except json.JSONDecodeError:
                continue

        if results:
            return results[0]

        return "No valid JSON found in text"

    except Exception as e:
        return f"Error extracting JSON: {e}"


def json_to_text(json_str: str) -> str:
    """
    Convert JSON to human-readable text format.

    Args:
        json_str: The JSON string to convert

    Returns:
        Human-readable text representation
    """

    def _to_text(obj, prefix="", depth: int = 0, max_depth: int = 50):
        if depth > max_depth:
            return [f"{prefix}... (max depth {max_depth} exceeded)"]
        lines = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(_to_text(value, prefix + "  ", depth + 1, max_depth))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i + 1}]:")
                    lines.extend(_to_text(item, prefix + "  ", depth + 1, max_depth))
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{obj}")
        return lines

    try:
        data = json.loads(json_str)
        return "\n".join(_to_text(data))
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e.msg}"
    except Exception as e:
        return f"Error converting JSON: {e}"


# Tool configurations for registry
TOOL_CONFIGS = [
    {
        "name": "parse_json",
        "description": "Parse and validate a JSON string. Returns structure information if valid.",
        "input_schema": ParseJsonInput,
        "requires_confirmation": False,
        "function": parse_json,
    },
    {
        "name": "format_json",
        "description": "Format/pretty-print a JSON string with proper indentation.",
        "input_schema": FormatJsonInput,
        "requires_confirmation": False,
        "function": format_json,
    },
    {
        "name": "query_json",
        "description": "Query a JSON structure using path expressions like 'data.users[0].name'.",
        "input_schema": QueryJsonInput,
        "requires_confirmation": False,
        "function": query_json,
    },
    {
        "name": "extract_json",
        "description": (
            "Extract JSON from text that may contain other content " "(like markdown or prose)."
        ),
        "input_schema": ExtractJsonInput,
        "requires_confirmation": False,
        "function": extract_json,
    },
    {
        "name": "json_to_text",
        "description": "Convert JSON to human-readable text format.",
        "input_schema": JsonToTextInput,
        "requires_confirmation": False,
        "function": json_to_text,
    },
]

# Default single tool config (for backwards compatibility)
TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "parse_json",
    "format_json",
    "query_json",
    "extract_json",
    "json_to_text",
    "ParseJsonInput",
    "FormatJsonInput",
    "QueryJsonInput",
    "ExtractJsonInput",
    "JsonToTextInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
