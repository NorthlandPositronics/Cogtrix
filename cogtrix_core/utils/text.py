"""Text utilities shared across Cogtrix modules.

These helpers have no layer dependencies and can be imported from any
package without creating circular or bidirectional coupling.
"""

_FALLBACK_MAX_CHARS = 30_000


def truncate_tool_output(text: str, max_chars: int) -> str:
    """Middle-truncate *text* if it exceeds *max_chars*."""
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    removed = len(text) - max_chars
    return (
        text[:keep] + f"\n\n[... {removed:,} chars truncated to fit context budget — "
        f"use start_line/max_lines to page through, or search to "
        f"find specific sections ...]\n\n" + text[-keep:]
    )
