"""Tests for cogtrix_core/tools/_path_policy — canonical path-policy error strings."""

from cogtrix_core.tools._path_policy import (
    ERR_PATH_TRAVERSAL,
    ERR_READ_OUTSIDE_PERMITTED_AREA,
    ERR_WRITE_OUTSIDE_PERMITTED_AREA,
    format_read_outside_error,
    format_traversal_error,
    format_write_outside_error,
    is_path_policy_error,
)


class TestFormatters:
    """Each formatter substitutes the offending path into the canonical template."""

    def test_format_write_outside_includes_path(self):
        msg = format_write_outside_error("/etc/passwd")
        assert "/etc/passwd" in msg
        assert msg.startswith("Error: ")
        assert "outside the permitted write area" in msg

    def test_format_read_outside_includes_path(self):
        msg = format_read_outside_error("/secret/data")
        assert "/secret/data" in msg
        assert msg.startswith("Error: ")
        assert "outside the permitted read area" in msg

    def test_format_traversal_includes_path(self):
        msg = format_traversal_error("../../../../etc/shadow")
        assert "../../../../etc/shadow" in msg
        assert msg.startswith("Error: ")
        assert "traversal not allowed" in msg.lower()

    def test_formatter_accepts_path_object(self):
        """Pathlib Path is accepted (will be str-ified via {path} formatting)."""
        from pathlib import Path

        msg = format_write_outside_error(Path("/tmp/foo"))
        assert "/tmp/foo" in msg


class TestIsPathPolicyError:
    """Classifier for downstream consumers (dispatcher, test harness)."""

    def test_recognises_write_outside_error(self):
        msg = format_write_outside_error("/etc")
        assert is_path_policy_error(msg) is True

    def test_recognises_read_outside_error(self):
        msg = format_read_outside_error("/etc")
        assert is_path_policy_error(msg) is True

    def test_recognises_traversal_error(self):
        msg = format_traversal_error("../etc")
        assert is_path_policy_error(msg) is True

    def test_rejects_unrelated_error(self):
        """OS-level errors (PermissionError, file not found, …) are NOT
        path-policy errors.  Keeping them distinct lets the agent
        disambiguate 'wrong path' from 'right path but OS denied'."""
        assert is_path_policy_error("Error: Permission denied: /tmp/foo") is False
        assert is_path_policy_error("Error: File not found: /tmp/foo") is False
        assert is_path_policy_error("Some random string") is False

    def test_rejects_non_string(self):
        assert is_path_policy_error(None) is False  # type: ignore[arg-type]
        assert is_path_policy_error(42) is False  # type: ignore[arg-type]
        assert is_path_policy_error([]) is False  # type: ignore[arg-type]

    def test_classifier_anchored_on_canonical_phrases(self):
        """Trying to fool the classifier with the canonical *prefix*
        but unrelated content does not work — the classifier looks
        for the canonical phrases, not arbitrary 'Error: ...' prefixes."""
        assert is_path_policy_error("Error: not a path policy violation") is False


class TestCanonicalConstants:
    """The exported constants are templates with a single ``{path}``
    placeholder so callers can format consistently."""

    def test_write_template_has_path_placeholder(self):
        assert "{path}" in ERR_WRITE_OUTSIDE_PERMITTED_AREA

    def test_read_template_has_path_placeholder(self):
        assert "{path}" in ERR_READ_OUTSIDE_PERMITTED_AREA

    def test_traversal_template_has_path_placeholder(self):
        assert "{path}" in ERR_PATH_TRAVERSAL

    def test_all_templates_have_error_prefix(self):
        """All canonical messages start with ``Error: `` so the existing
        ``_TOOL_ERROR_MARKERS`` allowlist in the Gate 2 harness still
        classifies them as tool errors."""
        for template in (
            ERR_WRITE_OUTSIDE_PERMITTED_AREA,
            ERR_READ_OUTSIDE_PERMITTED_AREA,
            ERR_PATH_TRAVERSAL,
        ):
            assert template.startswith("Error: ")
