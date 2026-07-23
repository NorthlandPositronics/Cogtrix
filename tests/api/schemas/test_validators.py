"""Tests for src/api/schemas/validators.py:validate_password_complexity()."""

from __future__ import annotations

import pytest

from src.api.schemas.validators import validate_password_complexity


class TestPasswordComplexityValid:
    """Valid passwords should pass without raising."""

    def test_password_complexity_valid_password(self) -> None:
        """Password with all 4 character classes passes."""
        result = validate_password_complexity("P@ssw0rd")
        assert result == "P@ssw0rd"

    def test_password_complexity_minimal_valid(self) -> None:
        """Minimal valid password (8 chars, one of each class)."""
        result = validate_password_complexity("A1b@")
        assert result == "A1b@"


class TestPasswordComplexityMissingLowercase:
    """Missing lowercase letter should raise ValueError."""

    def test_password_complexity_missing_lowercase_all_uppercase(self) -> None:
        """All uppercase password raises."""
        with pytest.raises(ValueError, match="lowercase"):
            validate_password_complexity("PASSW0RD!")

    def test_password_complexity_missing_lowercase_all_digits_special(self) -> None:
        """Digits + special only raises (fails lowercase check first)."""
        with pytest.raises(ValueError, match="lowercase"):
            validate_password_complexity("1234567890!@#$%^&*")

    def test_password_complexity_missing_lowercase_single_upper(self) -> None:
        """Single uppercase raises."""
        with pytest.raises(ValueError, match="lowercase"):
            validate_password_complexity("A1!")


class TestPasswordComplexityMissingUppercase:
    """Missing uppercase letter should raise ValueError."""

    def test_password_complexity_missing_uppercase_all_lowercase(self) -> None:
        """All lowercase password raises."""
        with pytest.raises(ValueError, match="uppercase"):
            validate_password_complexity("passw0rd!")

    def test_password_complexity_missing_uppercase_all_digits_special(self) -> None:
        """Digits + special only raises (fails uppercase check - lower is already present)."""
        with pytest.raises(ValueError, match="uppercase"):
            validate_password_complexity("a1234567890!@#$%^&*")

    def test_password_complexity_missing_uppercase_single_lower(self) -> None:
        """Single lowercase raises (fails uppercase check)."""
        with pytest.raises(ValueError, match="uppercase"):
            validate_password_complexity("a1!")


class TestPasswordComplexityMissingDigit:
    """Missing digit should raise ValueError."""

    def test_password_complexity_missing_digit_letters_special_only(self) -> None:
        """Letters + special only raises."""
        with pytest.raises(ValueError, match="digit"):
            validate_password_complexity("Password!")

    def test_password_complexity_missing_digit_all_letters(self) -> None:
        """All letters raises (fails digit check first)."""
        with pytest.raises(ValueError, match="digit"):
            validate_password_complexity("Password")

    def test_password_complexity_missing_digit_single_char_class(self) -> None:
        """Single char class raises."""
        with pytest.raises(ValueError, match="digit"):
            validate_password_complexity("Aa!")


class TestPasswordComplexityMissingSpecial:
    """Missing special character should raise ValueError."""

    def test_password_complexity_missing_special_letters_digits_only(self) -> None:
        """Letters + digits only raises."""
        with pytest.raises(ValueError, match="special"):
            validate_password_complexity("Password123")

    def test_password_complexity_missing_special_single_char_class(self) -> None:
        """Single char class raises (fails special check)."""
        with pytest.raises(ValueError, match="special"):
            validate_password_complexity("Aa1")


class TestPasswordComplexityEmptyString:
    """Empty string should raise ValueError."""

    def test_password_complexity_empty_string(self) -> None:
        """Empty string raises (fails all 4 checks)."""
        with pytest.raises(ValueError, match="lowercase"):
            validate_password_complexity("")


class TestPasswordComplexitySpecialChars:
    """Special character edge cases."""

    def test_password_complexity_punctuation_only(self) -> None:
        """All punctuation chars pass."""
        result = validate_password_complexity("Aa1!@#$%^&*()_+-=[]{}|;:',.<>?")
        assert result == "Aa1!@#$%^&*()_+-=[]{}|;:',.<>?"

    def test_password_complexity_multiple_special(self) -> None:
        """Multiple different special chars pass."""
        result = validate_password_complexity("Aa1!@#$%^&*")
        assert result == "Aa1!@#$%^&*"

    def test_password_complexity_space_special(self) -> None:
        """Space as special char passes."""
        result = validate_password_complexity("Aa1 ")
        assert result == "Aa1 "

    def test_password_complexity_unicode_special_chars(self) -> None:
        """Non-ASCII special chars pass."""
        result = validate_password_complexity("Aa1€")
        assert result == "Aa1€"
        result = validate_password_complexity("Aa1漢")
        assert result == "Aa1漢"
        result = validate_password_complexity("Aa1ñ")
        assert result == "Aa1ñ"
