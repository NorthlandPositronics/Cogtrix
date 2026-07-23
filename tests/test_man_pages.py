"""Smoke tests for man page infrastructure."""

import subprocess
import sys
from pathlib import Path


def test_man_dir_exists():
    assert Path("man").is_dir()
    assert Path("man/man5").is_dir()
    assert Path("man/man8").is_dir()


def test_hand_written_man5_pages_exist():
    assert Path("man/man5/cogtrix.5").exists()
    assert Path("man/man5/cogtrix-api.5").exists()


def test_hand_written_man8_pages_exist():
    assert Path("man/man8/cogtrix.8").exists()
    assert Path("man/man8/cogtrix-api.8").exists()


def test_man5_cogtrix_has_required_sections():
    content = Path("man/man5/cogtrix.5").read_text()
    assert ".TH COGTRIX 5" in content
    assert "ANTHROPIC_API_KEY" in content
    assert "services" in content


def test_man5_api_has_required_sections():
    content = Path("man/man5/cogtrix-api.5").read_text()
    assert ".TH COGTRIX-API 5" in content
    assert "DATABASE_URL" in content


def test_man8_cogtrix_has_required_sections():
    content = Path("man/man8/cogtrix.8").read_text()
    assert ".TH COGTRIX 8" in content


def test_man8_api_has_required_sections():
    content = Path("man/man8/cogtrix-api.8").read_text()
    assert ".TH COGTRIX-API 8" in content
    assert "systemd" in content


def test_build_parser_importable_from_cli_args():
    """argparse-manpage requires build_parser() to be importable."""
    from cogtrix_core.cli.args import build_parser

    parser = build_parser()
    assert parser is not None
    # Verify --version is present
    actions = {a.option_strings[0] for a in parser._actions if a.option_strings}
    assert "--version" in actions


def test_build_parser_importable_from_api_main():
    """argparse-manpage requires build_parser() to be importable."""
    from cogtrix_core.api.__main__ import build_parser

    parser = build_parser()
    assert parser is not None


def _argparse_manpage_cmd() -> list[str]:
    """Return the argparse-manpage command, preferring the venv binary."""
    import shutil

    binary = shutil.which("argparse-manpage")
    if binary:
        return [binary]
    return [sys.executable, "-c", "from argparse_manpage.cli import main; main()"]


def _subprocess_env() -> dict:
    """Return env with project root on PYTHONPATH so src.* imports resolve."""
    import os

    env = os.environ.copy()
    root = str(Path(__file__).parent.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}:{existing}" if existing else root
    return env


def test_argparse_manpage_generates_cogtrix1(tmp_path):
    """argparse-manpage can generate cogtrix(1) without error."""
    result = subprocess.run(
        _argparse_manpage_cmd()
        + [
            "--pyfile",
            "cogtrix_core/cli/args.py",
            "--function",
            "build_parser",
            "--project-name",
            "cogtrix",
            "--prog",
            "cogtrix",
            "--author",
            "Northland Positronics",
            "--manual-title",
            "Cogtrix Manual",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, f"argparse_manpage failed: {result.stderr}"
    assert ".TH" in result.stdout


def test_argparse_manpage_generates_cogtrix_api1(tmp_path):
    """argparse-manpage can generate cogtrix-api(1) without error."""
    result = subprocess.run(
        _argparse_manpage_cmd()
        + [
            "--pyfile",
            "cogtrix_core/api/__main__.py",
            "--function",
            "build_parser",
            "--project-name",
            "cogtrix",
            "--prog",
            "cogtrix-api",
            "--author",
            "Northland Positronics",
            "--manual-title",
            "Cogtrix Manual",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, f"argparse_manpage failed: {result.stderr}"
    assert ".TH" in result.stdout
