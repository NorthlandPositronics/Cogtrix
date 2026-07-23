"""Tests for symlink attack vectors in file_ops tools."""

from pathlib import Path

import pytest

from src.tools.file_ops import (
    list_directory,
    read_file,
    set_allowed_write_dirs,
    write_file,
)


@pytest.fixture()
def tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Change cwd to a temp directory so _validate_path allows paths inside it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestSymlinkAttackVectors:
    """Verify file_ops tools block symlink attack vectors."""

    def test_symlink_to_etc_passwd_read_rejected(self, tmp_cwd: Path) -> None:
        """Creating a symlink to /etc/passwd and reading through it should be rejected."""
        # Create a symlink pointing to /etc/passwd
        symlink = tmp_cwd / "symlink_to_passwd"
        try:
            symlink.symlink_to("/etc/passwd")
        except PermissionError:
            # Skip if we don't have permission to create symlinks
            pytest.skip("Cannot create symlinks without permission")

        result = read_file(str(symlink))
        assert result.startswith("Error:")

    def test_symlink_to_outside_dir_write_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Creating a symlink to an outside directory and writing through it should be rejected."""
        work_dir = tmp_path / "work"
        outside_dir = tmp_path / "outside"
        work_dir.mkdir()
        outside_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # Create a symlink pointing outside the working directory
        symlink = work_dir / "symlink_outside"
        try:
            symlink.symlink_to(outside_dir)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        # Try to write through the symlink - should fail because outside_dir is not in cwd
        result = write_file(str(symlink / "evil.txt"), "bad data")
        assert result.startswith("Error:")

    def test_dangling_symlink_then_swap_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Creating a dangling symlink then swapping it with a sensitive file should be rejected."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # Create a dangling symlink first
        symlink = work_dir / "symlink_to_swap"
        try:
            symlink.symlink_to("/nonexistent/file")
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        # Try to read through the dangling symlink
        result = read_file(str(symlink))
        assert result.startswith("Error:")

        # Create a real file in the work directory
        real_target = work_dir / "real_target.txt"
        real_target.write_text("legitimate content")

        # Now swap the symlink target
        symlink.unlink()
        symlink.symlink_to(real_target)

        # Try to read again - should succeed because the target is within cwd
        result = read_file(str(symlink))
        assert result == "legitimate content"

    def test_symlink_chain_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Creating a symlink chain escaping cwd should be rejected."""
        work_dir = tmp_path / "work"
        outside_dir = tmp_path / "outside"
        work_dir.mkdir()
        outside_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # Create intermediate symlinks - chain goes through outside_dir
        intermediate = work_dir / "intermediate"
        try:
            intermediate.symlink_to(outside_dir)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        chain = work_dir / "chain"
        try:
            chain.symlink_to(intermediate)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        # Try to read through the chain - should be rejected because target escapes cwd
        result = read_file(str(chain / "file.txt"))
        assert result.startswith("Error:")

    def test_symlink_to_dot_env_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading a .env file through a symlink should be rejected."""
        work_dir = tmp_path / "work"
        env_dir = tmp_path / "envdir"
        work_dir.mkdir()
        env_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # Create a .env file in a subdirectory
        env_file = env_dir / ".env"
        env_file.write_text("SECRET_KEY=supersecret")

        # Create a symlink to .env
        symlink = work_dir / "symlink_to_env"
        try:
            symlink.symlink_to(env_file)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        result = read_file(str(symlink))
        assert result.startswith("Error:")

    def test_symlink_to_dot_git_config_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading a .git/config file through a symlink should be rejected."""
        work_dir = tmp_path / "work"
        git_dir = tmp_path / "gitdir"
        work_dir.mkdir()
        git_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # Create a .git/config file in a subdirectory
        git_config = git_dir / ".git"
        git_config.mkdir()
        config_file = git_config / "config"
        config_file.write_text("[core]\n    bare = false")

        # Create a symlink to .git/config
        symlink = work_dir / "symlink_to_git_config"
        try:
            symlink.symlink_to(config_file)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        result = read_file(str(symlink))
        assert result.startswith("Error:")

    def test_symlink_to_ssh_key_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading an SSH key through a symlink should be rejected."""
        work_dir = tmp_path / "work"
        ssh_dir = tmp_path / "sshdir"
        work_dir.mkdir()
        ssh_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # Create an SSH key file in a subdirectory
        ssh_key = ssh_dir / "id_rsa"
        ssh_key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----")

        # Create a symlink to the SSH key
        symlink = work_dir / "symlink_to_ssh"
        try:
            symlink.symlink_to(ssh_key)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        result = read_file(str(symlink))
        assert result.startswith("Error:")

    def test_symlink_to_proc_self_environ_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading /proc/self/environ through a symlink should be rejected."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # /proc/self/environ usually requires special permissions, but we test the path validation
        symlink = work_dir / "symlink_to_proc"
        try:
            symlink.symlink_to("/proc/self/environ")
        except (PermissionError, FileNotFoundError):
            # /proc/self/environ may not exist or be accessible
            pytest.skip("/proc/self/environ not accessible or symlink creation failed")

        result = read_file(str(symlink))
        assert result.startswith("Error:")


class TestSymlinkAttackWritePaths:
    """Verify file_ops tools block symlink attacks when using extra write dirs."""

    @pytest.fixture(autouse=True)
    def _cleanup_extra_dirs(self) -> None:
        """Reset extra write dirs after each test."""
        set_allowed_write_dirs(None)

    def test_symlink_outside_extra_dir_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Creating a symlink that escapes the extra write dir should be rejected."""
        work_dir = tmp_path / "work"
        extra_dir = tmp_path / "extra"
        outside_dir = tmp_path / "outside"
        work_dir.mkdir()
        extra_dir.mkdir()
        outside_dir.mkdir()
        monkeypatch.chdir(work_dir)
        set_allowed_write_dirs([str(extra_dir)])

        # Create a symlink in extra_dir pointing outside
        symlink = extra_dir / "symlink_outside"
        try:
            symlink.symlink_to(outside_dir)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        # Try to write through the symlink
        result = write_file(str(symlink / "evil.txt"), "bad data")
        assert result.startswith("Error:")

    def test_symlink_chain_escaping_extra_dir_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Creating a symlink chain escaping the extra write dir should be rejected."""
        work_dir = tmp_path / "work"
        extra_dir = tmp_path / "extra"
        outside_dir = tmp_path / "outside"
        work_dir.mkdir()
        extra_dir.mkdir()
        outside_dir.mkdir()
        monkeypatch.chdir(work_dir)
        set_allowed_write_dirs([str(extra_dir)])

        # Create a chain of symlinks in extra_dir that points to outside_dir
        inner = extra_dir / "inner"
        inner.mkdir()

        chain1 = extra_dir / "chain1"
        chain2 = extra_dir / "chain2"
        try:
            chain1.symlink_to(outside_dir)
            chain2.symlink_to(chain1)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        # Try to write through the chain - should fail because target escapes extra_dir
        result = write_file(str(chain2 / "evil.txt"), "bad data")
        assert result.startswith("Error:")


class TestSymlinkListDirectory:
    """Verify list_directory handles symlinks safely."""

    def test_symlink_in_directory_listing_hidden(self, tmp_cwd: Path) -> None:
        """list_directory should filter out symlinks entirely."""
        real_file = tmp_cwd / "real.txt"
        real_file.write_text("content")

        symlink = tmp_cwd / "symlink"
        try:
            symlink.symlink_to(real_file)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        result = list_directory(".")

        assert "real.txt" in result
        assert "symlink" not in result

    def test_symlink_to_directory_in_listing_hidden(self, tmp_cwd: Path) -> None:
        """list_directory should hide symlinks pointing to directories."""
        real_dir = tmp_cwd / "realdir"
        real_dir.mkdir()
        (real_dir / "file.txt").write_text("content")

        symlink = tmp_cwd / "symlink_to_dir"
        try:
            symlink.symlink_to(real_dir)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        result = list_directory(".")

        assert "realdir" in result
        assert "symlink_to_dir" not in result

    def test_internal_symlink_does_not_leak_children(self, tmp_cwd: Path) -> None:
        """Symlink within allowed root must not leak children into listing."""
        real_dir = tmp_cwd / "subdir"
        real_dir.mkdir()
        (real_dir / "secret.txt").write_text("secret")

        symlink = tmp_cwd / "link_to_subdir"
        try:
            symlink.symlink_to(real_dir)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        result = list_directory(".")

        # real_dir itself should appear
        assert "subdir" in result
        # symlink itself should not appear
        assert "link_to_subdir" not in result
        # symlink children must not leak
        assert "secret.txt" not in result

    def test_symlink_loop_does_not_crash(self, tmp_cwd: Path) -> None:
        """Symlink loop within allowed root must not cause unhandled exception."""
        loop = tmp_cwd / "loop"
        try:
            loop.symlink_to(tmp_cwd)
        except PermissionError:
            pytest.skip("Cannot create symlinks without permission")

        # Should return normally, not crash with RuntimeError
        result = list_directory(".")
        assert "Contents of" in result or "empty or no files match" in result


class TestSymlinkNullByteInjection:
    """Verify file_ops tools block null byte injection in paths."""

    def test_null_byte_in_path_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Paths containing null bytes should be rejected."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)

        # Create a real file
        real_file = work_dir / "real.txt"
        real_file.write_text("content")

        # Try to inject null byte
        result = read_file("real.txt\x00.jpg")
        assert result.startswith("Error:")
