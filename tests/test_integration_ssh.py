"""Integration test: validate SSH connectivity and basic compilation on remote RISC-V host."""

import subprocess

import pytest

from src.config import SSH_HOST, REMOTE_DIR


SSH_TIMEOUT = 30


def _ssh_available() -> bool:
    """Check whether the remote RISC-V host is reachable."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", SSH_HOST, "echo ok"],
            capture_output=True,
            timeout=10,
            text=True,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ssh_available(),
    reason=f"SSH host '{SSH_HOST}' is not reachable",
)


class TestSSHConnectivity:
    """Basic SSH connectivity and toolchain tests."""

    def test_ssh_echo(self):
        """The remote host responds to a simple echo command."""
        result = subprocess.run(
            ["ssh", SSH_HOST, "echo hello"],
            capture_output=True,
            timeout=SSH_TIMEOUT,
            text=True,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_compiler_available(self):
        """A C++ compiler is available on the remote host."""
        result = subprocess.run(
            ["ssh", SSH_HOST, "which clang++ || which g++"],
            capture_output=True,
            timeout=SSH_TIMEOUT,
            text=True,
        )
        assert result.returncode == 0, "No C++ compiler found on remote host"

    def test_remote_dir_writable(self):
        """We can create the remote working directory."""
        result = subprocess.run(
            ["ssh", SSH_HOST, f"mkdir -p {REMOTE_DIR} && echo ok"],
            capture_output=True,
            timeout=SSH_TIMEOUT,
            text=True,
        )
        assert result.returncode == 0
        assert "ok" in result.stdout
