"""Generic compile-and-run validators for the SSE→RISC-V translation pipeline.

Two validators:
  - DockerValidator: compile + run inside Docker with QEMU emulation.
  - SSHValidator:    compile + run on real RISC-V hardware via SSH.

Both accept *commands* as parameters so they are not tied to any specific library.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    DOCKER_IMAGE,
    REMOTE_DIR,
    SSH_HOST,
    VALIDATION_TIMEOUT_SECONDS,
)
from src.logger import get_logger

logger = get_logger(__name__)

MAX_OUTPUT_CHARS = 16000


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    stage: str
    returncode: int | None
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        parts = [
            part.strip() for part in (self.stdout, self.stderr) if part and part.strip()
        ]
        return "\n".join(parts)

    def as_feedback(self, limit: int = MAX_OUTPUT_CHARS) -> str:
        details = self.combined_output or "No additional output was captured."
        if len(details) > limit:
            details = details[:limit] + "\n...[truncated]"
        return (
            f"Validation stage: {self.stage}\n"
            f"Return code: {self.returncode}\n"
            f"Failure details:\n{details}"
        )


def _infer_stage(stdout: str, stderr: str) -> str:
    """Infer the validation failure stage from combined output."""
    output = "\n".join(part for part in (stdout, stderr) if part).lower()
    if "error:" in output or "undefined reference" in output:
        return "compile"
    if "validation failed" in output or "mismatch" in output:
        return "correctness"
    return "runtime"


class DockerValidator:
    """Validates a workspace by building and running inside Docker/QEMU.

    The caller provides *build_command* — a shell command that compiles and
    runs the translated code.  This validator mounts the workspace into the
    Docker container and executes the command.
    """

    def __init__(self, docker_image: str = DOCKER_IMAGE):
        self.docker_image = docker_image

    def validate(
        self,
        workspace_dir: Path,
        build_command: str,
        timeout: int | None = None,
    ) -> ValidationResult:
        effective_timeout = timeout if timeout is not None else VALIDATION_TIMEOUT_SECONDS
        logger.debug("Running Docker validation in %s (timeout=%ds)", workspace_dir, effective_timeout)
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--mount",
                    f"type=bind,source={workspace_dir},target=/workspace",
                    "-w",
                    "/workspace",
                    self.docker_image,
                    "bash",
                    "-lc",
                    build_command,
                ],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raw_out = getattr(exc, "stdout", b"") or b""
            raw_err = getattr(exc, "stderr", b"") or b""
            return ValidationResult(
                ok=False,
                stage="timeout",
                returncode=None,
                stdout=raw_out.decode(errors="replace") if isinstance(raw_out, bytes) else raw_out,
                stderr=raw_err.decode(errors="replace") if isinstance(raw_err, bytes) else raw_err or "Docker validation timed out.",
            )
        except Exception as exc:
            logger.warning("Docker validation execution failed: %s", exc)
            return ValidationResult(
                ok=False,
                stage="internal-error",
                returncode=None,
                stdout="",
                stderr=str(exc),
            )

        if result.returncode == 0:
            logger.info("Docker validation passed")
            return ValidationResult(
                ok=True,
                stage="validation",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        stage = _infer_stage(result.stdout, result.stderr)
        logger.debug(
            "Docker validation failed at stage %s with return code %s",
            stage,
            result.returncode,
        )
        return ValidationResult(
            ok=False,
            stage=stage,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


class SSHValidator:
    """Validates translated code on real RISC-V hardware via SSH.

    The caller provides:
      - *local_files*: list of local file paths to upload.
      - *compile_command*: shell command to compile on the remote host.
      - *run_command*: shell command to run the compiled binary.

    Returns ok=True (skipped) when SSH is unavailable so the pipeline
    can still function without hardware.
    """

    def __init__(self, ssh_host: str = SSH_HOST, remote_dir: str = REMOTE_DIR):
        self.ssh_host = ssh_host
        self.remote_dir = remote_dir
        self._available = self._check_ssh()
        if self._available:
            self._setup_remote()

    def _check_ssh(self) -> bool:
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", self.ssh_host, "echo ok"],
                capture_output=True,
                timeout=10,
                text=True,
            )
            ok = result.returncode == 0 and "ok" in result.stdout
            if ok:
                logger.info("SSH host %s is reachable", self.ssh_host)
            else:
                logger.warning(
                    "SSH host %s is not reachable; SSH validation disabled",
                    self.ssh_host,
                )
            return ok
        except Exception as exc:
            logger.warning(
                "SSH connectivity check failed (%s); SSH validation disabled", exc
            )
            return False

    def _setup_remote(self) -> None:
        subprocess.run(
            ["ssh", self.ssh_host, f"mkdir -p {self.remote_dir}"],
            capture_output=True,
            timeout=30,
        )

    def upload(self, local_paths: list[Path]) -> ValidationResult | None:
        """Upload files and directories to the remote workspace. Returns a failure result or None on success."""
        for local_path in local_paths:
            if not local_path.exists():
                continue
            if local_path.is_dir():
                # Remove existing remote directory first to avoid nesting
                subprocess.run(
                    ["ssh", self.ssh_host, f"rm -rf {self.remote_dir}/{local_path.name}"],
                    capture_output=True,
                    timeout=30,
                )
            cmd = ["scp"]
            if local_path.is_dir():
                cmd.append("-r")
            cmd += [
                str(local_path),
                f"{self.ssh_host}:{self.remote_dir}/{local_path.name}",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,
                text=True,
            )
            if result.returncode != 0:
                return ValidationResult(
                    ok=False,
                    stage="ssh-upload",
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
        return None

    def validate(
        self,
        local_files: list[Path],
        compile_command: str,
        run_command: str,
    ) -> ValidationResult:
        """Upload, compile, and run on real hardware."""
        if not self._available:
            return ValidationResult(
                ok=True,
                stage="ssh-skipped",
                returncode=None,
                stdout="",
                stderr="SSH host not available; skipping hardware validation.",
            )

        try:
            upload_err = self.upload(local_files)
            if upload_err is not None:
                return upload_err

            comp = subprocess.run(
                ["ssh", self.ssh_host, f"cd {self.remote_dir} && {compile_command}"],
                capture_output=True,
                timeout=60,
                text=True,
            )
            if comp.returncode != 0:
                return ValidationResult(
                    ok=False,
                    stage="ssh-compile",
                    returncode=comp.returncode,
                    stdout=comp.stdout,
                    stderr=comp.stderr,
                )

            run = subprocess.run(
                ["ssh", self.ssh_host, f"cd {self.remote_dir} && {run_command}"],
                capture_output=True,
                timeout=300,
                text=True,
            )
            if run.returncode != 0:
                return ValidationResult(
                    ok=False,
                    stage="ssh-runtime",
                    returncode=run.returncode,
                    stdout=run.stdout,
                    stderr=run.stderr,
                )

            return ValidationResult(
                ok=True,
                stage="ssh-validation",
                returncode=run.returncode,
                stdout=run.stdout,
                stderr=run.stderr,
            )

        except subprocess.TimeoutExpired as exc:
            raw_out = getattr(exc, "stdout", b"") or b""
            raw_err = getattr(exc, "stderr", b"") or b""
            return ValidationResult(
                ok=False,
                stage="ssh-timeout",
                returncode=None,
                stdout=raw_out.decode(errors="replace") if isinstance(raw_out, bytes) else raw_out,
                stderr=raw_err.decode(errors="replace") if isinstance(raw_err, bytes) else raw_err or "SSH validation timed out.",
            )
        except Exception as exc:
            logger.warning("SSH validation error: %s", exc)
            return ValidationResult(
                ok=False,
                stage="ssh-error",
                returncode=None,
                stdout="",
                stderr=str(exc),
            )
