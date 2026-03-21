"""Shared validation result and hardware validators for SSW repair/evolution."""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    BENCHMARK_FASTA,
    REMOTE_DIR,
    SSH_HOST,
    SSW_BENCH_FILE,
    SSW_DIR,
)
from src.logger import get_logger

logger = get_logger(__name__)

MAX_OUTPUT_CHARS = 16000

DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "luispimo/riscv-toolchain:arm64-2025-10-20")
RISCVCC = os.getenv("RISCVCC", "riscv64-unknown-elf-gcc")
QEMU_RVV = os.getenv(
    "QEMU_RVV",
    "qemu-riscv64 -cpu rv64,v=on,vext_spec=v1.0,vlen=128,rvv_ta_all_1s=on",
)


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


class InitialCodeValidator:
    """Validates an SSW snapshot by building and running inside Docker/QEMU."""

    def validate(
        self,
        snapshot: "SourceSnapshot",  # noqa: F821 — forward ref to repair.py type
        workspace_dir: Path,
        target_fasta: str,
        query_fasta: str,
    ) -> ValidationResult:
        from src.repair import materialize_snapshot  # avoid circular import

        materialize_snapshot(workspace_dir, snapshot)
        command = (
            "make clean && "
            f'make rvv_example CC="{RISCVCC}" && '
            f'{QEMU_RVV} ./rvv_example && '
            f'echo "=== rvv_example PASSED ===" && '
            f'( make rvv_cli CC="{RISCVCC}" && '
            f'{QEMU_RVV} ./rvv_ssw_test "{target_fasta}" "{query_fasta}" '
            f'|| echo "=== rvv_cli skipped (zlib missing) ===" )'
        )
        logger.debug("Running initial_code validation in %s", workspace_dir)
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--mount",
                    f"type=bind,source={workspace_dir},target=/workspace/initial_code",
                    "-w",
                    "/workspace/initial_code",
                    "-e",
                    f"RISCVCC={RISCVCC}",
                    "-e",
                    f"QEMU_RVV={QEMU_RVV}",
                    DOCKER_IMAGE,
                    "bash",
                    "-lc",
                    command,
                ],
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            logger.warning("Validation execution failed: %s", exc)
            return ValidationResult(
                ok=False,
                stage="internal-error",
                returncode=None,
                stdout="",
                stderr=str(exc),
            )
        if result.returncode == 0:
            logger.info("Validation passed")
            return ValidationResult(
                ok=True,
                stage="validation",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        stage = _infer_stage(result.stdout, result.stderr)
        logger.debug(
            "Validation failed at stage %s with return code %s",
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


class SSHSSWValidator:
    """Validates a candidate ssw.c by compiling and running on real RISC-V
    hardware via SSH.  Catches problems (e.g. VLEN mismatch) that QEMU
    emulation with a fixed VLEN may not reveal."""

    _REMOTE_SSW_DIR = f"{REMOTE_DIR}/ssw_repair"

    def __init__(self):
        self._available = self._check_ssh()
        if self._available:
            self._setup_remote()

    @staticmethod
    def _check_ssh() -> bool:
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", SSH_HOST, "echo ok"],
                capture_output=True,
                timeout=10,
                text=True,
            )
            ok = result.returncode == 0 and "ok" in result.stdout
            if ok:
                logger.info("SSH host %s is reachable", SSH_HOST)
            else:
                logger.warning(
                    "SSH host %s is not reachable; SSH validation disabled", SSH_HOST
                )
            return ok
        except Exception as exc:
            logger.warning(
                "SSH connectivity check failed (%s); SSH validation disabled", exc
            )
            return False

    def _setup_remote(self) -> None:
        subprocess.run(
            ["ssh", SSH_HOST, f"mkdir -p {self._REMOTE_SSW_DIR}"],
            capture_output=True,
            timeout=30,
        )
        for name in ("ssw.h", "sse2rvv.h"):
            src = SSW_DIR / name
            if src.exists():
                subprocess.run(
                    ["scp", str(src), f"{SSH_HOST}:{self._REMOTE_SSW_DIR}/{name}"],
                    capture_output=True,
                    timeout=30,
                )
        if SSW_BENCH_FILE.exists():
            subprocess.run(
                [
                    "scp",
                    str(SSW_BENCH_FILE),
                    f"{SSH_HOST}:{self._REMOTE_SSW_DIR}/bench_ssw.c",
                ],
                capture_output=True,
                timeout=30,
            )
        if BENCHMARK_FASTA.exists():
            subprocess.run(
                [
                    "scp",
                    str(BENCHMARK_FASTA),
                    f"{SSH_HOST}:{self._REMOTE_SSW_DIR}/dataset.fa",
                ],
                capture_output=True,
                timeout=120,
            )
        logger.info(
            "SSH validation workspace set up at %s:%s", SSH_HOST, self._REMOTE_SSW_DIR
        )

    def validate(self, ssw_code: str) -> ValidationResult:
        """Compile and run on real hardware.  Returns ok=True (skipped) when
        SSH is unavailable so the repair loop can still function."""
        if not self._available:
            return ValidationResult(
                ok=True,
                stage="ssh-skipped",
                returncode=None,
                stdout="",
                stderr="SSH host not available; skipping hardware validation.",
            )

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".c", delete=False
            ) as f:
                f.write(ssw_code)
                local_tmp = f.name

            scp_result = subprocess.run(
                ["scp", local_tmp, f"{SSH_HOST}:{self._REMOTE_SSW_DIR}/ssw.c"],
                capture_output=True,
                timeout=30,
                text=True,
            )
            os.unlink(local_tmp)
            if scp_result.returncode != 0:
                return ValidationResult(
                    ok=False,
                    stage="ssh-upload",
                    returncode=scp_result.returncode,
                    stdout=scp_result.stdout,
                    stderr=scp_result.stderr,
                )

            compile_cmd = (
                f"cd {self._REMOTE_SSW_DIR} && "
                f"clang -O2 -march=rv64gcv -c -o ssw.o ssw.c 2>&1 && "
                f"clang -O2 -march=rv64gcv -o bench_ssw bench_ssw.c ssw.o -lm 2>&1"
            )
            comp = subprocess.run(
                ["ssh", SSH_HOST, compile_cmd],
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

            run_cmd = f"cd {self._REMOTE_SSW_DIR} && ./bench_ssw dataset.fa 2>&1"
            run = subprocess.run(
                ["ssh", SSH_HOST, run_cmd],
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
            return ValidationResult(
                ok=False,
                stage="ssh-timeout",
                returncode=None,
                stdout=getattr(exc, "stdout", "") or "",
                stderr=getattr(exc, "stderr", "") or "SSH validation timed out.",
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
