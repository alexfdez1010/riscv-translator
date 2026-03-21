"""SSW validation for the repair/evolution pipeline."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.config import (
    DOCKER_IMAGE,
    RISCVCC,
    SSW_DIR,
    VALIDATION_TIMEOUT_SECONDS,
)
from src.logger import get_logger
from src.validators import ValidationResult

logger = get_logger(__name__)

_QEMU_RVV = os.getenv(
    "QEMU_RVV",
    "qemu-riscv64 -cpu rv64,v=on,vext_spec=v1.0,vlen=128,rvv_ta_all_1s=on",
)


def _create_ssw_workspace(ssw_code: str) -> Path:
    """Create a temp workspace with all SSW files, replacing ssw.c with the candidate."""
    workdir = Path(tempfile.mkdtemp(prefix="ssw-ea-"))
    for item in SSW_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, workdir / item.name)
        elif item.is_dir():
            shutil.copytree(item, workdir / item.name)
    (workdir / "ssw.c").write_text(ssw_code)
    return workdir


class SSWValidator:
    """Validates a candidate ssw.c by building and running rvv_example in Docker/QEMU."""

    def validate(self, ssw_code: str) -> ValidationResult:
        workdir = _create_ssw_workspace(ssw_code)
        try:
            command = (
                "make clean && "
                f'make rvv_example CC="{RISCVCC}" && '
                f"{_QEMU_RVV} ./rvv_example"
            )
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--mount",
                    f"type=bind,source={workdir},target=/workspace/initial_code",
                    "-w",
                    "/workspace/initial_code",
                    "-e",
                    f"RISCVCC={RISCVCC}",
                    "-e",
                    f"QEMU_RVV={_QEMU_RVV}",
                    DOCKER_IMAGE,
                    "bash",
                    "-lc",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=VALIDATION_TIMEOUT_SECONDS,
            )

            if result.returncode == 0:
                return ValidationResult(
                    ok=True,
                    stage="validation",
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

            stage = self._infer_stage(result.stdout, result.stderr)
            return ValidationResult(
                ok=False,
                stage=stage,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return ValidationResult(
                ok=False,
                stage="timeout",
                returncode=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "Validation timed out.",
            )
        except Exception as exc:
            return ValidationResult(
                ok=False,
                stage="internal-error",
                returncode=None,
                stdout="",
                stderr=str(exc),
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _infer_stage(stdout: str, stderr: str) -> str:
        output = "\n".join(part for part in (stdout, stderr) if part).lower()
        if "error:" in output or "undefined reference" in output:
            return "compile"
        if "validation failed" in output or "mismatch" in output:
            return "correctness"
        return "runtime"
