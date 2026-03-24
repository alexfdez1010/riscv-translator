"""Validate a translation output folder against the simulator and/or SSH hardware.

Usage:
    uv run python -m src.check <output_dir> [--build-command CMD]
        [--ssh-compile CMD] [--ssh-run CMD] [--target-file FILE]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from src.config import DATASETS_DIR
from src.logger import configure_logging, get_logger
from src.repair import default_build_command, default_ssh_compile_command, default_ssh_run_command
from src.validators import DockerValidator, SSHValidator

logger = get_logger(__name__)


def check(
    output_dir: Path,
    target_file: str | None = None,
    build_command: str | None = None,
    ssh_compile_command: str | None = None,
    ssh_run_command: str | None = None,
    test_data_dir: Path | None = None,
) -> int:
    """Run Docker/QEMU and optionally SSH validation on an output folder.

    Returns 0 if all validations pass, 1 otherwise.
    """
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return 1

    # Auto-detect target file if not provided (first .c or .cpp file)
    if target_file is None:
        candidates = sorted(
            p.name
            for p in output_dir.iterdir()
            if p.is_file() and p.suffix in (".c", ".cpp", ".cc", ".cxx")
        )
        if not candidates:
            logger.error("No C/C++ source files found in %s", output_dir)
            return 1
        target_file = candidates[0]
        logger.info("Auto-detected target file: %s", target_file)

    if build_command is None:
        build_command = default_build_command(target_file)
    if ssh_compile_command is None:
        ssh_compile_command = default_ssh_compile_command()
    if ssh_run_command is None:
        ssh_run_command = default_ssh_run_command()

    # If test data is provided, create a temporary copy of the output dir
    # with demo/ subdirectory so the build command can find the datasets.
    cleanup_dir = None
    validation_dir = output_dir
    if test_data_dir is not None and test_data_dir.is_dir():
        demo_in_output = output_dir / "demo"
        demo_has_files = demo_in_output.is_dir() and any(demo_in_output.iterdir())
        if not demo_has_files:
            cleanup_dir = Path(tempfile.mkdtemp(prefix="sse2rvv-check-"))
            validation_dir = cleanup_dir / "workspace"
            shutil.copytree(output_dir, validation_dir)
            demo_dest = validation_dir / "demo"
            if demo_dest.exists():
                shutil.rmtree(demo_dest)
            shutil.copytree(test_data_dir, demo_dest)
            logger.info("Copied test data into temporary validation workspace")

    try:
        exit_code = 0

        # --- Docker/QEMU validation ---
        logger.info("Running Docker/QEMU validation on %s ...", validation_dir)
        docker = DockerValidator()
        docker_result = docker.validate(validation_dir, build_command)

        if docker_result.ok:
            logger.info("Docker/QEMU: PASSED")
        else:
            logger.error(
                "Docker/QEMU: FAILED (stage=%s, rc=%s)\n%s",
                docker_result.stage,
                docker_result.returncode,
                docker_result.combined_output,
            )
            exit_code = 1

        # --- SSH validation ---
        logger.info("Running SSH hardware validation ...")
        ssh = SSHValidator()
        local_files = [p for p in validation_dir.iterdir()]
        ssh_result = ssh.validate(local_files, ssh_compile_command, ssh_run_command)

        if ssh_result.ok:
            if ssh_result.stage == "ssh-skipped":
                logger.warning("SSH: SKIPPED (host not reachable)")
            else:
                logger.info("SSH: PASSED")
        else:
            logger.error(
                "SSH: FAILED (stage=%s, rc=%s)\n%s",
                ssh_result.stage,
                ssh_result.returncode,
                ssh_result.combined_output,
            )
            exit_code = 1

        return exit_code
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a translation output folder against simulator and/or SSH hardware"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing translated output files",
    )
    parser.add_argument(
        "--target-file",
        default=None,
        help="Target source file name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--build-command",
        default=None,
        help="Shell command to compile and test in Docker (auto-generated if omitted)",
    )
    parser.add_argument(
        "--ssh-compile",
        default=None,
        help="Shell command to compile on SSH hardware",
    )
    parser.add_argument(
        "--ssh-run",
        default=None,
        help="Shell command to run on SSH hardware",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DATASETS_DIR,
        help="Directory with test data files; copied into workspace as demo/",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging(level="INFO")
    args = parse_args()
    return check(
        output_dir=args.output_dir,
        target_file=args.target_file,
        build_command=args.build_command,
        ssh_compile_command=args.ssh_compile,
        ssh_run_command=args.ssh_run,
        test_data_dir=args.test_data,
    )


if __name__ == "__main__":
    raise SystemExit(main())
