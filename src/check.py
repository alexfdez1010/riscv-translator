"""Validate a translation output folder against the simulator and/or SSH hardware.

Runs the translated code under Spike at multiple VLEN sizes (128..max),
compares each output against an SSE reference (original code on Intel),
then validates on real RISC-V hardware via SSH.

Usage:
    uv run python -m src.check <output_dir> [--build-command CMD]
        [--ssh-compile CMD] [--ssh-run CMD] [--target-file FILE]
        [--max-vlen N] [--dataset FILE]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from src.benchmark import (
    BenchmarkResult,
    REFERENCE_FILE as BENCH_REFERENCE_FILE,
    check_ssh,
    compare_outputs,
    prepare_local_dir,
    run_locally,
    run_on_host,
    upload_datasets,
    upload_to_host,
)
from src.config import (
    DATASETS_DIR,
    DOCKER_IMAGE,
    PROJECT_DIR,
    REMOTE_DIR,
    RISCVCC,
    SSH_CC,
    SSH_HOST,
    SSH_JUMP_HOST,
)
from src.logger import configure_logging, get_logger
from src.repair import default_build_command, default_ssh_compile_command, default_ssh_run_command
from src.validators import DockerValidator, SSHValidator, ValidationResult

logger = get_logger(__name__)

DEFAULT_ORIGINAL_DIR = PROJECT_DIR / "initial_code"
DEFAULT_CHECK_DATASET = "10k.fa"

# VLEN must be a power of 2, minimum 128, Spike supports up to 4096.
MIN_VLEN = 128
MAX_VLEN_DEFAULT = 4096
# Base timeout (seconds) for VLEN=128; doubles for each VLEN doubling.
BASE_TIMEOUT = 120


def _vlen_range(max_vlen: int) -> list[int]:
    """Return list of VLEN values: 128, 256, 512, ... up to max_vlen."""
    vlens = []
    v = MIN_VLEN
    while v <= max_vlen:
        vlens.append(v)
        v *= 2
    return vlens


def _timeout_for_vlen(vlen: int) -> int:
    """Scale timeout with VLEN — simulation slows significantly at wider widths."""
    factor = vlen // MIN_VLEN  # 1x at 128, 2x at 256, …, 32x at 4096
    return BASE_TIMEOUT * factor


def _build_command_for_vlen(target_file: str, vlen: int, dataset: str) -> str:
    """Build command for Docker/Spike at a specific VLEN."""
    cflags = "-O2 -I. -march=rv64gcv -mabi=lp64d"
    ldflags = "-lm"
    simulator = f"spike --isa=rv64gcv_zvl{vlen}b pk64"
    return (
        f'echo "=== Compiling ===" && '
        f'{RISCVCC} {cflags} main.c ssw.c -o ssw_test {ldflags} 2>&1 && '
        f'echo "=== Compilation succeeded, running under Spike (VLEN={vlen}) ===" && '
        f'{simulator} ./ssw_test demo/{dataset} demo/{BENCH_REFERENCE_FILE} 2>&1 && '
        f'echo "=== Execution succeeded ==="'
    )


def _get_sse_reference(
    original_dir: Path,
    dataset_dir: Path,
    dataset: str,
) -> BenchmarkResult | None:
    """Run the original SSE code on Intel to get reference output.

    Uses SSH_JUMP_HOST if set, otherwise runs locally.
    """
    run_cmd = f"./ssw_test demo/{dataset} demo/{BENCH_REFERENCE_FILE} 2>&1"
    intel_compile = "gcc -O2 -o ssw_test main.c ssw.c -lm 2>&1"

    if SSH_JUMP_HOST:
        logger.info("Getting SSE reference from %s ...", SSH_JUMP_HOST)
        if not check_ssh(SSH_JUMP_HOST):
            logger.error("Cannot reach jump host: %s", SSH_JUMP_HOST)
            return None
        remote = f"{REMOTE_DIR}-check-ref"
        original_files = [p for p in original_dir.iterdir() if p.is_file()]
        if not upload_to_host(SSH_JUMP_HOST, remote, original_files):
            return None
        if not upload_datasets(SSH_JUMP_HOST, remote, dataset_dir, dataset):
            return None
        return run_on_host(SSH_JUMP_HOST, remote, intel_compile, run_cmd, "Intel (SSE reference)")
    else:
        logger.info("Getting SSE reference locally ...")
        local_dir = prepare_local_dir(original_dir, dataset_dir, dataset)
        try:
            return run_locally(local_dir, intel_compile, run_cmd, "Intel (SSE reference)")
        finally:
            shutil.rmtree(local_dir, ignore_errors=True)


def check(
    output_dir: Path,
    target_file: str | None = None,
    build_command: str | None = None,
    ssh_compile_command: str | None = None,
    ssh_run_command: str | None = None,
    test_data_dir: Path | None = None,
    max_vlen: int = MAX_VLEN_DEFAULT,
    dataset: str = DEFAULT_CHECK_DATASET,
    original_dir: Path = DEFAULT_ORIGINAL_DIR,
) -> int:
    """Run multi-VLEN simulator checks with SSE comparison, then SSH validation.

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

    if ssh_compile_command is None:
        ssh_compile_command = default_ssh_compile_command()
    if ssh_run_command is None:
        ssh_run_command = default_ssh_run_command()

    # Prepare validation workspace with test data
    dataset_dir = test_data_dir if test_data_dir is not None else DATASETS_DIR
    cleanup_dir = None
    validation_dir = output_dir

    demo_in_output = output_dir / "demo"
    demo_has_files = demo_in_output.is_dir() and any(demo_in_output.iterdir())
    if dataset_dir.is_dir() and not demo_has_files:
        cleanup_dir = Path(tempfile.mkdtemp(prefix="sse2rvv-check-"))
        validation_dir = cleanup_dir / "workspace"
        shutil.copytree(output_dir, validation_dir)
        demo_dest = validation_dir / "demo"
        if demo_dest.exists():
            shutil.rmtree(demo_dest)
        shutil.copytree(dataset_dir, demo_dest)
        logger.info("Copied test data into temporary validation workspace")

    try:
        exit_code = 0
        vlens = _vlen_range(max_vlen)

        # --- Step 1: Get SSE reference output ---
        print("\n" + "=" * 60)
        print("STEP 1: SSE REFERENCE")
        print("=" * 60)

        sse_ref = _get_sse_reference(original_dir, dataset_dir, dataset)
        if sse_ref is None or not sse_ref.ok:
            logger.error("Failed to get SSE reference output")
            if sse_ref:
                logger.error("stderr: %s", sse_ref.stderr)
            return 1
        logger.info("SSE reference: OK")

        # --- Step 2: Simulator checks at each VLEN ---
        print("\n" + "=" * 60)
        print("STEP 2: SIMULATOR CHECKS (VLEN sweep)")
        print("=" * 60)

        docker = DockerValidator()
        sim_results: list[tuple[int, ValidationResult, bool, str]] = []

        for vlen in vlens:
            vlen_build_cmd = _build_command_for_vlen(target_file, vlen, dataset)
            logger.info("Running simulator at VLEN=%d ...", vlen)
            timeout = _timeout_for_vlen(vlen)
            logger.info("Timeout for VLEN=%d: %ds", vlen, timeout)
            result = docker.validate(validation_dir, vlen_build_cmd, timeout=timeout)

            if not result.ok:
                logger.error("Simulator VLEN=%d: FAILED (stage=%s)", vlen, result.stage)
                sim_results.append((vlen, result, False, f"Execution failed: {result.stage}"))
                exit_code = 1
                continue

            # Compare output against SSE reference
            sim_bench = BenchmarkResult(
                host="simulator",
                label=f"Simulator (VLEN={vlen})",
                ok=True,
                elapsed_seconds=0,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            match, details = compare_outputs(sse_ref, sim_bench)
            if match:
                logger.info("Simulator VLEN=%d: PASSED (output matches SSE)", vlen)
            else:
                logger.error("Simulator VLEN=%d: MISMATCH with SSE reference", vlen)
                logger.error("%s", details)
                exit_code = 1
            sim_results.append((vlen, result, match, details))

        # Print simulator summary
        print("\n" + "-" * 60)
        print(f"{'VLEN':>6}  {'Compile':>8}  {'Run':>8}  {'vs SSE':>10}")
        print("-" * 60)
        for vlen, result, match, details in sim_results:
            if not result.ok:
                print(f"{vlen:>6}  {'FAIL':>8}  {'---':>8}  {'---':>10}")
            else:
                print(f"{vlen:>6}  {'OK':>8}  {'OK':>8}  {'MATCH' if match else 'MISMATCH':>10}")
        print("-" * 60)

        # --- Step 3: SSH hardware validation ---
        print("\n" + "=" * 60)
        print("STEP 3: SSH HARDWARE VALIDATION")
        print("=" * 60)

        ssh = SSHValidator()
        local_files = list(validation_dir.iterdir())
        ssh_result = ssh.validate(local_files, ssh_compile_command, ssh_run_command)

        if ssh_result.ok:
            if ssh_result.stage == "ssh-skipped":
                logger.warning("SSH: SKIPPED (host not reachable)")
            else:
                # Compare SSH output against SSE reference
                ssh_bench = BenchmarkResult(
                    host=SSH_HOST,
                    label="SSH (RISC-V hardware)",
                    ok=True,
                    elapsed_seconds=0,
                    stdout=ssh_result.stdout,
                    stderr=ssh_result.stderr,
                )
                match, details = compare_outputs(sse_ref, ssh_bench)
                if match:
                    logger.info("SSH: PASSED (output matches SSE)")
                else:
                    logger.error("SSH: MISMATCH with SSE reference")
                    logger.error("%s", details)
                    exit_code = 1
                print(f"\nSSH hardware: {'MATCH' if match else 'MISMATCH'} with SSE reference")
                print(details)
        else:
            logger.error(
                "SSH: FAILED (stage=%s, rc=%s)\n%s",
                ssh_result.stage,
                ssh_result.returncode,
                ssh_result.combined_output,
            )
            exit_code = 1

        # --- Final summary ---
        print("\n" + "=" * 60)
        if exit_code == 0:
            print("CHECK PASSED: all VLEN sizes match SSE reference.")
        else:
            print("CHECK FAILED: see details above.")
        print("=" * 60)

        return exit_code
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a translation output folder against simulator (multi-VLEN) and SSH hardware"
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
    parser.add_argument(
        "--max-vlen",
        type=int,
        default=MAX_VLEN_DEFAULT,
        help=f"Maximum VLEN to test (powers of 2, default: {MAX_VLEN_DEFAULT})",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_CHECK_DATASET,
        help=f"Dataset file to use for checking (default: {DEFAULT_CHECK_DATASET})",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=DEFAULT_ORIGINAL_DIR,
        help="Directory with original SSE source code for reference output",
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
        max_vlen=args.max_vlen,
        dataset=args.dataset,
        original_dir=args.original_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
