"""Benchmark: run original SSE code on Intel (jump) and translated RVV code on RISC-V (final).

Validates that both produce identical alignment output, then compares execution time.

Usage:
    uv run python -m src.benchmark [--dataset FILE] [--original-dir DIR] [--translated-dir DIR]
"""

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    DATASETS_DIR,
    PROJECT_DIR,
    REMOTE_DIR,
    SSH_CC,
    SSH_HOST,
    SSH_JUMP_HOST,
)
from src.logger import configure_logging, get_logger

logger = get_logger(__name__)

DEFAULT_ORIGINAL_DIR = PROJECT_DIR / "initial_code"
DEFAULT_TRANSLATED_DIR = PROJECT_DIR / "translations" / "sequence-alignment"
DEFAULT_DATASET = "1M.fa"
REFERENCE_FILE = "54mer_hap1_1.100.fa"


@dataclass(slots=True)
class BenchmarkResult:
    host: str
    label: str
    ok: bool
    elapsed_seconds: float
    stdout: str
    stderr: str


def check_ssh(host: str) -> bool:
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", host, "echo ok"],
            capture_output=True, timeout=10, text=True,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def upload_to_host(host: str, remote_dir: str, local_paths: list[Path]) -> bool:
    subprocess.run(
        ["ssh", host, f"mkdir -p {remote_dir}/demo"],
        capture_output=True, timeout=30,
    )
    for p in local_paths:
        if not p.exists():
            continue
        cmd = ["scp"]
        if p.is_dir():
            cmd.append("-r")
        cmd += [str(p), f"{host}:{remote_dir}/{p.name}"]
        r = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        if r.returncode != 0:
            logger.error("Upload failed for %s: %s", p, r.stderr)
            return False
    return True


def upload_datasets(host: str, remote_dir: str, dataset_dir: Path, dataset: str) -> bool:
    """Upload dataset files into remote demo/ subdirectory."""
    for fname in [dataset, REFERENCE_FILE]:
        src = dataset_dir / fname
        if not src.exists():
            logger.error("Dataset file not found: %s", src)
            return False
        cmd = ["scp", str(src), f"{host}:{remote_dir}/demo/{fname}"]
        r = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        if r.returncode != 0:
            logger.error("Dataset upload failed for %s: %s", fname, r.stderr)
            return False
    return True


def run_on_host(
    host: str,
    remote_dir: str,
    compile_cmd: str,
    run_cmd: str,
    label: str,
) -> BenchmarkResult:
    """Compile and run on a remote host, measuring execution time."""
    # Compile
    comp = subprocess.run(
        ["ssh", host, f"cd {remote_dir} && {compile_cmd}"],
        capture_output=True, timeout=120, text=True,
    )
    if comp.returncode != 0:
        logger.error("[%s] Compilation failed:\n%s\n%s", label, comp.stdout, comp.stderr)
        return BenchmarkResult(
            host=host, label=label, ok=False, elapsed_seconds=0,
            stdout=comp.stdout, stderr=comp.stderr,
        )

    # Run with timing
    start = time.monotonic()
    run = subprocess.run(
        ["ssh", host, f"cd {remote_dir} && {run_cmd}"],
        capture_output=True, timeout=600, text=True,
    )
    elapsed = time.monotonic() - start

    ok = run.returncode == 0
    if not ok:
        logger.error("[%s] Execution failed (rc=%d):\n%s\n%s", label, run.returncode, run.stdout, run.stderr)

    return BenchmarkResult(
        host=host, label=label, ok=ok, elapsed_seconds=elapsed,
        stdout=run.stdout, stderr=run.stderr,
    )


def normalize_output(raw: str) -> str:
    """Normalize alignment output for comparison: strip trailing whitespace per line, skip empty lines."""
    lines = [line.rstrip() for line in raw.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def compare_outputs(intel: BenchmarkResult, riscv: BenchmarkResult) -> tuple[bool, str]:
    """Compare alignment outputs from both runs. Returns (match, details)."""
    intel_norm = normalize_output(intel.stdout)
    riscv_norm = normalize_output(riscv.stdout)

    if intel_norm == riscv_norm:
        return True, "Outputs match exactly."

    # Show diff summary
    intel_lines = intel_norm.splitlines()
    riscv_lines = riscv_norm.splitlines()

    details = []
    details.append(f"Intel output: {len(intel_lines)} lines")
    details.append(f"RISC-V output: {len(riscv_lines)} lines")

    max_lines = max(len(intel_lines), len(riscv_lines))
    diff_count = 0
    first_diffs = []
    for i in range(max_lines):
        a = intel_lines[i] if i < len(intel_lines) else "<missing>"
        b = riscv_lines[i] if i < len(riscv_lines) else "<missing>"
        if a != b:
            diff_count += 1
            if len(first_diffs) < 5:
                first_diffs.append(f"  line {i+1}:\n    Intel:  {a}\n    RISC-V: {b}")

    details.append(f"Differing lines: {diff_count}")
    details.append("First differences:")
    details.extend(first_diffs)

    return False, "\n".join(details)


def benchmark(
    dataset: str = DEFAULT_DATASET,
    original_dir: Path = DEFAULT_ORIGINAL_DIR,
    translated_dir: Path = DEFAULT_TRANSLATED_DIR,
    dataset_dir: Path = DATASETS_DIR,
) -> int:
    """Run benchmark comparing original SSE code on Intel vs translated RVV code on RISC-V."""
    jump_host = SSH_JUMP_HOST
    final_host = SSH_HOST
    jump_remote = f"{REMOTE_DIR}-bench-original"
    final_remote = f"{REMOTE_DIR}-bench-translated"

    logger.info("Benchmark dataset: %s", dataset)
    logger.info("Original code: %s -> %s (Intel)", original_dir, jump_host)
    logger.info("Translated code: %s -> %s (RISC-V)", translated_dir, final_host)

    # Check connectivity
    if not check_ssh(jump_host):
        logger.error("Cannot reach jump host: %s", jump_host)
        return 1
    logger.info("SSH to %s: OK", jump_host)

    if not check_ssh(final_host):
        logger.error("Cannot reach final host: %s", final_host)
        return 1
    logger.info("SSH to %s: OK", final_host)

    # Upload original code to jump (Intel)
    original_files = [p for p in original_dir.iterdir() if p.is_file()]
    logger.info("Uploading original code to %s:%s ...", jump_host, jump_remote)
    if not upload_to_host(jump_host, jump_remote, original_files):
        return 1
    if not upload_datasets(jump_host, jump_remote, dataset_dir, dataset):
        return 1

    # Upload translated code to final (RISC-V)
    translated_files = [p for p in translated_dir.iterdir() if p.is_file()]
    translated_dirs = [p for p in translated_dir.iterdir() if p.is_dir()]
    logger.info("Uploading translated code to %s:%s ...", final_host, final_remote)
    if not upload_to_host(final_host, final_remote, translated_files + translated_dirs):
        return 1
    if not upload_datasets(final_host, final_remote, dataset_dir, dataset):
        return 1

    run_cmd = f"./ssw_test demo/{dataset} demo/{REFERENCE_FILE} 2>&1"

    # Run original on Intel (jump) — compile with gcc (x86)
    logger.info("Running original code on Intel (%s) ...", jump_host)
    intel_compile = "gcc -O2 -o ssw_test main.c ssw.c -lm -lz 2>&1"
    intel_result = run_on_host(
        jump_host, jump_remote, intel_compile, run_cmd, "Intel (original SSE)",
    )

    # Run translated on RISC-V (final)
    logger.info("Running translated code on RISC-V (%s) ...", final_host)
    riscv_compile = f"{SSH_CC} -o ssw_test main.c ssw.c --target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm -lz 2>&1"
    riscv_result = run_on_host(
        final_host, final_remote, riscv_compile, run_cmd, "RISC-V (translated RVV)",
    )

    # Report
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print("-" * 60)

    for r in [intel_result, riscv_result]:
        status = "PASS" if r.ok else "FAIL"
        print(f"  {r.label:30s}  {status:5s}  {r.elapsed_seconds:8.2f}s  ({r.host})")

    print("-" * 60)

    if not intel_result.ok or not riscv_result.ok:
        print("\nOne or more executions failed.")
        if not intel_result.ok:
            print(f"\n[Intel stderr]\n{intel_result.stderr}")
        if not riscv_result.ok:
            print(f"\n[RISC-V stderr]\n{riscv_result.stderr}")
        return 1

    # Compare outputs for correctness
    match, details = compare_outputs(intel_result, riscv_result)

    print(f"\nOutput comparison: {'MATCH' if match else 'MISMATCH'}")
    print(details)

    if not match:
        print("\nBenchmark FAILED: outputs differ between Intel and RISC-V.")
        return 1

    # Timing comparison
    if intel_result.elapsed_seconds > 0:
        ratio = riscv_result.elapsed_seconds / intel_result.elapsed_seconds
        print(f"\nRISC-V / Intel time ratio: {ratio:.2f}x")

    print("\nBenchmark PASSED: both produce identical output.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark original SSE code on Intel vs translated RVV code on RISC-V"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset file name to use (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=DEFAULT_ORIGINAL_DIR,
        help="Directory with original SSE source code",
    )
    parser.add_argument(
        "--translated-dir",
        type=Path,
        default=DEFAULT_TRANSLATED_DIR,
        help="Directory with translated RVV source code",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASETS_DIR,
        help="Directory containing dataset files",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging(level="INFO")
    args = parse_args()
    return benchmark(
        dataset=args.dataset,
        original_dir=args.original_dir,
        translated_dir=args.translated_dir,
        dataset_dir=args.dataset_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
