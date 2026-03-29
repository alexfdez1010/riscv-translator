"""Benchmark: run original SSE code on Intel and translated RVV code on RISC-V.

When SSH_JUMP_HOST is set, the Intel reference runs on that remote host.
When SSH_JUMP_HOST is unset (default), the Intel reference runs locally.

Validates that both produce identical alignment output, then compares execution time.

Usage:
    uv run python -m src.benchmark [--dataset FILE] [--original-dir DIR] [--translated-dir DIR]
"""

import argparse
import shutil
import subprocess
import tempfile
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


def run_locally(
    work_dir: Path,
    compile_cmd: str,
    run_cmd: str,
    label: str,
) -> BenchmarkResult:
    """Compile and run locally, measuring execution time."""
    # Compile
    comp = subprocess.run(
        compile_cmd, shell=True, cwd=work_dir,
        capture_output=True, timeout=120, text=True,
    )
    if comp.returncode != 0:
        logger.error("[%s] Compilation failed:\n%s\n%s", label, comp.stdout, comp.stderr)
        return BenchmarkResult(
            host="localhost", label=label, ok=False, elapsed_seconds=0,
            stdout=comp.stdout, stderr=comp.stderr,
        )

    # Run with timing
    start = time.monotonic()
    run = subprocess.run(
        run_cmd, shell=True, cwd=work_dir,
        capture_output=True, timeout=600, text=True,
    )
    elapsed = time.monotonic() - start

    ok = run.returncode == 0
    if not ok:
        logger.error("[%s] Execution failed (rc=%d):\n%s\n%s", label, run.returncode, run.stdout, run.stderr)

    return BenchmarkResult(
        host="localhost", label=label, ok=ok, elapsed_seconds=elapsed,
        stdout=run.stdout, stderr=run.stderr,
    )


def prepare_local_dir(
    original_dir: Path,
    dataset_dir: Path,
    dataset: str,
) -> Path:
    """Copy original source and dataset files into a temporary directory."""
    tmp = Path(tempfile.mkdtemp(prefix="bench-intel-"))
    # Copy source files
    for p in original_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, tmp / p.name)
    # Create demo/ subdirectory with datasets
    demo = tmp / "demo"
    demo.mkdir()
    for fname in [dataset, REFERENCE_FILE]:
        src = dataset_dir / fname
        if src.exists():
            shutil.copy2(src, demo / fname)
        else:
            logger.error("Dataset file not found: %s", src)
    return tmp


def normalize_output(raw: str) -> str:
    """Normalize alignment output for comparison: strip trailing whitespace per line, skip empty lines,
    and remove CPU time measurements (which naturally differ between platforms)."""
    import re
    lines = [line.rstrip() for line in raw.strip().splitlines()]
    # Remove CPU time suffixes like "CPU time: 0.074732 seconds"
    lines = [re.sub(r'CPU time:\s*[\d.]+\s*seconds', '', line).rstrip() for line in lines]
    return "\n".join(line for line in lines if line)


def _parse_alignment_records(text: str) -> list[dict[str, str]]:
    """Parse SSW alignment output into a list of records.

    Each record is a dict of field name → value extracted from the tab-separated
    score line.  The ``target_name`` and ``query_name`` lines that precede it are
    included as fields too.

    Recognised fields (all optional except optimal_alignment_score):
        target_name, query_name, optimal_alignment_score,
        suboptimal_alignment_score, strand, target_begin, target_end,
        query_begin, query_end.
    """
    import re

    norm = normalize_output(text)
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for line in norm.splitlines():
        # "target_name: <value>" starts a new record
        m = re.match(r"target_name:\s*(.+)", line)
        if m:
            if current:
                records.append(current)
            current = {"target_name": m.group(1).strip()}
            continue

        m = re.match(r"query_name:\s*(.+)", line)
        if m:
            current["query_name"] = m.group(1).strip()
            continue

        # Tab-separated key: value pairs on the score line
        if "optimal_alignment_score:" in line:
            for part in re.split(r"\t+", line):
                kv = re.match(r"(\S+):\s*(.+)", part.strip())
                if kv:
                    current[kv.group(1)] = kv.group(2).strip()

    if current:
        records.append(current)

    return records


# Fields that must match for correctness.
_REQUIRED_FIELDS = [
    "target_name",
    "query_name",
    "optimal_alignment_score",
    "strand",
    "target_end",
    "query_end",
]

# Fields compared only when strict_suboptimal is True.
_OPTIONAL_FIELDS = [
    "suboptimal_alignment_score",
]

# Fields that are compared when present in both records but not required.
_EXTRA_FIELDS = [
    "target_begin",
    "query_begin",
]


def compare_outputs(
    intel: BenchmarkResult,
    riscv: BenchmarkResult,
    *,
    strict_suboptimal: bool = False,
) -> tuple[bool, str]:
    """Compare alignment outputs from both runs.

    Parses each output into structured alignment records and compares the key
    fields (optimal score, strand, target/query end positions).

    When *strict_suboptimal* is ``True`` the ``suboptimal_alignment_score`` field
    is also compared; otherwise it is ignored (it is implementation-dependent).

    Returns ``(match, details)``.
    """
    intel_recs = _parse_alignment_records(intel.stdout)
    riscv_recs = _parse_alignment_records(riscv.stdout)

    if len(intel_recs) != len(riscv_recs):
        return False, (
            f"Record count mismatch: Intel={len(intel_recs)}, RISC-V={len(riscv_recs)}"
        )

    fields_to_check = list(_REQUIRED_FIELDS)
    if strict_suboptimal:
        fields_to_check += _OPTIONAL_FIELDS
    fields_to_check += _EXTRA_FIELDS

    mismatches: list[str] = []
    for idx, (a, b) in enumerate(zip(intel_recs, riscv_recs)):
        for field in fields_to_check:
            va = a.get(field)
            vb = b.get(field)
            # Skip fields missing from both sides.
            if va is None and vb is None:
                continue
            # For extra (non-required) fields, skip if either side is missing.
            if field in _EXTRA_FIELDS and (va is None or vb is None):
                continue
            if va != vb:
                query = a.get("query_name", f"record #{idx+1}")
                mismatches.append(
                    f"  record {idx+1} ({query}): "
                    f"{field} Intel={va!r} RISC-V={vb!r}"
                )

    if not mismatches:
        ignored = "" if strict_suboptimal else " (suboptimal_alignment_score ignored)"
        return True, f"Outputs match: {len(intel_recs)} alignment records compared.{ignored}"

    details = [
        f"Alignment records compared: {len(intel_recs)}",
        f"Mismatched fields: {len(mismatches)}",
        "First differences:",
    ]
    details.extend(mismatches[:10])
    if len(mismatches) > 10:
        details.append(f"  ... and {len(mismatches) - 10} more")

    return False, "\n".join(details)


def benchmark(
    dataset: str = DEFAULT_DATASET,
    original_dir: Path = DEFAULT_ORIGINAL_DIR,
    translated_dir: Path = DEFAULT_TRANSLATED_DIR,
    dataset_dir: Path = DATASETS_DIR,
    strict_suboptimal: bool = False,
) -> int:
    """Run benchmark comparing original SSE code on Intel vs translated RVV code on RISC-V."""
    run_intel_locally = not SSH_JUMP_HOST
    jump_host = SSH_JUMP_HOST
    final_host = SSH_HOST
    final_remote = f"{REMOTE_DIR}-bench-translated"

    logger.info("Benchmark dataset: %s", dataset)
    if run_intel_locally:
        logger.info("Original code: %s -> localhost (Intel, local)", original_dir)
    else:
        logger.info("Original code: %s -> %s (Intel, SSH)", original_dir, jump_host)
    logger.info("Translated code: %s -> %s (RISC-V)", translated_dir, final_host)

    # Check connectivity for remote hosts
    if not run_intel_locally:
        if not check_ssh(jump_host):
            logger.error("Cannot reach jump host: %s", jump_host)
            return 1
        logger.info("SSH to %s: OK", jump_host)

    if not check_ssh(final_host):
        logger.error("Cannot reach final host: %s", final_host)
        return 1
    logger.info("SSH to %s: OK", final_host)

    run_cmd_suffix = f"./ssw_test demo/{dataset} demo/{REFERENCE_FILE} 2>&1"

    # --- Intel (original SSE) ---
    intel_compile = "gcc -O2 -o ssw_test main.c ssw.c -lm 2>&1"

    if run_intel_locally:
        logger.info("Running original code locally (Intel) ...")
        local_dir = prepare_local_dir(original_dir, dataset_dir, dataset)
        try:
            intel_result = run_locally(
                local_dir, intel_compile, run_cmd_suffix, "Intel (original SSE)",
            )
        finally:
            shutil.rmtree(local_dir, ignore_errors=True)
    else:
        jump_remote = f"{REMOTE_DIR}-bench-original"
        original_files = [p for p in original_dir.iterdir() if p.is_file()]
        logger.info("Uploading original code to %s:%s ...", jump_host, jump_remote)
        if not upload_to_host(jump_host, jump_remote, original_files):
            return 1
        if not upload_datasets(jump_host, jump_remote, dataset_dir, dataset):
            return 1
        logger.info("Running original code on Intel (%s) ...", jump_host)
        intel_result = run_on_host(
            jump_host, jump_remote, intel_compile, run_cmd_suffix, "Intel (original SSE)",
        )

    # --- RISC-V (translated RVV) ---
    translated_files = [p for p in translated_dir.iterdir() if p.is_file()]
    translated_dirs = [p for p in translated_dir.iterdir() if p.is_dir()]
    logger.info("Uploading translated code to %s:%s ...", final_host, final_remote)
    if not upload_to_host(final_host, final_remote, translated_files + translated_dirs):
        return 1
    if not upload_datasets(final_host, final_remote, dataset_dir, dataset):
        return 1

    logger.info("Running translated code on RISC-V (%s) ...", final_host)
    riscv_compile = f"{SSH_CC} -o ssw_test main.c ssw.c --target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm 2>&1"
    riscv_result = run_on_host(
        final_host, final_remote, riscv_compile, run_cmd_suffix, "RISC-V (translated RVV)",
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
    match, details = compare_outputs(
        intel_result, riscv_result, strict_suboptimal=strict_suboptimal,
    )

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
    parser.add_argument(
        "--strict-suboptimal",
        action="store_true",
        default=False,
        help="Also compare suboptimal_alignment_score (implementation-dependent)",
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
        strict_suboptimal=args.strict_suboptimal,
    )


if __name__ == "__main__":
    raise SystemExit(main())
