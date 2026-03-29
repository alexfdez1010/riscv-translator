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
DEFAULT_NAIVE_DIR = PROJECT_DIR / "naive" / "code"
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
        capture_output=True, timeout=60 * 60 * 24, text=True,
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
    """Normalize alignment output for comparison.

    Strips trailing whitespace, removes empty lines, and removes CPU time
    measurements (which naturally differ between platforms).

    CPU time messages from stderr can interleave mid-line when ``2>&1`` is
    used (e.g. ``"optimal_alignmCPU time: 1.23 seconds\\nent_score: …"``),
    so we strip them from the raw string *before* splitting into lines and
    then rejoin any line that was split by the removal.
    """
    import re
    # Strip CPU time stamps that may appear anywhere (including mid-line).
    # The newline that the CPU time fprintf adds is consumed by \s* so the
    # two halves of the interrupted line get glued back together.
    cleaned = re.sub(r'CPU time:\s*[\d.]+\s*seconds\s*\n?', '', raw)
    lines = [line.rstrip() for line in cleaned.strip().splitlines()]
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
    reference: BenchmarkResult,
    other: BenchmarkResult,
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
    ref_label = reference.label
    other_label = other.label
    ref_recs = _parse_alignment_records(reference.stdout)
    other_recs = _parse_alignment_records(other.stdout)

    if len(ref_recs) != len(other_recs):
        return False, (
            f"Record count mismatch: {ref_label}={len(ref_recs)}, {other_label}={len(other_recs)}"
        )

    fields_to_check = list(_REQUIRED_FIELDS)
    if strict_suboptimal:
        fields_to_check += _OPTIONAL_FIELDS
    fields_to_check += _EXTRA_FIELDS

    mismatches: list[str] = []
    for idx, (a, b) in enumerate(zip(ref_recs, other_recs)):
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
                    f"{field} {ref_label}={va!r} {other_label}={vb!r}"
                )

    if not mismatches:
        ignored = "" if strict_suboptimal else " (suboptimal_alignment_score ignored)"
        return True, f"Outputs match: {len(ref_recs)} alignment records compared.{ignored}"

    details = [
        f"Alignment records compared: {len(ref_recs)}",
        f"Mismatched fields: {len(mismatches)}",
        "First differences:",
    ]
    details.extend(mismatches[:10])
    if len(mismatches) > 10:
        details.append(f"  ... and {len(mismatches) - 10} more")

    return False, "\n".join(details)


def _run_riscv_entry(
    label: str,
    source_dir: Path,
    final_host: str,
    remote_dir: str,
    dataset_dir: Path,
    dataset: str,
    run_cmd_suffix: str,
) -> BenchmarkResult | None:
    """Upload, compile and run one RISC-V implementation."""
    src_files = [p for p in source_dir.iterdir() if p.is_file()]
    src_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
    logger.info("Uploading %s to %s:%s ...", label, final_host, remote_dir)
    if not upload_to_host(final_host, remote_dir, src_files + src_dirs):
        return None
    if not upload_datasets(final_host, remote_dir, dataset_dir, dataset):
        return None

    logger.info("Running %s on RISC-V (%s) ...", label, final_host)
    compile_cmd = f"{SSH_CC} -o ssw_test main.c ssw.c --target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm 2>&1"
    return run_on_host(final_host, remote_dir, compile_cmd, run_cmd_suffix, label)


def benchmark(
    dataset: str = DEFAULT_DATASET,
    original_dir: Path = DEFAULT_ORIGINAL_DIR,
    translated_dir: Path = DEFAULT_TRANSLATED_DIR,
    naive_dir: Path | None = DEFAULT_NAIVE_DIR,
    dataset_dir: Path = DATASETS_DIR,
    strict_suboptimal: bool = False,
    rivals: list[tuple[str, Path]] | None = None,
) -> int:
    """Run benchmark comparing implementations.

    When *naive_dir* is provided, the naive RISC-V implementation is also
    compiled and run on the RISC-V host and included in the comparison.
    Pass ``None`` to skip the naive run.

    *rivals* is an optional list of ``(label, path)`` pairs for additional
    RISC-V implementations to include in the comparison.
    """
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
    if naive_dir:
        logger.info("Naive code: %s -> %s (RISC-V)", naive_dir, final_host)
    for label, path in (rivals or []):
        logger.info("Rival %s: %s -> %s (RISC-V)", label, path, final_host)

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

    # --- RISC-V (naive implementation) ---
    naive_result = None
    if naive_dir:
        naive_result = _run_riscv_entry(
            "RISC-V (naive)", naive_dir, final_host,
            f"{REMOTE_DIR}-bench-naive", dataset_dir, dataset, run_cmd_suffix,
        )
        if naive_result is None:
            return 1

    # --- RISC-V (translated RVV) ---
    riscv_result = _run_riscv_entry(
        "RISC-V (translated RVV)", translated_dir, final_host,
        final_remote, dataset_dir, dataset, run_cmd_suffix,
    )
    if riscv_result is None:
        return 1

    # --- RISC-V (rivals) ---
    rival_results: list[BenchmarkResult] = []
    for idx, (label, path) in enumerate(rivals or []):
        rival_remote = f"{REMOTE_DIR}-bench-rival-{idx}"
        result = _run_riscv_entry(
            label, path, final_host,
            rival_remote, dataset_dir, dataset, run_cmd_suffix,
        )
        if result is None:
            return 1
        rival_results.append(result)

    # Report
    all_results = [intel_result]
    if naive_result:
        all_results.append(naive_result)
    all_results.append(riscv_result)
    all_results.extend(rival_results)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print("-" * 60)

    for r in all_results:
        status = "PASS" if r.ok else "FAIL"
        print(f"  {r.label:30s}  {status:5s}  {r.elapsed_seconds:8.2f}s  ({r.host})")

    print("-" * 60)

    failed = [r for r in all_results if not r.ok]
    if failed:
        print("\nOne or more executions failed.")
        for r in failed:
            print(f"\n[{r.label} stderr]\n{r.stderr}")
        return 1

    # Compare outputs for correctness (all vs Intel reference)
    all_match = True
    for r in all_results[1:]:
        match, details = compare_outputs(
            intel_result, r, strict_suboptimal=strict_suboptimal,
        )
        label = r.label
        print(f"\nOutput comparison (Intel vs {label}): {'MATCH' if match else 'MISMATCH'}")
        print(details)
        if not match:
            all_match = False

    # Timing comparison vs Intel
    if intel_result.elapsed_seconds > 0:
        for r in all_results[1:]:
            ratio = r.elapsed_seconds / intel_result.elapsed_seconds
            print(f"\n{r.label} / Intel time ratio: {ratio:.2f}x")

    if naive_result and naive_result.ok and riscv_result.ok and riscv_result.elapsed_seconds > 0:
        speedup = naive_result.elapsed_seconds / riscv_result.elapsed_seconds
        print(f"\nRVV speedup over naive: {speedup:.2f}x")

    # Cross-comparison between RISC-V implementations
    riscv_entries = [r for r in all_results[1:] if r.ok]
    if len(riscv_entries) >= 2:
        ref = riscv_entries[0]
        print(f"\n--- RISC-V cross-comparison (reference: {ref.label}) ---")
        for r in riscv_entries[1:]:
            if ref.elapsed_seconds > 0:
                ratio = r.elapsed_seconds / ref.elapsed_seconds
                print(f"  {r.label:30s}  {ratio:.3f}x  ({r.elapsed_seconds:.2f}s vs {ref.elapsed_seconds:.2f}s)")

    if all_match:
        print("\nBenchmark PASSED: all implementations produce identical output.")
    else:
        print("\nBenchmark FINISHED: some outputs differ (see above).")
    return 0


def _parse_rival(value: str) -> tuple[str, Path]:
    """Parse a ``LABEL:PATH`` rival specification."""
    if ":" in value:
        label, path_str = value.split(":", 1)
        return label.strip(), Path(path_str.strip())
    # No label — derive from directory name
    p = Path(value)
    return f"RISC-V ({p.name})", p


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
        "--naive-dir",
        type=Path,
        default=DEFAULT_NAIVE_DIR,
        help="Directory with naive RISC-V implementation (omit to skip)",
    )
    parser.add_argument(
        "--no-naive",
        action="store_true",
        default=False,
        help="Skip the naive RISC-V implementation benchmark",
    )
    parser.add_argument(
        "--rival",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Additional RISC-V implementation to compare (repeatable). "
             "Format: 'My Label:path/to/dir' or just 'path/to/dir'",
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
    naive = None if args.no_naive else args.naive_dir
    rivals = [_parse_rival(r) for r in args.rival]
    return benchmark(
        dataset=args.dataset,
        original_dir=args.original_dir,
        translated_dir=args.translated_dir,
        naive_dir=naive,
        dataset_dir=args.dataset_dir,
        strict_suboptimal=args.strict_suboptimal,
        rivals=rivals,
    )


if __name__ == "__main__":
    raise SystemExit(main())
