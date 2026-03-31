"""Benchmark: run all paper experiments on RISC-V (ssh final), validate against SSE.

Comparisons:
  1. naive vs sequence-alignment — 1k, 10k, 100k
  2. sequence-alignment vs sequence-alignment-widened-auto — 10k, 100k, 1M
  3. sequence-alignment-widened vs sequence-alignment-widened-auto — 10k, 100k, 1M

Each (code_variant, dataset) pair is run 10 times.  Results are deduplicated
and written to benchmarks.csv.  Correctness is validated against the original
SSE code (initial_code) on Intel.

Usage:
    uv run python -m src.benchmark
"""

import csv
import math
import re
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

REFERENCE_FILE = "54mer_hap1_1.100.fa"
DEFAULT_ORIGINAL_DIR = PROJECT_DIR / "initial_code"
RUNS = 10
CSV_PATH = PROJECT_DIR / "benchmarks.csv"

# All RISC-V variants: label -> source dir (relative to PROJECT_DIR)
VARIANTS: dict[str, str] = {
    "naive": "naive/code",
    "sequence-alignment": "translations/sequence-alignment",
    "sequence-alignment-widened": "translations/sequence-alignment-widened",
    "sequence-alignment-widened-auto": "translations/sequence-alignment-widened-auto",
}

# Deduplicated (variant, datasets) for all three comparison groups
EXPERIMENT_PLAN: dict[str, list[str]] = {
    "naive": ["1k.fa", "10k.fa", "100k.fa"],
    "sequence-alignment": ["1k.fa", "10k.fa", "100k.fa", "1M.fa"],
    "sequence-alignment-widened": ["10k.fa", "100k.fa", "1M.fa"],
    "sequence-alignment-widened-auto": ["10k.fa", "100k.fa", "1M.fa"],
}

RISCV_COMPILE = (
    f"{SSH_CC} -o ssw_test main.c ssw.c "
    f"--target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm 2>&1"
)
INTEL_COMPILE = "gcc -O2 -o ssw_test main.c ssw.c -lm 2>&1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_ssh(host: str) -> bool:
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", host, "echo ok"],
            capture_output=True, timeout=10, text=True,
        )
        return r.returncode == 0 and "ok" in r.stdout
    except Exception:
        return False


def upload(host: str, remote_dir: str, local_paths: list[Path]) -> bool:
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


def upload_dataset(host: str, remote_dir: str, dataset_dir: Path, dataset: str) -> bool:
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


def compile_on_host(host: str, remote_dir: str, compile_cmd: str, label: str) -> bool:
    comp = subprocess.run(
        ["ssh", host, f"cd {remote_dir} && {compile_cmd}"],
        capture_output=True, timeout=120, text=True,
    )
    if comp.returncode != 0:
        logger.error("[%s] Compilation failed:\n%s\n%s", label, comp.stdout, comp.stderr)
        return False
    return True


def run_once(host: str, remote_dir: str, run_cmd: str) -> tuple[float, str, bool]:
    """Run a command on host, return (elapsed, stdout, ok)."""
    start = time.monotonic()
    r = subprocess.run(
        ["ssh", host, f"cd {remote_dir} && {run_cmd}"],
        capture_output=True, timeout=60 * 60 * 24, text=True,
    )
    elapsed = time.monotonic() - start
    return elapsed, r.stdout, r.returncode == 0


# ---------------------------------------------------------------------------
# Shared types and functions used by other modules (check, repair, widen)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BenchmarkResult:
    host: str
    label: str
    ok: bool
    elapsed_seconds: float
    stdout: str
    stderr: str


def upload_to_host(host: str, remote_dir: str, local_paths: list[Path]) -> bool:
    return upload(host, remote_dir, local_paths)


def upload_datasets(host: str, remote_dir: str, dataset_dir: Path, dataset: str) -> bool:
    return upload_dataset(host, remote_dir, dataset_dir, dataset)


def run_on_host(
    host: str,
    remote_dir: str,
    compile_cmd: str,
    run_cmd: str,
    label: str,
) -> BenchmarkResult:
    """Compile and run on a remote host, measuring execution time."""
    if not compile_on_host(host, remote_dir, compile_cmd, label):
        return BenchmarkResult(host=host, label=label, ok=False, elapsed_seconds=0, stdout="", stderr="compile failed")
    elapsed, stdout, ok = run_once(host, remote_dir, run_cmd)
    return BenchmarkResult(host=host, label=label, ok=ok, elapsed_seconds=elapsed, stdout=stdout, stderr="" if ok else "run failed")


def run_locally(
    work_dir: Path,
    compile_cmd: str,
    run_cmd: str,
    label: str,
) -> BenchmarkResult:
    """Compile and run locally, measuring execution time."""
    comp = subprocess.run(compile_cmd, shell=True, cwd=work_dir, capture_output=True, timeout=120, text=True)
    if comp.returncode != 0:
        return BenchmarkResult(host="localhost", label=label, ok=False, elapsed_seconds=0, stdout=comp.stdout, stderr=comp.stderr)
    start = time.monotonic()
    run = subprocess.run(run_cmd, shell=True, cwd=work_dir, capture_output=True, timeout=600, text=True)
    elapsed = time.monotonic() - start
    return BenchmarkResult(host="localhost", label=label, ok=run.returncode == 0, elapsed_seconds=elapsed, stdout=run.stdout, stderr=run.stderr)


def prepare_local_dir(original_dir: Path, dataset_dir: Path, dataset: str) -> Path:
    """Copy original source and dataset files into a temporary directory."""
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="bench-intel-"))
    for p in original_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, tmp / p.name)
    demo = tmp / "demo"
    demo.mkdir()
    for fname in [dataset, REFERENCE_FILE]:
        src = dataset_dir / fname
        if src.exists():
            shutil.copy2(src, demo / fname)
    return tmp


def normalize_output(raw: str) -> str:
    return _normalize(raw)


def compare_outputs(
    reference: BenchmarkResult,
    other: BenchmarkResult,
    *,
    strict_suboptimal: bool = False,
) -> tuple[bool, str]:
    """Compare alignment outputs from both runs."""
    match, details = validate_output(reference.stdout, other.stdout)
    return match, details


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[float], p: float) -> float:
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def _stats_row(times: list[float], max_runs: int) -> dict:
    s = sorted(times)
    n = len(times)
    mean = sum(times) / n if n else 0.0
    mid = n // 2
    median = ((s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]) if n else 0.0
    stdev = (math.sqrt(sum((t - mean) ** 2 for t in times) / (n - 1)) if n > 1 else 0.0)
    q1 = _percentile(s, 25)
    q3 = _percentile(s, 75)
    return {
        "n_runs": n,
        "mean": mean,
        "median": median,
        "min": min(times) if times else 0.0,
        "max": max(times) if times else 0.0,
        "stdev": stdev,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "times": times,
    }


# ---------------------------------------------------------------------------
# Output comparison (SSE validation)
# ---------------------------------------------------------------------------

def _normalize(raw: str) -> str:
    cleaned = re.sub(r'CPU time:\s*[\d.]+\s*seconds\s*\n?', '', raw)
    lines = [line.rstrip() for line in cleaned.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def _parse_records(text: str) -> list[dict[str, str]]:
    norm = _normalize(text)
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in norm.splitlines():
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
        if "optimal_alignment_score:" in line:
            for part in re.split(r"\t+", line):
                kv = re.match(r"(\S+):\s*(.+)", part.strip())
                if kv:
                    current[kv.group(1)] = kv.group(2).strip()
    if current:
        records.append(current)
    return records


_CHECK_FIELDS = [
    "target_name", "query_name", "optimal_alignment_score",
    "strand", "target_end", "query_end",
]


def validate_output(ref_stdout: str, test_stdout: str) -> tuple[bool, str]:
    ref = _parse_records(ref_stdout)
    test = _parse_records(test_stdout)
    if len(ref) != len(test):
        return False, f"record count mismatch: {len(ref)} vs {len(test)}"
    mismatches = []
    for i, (a, b) in enumerate(zip(ref, test)):
        for f in _CHECK_FIELDS:
            va, vb = a.get(f), b.get(f)
            if va is None and vb is None:
                continue
            if va != vb:
                mismatches.append(f"rec {i+1} {f}: {va!r} vs {vb!r}")
    if not mismatches:
        return True, f"{len(ref)} records match"
    return False, "; ".join(mismatches[:5])


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def write_csv(
    results: dict[tuple[str, str], dict],
    csv_path: Path,
    max_runs: int,
) -> None:
    run_cols = [f"run_{i + 1}" for i in range(max_runs)]
    header = (
        ["code_variant", "dataset", "n_runs"]
        + run_cols
        + ["mean", "median", "min", "max", "stdev", "q1", "q3", "iqr", "correct"]
    )
    rows: list[list[str]] = []
    for (variant, ds), data in sorted(results.items()):
        st = data["stats"]
        row: list[str] = [variant, ds, str(st["n_runs"])]
        for i in range(max_runs):
            row.append(f"{st['times'][i]:.6f}" if i < st["n_runs"] else "")
        for k in ["mean", "median", "min", "max", "stdev", "q1", "q3", "iqr"]:
            row.append(f"{st[k]:.6f}")
        row.append(str(data.get("correct", "")))
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    configure_logging(level="INFO")

    final_host = SSH_HOST
    jump_host = SSH_JUMP_HOST

    # --- Connectivity ---
    if not check_ssh(final_host):
        logger.error("Cannot reach RISC-V host: %s", final_host)
        return 1
    logger.info("SSH to %s (RISC-V): OK", final_host)

    has_intel = bool(jump_host) and check_ssh(jump_host)
    if has_intel:
        logger.info("SSH to %s (Intel): OK", jump_host)
    else:
        logger.warning("No Intel host — skipping SSE validation")

    # --- Phase 1: RISC-V benchmarks ---
    results: dict[tuple[str, str], dict] = {}
    all_datasets: set[str] = set()

    for variant, datasets in EXPERIMENT_PLAN.items():
        source_dir = PROJECT_DIR / VARIANTS[variant]
        remote_dir = f"{REMOTE_DIR}-paper-{variant}"

        # Upload source
        src_items = list(source_dir.iterdir())
        logger.info("Uploading %s ...", variant)
        if not upload(final_host, remote_dir, src_items):
            return 1

        # Compile once
        if not compile_on_host(final_host, remote_dir, RISCV_COMPILE, variant):
            return 1
        logger.info("Compiled %s OK", variant)

        # Run each dataset
        for ds in datasets:
            all_datasets.add(ds)
            if not upload_dataset(final_host, remote_dir, DATASETS_DIR, ds):
                return 1

            run_cmd = f"./ssw_test demo/{ds} demo/{REFERENCE_FILE} 2>&1"
            label = f"{variant} | {ds}"
            logger.info("Benchmarking %s (%d runs) ...", label, RUNS)

            times: list[float] = []
            first_stdout = ""
            for i in range(RUNS):
                elapsed, stdout, ok = run_once(final_host, remote_dir, run_cmd)
                if not ok:
                    logger.error("[%s] run %d failed", label, i + 1)
                    continue
                times.append(elapsed)
                if i == 0:
                    first_stdout = stdout
                logger.info("  %s  run %d/%d: %.2fs", label, i + 1, RUNS, elapsed)

            if not times:
                logger.error("All runs failed for %s", label)
                return 1

            results[(variant, ds)] = {
                "stats": _stats_row(times, RUNS),
                "stdout": first_stdout,
            }

    # --- Phase 2: SSE validation ---
    if has_intel:
        intel_dir = PROJECT_DIR / "initial_code"
        intel_remote = f"{REMOTE_DIR}-paper-sse-ref"

        src_files = [p for p in intel_dir.iterdir() if p.is_file()]
        logger.info("Uploading SSE reference to %s ...", jump_host)
        if upload(jump_host, intel_remote, src_files) and compile_on_host(
            jump_host, intel_remote, INTEL_COMPILE, "SSE ref"
        ):
            for ds in sorted(all_datasets):
                if not upload_dataset(jump_host, intel_remote, DATASETS_DIR, ds):
                    continue
                run_cmd = f"./ssw_test demo/{ds} demo/{REFERENCE_FILE} 2>&1"
                _, ref_stdout, ok = run_once(jump_host, intel_remote, run_cmd)
                if not ok:
                    logger.warning("SSE ref run failed for %s", ds)
                    continue
                logger.info("SSE reference for %s: OK", ds)

                # Validate all variants that used this dataset
                for (v, d), data in results.items():
                    if d != ds:
                        continue
                    match, details = validate_output(ref_stdout, data["stdout"])
                    data["correct"] = match
                    status = "PASS" if match else "FAIL"
                    logger.info("  Validation %s | %s: %s — %s", v, ds, status, details)

    # --- Phase 3: Write CSV ---
    write_csv(results, CSV_PATH, RUNS)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("PAPER BENCHMARK RESULTS")
    print("=" * 80)
    for (variant, ds), data in sorted(results.items()):
        st = data["stats"]
        correct = data.get("correct", "")
        tag = " [OK]" if correct is True else " [MISMATCH]" if correct is False else ""
        print(
            f"  {variant:35s} {ds:10s}  "
            f"mean={st['mean']:.2f}s  median={st['median']:.2f}s  "
            f"stdev={st['stdev']:.2f}s  min={st['min']:.2f}s  max={st['max']:.2f}s"
            f"{tag}"
        )
    print("=" * 80)
    print(f"CSV written to: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
