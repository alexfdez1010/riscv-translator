"""Integration test: compile and run the baseline kernel on the remote RISC-V host via SSH."""

import subprocess
from pathlib import Path

import pytest

from src.config import (
    BENCH_FILE,
    BENCHMARK_FASTA,
    BLOSUM62_FILE,
    FASTA_DATASET,
    KERNEL_FILE,
    REMOTE_DIR,
    SSH_HOST,
    SSW_BENCH_FILE,
    SSW_DIR,
    SSW_FILE,
)
try:
    from src.fitness import FitnessEvaluator  # type: ignore[attr-defined]
except ImportError:
    FitnessEvaluator = None  # type: ignore[misc,assignment]

try:
    from src.fitness import SSWFitnessEvaluator  # type: ignore[attr-defined]
except ImportError:
    SSWFitnessEvaluator = None  # type: ignore[misc,assignment]

_needs_fitness_evaluator = pytest.mark.skipif(
    FitnessEvaluator is None,
    reason="FitnessEvaluator not available in src.fitness (evolutionary algorithm not in this repo)",
)
_needs_ssw_fitness_evaluator = pytest.mark.skipif(
    SSWFitnessEvaluator is None,
    reason="SSWFitnessEvaluator not available in src.fitness (evolutionary algorithm not in this repo)",
)


SSH_TIMEOUT = 30
EXEC_TIMEOUT = 300
REMOTE_KERNEL = f"{REMOTE_DIR}/test_baseline_kernel.c"
REMOTE_BIN = f"{REMOTE_DIR}/test_baseline_bench"
REMOTE_SSW_DIR = f"{REMOTE_DIR}/ssw"
REMOTE_SSW_BIN = f"{REMOTE_SSW_DIR}/bench_ssw"


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


_BASELINE_LOCAL_FILES = [BENCH_FILE, BLOSUM62_FILE, KERNEL_FILE, FASTA_DATASET]


@pytest.fixture(scope="module", autouse=True)
def setup_remote():
    """Upload benchmark files to the remote host once for the whole module."""
    missing = [f for f in _BASELINE_LOCAL_FILES if not Path(f).exists()]
    if missing:
        pytest.skip(
            f"Local benchmark files not found (csrc/ or dataset/ missing): {missing}"
        )

    subprocess.run(
        ["ssh", SSH_HOST, f"mkdir -p {REMOTE_DIR}"],
        capture_output=True,
        timeout=SSH_TIMEOUT,
        check=True,
    )
    for local, remote_name in [
        (BENCH_FILE, "bench_remote.c"),
        (BLOSUM62_FILE, "blosum62.h"),
        (KERNEL_FILE, REMOTE_KERNEL.split("/")[-1]),
        (FASTA_DATASET, "dataset.fasta"),
    ]:
        subprocess.run(
            ["scp", str(local), f"{SSH_HOST}:{REMOTE_DIR}/{remote_name}"],
            capture_output=True,
            timeout=60,
            check=True,
        )

    yield

    # Cleanup
    subprocess.run(
        ["ssh", SSH_HOST, f"rm -f {REMOTE_KERNEL} {REMOTE_BIN}"],
        capture_output=True,
        timeout=SSH_TIMEOUT,
    )


class TestBaselineKernelSSH:
    """Compile and run the baseline kernel on the remote RISC-V host."""

    def test_compile(self):
        """The baseline kernel compiles without errors on the remote host."""
        compile_cmd = (
            f"clang -o {REMOTE_BIN} {REMOTE_DIR}/bench_remote.c {REMOTE_KERNEL} "
            f"-I{REMOTE_DIR} -march=rv64gcv -O2 2>&1"
        )
        result = subprocess.run(
            ["ssh", SSH_HOST, compile_cmd],
            capture_output=True,
            timeout=60,
            text=True,
        )
        assert result.returncode == 0, (
            f"Compilation failed:\n{result.stdout}\n{result.stderr}"
        )

    @_needs_fitness_evaluator
    def test_correctness(self):
        """The baseline kernel produces correct alignment results (status=OK)."""
        # Ensure binary exists (compile first)
        compile_cmd = (
            f"clang -o {REMOTE_BIN} {REMOTE_DIR}/bench_remote.c {REMOTE_KERNEL} "
            f"-I{REMOTE_DIR} -march=rv64gcv -O2 2>&1"
        )
        comp = subprocess.run(
            ["ssh", SSH_HOST, compile_cmd],
            capture_output=True,
            timeout=60,
            text=True,
        )
        assert comp.returncode == 0, (
            f"Compilation failed:\n{comp.stdout}\n{comp.stderr}"
        )

        # Run the benchmark
        result = subprocess.run(
            ["ssh", SSH_HOST, f"{REMOTE_BIN} {REMOTE_DIR}/dataset.fasta 2>&1"],
            capture_output=True,
            timeout=EXEC_TIMEOUT,
            text=True,
        )
        assert result.returncode == 0, (
            f"Execution failed:\n{result.stdout}\n{result.stderr}"
        )

        status = FitnessEvaluator._parse_field(result.stdout, "status")
        assert status == "OK", (
            f"Correctness check failed (status={status}):\n{result.stdout}"
        )

    @_needs_fitness_evaluator
    def test_fitness_positive(self):
        """The baseline kernel gets a positive fitness score."""
        # Ensure binary exists
        compile_cmd = (
            f"clang -o {REMOTE_BIN} {REMOTE_DIR}/bench_remote.c {REMOTE_KERNEL} "
            f"-I{REMOTE_DIR} -march=rv64gcv -O2 2>&1"
        )
        comp = subprocess.run(
            ["ssh", SSH_HOST, compile_cmd],
            capture_output=True,
            timeout=60,
            text=True,
        )
        assert comp.returncode == 0, (
            f"Compilation failed:\n{comp.stdout}\n{comp.stderr}"
        )

        result = subprocess.run(
            ["ssh", SSH_HOST, f"{REMOTE_BIN} {REMOTE_DIR}/dataset.fasta 2>&1"],
            capture_output=True,
            timeout=EXEC_TIMEOUT,
            text=True,
        )
        assert result.returncode == 0, (
            f"Execution failed:\n{result.stdout}\n{result.stderr}"
        )

        time_ns_str = FitnessEvaluator._parse_field(result.stdout, "time_ns")
        assert time_ns_str is not None, f"Could not parse time_ns:\n{result.stdout}"

        time_ns = float(time_ns_str)
        assert time_ns > 0, f"Expected positive time_ns, got {time_ns}"

        fitness = 1.0 / (time_ns + 1.0)
        assert fitness > 0, f"Expected positive fitness, got {fitness}"


# ---------------------------------------------------------------------------
# SSW library (ssw.repaired.c) on remote RISC-V host
# ---------------------------------------------------------------------------

_SSW_UPLOAD_FILES = [
    (SSW_FILE, "ssw.c"),
    (SSW_DIR / "ssw.h", "ssw.h"),
    (SSW_DIR / "sse2rvv.h", "sse2rvv.h"),
    (SSW_BENCH_FILE, "bench_ssw.c"),
    (BENCHMARK_FASTA, "dataset.fa"),
]


@pytest.fixture(scope="module", autouse=False)
def setup_remote_ssw():
    """Upload SSW files + benchmark to the remote host."""
    missing = [f for f, _ in _SSW_UPLOAD_FILES if not Path(f).exists()]
    if missing:
        pytest.skip(
            f"Local SSW files not found: {missing}"
        )

    subprocess.run(
        ["ssh", SSH_HOST, f"mkdir -p {REMOTE_SSW_DIR}"],
        capture_output=True,
        timeout=SSH_TIMEOUT,
        check=True,
    )
    for local, remote_name in _SSW_UPLOAD_FILES:
        subprocess.run(
            ["scp", str(local), f"{SSH_HOST}:{REMOTE_SSW_DIR}/{remote_name}"],
            capture_output=True,
            timeout=60,
            check=True,
        )

    yield

    subprocess.run(
        ["ssh", SSH_HOST, f"rm -rf {REMOTE_SSW_DIR}"],
        capture_output=True,
        timeout=SSH_TIMEOUT,
    )


class TestBaselineSSWSSH:
    """Compile and run the baseline ssw.repaired.c on the remote RISC-V host."""

    def test_compile(self, setup_remote_ssw):
        """The SSW library compiles without errors on the remote host."""
        compile_cmd = (
            f"cd {REMOTE_SSW_DIR} && "
            f"clang -O2 -march=rv64gcv -c -o ssw.o ssw.c 2>&1 && "
            f"clang -O2 -march=rv64gcv -o bench_ssw bench_ssw.c ssw.o -lm 2>&1"
        )
        result = subprocess.run(
            ["ssh", SSH_HOST, compile_cmd],
            capture_output=True,
            timeout=60,
            text=True,
        )
        assert result.returncode == 0, (
            f"SSW compilation failed:\n{result.stdout}\n{result.stderr}"
        )

    @_needs_ssw_fitness_evaluator
    def test_correctness(self, setup_remote_ssw):
        """The SSW benchmark produces status=OK on the remote host."""
        compile_cmd = (
            f"cd {REMOTE_SSW_DIR} && "
            f"clang -O2 -march=rv64gcv -c -o ssw.o ssw.c 2>&1 && "
            f"clang -O2 -march=rv64gcv -o bench_ssw bench_ssw.c ssw.o -lm 2>&1"
        )
        comp = subprocess.run(
            ["ssh", SSH_HOST, compile_cmd],
            capture_output=True,
            timeout=60,
            text=True,
        )
        assert comp.returncode == 0, (
            f"SSW compilation failed:\n{comp.stdout}\n{comp.stderr}"
        )

        result = subprocess.run(
            ["ssh", SSH_HOST, f"cd {REMOTE_SSW_DIR} && ./bench_ssw dataset.fa 2>&1"],
            capture_output=True,
            timeout=EXEC_TIMEOUT,
            text=True,
        )
        assert result.returncode == 0, (
            f"SSW execution failed:\n{result.stdout}\n{result.stderr}"
        )

        status = SSWFitnessEvaluator._parse_field(result.stdout, "status")
        assert status == "OK", (
            f"SSW correctness check failed (status={status}):\n{result.stdout}"
        )

    @_needs_ssw_fitness_evaluator
    def test_fitness_positive(self, setup_remote_ssw):
        """The SSW benchmark reports a positive time_ns."""
        compile_cmd = (
            f"cd {REMOTE_SSW_DIR} && "
            f"clang -O2 -march=rv64gcv -c -o ssw.o ssw.c 2>&1 && "
            f"clang -O2 -march=rv64gcv -o bench_ssw bench_ssw.c ssw.o -lm 2>&1"
        )
        comp = subprocess.run(
            ["ssh", SSH_HOST, compile_cmd],
            capture_output=True,
            timeout=60,
            text=True,
        )
        assert comp.returncode == 0, (
            f"SSW compilation failed:\n{comp.stdout}\n{comp.stderr}"
        )

        result = subprocess.run(
            ["ssh", SSH_HOST, f"cd {REMOTE_SSW_DIR} && ./bench_ssw dataset.fa 2>&1"],
            capture_output=True,
            timeout=EXEC_TIMEOUT,
            text=True,
        )
        assert result.returncode == 0, (
            f"SSW execution failed:\n{result.stdout}\n{result.stderr}"
        )

        time_ns_str = SSWFitnessEvaluator._parse_field(result.stdout, "time_ns")
        assert time_ns_str is not None, (
            f"Could not parse time_ns from SSW benchmark:\n{result.stdout}"
        )

        time_ns = float(time_ns_str)
        assert time_ns > 0, f"Expected positive time_ns, got {time_ns}"
