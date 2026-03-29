"""Vector-width optimization pipeline for RISC-V Vector (RVV) code.

Takes translated RVV code that is constrained to 128-bit (SSE-compatible)
vector operations and iteratively widens it to exploit the full hardware
VLEN.  Uses the same LLM compile-fix loop as ``repair.py``:

  1. Start with 128-bit-constrained translated code.
  2. LLM proposes diffs to widen a portion of the code.
  3. Compile + run on simulator (VLEN=128) to verify no regression.
  4. Compile + run on SSH hardware (VLEN=256) to verify wider execution.
  5. Compare both outputs against Intel reference for correctness.
  6. Feed errors back to LLM and iterate.

On success the widened code is written to the output directory.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from src.llm_types import LLM, Message
from src.config import (
    DATASETS_DIR,
    LLM_VALIDATION_RETRIES,
    PROJECT_DIR,
    REACT_MAX_STEPS,
    REMOTE_DIR,
    RISCVCC,
    SIMULATOR,
    SSH_CC,
    SSH_HOST,
    SSH_JUMP_HOST,
)
from src.benchmark import (
    BenchmarkResult,
    REFERENCE_FILE as BENCH_REFERENCE_FILE,
    check_ssh,
    compare_outputs,
    run_on_host,
    upload_datasets,
    upload_to_host,
)
from src.search_replace import (
    apply_search_replace,
    extract_search_replace,
)
from src.widen_prompts import (
    build_widen_continue_prompt,
    build_widen_edit_format_feedback,
    build_widen_initial_prompt,
    build_widen_repair_prompt,
    build_widen_system_prompt,
)
from src.llm_utils import create_llm
from src.logger import configure_logging, get_logger
from src.validators import (
    DockerValidator,
    SSHValidator,
    ValidationResult,
)

logger = get_logger(__name__)

MAX_OUTPUT_CHARS = 16000
CORRECTNESS_DATASET = "10k.fa"

DEFAULT_SOURCE_DIR = PROJECT_DIR / "translations" / "sequence-alignment"
ORIGINAL_SOURCE_DIR = PROJECT_DIR / "initial_code"

# Separator used to concatenate .h and .c files for the LLM
_FILE_SEPARATOR = "\n/* ===== END ssw.h / BEGIN ssw.c ===== */\n"
HEADER_FILE = "ssw.h"


def truncate_for_log(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


# ---------------------------------------------------------------------------
# Snapshot and workspace (reused from repair.py pattern)
# ---------------------------------------------------------------------------


class SourceSnapshot:
    __slots__ = ("files",)

    def __init__(self, files: dict[str, str]):
        self.files = files


class WorkspaceSet:
    __slots__ = ("root", "workspace_dir")

    def __init__(self, root: Path, workspace_dir: Path):
        self.root = root
        self.workspace_dir = workspace_dir


def materialize_snapshot(workspace_dir: Path, snapshot: SourceSnapshot) -> None:
    for name, content in snapshot.files.items():
        (workspace_dir / name).write_text(content)


def create_workspace(
    source_dir: Path,
    snapshot: SourceSnapshot,
    test_data_dir: Path | None = None,
) -> WorkspaceSet:
    root = Path(tempfile.mkdtemp(prefix="widen-"))
    workspace_dir = root / "workspace"
    shutil.copytree(source_dir, workspace_dir)
    materialize_snapshot(workspace_dir, snapshot)
    if test_data_dir is not None and test_data_dir.is_dir():
        demo_dir = workspace_dir / "demo"
        shutil.copytree(test_data_dir, demo_dir)
    return WorkspaceSet(root=root, workspace_dir=workspace_dir)


def apply_content_to_snapshot(
    snapshot: SourceSnapshot, file_name: str, content: str
) -> SourceSnapshot:
    if file_name not in snapshot.files:
        raise ValueError(f"Unknown target file: {file_name}")
    updated = dict(snapshot.files)
    updated[file_name] = content
    return SourceSnapshot(files=updated)


def write_output(output_dir: Path, snapshot: SourceSnapshot) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in snapshot.files.items():
        (output_dir / name).write_text(content)
    logger.info("Wrote %d file(s) to %s", len(snapshot.files), output_dir)


def concat_header_and_source(snapshot: SourceSnapshot, target_file: str) -> str:
    """Concatenate ssw.h + ssw.c into a single string for the LLM."""
    header = snapshot.files.get(HEADER_FILE, "")
    source = snapshot.files[target_file]
    return header + _FILE_SEPARATOR + source


def split_header_and_source(combined: str) -> tuple[str, str]:
    """Split the concatenated string back into (ssw.h, ssw.c)."""
    parts = combined.split(_FILE_SEPARATOR, 1)
    if len(parts) != 2:
        raise ValueError("Could not find file separator in LLM output")
    return parts[0], parts[1]


def _to_messages(raw: list[dict[str, str]]) -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in raw]


# ---------------------------------------------------------------------------
# Build commands
# ---------------------------------------------------------------------------


def default_docker_build_command() -> str:
    cflags = "-O2 -I. -march=rv64gcv -mabi=lp64d"
    ldflags = "-lm"
    return (
        f'echo "=== Compiling ===" && '
        f'{RISCVCC} {cflags} main.c ssw.c -o ssw_test {ldflags} 2>&1 && '
        f'echo "=== Compilation succeeded, running under QEMU ===" && '
        f'{SIMULATOR} ./ssw_test demo/10k.fa demo/54mer_hap1_1.100.fa >/dev/null 2>&1 && '
        f'echo "=== Execution succeeded ==="'
    )


def default_ssh_compile_command() -> str:
    return f"{SSH_CC} -o ssw_test main.c ssw.c --target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm 2>&1"


def default_ssh_run_command() -> str:
    return "./ssw_test demo/10k.fa demo/54mer_hap1_1.100.fa 2>&1"


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def _detect_target_file(source_dir: Path, file_names: list[str]) -> str:
    """Find the .c file that contains SSE intrinsics (_mm_*)."""
    candidates = []
    for name in file_names:
        if not name.endswith(".c"):
            continue
        content = (source_dir / name).read_text()
        if "_mm_" in content:
            candidates.append(name)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # If multiple .c files have SSE intrinsics, pick the one that is NOT main.c
        non_main = [c for c in candidates if c != "main.c"]
        if len(non_main) == 1:
            return non_main[0]
        raise ValueError(
            f"Multiple .c files contain SSE intrinsics: {candidates}. "
            "Specify target_file explicitly."
        )
    raise ValueError(
        f"No .c file with SSE intrinsics (_mm_*) found in {source_dir}. "
        "Specify target_file explicitly."
    )


# ---------------------------------------------------------------------------
# Widening agent
# ---------------------------------------------------------------------------


class WidenAgent:
    """LLM-driven vector-width optimization with compile-fix loop."""

    def __init__(self):
        self.docker_validator = DockerValidator()
        self.ssh_validator = SSHValidator()
        self.llm: LLM = create_llm()
        self._intel_reference: BenchmarkResult | None = None
        self._intel_reference_computed = False

    def _get_intel_reference(self, original_dir: Path, test_data_dir: Path | None) -> BenchmarkResult | None:
        """Compute Intel reference by compiling the *original* SSE code on the jump host.

        The original_dir must contain the unmodified SSE source (e.g. initial_code/)
        — NOT the translated RVV code, which uses sse2rvv.h and won't compile on x86.
        """
        if self._intel_reference_computed:
            return self._intel_reference

        self._intel_reference_computed = True
        jump_host = SSH_JUMP_HOST
        if not check_ssh(jump_host):
            logger.warning("Jump host %s not reachable; correctness check disabled", jump_host)
            return None

        jump_remote = f"{REMOTE_DIR}-widen-ref"
        dataset_dir = test_data_dir if test_data_dir and test_data_dir.is_dir() else DATASETS_DIR

        dataset_file = dataset_dir / CORRECTNESS_DATASET
        if not dataset_file.exists():
            logger.warning("Correctness dataset %s not found; correctness check disabled", dataset_file)
            return None

        # Upload the ORIGINAL SSE source (not the translated RVV code)
        original_files = [p for p in original_dir.iterdir() if p.is_file()]
        logger.info("Uploading original SSE code to %s for Intel reference ...", jump_host)
        if not upload_to_host(jump_host, jump_remote, original_files):
            logger.warning("Failed to upload reference code; correctness check disabled")
            return None
        if not upload_datasets(jump_host, jump_remote, dataset_dir, CORRECTNESS_DATASET):
            logger.warning("Failed to upload datasets; correctness check disabled")
            return None

        run_cmd = f"./ssw_test demo/{CORRECTNESS_DATASET} demo/{BENCH_REFERENCE_FILE} 2>&1"
        compile_cmd = "gcc -O2 -o ssw_test main.c ssw.c -lm 2>&1"

        result = run_on_host(jump_host, jump_remote, compile_cmd, run_cmd, "Intel reference")
        if not result.ok:
            logger.warning("Intel reference run failed; correctness check disabled\n%s", result.stderr)
            return None

        logger.info("Intel reference output computed (%d chars)", len(result.stdout))
        self._intel_reference = result
        return result

    def _validate_correctness(self, workspace_dir: Path, original_code: str | None = None) -> ValidationResult | None:
        if self._intel_reference is None:
            return None

        final_host = SSH_HOST
        final_remote = f"{REMOTE_DIR}-widen-correctness"

        all_paths = [p for p in workspace_dir.iterdir()]
        if not upload_to_host(final_host, final_remote, all_paths):
            logger.warning("Failed to upload widened code for correctness check")
            return None

        dataset_dir = DATASETS_DIR
        if not upload_datasets(final_host, final_remote, dataset_dir, CORRECTNESS_DATASET):
            logger.warning("Failed to upload datasets for correctness check")
            return None

        run_cmd = f"./ssw_test demo/{CORRECTNESS_DATASET} demo/{BENCH_REFERENCE_FILE} 2>&1"
        compile_cmd = default_ssh_compile_command()

        riscv_result = run_on_host(final_host, final_remote, compile_cmd, run_cmd, "RISC-V widened")
        if not riscv_result.ok:
            return ValidationResult(
                ok=False,
                stage="correctness",
                returncode=riscv_result.ok,
                stdout=riscv_result.stdout,
                stderr=f"RISC-V widened run failed:\n{riscv_result.stderr}",
            )

        match, details = compare_outputs(self._intel_reference, riscv_result)
        if match:
            logger.info("Correctness check PASSED: widened output matches Intel reference")
            return ValidationResult(ok=True, stage="correctness", returncode=0, stdout=details, stderr="")

        logger.warning("Correctness check FAILED: widened output differs from Intel reference")

        original_hint = ""
        if original_code:
            original_hint = (
                "\n\n## Original 128-bit code (for comparison)\n"
                "Compare your widened code against this to find where the "
                "data layout or loop bounds diverged:\n"
                f"```c\n{original_code}\n```\n"
            )

        return ValidationResult(
            ok=False,
            stage="correctness",
            returncode=1,
            stdout="",
            stderr=(
                "CORRECTNESS FAILURE: The widened RISC-V code produces different "
                "alignment results than the Intel SSE reference.\n\n"
                f"{details}\n\n"
                "The widening introduced a bug. Common causes:\n"
                "1. Segment length (segLen) not updated consistently — profile "
                "construction and DP loop must use the same vector length.\n"
                "2. Memory allocation still uses hardcoded 16 instead of runtime "
                "vector byte size.\n"
                "3. Pointer arithmetic stride mismatch between write and read paths.\n"
                "4. Shuffle/slide operations not adjusted for wider vectors.\n"
                "5. Boundary conditions (e.g. segLen calculation, loop termination) "
                "not updated for the new vector length.\n"
                "6. Lazy_F loop exit: must check ALL lanes (use vcpop on mask), "
                "not just lane 0.\n"
                "7. Profile builder writes vl bytes per segment but consumer reads "
                "with a different stride.\n\n"
                "Fix the bug while keeping the code VLEN-agnostic."
                f"{original_hint}"
            ),
        )

    def _benchmark_on_ssh(
        self,
        workspace_dir: Path,
        ssh_compile_cmd: str,
        ssh_run_cmd: str,
        label: str,
    ) -> float | None:
        """Run a timed execution on SSH hardware. Returns elapsed seconds or None on failure."""
        final_host = SSH_HOST
        if not check_ssh(final_host):
            return None

        bench_remote = f"{REMOTE_DIR}-widen-bench"
        all_paths = [p for p in workspace_dir.iterdir()]
        if not upload_to_host(final_host, bench_remote, all_paths):
            return None

        dataset_dir = DATASETS_DIR
        if not upload_datasets(final_host, bench_remote, dataset_dir, CORRECTNESS_DATASET):
            return None

        result = run_on_host(final_host, bench_remote, ssh_compile_cmd, ssh_run_cmd, label)
        if not result.ok:
            logger.warning("Benchmark run failed for %s: %s", label, result.stderr)
            return None

        return result.elapsed_seconds

    def _generate_valid_file(
        self,
        messages: list[dict[str, str]],
        snapshot: SourceSnapshot,
        file_name: str,
        workspaces: WorkspaceSet,
        build_command: str,
    ) -> tuple[SourceSnapshot | None, ValidationResult]:
        """Run one LLM request cycle with retries on diff/validation failure."""
        active_messages = list(messages)
        current_snapshot = snapshot
        latest_validation = ValidationResult(
            ok=False,
            stage="edit-failure",
            returncode=None,
            stdout="",
            stderr="No validation attempted.",
        )

        for attempt in range(LLM_VALIDATION_RETRIES + 1):
            try:
                logger.info(
                    "LLM request attempt %d with %d message(s)",
                    attempt + 1,
                    len(active_messages),
                )
                response = self.llm(_to_messages(active_messages))
            except Exception as exc:
                latest_validation = ValidationResult(
                    ok=False,
                    stage="internal-error",
                    returncode=None,
                    stdout="",
                    stderr=str(exc),
                )
                logger.warning("LLM generation failed on attempt %d: %s", attempt + 1, exc)
                return None, latest_validation

            logger.info(
                "LLM response (attempt %d, %d chars):\n%s",
                attempt + 1,
                len(response),
                truncate_for_log(response, 3000),
            )

            # Check if LLM says all widening is done
            if "ALL_WIDENED" in response and "No more SSE intrinsics" in response:
                latest_validation = ValidationResult(
                    ok=True,
                    stage="all-widened",
                    returncode=0,
                    stdout="LLM reports all SSE intrinsics have been widened.",
                    stderr="",
                )
                return None, latest_validation

            # Extract and apply search/replace blocks
            candidate_snapshot = None
            edit_error = None

            sr_blocks = extract_search_replace(response)
            if sr_blocks is not None:
                logger.info("Extracted %d search/replace block(s)", len(sr_blocks))
                try:
                    combined = concat_header_and_source(current_snapshot, file_name)
                    new_combined = apply_search_replace(combined, sr_blocks)
                    new_header, new_source = split_header_and_source(new_combined)
                    updated_files = dict(current_snapshot.files)
                    updated_files[HEADER_FILE] = new_header
                    updated_files[file_name] = new_source
                    candidate_snapshot = SourceSnapshot(files=updated_files)
                except ValueError as exc:
                    logger.warning("Search/replace failed on attempt %d: %s", attempt + 1, exc)
                    edit_error = str(exc)

            if candidate_snapshot is None:
                error_msg = edit_error or (
                    "Could not extract edits from the response. "
                    "Use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks."
                )
                logger.warning("No valid edits on attempt %d: %s", attempt + 1, error_msg)
                if attempt >= LLM_VALIDATION_RETRIES:
                    return None, latest_validation
                active_messages = active_messages + [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": build_widen_edit_format_feedback(
                            file_name,
                            concat_header_and_source(current_snapshot, file_name),
                            error_msg,
                        ),
                    },
                ]
                continue

            # Validate in Docker/QEMU (VLEN=128 regression check)
            materialize_snapshot(workspaces.workspace_dir, candidate_snapshot)
            latest_validation = self.docker_validator.validate(
                workspaces.workspace_dir, build_command,
            )
            logger.info(
                "Validation result (attempt %d): ok=%s stage=%s rc=%s\n%s",
                attempt + 1,
                latest_validation.ok,
                latest_validation.stage,
                latest_validation.returncode,
                truncate_for_log(latest_validation.combined_output, 2000),
            )

            if latest_validation.ok:
                return candidate_snapshot, latest_validation

            if attempt >= LLM_VALIDATION_RETRIES:
                logger.warning(
                    "Validation failed after %d attempt(s); returning latest snapshot",
                    attempt + 1,
                )
                return candidate_snapshot, latest_validation

            # Feed errors back
            active_messages = active_messages + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": build_widen_repair_prompt(
                        file_name,
                        concat_header_and_source(candidate_snapshot, file_name),
                        latest_validation.as_feedback(),
                    ),
                },
            ]
            current_snapshot = candidate_snapshot

        return None, latest_validation

    def run(
        self,
        source_dir: Path,
        output_dir: Path,
        target_file: str | None = None,
        build_command: str | None = None,
        ssh_compile_command: str | None = None,
        ssh_run_command: str | None = None,
        max_steps: int = REACT_MAX_STEPS,
        test_data_dir: Path | None = None,
    ) -> int:
        """Run the widening pipeline.

        Args:
            source_dir: Directory with the 128-bit translated code.
            output_dir: Where to write the widened output.
            target_file: File to widen (e.g. "ssw.c").  If None, auto-detected
                as the .c file containing ``_mm_`` SSE intrinsics.
            build_command: Docker build+test command. Auto-generated if None.
            ssh_compile_command: SSH compile command. Auto-generated if None.
            ssh_run_command: SSH run command. Auto-generated if None.
            max_steps: Maximum widening passes.
            test_data_dir: Directory with test data; copied as demo/.
        """
        if build_command is None:
            build_command = default_docker_build_command()
        if ssh_compile_command is None:
            ssh_compile_command = default_ssh_compile_command()
        if ssh_run_command is None:
            ssh_run_command = default_ssh_run_command()

        file_names = [f.name for f in source_dir.iterdir() if f.is_file()]

        # Auto-detect target file if not specified
        if target_file is None:
            target_file = _detect_target_file(source_dir, file_names)

        if target_file not in file_names:
            raise ValueError(f"Target file {target_file} not found in {source_dir}")

        logger.info("Starting widening for %s in %s", target_file, source_dir)

        snapshot = SourceSnapshot(
            files={name: (source_dir / name).read_text() for name in file_names}
        )

        workspaces = create_workspace(source_dir, snapshot, test_data_dir)

        # Compute Intel reference using the ORIGINAL SSE source (initial_code/),
        # not the translated RVV code which won't compile on x86.
        self._get_intel_reference(ORIGINAL_SOURCE_DIR, test_data_dir)

        try:
            # Baseline: verify the input code compiles and runs
            baseline = self.docker_validator.validate(
                workspaces.workspace_dir, build_command,
            )
            logger.info(
                "Baseline validation: ok=%s stage=%s\n%s",
                baseline.ok,
                baseline.stage,
                truncate_for_log(baseline.combined_output, 3000),
            )

            if not baseline.ok:
                logger.error(
                    "Input code does not pass baseline validation; "
                    "fix compilation/runtime errors before widening."
                )
                return 1

            # Benchmark the original translated code to establish baseline timing
            baseline_elapsed = self._benchmark_on_ssh(
                workspaces.workspace_dir, ssh_compile_command, ssh_run_command,
                "original translated (baseline)",
            )
            if baseline_elapsed is not None:
                logger.info("Baseline timing: %.2fs", baseline_elapsed)
            else:
                logger.warning("Could not establish baseline timing; speedup tracking disabled")

            # Main widening loop — each step is one widening pass
            current_snapshot = snapshot
            last_known_good = snapshot
            successful_passes = 0
            pending_feedback: str | None = None  # validation errors to feed forward

            for step in range(1, max_steps + 1):
                logger.info("Widening pass %d/%d for %s", step, max_steps, target_file)
                current_code = concat_header_and_source(current_snapshot, target_file)

                if step == 1:
                    user_content = build_widen_initial_prompt(
                        target_file,
                        current_code,
                        build_command,
                        validation_feedback=pending_feedback,
                    )
                else:
                    user_content = build_widen_continue_prompt(
                        target_file,
                        current_code,
                        build_command,
                        pass_number=step,
                        validation_feedback=pending_feedback,
                    )

                messages = [
                    {"role": "system", "content": build_widen_system_prompt(target_file)},
                    {"role": "user", "content": user_content},
                ]

                repaired_snapshot, latest_validation = self._generate_valid_file(
                    messages,
                    current_snapshot,
                    target_file,
                    workspaces,
                    build_command,
                )

                # Check if LLM says everything is widened
                if latest_validation.stage == "all-widened":
                    logger.info("LLM reports all widening complete after %d pass(es)", successful_passes)
                    write_output(output_dir, last_known_good)
                    return 0

                if repaired_snapshot is None:
                    if latest_validation.stage == "internal-error":
                        logger.warning("Stopping due to unrecoverable error: %s", latest_validation.stderr)
                        write_output(output_dir, last_known_good)
                        return 1
                    logger.info("Pass %d did not yield a valid candidate, continuing with pending feedback", step)
                    pending_feedback = latest_validation.as_feedback() if not latest_validation.ok else None
                    continue

                if not latest_validation.ok:
                    logger.warning(
                        "Pass %d failed Docker/QEMU validation; keeping snapshot and feeding errors forward",
                        step,
                    )
                    current_snapshot = repaired_snapshot
                    pending_feedback = latest_validation.as_feedback()
                    continue

                # Docker/QEMU passed — try SSH hardware
                pending_feedback = None
                if ssh_compile_command and ssh_run_command:
                    ssh_files = [p for p in workspaces.workspace_dir.iterdir()]
                    ssh_result = self.ssh_validator.validate(
                        ssh_files, ssh_compile_command, ssh_run_command,
                    )
                    if not ssh_result.ok:
                        logger.warning(
                            "Pass %d passed QEMU but failed SSH at stage %s; "
                            "keeping snapshot and feeding errors forward\n%s",
                            step, ssh_result.stage, ssh_result.combined_output,
                        )
                        current_snapshot = repaired_snapshot
                        pending_feedback = ssh_result.as_feedback()
                        continue

                    # SSH passed — correctness check (pass original code for diff hints)
                    original_code = concat_header_and_source(snapshot, target_file)
                    correctness = self._validate_correctness(workspaces.workspace_dir, original_code=original_code)
                    if correctness is not None and not correctness.ok:
                        logger.warning(
                            "Pass %d passed SSH but failed correctness; "
                            "keeping snapshot and feeding errors forward\n%s",
                            step, correctness.combined_output,
                        )
                        current_snapshot = repaired_snapshot
                        pending_feedback = correctness.as_feedback()
                        continue

                # This pass fully validated
                current_snapshot = repaired_snapshot
                last_known_good = repaired_snapshot
                successful_passes += 1

                # Measure speedup vs original translated code
                step_elapsed = self._benchmark_on_ssh(
                    workspaces.workspace_dir, ssh_compile_command, ssh_run_command,
                    f"widened (pass {step})",
                )
                if step_elapsed is not None and baseline_elapsed is not None and baseline_elapsed > 0:
                    speedup = baseline_elapsed / step_elapsed
                    logger.info(
                        "Pass %d succeeded — %.2fs (%.2fx vs baseline %.2fs)",
                        step, step_elapsed, speedup, baseline_elapsed,
                    )
                elif step_elapsed is not None:
                    logger.info(
                        "Pass %d succeeded — %.2fs",
                        step, step_elapsed,
                    )
                else:
                    logger.info(
                        "Widening pass %d succeeded (%d total successful passes)",
                        step, successful_passes,
                    )

            logger.info(
                "Widening completed after %d step(s) (%d successful passes)",
                max_steps, successful_passes,
            )
            write_output(output_dir, last_known_good)
            return 0 if successful_passes > 0 else 1

        finally:
            shutil.rmtree(workspaces.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Widen RISC-V Vector code from 128-bit to full VLEN"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_SOURCE_DIR,
        help="Directory with translated 128-bit code (default: translations/sequence-alignment/)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=PROJECT_DIR / "widened",
        help="Directory for widened output (default: widened/)",
    )
    parser.add_argument(
        "--target-file",
        default=None,
        help="File to widen (default: auto-detect .c file with SSE intrinsics)",
    )
    parser.add_argument(
        "--build-command",
        default=None,
        help="Shell command to compile and test (Docker container)",
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
        "--max-steps",
        type=int,
        default=REACT_MAX_STEPS,
        help="Maximum widening passes",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DATASETS_DIR,
        help="Directory with test data files; copied as demo/",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging(level="INFO")
    args = parse_args()
    return WidenAgent().run(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_file=args.target_file,
        build_command=args.build_command,
        ssh_compile_command=args.ssh_compile,
        ssh_run_command=args.ssh_run,
        max_steps=args.max_steps,
        test_data_dir=args.test_data,
    )


if __name__ == "__main__":
    raise SystemExit(main())
