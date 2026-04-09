"""Generic SSE→RISC-V translation pipeline using sse2rvv.h.

Ports C/C++ source files that use x86 SSE intrinsics to RISC-V by swapping
SSE headers with the sse2rvv.h drop-in compatibility header and fixing any
remaining compiler errors via a ReAct agent loop:

  1. Start with SSE source code in a workspace.
  2. Pre-process: replace SSE headers with ``#include "sse2rvv.h"``.
  3. ReAct loop — each iteration:
     a. **Reason**: LLM analyses feedback from the previous iteration
        and proposes a minimal search-and-replace patch.
     b. **Act**: The patch is applied and the code is run through a
        two-stage validation chain:
        - Stage 1 — Simulator validation: cross-compile + Spike execution.
        - Stage 2 — Hardware validation: native compile + run on real
          RISC-V hardware + output comparison against Intel SSE reference.
        The chain stops at the first failure; diagnostics feed the next
        Reason step.
  4. The loop terminates when both stages pass in a single iteration,
     or after a hard limit of iterations.

On success the entire workspace (all source files) is written to the
output directory so the result is self-contained and ready to compile.
"""

import argparse
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
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
from src.search_replace import apply_search_replace, extract_search_replace
from src.llm_utils import create_llm
from src.logger import configure_logging, get_logger
from src.prompts import (
    build_edit_format_feedback,
    build_initial_translation_prompt,
    build_repair_prompt,
    build_system_prompt,
)
from src.validators import (
    DockerValidator,
    SSHValidator,
    ValidationResult,
)

logger = get_logger(__name__)

SSE2RVV_HEADER = PROJECT_DIR / "initial_code" / "sse2rvv.h"
MAX_OUTPUT_CHARS = 16000


def truncate_for_log(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


# ---------------------------------------------------------------------------
# SSE → sse2rvv.h header pre-processing
# ---------------------------------------------------------------------------

# SSE headers to replace with sse2rvv.h
_SSE_HEADER_RE = re.compile(
    r'^\s*#\s*include\s*<\s*('
    r'emmintrin\.h|xmmintrin\.h|smmintrin\.h|immintrin\.h|'
    r'nmmintrin\.h|pmmintrin\.h|tmmintrin\.h'
    r')\s*>',
    re.MULTILINE,
)

_SSE2RVV_INCLUDE = '#include "sse2rvv.h"'


def replace_sse_headers(content: str) -> str:
    """Mechanically replace SSE #include directives with sse2rvv.h.

    This is a best-effort pre-processing step that runs before the LLM.
    It only touches the #include lines — sse2rvv.h provides all SSE intrinsics.
    """
    if not _SSE_HEADER_RE.search(content):
        return content

    # Replace the first SSE header with sse2rvv.h include, remove the rest
    replaced_first = False

    def _replacer(m: re.Match) -> str:
        nonlocal replaced_first
        if not replaced_first:
            replaced_first = True
            return _SSE2RVV_INCLUDE
        return ""

    return _SSE_HEADER_RE.sub(_replacer, content)


def preprocess_snapshot(snapshot: "SourceSnapshot") -> "SourceSnapshot":
    """Apply mechanical SSE→sse2rvv.h header replacement to all source files."""
    updated = {}
    for name, content in snapshot.files.items():
        if name.endswith((".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")):
            new_content = replace_sse_headers(content)
            if new_content != content:
                logger.info("Pre-processed %s", name)
            updated[name] = new_content
        else:
            updated[name] = content
    return SourceSnapshot(files=updated)


# ---------------------------------------------------------------------------
# Snapshot and workspace management
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SourceSnapshot:
    """In-memory snapshot of all tracked files in the workspace."""
    files: dict[str, str]


@dataclass(slots=True)
class WorkspaceSet:
    root: Path
    workspace_dir: Path


def materialize_snapshot(workspace_dir: Path, snapshot: SourceSnapshot) -> None:
    """Write all snapshot files into the workspace directory."""
    for name, content in snapshot.files.items():
        (workspace_dir / name).write_text(content)


def create_workspace(
    source_dir: Path,
    snapshot: SourceSnapshot,
    test_data_dir: Path | None = None,
) -> WorkspaceSet:
    """Create a temporary workspace by copying the source directory."""
    root = Path(tempfile.mkdtemp(prefix="sse2rvv-"))
    workspace_dir = root / "workspace"
    shutil.copytree(source_dir, workspace_dir)
    # Copy sse2rvv.h into the workspace so #include "sse2rvv.h" works
    sse2rvv_dest = workspace_dir / "sse2rvv.h"
    if SSE2RVV_HEADER.exists() and not sse2rvv_dest.exists():
        shutil.copy2(SSE2RVV_HEADER, sse2rvv_dest)
    # Copy test data into demo/ subdirectory if provided
    if test_data_dir is not None and test_data_dir.is_dir():
        demo_dir = workspace_dir / "demo"
        shutil.copytree(test_data_dir, demo_dir)
        logger.debug("Copied test data from %s to %s", test_data_dir, demo_dir)
    materialize_snapshot(workspace_dir, snapshot)
    logger.debug("Created workspace at %s", workspace_dir)
    return WorkspaceSet(root=root, workspace_dir=workspace_dir)


def apply_content_to_snapshot(
    snapshot: SourceSnapshot, file_name: str, content: str
) -> SourceSnapshot:
    if file_name not in snapshot.files:
        raise ValueError(f"Unknown target file: {file_name}")
    updated = dict(snapshot.files)
    updated[file_name] = content
    return SourceSnapshot(files=updated)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_output(output_dir: Path, snapshot: SourceSnapshot) -> None:
    """Write all snapshot files into the output directory.

    Creates the directory if needed.  The result is a self-contained
    directory with every file needed to compile and run the translated code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in snapshot.files.items():
        (output_dir / name).write_text(content)
    # Copy sse2rvv.h so the output is standalone
    sse2rvv_dest = output_dir / "sse2rvv.h"
    if SSE2RVV_HEADER.exists() and not sse2rvv_dest.exists():
        shutil.copy2(SSE2RVV_HEADER, sse2rvv_dest)
    logger.info("Wrote %d file(s) to %s", len(snapshot.files), output_dir)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _to_messages(raw: list[dict[str, str]]) -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in raw]


# ---------------------------------------------------------------------------
# Build command generation
# ---------------------------------------------------------------------------


def default_ssh_compile_command() -> str:
    """Default compile command for real RISC-V hardware via SSH."""
    return f"{SSH_CC} -o ssw_test main.c ssw.c --target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm 2>&1"


def default_ssh_run_command() -> str:
    """Default run command for real RISC-V hardware via SSH."""
    return "./ssw_test demo/10k.fa demo/54mer_hap1_1.100.fa 2>&1"


def default_build_command(target_file: str) -> str:
    """Generate a default build + run command for sse2rvv translation.

    Compiles main.c and ssw.c into ssw_test, links with -lm, and
    runs it under QEMU with the demo datasets.  Callers can override via
    --build-command.
    """
    cflags = "-O2 -I. -march=rv64gcv -mabi=lp64d"
    ldflags = "-lm"
    return (
        f'echo "=== Compiling ===" && '
        f'{RISCVCC} {cflags} main.c ssw.c -o ssw_test {ldflags} 2>&1 && '
        f'echo "=== Compilation succeeded, running under QEMU ===" && '
        f'ls demo/ 2>&1 && '
        f'{SIMULATOR} ./ssw_test demo/10k.fa demo/54mer_hap1_1.100.fa 2>&1 && '
        f'echo "=== Execution succeeded ==="'
    )


# ---------------------------------------------------------------------------
# Translation agent
# ---------------------------------------------------------------------------


CORRECTNESS_DATASET = "10k.fa"


class TranslationAgent:
    """LLM-driven SSE→RISC-V translation with compile-fix loop using sse2rvv.h."""

    def __init__(self):
        self.docker_validator = DockerValidator()
        self.ssh_validator = SSHValidator()
        self.llm: LLM = create_llm()
        self._intel_reference: BenchmarkResult | None = None
        self._intel_reference_computed = False

    def _get_intel_reference(self, source_dir: Path, test_data_dir: Path | None) -> BenchmarkResult | None:
        """Compute reference output by running original code on Intel (jump host). Cached after first call."""
        if self._intel_reference_computed:
            return self._intel_reference

        self._intel_reference_computed = True
        jump_host = SSH_JUMP_HOST
        if not check_ssh(jump_host):
            logger.warning("Jump host %s not reachable; correctness check disabled", jump_host)
            return None

        jump_remote = f"{REMOTE_DIR}-correctness-ref"
        dataset_dir = test_data_dir if test_data_dir and test_data_dir.is_dir() else DATASETS_DIR

        dataset_file = dataset_dir / CORRECTNESS_DATASET
        if not dataset_file.exists():
            logger.warning("Correctness dataset %s not found; correctness check disabled", dataset_file)
            return None

        # Upload original source files
        original_files = [p for p in source_dir.iterdir() if p.is_file()]
        logger.info("Uploading original code to %s for correctness reference ...", jump_host)
        if not upload_to_host(jump_host, jump_remote, original_files):
            logger.warning("Failed to upload original code to jump; correctness check disabled")
            return None
        if not upload_datasets(jump_host, jump_remote, dataset_dir, CORRECTNESS_DATASET):
            logger.warning("Failed to upload datasets to jump; correctness check disabled")
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

    def _validate_correctness(self, workspace_dir: Path) -> ValidationResult | None:
        """Run translated code on RISC-V with correctness dataset and compare against Intel reference.

        Returns None if correctness check is not available, a ValidationResult otherwise.
        """
        if self._intel_reference is None:
            return None

        final_host = SSH_HOST
        final_remote = f"{REMOTE_DIR}-correctness-check"

        # Upload translated code
        all_paths = [p for p in workspace_dir.iterdir()]
        if not upload_to_host(final_host, final_remote, all_paths):
            logger.warning("Failed to upload translated code for correctness check")
            return None

        dataset_dir = DATASETS_DIR
        if not upload_datasets(final_host, final_remote, dataset_dir, CORRECTNESS_DATASET):
            logger.warning("Failed to upload datasets for correctness check")
            return None

        run_cmd = f"./ssw_test demo/{CORRECTNESS_DATASET} demo/{BENCH_REFERENCE_FILE} 2>&1"
        compile_cmd = f"{SSH_CC} -o ssw_test main.c ssw.c --target=riscv64-linux-gnu -march=rv64imafdcv -O2 -I. -lm 2>&1"

        riscv_result = run_on_host(final_host, final_remote, compile_cmd, run_cmd, "RISC-V correctness")
        if not riscv_result.ok:
            return ValidationResult(
                ok=False,
                stage="correctness",
                returncode=riscv_result.ok,
                stdout=riscv_result.stdout,
                stderr=f"RISC-V correctness run failed:\n{riscv_result.stderr}",
            )

        match, details = compare_outputs(self._intel_reference, riscv_result)
        if match:
            logger.info("Correctness check PASSED: outputs match Intel reference")
            return ValidationResult(ok=True, stage="correctness", returncode=0, stdout=details, stderr="")

        logger.warning("Correctness check FAILED: outputs differ from Intel reference")
        return ValidationResult(
            ok=False,
            stage="correctness",
            returncode=1,
            stdout="",
            stderr=(
                "CORRECTNESS FAILURE: The translated RISC-V code produces different "
                "alignment results than the original Intel SSE code.\n\n"
                f"{details}\n\n"
                "The SIMD translation has a bug that causes incorrect computation. "
                "Do NOT change the algorithm — the bug is in how SIMD data is "
                "accessed in memory.\n\n"
                "LIKELY ROOT CAUSES (check and fix ALL of these):\n"
                "1. Direct __m128i pointer dereference (`*ptr = v` or `v = *ptr`) "
                "reads/writes the full hardware register (32+ bytes on VLEN>128), "
                "corrupting adjacent memory. Replace ALL direct dereferences with "
                "`_mm_store_si128(ptr, v)` and `v = _mm_load_si128(ptr)`.\n"
                "2. Using SSE2RVV_VTYPE_SIZE or vlenb for allocation/pointer "
                "arithmetic instead of the constant 16. SSE intrinsics always "
                "operate on exactly 16 bytes regardless of hardware VLEN. Use 16 "
                "for all sizeof(__m128i) replacements.\n"
                "3. Profile/scoring arrays written sequentially with byte pointer "
                "(stride=16) but read with SSE2RVV_VTYPE_SIZE stride — data "
                "misalignment. Ensure both write and read use 16-byte stride."
            ),
        )

    def _validate_all_stages(
        self,
        workspace_dir: Path,
        build_command: str,
        ssh_compile_command: str | None,
        ssh_run_command: str | None,
    ) -> ValidationResult:
        """Run the two-stage validation chain: simulator → hardware + correctness.

        Stages execute strictly in order; the chain stops at the first failure
        and returns its diagnostic output.
        """
        # Stage 1: Simulator validation (Docker/QEMU + Spike)
        result = self.docker_validator.validate(workspace_dir, build_command)
        logger.info(
            "Simulator validation: ok=%s stage=%s rc=%s\n%s",
            result.ok, result.stage, result.returncode,
            truncate_for_log(result.combined_output, 2000),
        )
        if not result.ok:
            return result

        # Stage 2: Hardware validation (SSH compile + run + correctness check)
        if not (ssh_compile_command and ssh_run_command):
            return result

        ssh_files = [p for p in workspace_dir.iterdir()]
        ssh_result = self.ssh_validator.validate(
            ssh_files, ssh_compile_command, ssh_run_command
        )
        if not ssh_result.ok:
            logger.warning(
                "Passed simulator but failed hardware at stage %s\n%s",
                ssh_result.stage, ssh_result.combined_output,
            )
            return ValidationResult(
                ok=False,
                stage=ssh_result.stage,
                returncode=ssh_result.returncode,
                stdout=ssh_result.stdout,
                stderr=(
                    "Passed QEMU emulation but FAILED on real RISC-V hardware.\n"
                    f"{ssh_result.combined_output}"
                ),
            )

        # Correctness check against Intel reference
        correctness = self._validate_correctness(workspace_dir)
        if correctness is not None and not correctness.ok:
            return correctness

        logger.info("All validation stages passed")
        return ValidationResult(
            ok=True,
            stage="all-passed",
            returncode=0,
            stdout=ssh_result.stdout,
            stderr="",
        )

    def _apply_llm_edits(
        self,
        messages: list[dict[str, str]],
        snapshot: SourceSnapshot,
        file_name: str,
    ) -> SourceSnapshot | None:
        """Call LLM and apply search/replace edits.

        Retries up to LLM_VALIDATION_RETRIES times on edit format failures.
        Returns the updated snapshot, or None if no valid edits could be
        extracted.  Raises on unrecoverable LLM errors.
        """
        active_messages = list(messages)

        for attempt in range(LLM_VALIDATION_RETRIES + 1):
            logger.info(
                "LLM request attempt %d with %d message(s)",
                attempt + 1,
                len(active_messages),
            )
            response = self.llm(_to_messages(active_messages))

            logger.info(
                "LLM response (attempt %d, %d chars):\n%s",
                attempt + 1,
                len(response),
                truncate_for_log(response, 3000),
            )

            # --- Extract and apply search/replace blocks ---
            edit_error = None
            sr_blocks = extract_search_replace(response)
            if sr_blocks is not None:
                logger.info(
                    "Extracted %d search/replace block(s) from response",
                    len(sr_blocks),
                )
                try:
                    new_content = apply_search_replace(
                        snapshot.files[file_name], sr_blocks,
                    )
                    return apply_content_to_snapshot(
                        snapshot, file_name, new_content,
                    )
                except ValueError as exc:
                    logger.warning(
                        "Search/replace failed on attempt %d: %s",
                        attempt + 1, exc,
                    )
                    edit_error = str(exc)

            error_msg = edit_error or (
                "Could not extract edits from the response. "
                "Use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks."
            )
            logger.warning(
                "No valid edits on attempt %d: %s", attempt + 1, error_msg,
            )

            if attempt >= LLM_VALIDATION_RETRIES:
                return None

            active_messages = active_messages + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": build_edit_format_feedback(
                        file_name,
                        snapshot.files[file_name],
                        error_msg,
                    ),
                },
            ]

        return None

    def run(
        self,
        source_dir: Path,
        target_file: str,
        output_dir: Path,
        build_command: str | None = None,
        ssh_compile_command: str | None = None,
        ssh_run_command: str | None = None,
        max_steps: int = REACT_MAX_STEPS,
        test_data_dir: Path | None = None,
    ) -> int:
        """Run the full translation pipeline.

        Args:
            source_dir: Directory containing the source files to translate.
            target_file: Name of the file to translate (e.g. "ssw.c").
            output_dir: Directory to write all translated files into.
            build_command: Shell command to build+test in Docker. Auto-generated if None.
            ssh_compile_command: Shell command to compile on SSH hardware. Skipped if None.
            ssh_run_command: Shell command to run on SSH hardware. Skipped if None.
            max_steps: Maximum LLM repair iterations.
            test_data_dir: Directory with test data files; copied into workspace as demo/.
        """
        if build_command is None:
            build_command = default_build_command(target_file)
        if ssh_compile_command is None:
            ssh_compile_command = default_ssh_compile_command()
        if ssh_run_command is None:
            ssh_run_command = default_ssh_run_command()

        logger.info("Starting translation for %s in %s", target_file, source_dir)

        # Load all files from the source directory
        file_names = [
            f.name for f in source_dir.iterdir() if f.is_file()
        ]
        if target_file not in file_names:
            raise ValueError(
                f"Target file {target_file} not found in {source_dir}"
            )

        snapshot = SourceSnapshot(
            files={name: (source_dir / name).read_text() for name in file_names}
        )

        # --- Pre-processing: replace SSE headers with sse2rvv.h ---
        snapshot = preprocess_snapshot(snapshot)

        workspaces = create_workspace(source_dir, snapshot, test_data_dir)

        # Compute Intel reference output for correctness checking
        self._get_intel_reference(source_dir, test_data_dir)

        try:
            # --- Baseline validation ---
            latest_validation = self._validate_all_stages(
                workspaces.workspace_dir, build_command,
                ssh_compile_command, ssh_run_command,
            )
            if latest_validation.ok:
                logger.info("Input already passes all validations")
                write_output(output_dir, snapshot)
                return 0

            # --- ReAct loop ---
            current_snapshot = snapshot

            for step in range(1, max_steps + 1):
                logger.info(
                    "ReAct step %d/%d for %s", step, max_steps, target_file
                )

                # -- Reason: LLM analyses feedback and proposes a patch --
                current_code = current_snapshot.files[target_file]

                if step == 1:
                    user_content = build_initial_translation_prompt(
                        target_file,
                        current_code,
                        build_command,
                        latest_validation.as_feedback(),
                    )
                else:
                    user_content = build_repair_prompt(
                        target_file,
                        current_code,
                        latest_validation.as_feedback(),
                    )

                messages = [
                    {"role": "system", "content": build_system_prompt(target_file)},
                    {"role": "user", "content": user_content},
                ]

                try:
                    candidate = self._apply_llm_edits(
                        messages, current_snapshot, target_file,
                    )
                except Exception as exc:
                    logger.warning(
                        "Stopping early due to unrecoverable error: %s", exc
                    )
                    return 1

                if candidate is None:
                    logger.info(
                        "Step %d did not yield valid edits, continuing", step
                    )
                    continue

                current_snapshot = candidate

                # -- Act: apply patch and run two-stage validation chain --
                materialize_snapshot(workspaces.workspace_dir, current_snapshot)

                latest_validation = self._validate_all_stages(
                    workspaces.workspace_dir, build_command,
                    ssh_compile_command, ssh_run_command,
                )

                if latest_validation.ok:
                    write_output(output_dir, current_snapshot)
                    logger.info(
                        "Translation succeeded at step %d; wrote output to %s",
                        step, output_dir,
                    )
                    return 0

                logger.info(
                    "Step %d: validation failed at stage %s; feeding back to LLM",
                    step, latest_validation.stage,
                )

            logger.warning("Translation failed after %d step(s)", max_steps)
            write_output(output_dir, current_snapshot)
            logger.info("Wrote best-effort output to %s", output_dir)
            return 1

        finally:
            logger.debug("Cleaning up workspace at %s", workspaces.root)
            shutil.rmtree(workspaces.root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate SSE C/C++ code to RISC-V using sse2rvv.h"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing source files to translate",
    )
    parser.add_argument(
        "target_file",
        help="Name of the file to translate (e.g. ssw.c)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write translated files into (created if needed)",
    )
    parser.add_argument(
        "--build-command",
        default=None,
        help="Shell command to compile and test (run inside Docker container)",
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
    return TranslationAgent().run(
        source_dir=args.source_dir,
        target_file=args.target_file,
        output_dir=args.output_dir,
        build_command=args.build_command,
        ssh_compile_command=args.ssh_compile,
        ssh_run_command=args.ssh_run,
        max_steps=args.max_steps,
        test_data_dir=args.test_data,
    )


if __name__ == "__main__":
    raise SystemExit(main())
