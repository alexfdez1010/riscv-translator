"""Generic SSE→RISC-V translation pipeline using sse2rvv.h.

Ports C/C++ source files that use x86 SSE intrinsics to RISC-V by swapping
SSE headers with the sse2rvv.h drop-in compatibility header and fixing any
remaining compiler errors via an LLM compile-fix loop:

  1. Start with SSE source code in a workspace.
  2. Pre-process: replace SSE headers with ``#include "sse2rvv.h"``.
  3. LLM proposes minimal diffs to fix remaining compilation issues.
  4. Validate by compiling + running in Docker/QEMU (simulator).
  5. If errors, feed compiler output back to the LLM and loop.
  6. Once the simulator passes, validate on real hardware via SSH.
  7. Loop until fully passing or max steps exhausted.

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
    LLM_VALIDATION_RETRIES,
    PROJECT_DIR,
    REACT_MAX_STEPS,
    RISCVCXX,
    SIMULATOR,
)
from src.diff_utils import apply_search_replace, extract_search_replace
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


def create_workspace(source_dir: Path, snapshot: SourceSnapshot) -> WorkspaceSet:
    """Create a temporary workspace by copying the source directory."""
    root = Path(tempfile.mkdtemp(prefix="sse2rvv-"))
    workspace_dir = root / "workspace"
    shutil.copytree(source_dir, workspace_dir)
    # Copy sse2rvv.h into the workspace so #include "sse2rvv.h" works
    sse2rvv_dest = workspace_dir / "sse2rvv.h"
    if SSE2RVV_HEADER.exists() and not sse2rvv_dest.exists():
        shutil.copy2(SSE2RVV_HEADER, sse2rvv_dest)
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
    return "g++ -O2 -std=c++17 -I. -march=rv64gcv -mabi=lp64d *.c -o test_binary 2>&1"


def default_ssh_run_command() -> str:
    """Default run command for real RISC-V hardware via SSH."""
    return "./test_binary 2>&1"


def default_build_command(target_file: str) -> str:
    """Generate a default build + run command for sse2rvv translation.

    Compiles all C/C++ source files in the workspace, links them into a
    single binary, and runs it under QEMU.  Callers can override via
    --build-command.
    """
    cflags = "-O2 -std=c++17 -I. -march=rv64gcv -mabi=lp64d"
    return (
        f'echo "=== Compiling ===" && '
        f'{RISCVCXX} {cflags} *.c -o test_binary 2>&1 && '
        f'echo "=== Compilation succeeded, running under QEMU ===" && '
        f'{SIMULATOR} ./test_binary 2>&1 && '
        f'echo "=== Execution succeeded ==="'
    )


# ---------------------------------------------------------------------------
# Translation agent
# ---------------------------------------------------------------------------


class TranslationAgent:
    """LLM-driven SSE→RISC-V translation with compile-fix loop using sse2rvv.h."""

    def __init__(self):
        self.docker_validator = DockerValidator()
        self.ssh_validator = SSHValidator()
        self.llm: LLM = create_llm()

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
            stage="internal-error",
            returncode=None,
            stdout="",
            stderr="No validation attempted.",
        )

        for attempt in range(LLM_VALIDATION_RETRIES + 1):
            # --- LLM request ---
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
                logger.warning(
                    "LLM generation failed on attempt %d: %s", attempt + 1, exc
                )
                return None, latest_validation

            logger.info(
                "LLM response (attempt %d, %d chars):\n%s",
                attempt + 1,
                len(response),
                truncate_for_log(response, 3000),
            )

            # --- Extract and apply search/replace blocks ---
            candidate_snapshot = None
            edit_error = None

            sr_blocks = extract_search_replace(response)
            if sr_blocks is not None:
                logger.info(
                    "Extracted %d search/replace block(s) from response",
                    len(sr_blocks),
                )
                try:
                    new_content = apply_search_replace(
                        current_snapshot.files[file_name], sr_blocks,
                    )
                    candidate_snapshot = apply_content_to_snapshot(
                        current_snapshot, file_name, new_content,
                    )
                except ValueError as exc:
                    logger.warning(
                        "Search/replace failed on attempt %d: %s",
                        attempt + 1, exc,
                    )
                    edit_error = str(exc)

            if candidate_snapshot is None:
                error_msg = edit_error or (
                    "Could not extract edits from the response. "
                    "Use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks."
                )
                logger.warning(
                    "No valid edits on attempt %d: %s", attempt + 1, error_msg,
                )
                if attempt >= LLM_VALIDATION_RETRIES:
                    return None, latest_validation
                active_messages = active_messages + [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": build_edit_format_feedback(
                            file_name,
                            current_snapshot.files[file_name],
                            error_msg,
                        ),
                    },
                ]
                continue

            # --- Validate in Docker/QEMU ---
            materialize_snapshot(workspaces.workspace_dir, candidate_snapshot)
            latest_validation = self.docker_validator.validate(
                workspaces.workspace_dir,
                build_command,
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
                if attempt > 0:
                    logger.info(
                        "Candidate fixed after %d retry(s)", attempt
                    )
                return candidate_snapshot, latest_validation

            if attempt >= LLM_VALIDATION_RETRIES:
                logger.warning(
                    "Validation failed after %d attempt(s); "
                    "returning latest snapshot for next step",
                    attempt + 1,
                )
                return candidate_snapshot, latest_validation

            # Feed errors back for retry
            active_messages = active_messages + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": build_repair_prompt(
                        file_name,
                        candidate_snapshot.files[file_name],
                        latest_validation.as_feedback(),
                    ),
                },
            ]
            current_snapshot = candidate_snapshot

        return None, latest_validation

    def run(
        self,
        source_dir: Path,
        target_file: str,
        output_dir: Path,
        build_command: str | None = None,
        ssh_compile_command: str | None = None,
        ssh_run_command: str | None = None,
        max_steps: int = REACT_MAX_STEPS,
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

        # --- Pre-processing: replace SSE headers with Highway includes ---
        snapshot = preprocess_snapshot(snapshot)

        workspaces = create_workspace(source_dir, snapshot)

        try:
            # --- Baseline validation ---
            baseline = self.docker_validator.validate(
                workspaces.workspace_dir, build_command
            )
            logger.info(
                "Baseline validation: ok=%s stage=%s rc=%s\n%s",
                baseline.ok,
                baseline.stage,
                baseline.returncode,
                truncate_for_log(baseline.combined_output, 3000),
            )

            if baseline.ok:
                # Already compiles + runs; try SSH hardware
                if ssh_compile_command and ssh_run_command:
                    ssh_files = [
                        p for p in workspaces.workspace_dir.iterdir()
                    ]
                    ssh_result = self.ssh_validator.validate(
                        ssh_files, ssh_compile_command, ssh_run_command
                    )
                    if ssh_result.ok:
                        logger.info("Input already passes all validations")
                        write_output(output_dir, snapshot)
                        return 0
                    logger.warning(
                        "Baseline passes QEMU but fails SSH at stage %s; "
                        "proceeding with repair",
                        ssh_result.stage,
                    )
                    baseline = ValidationResult(
                        ok=False,
                        stage=ssh_result.stage,
                        returncode=ssh_result.returncode,
                        stdout=ssh_result.stdout,
                        stderr=(
                            "Passed QEMU emulation but FAILED on real RISC-V hardware.\n"
                            f"{ssh_result.combined_output}"
                        ),
                    )
                else:
                    logger.info("Input already passes Docker/QEMU validation")
                    write_output(output_dir, snapshot)
                    return 0

            # --- Main repair loop ---
            current_snapshot = snapshot
            latest_validation = baseline

            for step in range(1, max_steps + 1):
                logger.info(
                    "Translation step %d/%d for %s", step, max_steps, target_file
                )
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

                repaired_snapshot, latest_validation = self._generate_valid_file(
                    messages,
                    current_snapshot,
                    target_file,
                    workspaces,
                    build_command,
                )

                if repaired_snapshot is None:
                    if latest_validation.stage == "internal-error":
                        logger.warning(
                            "Stopping early due to unrecoverable error: %s",
                            latest_validation.stderr,
                        )
                        return 1
                    logger.info("Step %d did not yield a valid candidate", step)
                    continue

                if latest_validation.ok:
                    current_snapshot = repaired_snapshot

                    # Try SSH hardware validation
                    if ssh_compile_command and ssh_run_command:
                        ssh_files = [
                            p for p in workspaces.workspace_dir.iterdir()
                        ]
                        ssh_result = self.ssh_validator.validate(
                            ssh_files, ssh_compile_command, ssh_run_command
                        )
                        if ssh_result.ok:
                            write_output(output_dir, current_snapshot)
                            logger.info(
                                "Translation succeeded; wrote output to %s",
                                output_dir,
                            )
                            return 0
                        logger.warning(
                            "Step %d passed QEMU but failed SSH at stage %s",
                            step,
                            ssh_result.stage,
                        )
                        latest_validation = ValidationResult(
                            ok=False,
                            stage=ssh_result.stage,
                            returncode=ssh_result.returncode,
                            stdout=ssh_result.stdout,
                            stderr=(
                                "Passed QEMU emulation but FAILED on real hardware.\n"
                                f"{ssh_result.combined_output}"
                            ),
                        )
                    else:
                        # No SSH configured — QEMU pass is success
                        write_output(output_dir, current_snapshot)
                        logger.info(
                            "Translation succeeded (QEMU only); wrote output to %s",
                            output_dir,
                        )
                        return 0

                # Keep partial progress for next step
                logger.info(
                    "Step %d made progress but validation still fails; continuing",
                    step,
                )
                current_snapshot = repaired_snapshot

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
    )


if __name__ == "__main__":
    raise SystemExit(main())
