"""ReAct-style repair agent for incremental LLM-driven C file patching."""

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.llm_types import LLM, Message

from src.config import (
    LLM_VALIDATION_RETRIES,
    PROJECT_DIR,
)
from src.diff_utils import apply_patch, extract_diff
from src.fitness import SSWValidator as _FitnessSSWValidator
from src.llm_utils import create_llm
from src.logger import configure_logging, get_logger
from src.prompts import (
    build_diff_format_feedback,
    build_initial_user_prompt,
    build_repair_prompt,
    build_system_prompt,
    preprocess_rvv_compat,
)
from src.validators import (
    InitialCodeValidator,
    SSHSSWValidator,
    ValidationResult,
)

logger = get_logger(__name__)

INITIAL_CODE_DIR = PROJECT_DIR / "initial_code"
TARGET_FILE_NAMES = (
    "ssw.c",
    "ssw.h",
    "main.c",
    "Makefile",
    "kseq.h",
    "example.c",
    "sse2rvv.h",
)
DATASET_DIR = PROJECT_DIR / "dataset"
DEFAULT_TARGET = DATASET_DIR / "10k.fa"
DEFAULT_QUERY = DATASET_DIR / "10k.fa"
REACT_MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "15"))
MAX_OUTPUT_CHARS = 16000


def truncate_for_log(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


# ---------------------------------------------------------------------------
# Snapshot and workspace management
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SourceSnapshot:
    files: dict[str, str]


@dataclass(slots=True)
class WorkspaceSet:
    root: Path
    current_dir: Path


def load_initial_sources() -> SourceSnapshot:
    snapshot = SourceSnapshot(
        files={
            name: (INITIAL_CODE_DIR / name).read_text() for name in TARGET_FILE_NAMES
        }
    )
    logger.debug("Loaded initial sources: %d file(s)", len(snapshot.files))
    return snapshot


def materialize_snapshot(base_dir: Path, snapshot: SourceSnapshot) -> None:
    for name, content in snapshot.files.items():
        (base_dir / name).write_text(content)


def create_workspace(initial_snapshot: SourceSnapshot) -> WorkspaceSet:
    root = Path(tempfile.mkdtemp(prefix="ssw-repair-"))
    current_dir = root / "initial_code"
    shutil.copytree(INITIAL_CODE_DIR, current_dir)
    materialize_snapshot(current_dir, initial_snapshot)
    logger.debug("Created repair workspace at %s", current_dir)
    return WorkspaceSet(root=root, current_dir=current_dir)


def apply_content_to_snapshot(
    snapshot: SourceSnapshot, file_name: str, content: str
) -> SourceSnapshot:
    if file_name not in snapshot.files:
        raise ValueError(f"Unsupported target path: {file_name}")
    updated = dict(snapshot.files)
    updated[file_name] = content
    return SourceSnapshot(files=updated)


def apply_patch_to_snapshot(
    snapshot: SourceSnapshot,
    file_name: str,
    patch: str,
) -> SourceSnapshot:
    updated_content = apply_patch(snapshot.files[file_name], patch, file_name)
    return apply_content_to_snapshot(snapshot, file_name, updated_content)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _to_messages(raw: list[dict[str, str]]) -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in raw]


# ---------------------------------------------------------------------------
# Repair agent
# ---------------------------------------------------------------------------


class RepairAgent:
    def __init__(self):
        self.validator = InitialCodeValidator()
        self.ssw_validator = _FitnessSSWValidator()
        self.ssh_validator = SSHSSWValidator()
        self.llm: LLM = create_llm()

    def _passes_ssw_validation(self, ssw_code: str) -> bool:
        """Verify with the same validator used by mutation/crossover operators."""
        logger.info("Running SSW validation (same as mutation/crossover)")
        result = self.ssw_validator.validate(ssw_code)
        if result.ok:
            logger.info("SSW validation passed")
            return True
        logger.warning(
            "SSW validation failed at stage %s: %s",
            result.stage,
            truncate_for_log(result.combined_output, 2000),
        )
        return False

    def _generate_valid_file(
        self,
        messages: list[dict[str, str]],
        snapshot: SourceSnapshot,
        file_name: str,
        workspaces: WorkspaceSet,
        target_fasta: str,
        query_fasta: str,
    ) -> tuple[SourceSnapshot | None, ValidationResult]:
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
            patch = extract_diff(response)
            if patch is not None:
                logger.info(
                    "Extracted patch (%d chars):\n%s",
                    len(patch),
                    truncate_for_log(patch, 2000),
                )
            if patch is None:
                logger.warning(
                    "Could not extract diff from LLM response on attempt %d",
                    attempt + 1,
                )
                if attempt >= LLM_VALIDATION_RETRIES:
                    return None, latest_validation
                active_messages = active_messages + [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": build_diff_format_feedback(
                            file_name,
                            current_snapshot.files[file_name],
                            "Could not extract a unified diff from the previous response.",
                        ),
                    },
                ]
                continue
            try:
                candidate_snapshot = apply_patch_to_snapshot(
                    current_snapshot,
                    file_name,
                    patch,
                )
            except ValueError as exc:
                logger.warning(
                    "Could not apply diff from LLM response on attempt %d: %s",
                    attempt + 1,
                    exc,
                )
                if attempt >= LLM_VALIDATION_RETRIES:
                    return None, latest_validation
                active_messages = active_messages + [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": build_diff_format_feedback(
                            file_name,
                            current_snapshot.files[file_name],
                            str(exc),
                        ),
                    },
                ]
                continue
            logger.debug(
                "Validating repair candidate for %s on attempt %d",
                file_name,
                attempt + 1,
            )
            latest_validation = self.validator.validate(
                candidate_snapshot,
                workspaces.current_dir,
                target_fasta,
                query_fasta,
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
                        "Candidate repaired successfully after %d retry(s)", attempt
                    )
                return candidate_snapshot, latest_validation
            logger.debug(
                "Candidate validation failed on attempt %d at stage %s",
                attempt + 1,
                latest_validation.stage,
            )
            if attempt >= LLM_VALIDATION_RETRIES:
                logger.warning(
                    "Candidate validation failed after %d attempt(s); "
                    "returning latest snapshot for next step",
                    attempt + 1,
                )
                return candidate_snapshot, latest_validation
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
        input_file: Path,
        output_file: Path,
        target_fasta: str,
        query_fasta: str,
        max_steps: int = REACT_MAX_STEPS,
    ) -> int:
        file_name = input_file.name
        if file_name not in TARGET_FILE_NAMES:
            raise ValueError(f"Unsupported repair target: {file_name}")

        logger.info("Starting repair for %s -> %s", input_file, output_file)
        snapshot = load_initial_sources()
        raw_code = input_file.read_text()
        preprocessed_code = preprocess_rvv_compat(raw_code)
        snapshot = apply_content_to_snapshot(
            snapshot, file_name, preprocessed_code
        )
        workspaces = create_workspace(snapshot)
        try:
            baseline = self.validator.validate(
                snapshot,
                workspaces.current_dir,
                target_fasta,
                query_fasta,
            )
            logger.info(
                "Baseline validation: ok=%s stage=%s rc=%s\n%s",
                baseline.ok,
                baseline.stage,
                baseline.returncode,
                truncate_for_log(baseline.combined_output, 3000),
            )
            if baseline.ok:
                ssw_code = snapshot.files[file_name]
                if self._passes_ssw_validation(ssw_code):
                    ssh_result = self.ssh_validator.validate(ssw_code)
                    if ssh_result.ok:
                        logger.info(
                            "Input already passes all validations; writing output without repair"
                        )
                        output_file.write_text(ssw_code)
                        return 0
                    logger.warning(
                        "Baseline passes QEMU validation but fails SSH hardware "
                        "validation at stage %s; proceeding with repair",
                        ssh_result.stage,
                    )
                    baseline = ValidationResult(
                        ok=False,
                        stage=ssh_result.stage,
                        returncode=ssh_result.returncode,
                        stdout=ssh_result.stdout,
                        stderr=(
                            "Passed QEMU emulation but FAILED on real RISC-V hardware "
                            f"(SSH, VLEN may differ from emulator).\n"
                            f"{ssh_result.combined_output}"
                        ),
                    )
                else:
                    logger.warning(
                        "Baseline passes initial validation but fails SSW validation; "
                        "proceeding with repair"
                    )

            current_snapshot = snapshot
            latest_validation = baseline
            for step in range(1, max_steps + 1):
                logger.info("Repair step %d/%d for %s", step, max_steps, file_name)
                current_code = current_snapshot.files[file_name]
                messages = [
                    {"role": "system", "content": build_system_prompt(file_name)},
                    {
                        "role": "user",
                        "content": build_initial_user_prompt(
                            file_name,
                            current_code,
                            target_fasta,
                            query_fasta,
                            latest_validation.as_feedback(),
                        )
                        if step == 1
                        else build_repair_prompt(
                            file_name,
                            current_code,
                            latest_validation.as_feedback(),
                        ),
                    },
                ]
                repaired_snapshot, latest_validation = self._generate_valid_file(
                    messages,
                    current_snapshot,
                    file_name,
                    workspaces,
                    target_fasta,
                    query_fasta,
                )
                if repaired_snapshot is None:
                    if latest_validation.stage == "internal-error":
                        logger.warning(
                            "Stopping repair early due to unrecoverable internal error: %s",
                            latest_validation.stderr,
                        )
                        return 1
                    logger.info("Repair step %d did not yield a valid candidate", step)
                    continue
                if latest_validation.ok:
                    current_snapshot = repaired_snapshot
                    ssw_code = current_snapshot.files[file_name]
                    if self._passes_ssw_validation(ssw_code):
                        # Also validate on real hardware via SSH
                        ssh_result = self.ssh_validator.validate(ssw_code)
                        if ssh_result.ok:
                            output_file.write_text(ssw_code)
                            logger.info("Repair succeeded; wrote output to %s", output_file)
                            return 0
                        logger.warning(
                            "Repair step %d passed QEMU but failed SSH hardware "
                            "validation at stage %s; continuing",
                            step,
                            ssh_result.stage,
                        )
                        latest_validation = ValidationResult(
                            ok=False,
                            stage=ssh_result.stage,
                            returncode=ssh_result.returncode,
                            stdout=ssh_result.stdout,
                            stderr=(
                                "Passed QEMU emulation but FAILED on real RISC-V "
                                f"hardware (SSH, VLEN may differ from emulator).\n"
                                f"{ssh_result.combined_output}"
                            ),
                        )
                    else:
                        logger.info(
                            "Repair step %d passed initial validation but failed "
                            "SSW validation; continuing",
                            step,
                        )
                        latest_validation = ValidationResult(
                            ok=False,
                            stage="ssw-validation",
                            returncode=None,
                            stdout="",
                            stderr=(
                                "Passed initial validation but failed SSW validation "
                                "(same test used by mutation/crossover)"
                            ),
                        )
                # Partial progress — keep the latest snapshot for the next step.
                logger.info(
                    "Repair step %d made progress but validation still fails; "
                    "continuing from updated snapshot",
                    step,
                )
                current_snapshot = repaired_snapshot
            logger.warning("Repair failed after %d step(s)", max_steps)
            return 1
        finally:
            logger.debug("Cleaning up repair workspace at %s", workspaces.root)
            shutil.rmtree(workspaces.root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--target-fasta", default=str(DEFAULT_TARGET))
    parser.add_argument("--query-fasta", default=str(DEFAULT_QUERY))
    parser.add_argument("--max-steps", type=int, default=REACT_MAX_STEPS)
    return parser.parse_args()


def main() -> int:
    configure_logging(level="INFO")
    args = parse_args()
    return RepairAgent().run(
        input_file=args.input_file,
        output_file=args.output_file,
        target_fasta=str(args.target_fasta),
        query_fasta=str(args.query_fasta),
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
