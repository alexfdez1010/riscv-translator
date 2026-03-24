"""Tests for the generic SSE→RISC-V translation pipeline using sse2rvv.h (src/repair.py)."""

import pytest

from src import repair
from src.config import REFERENCE_FILE
from src.search_replace import search_replace_error_feedback
from src.prompts import (
    build_initial_translation_prompt,
    build_repair_prompt,
    build_system_prompt,
)
from src.validators import (
    DockerValidator,
    SSHValidator,
    ValidationResult,
    _infer_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAMPLE_FILES = {
    "main.cpp": '#include "lib.h"\nint main() { return 0; }\n',
    "lib.h": "void foo();\n",
    "lib.cpp": '#include "lib.h"\nvoid foo() {}\n',
}


def _make_snapshot(
    target_code: str = "void foo() {}\n",
    target_file: str = "lib.cpp",
) -> repair.SourceSnapshot:
    return repair.SourceSnapshot(
        files={**_EXAMPLE_FILES, target_file: target_code}
    )


@pytest.fixture(autouse=True)
def _mock_ssh_validator_init(monkeypatch):
    """Prevent SSHValidator from attempting real SSH connections in tests."""
    monkeypatch.setattr(
        SSHValidator,
        "__init__",
        lambda self, **kw: (
            setattr(self, "_available", False)
            or setattr(self, "ssh_host", "test")
            or setattr(self, "remote_dir", "/tmp/test")
        ),
    )


def _patch_infra(monkeypatch, tmp_path, *, docker_ok=False):
    """Patch workspace creation and Docker validator for unit tests."""
    monkeypatch.setattr(
        repair,
        "create_workspace",
        lambda source_dir, snapshot, test_data_dir=None: repair.WorkspaceSet(tmp_path / "root", tmp_path),
    )
    monkeypatch.setattr(repair.shutil, "rmtree", lambda *a, **kw: None)
    monkeypatch.setattr(
        DockerValidator,
        "validate",
        lambda self, workspace_dir, build_command: ValidationResult(
            docker_ok,
            "validation" if docker_ok else "compile",
            0 if docker_ok else 1,
            "",
            "" if docker_ok else "error: broken",
        ),
    )


# ---------------------------------------------------------------------------
# Config & reference tests
# ---------------------------------------------------------------------------


def test_build_system_prompt_includes_sse2rvv_guidance():
    prompt = build_system_prompt("lib.cpp")
    assert "sse2rvv.h" in prompt
    assert "Modify only `lib.cpp`" in prompt
    assert "<<<<<<< SEARCH" in prompt
    assert ">>>>>>> REPLACE" in prompt


def test_build_system_prompt_includes_rvv_reference():
    prompt = build_system_prompt("lib.cpp")
    if REFERENCE_FILE.exists():
        assert str(REFERENCE_FILE) in prompt


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


def test_apply_content_to_snapshot_updates_only_requested_file():
    snapshot = repair.SourceSnapshot(files={"lib.cpp": "old", "lib.h": "keep"})
    updated = repair.apply_content_to_snapshot(snapshot, "lib.cpp", "new")
    assert updated.files["lib.cpp"] == "new"
    assert updated.files["lib.h"] == "keep"


def test_apply_content_to_snapshot_rejects_unknown_file():
    snapshot = repair.SourceSnapshot(files={"lib.cpp": "old"})
    with pytest.raises(ValueError, match="Unknown target file"):
        repair.apply_content_to_snapshot(snapshot, "other.cpp", "new")


# ---------------------------------------------------------------------------
# Prompt format tests
# ---------------------------------------------------------------------------


def test_build_initial_translation_prompt_structure():
    prompt = build_initial_translation_prompt(
        "lib.cpp", "int main() {}\n", "make all"
    )
    assert "sse2rvv.h" in prompt
    assert "lib.cpp" in prompt
    assert "make all" in prompt
    assert "<<<<<<< SEARCH" in prompt


def test_build_initial_translation_prompt_with_feedback():
    prompt = build_initial_translation_prompt(
        "lib.cpp", "int main() {}\n", "make all",
        validation_feedback="error: undefined reference"
    )
    assert "error: undefined reference" in prompt


def test_build_repair_prompt_structure():
    prompt = build_repair_prompt(
        "lib.cpp",
        "int main() {}\n",
        "Validation stage: compile\nReturn code: 1\nFailure details:\nerror: broken",
    )
    assert "Fix the current validation failure" in prompt
    assert "error: broken" in prompt
    assert "<<<<<<< SEARCH" in prompt


def test_build_edit_format_feedback_structure():
    prompt = search_replace_error_feedback("lib.cpp", "int main() {}\n", "search text not found")
    assert "search text not found" in prompt
    assert "<<<<<<< SEARCH" in prompt


# ---------------------------------------------------------------------------
# Validation result tests
# ---------------------------------------------------------------------------


def test_validation_result_as_feedback():
    vr = ValidationResult(
        ok=False, stage="compile", returncode=1,
        stdout="", stderr="error: unknown type"
    )
    feedback = vr.as_feedback()
    assert "compile" in feedback
    assert "error: unknown type" in feedback


def test_infer_stage_compile():
    assert _infer_stage("", "error: unknown type") == "compile"
    assert _infer_stage("", "undefined reference to 'foo'") == "compile"


def test_infer_stage_correctness():
    assert _infer_stage("validation failed", "") == "correctness"
    assert _infer_stage("mismatch detected", "") == "correctness"


def test_infer_stage_runtime():
    assert _infer_stage("segfault", "") == "runtime"


# ---------------------------------------------------------------------------
# Generate valid file tests
# ---------------------------------------------------------------------------


def test_generate_valid_file_retries_with_feedback(monkeypatch, tmp_path):
    responses = iter(
        [
            "Add a required include before changing anything else.\n\n"
            "<<<<<<< SEARCH\nseed\n=======\n#include <stdio.h>\nseed\n>>>>>>> REPLACE",
            "Replace the placeholder line with a stub.\n\n"
            "<<<<<<< SEARCH\nseed\n=======\nvoid foo() {}\n>>>>>>> REPLACE",
        ]
    )
    validations = iter(
        [
            ValidationResult(False, "compile", 1, "", "error: compile failed"),
            ValidationResult(True, "validation", 0, "ok", ""),
        ]
    )
    calls = []

    class FakeAgent(repair.TranslationAgent):
        def __init__(self):
            super().__init__()
            self.llm = lambda messages: calls.append(messages) or next(responses)

    monkeypatch.setattr(repair, "LLM_VALIDATION_RETRIES", 2)
    monkeypatch.setattr(
        DockerValidator,
        "validate",
        lambda self, workspace_dir, build_command: next(validations),
    )
    monkeypatch.setattr(
        repair,
        "materialize_snapshot",
        lambda workspace_dir, snapshot: None,
    )

    agent = FakeAgent()
    snapshot = repair.SourceSnapshot(files={"lib.cpp": "seed\n", "lib.h": "h"})
    workspaces = repair.WorkspaceSet(tmp_path, tmp_path)

    result, validation = agent._generate_valid_file(
        [{"role": "user", "content": "start"}],
        snapshot,
        "lib.cpp",
        workspaces,
        "make all",
    )

    assert result is not None
    assert result.files["lib.cpp"] == "#include <stdio.h>\nvoid foo() {}\n"
    assert validation.ok is True
    assert len(calls) == 2
    assert "compile failed" in calls[1][-1].content


def test_run_writes_output_on_success(monkeypatch, tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "lib.cpp").write_text("seed\n")
    (source_dir / "lib.h").write_text("h\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=False)
    monkeypatch.setattr(repair, "write_output", lambda d, s: d.mkdir(parents=True, exist_ok=True) or (d / "lib.cpp").write_text(s.files["lib.cpp"]))

    monkeypatch.setattr(
        repair.TranslationAgent,
        "_generate_valid_file",
        lambda self, messages, snapshot, file_name, workspaces, build_command: (
            repair.SourceSnapshot(
                files={**snapshot.files, "lib.cpp": "translated code"}
            ),
            ValidationResult(True, "validation", 0, "ok", ""),
        ),
    )

    rc = repair.TranslationAgent().run(
        source_dir=source_dir,
        target_file="lib.cpp",
        output_dir=output_dir,
    )

    assert rc == 0
    assert (output_dir / "lib.cpp").read_text() == "translated code"


def test_run_rejects_missing_target(monkeypatch, tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "lib.cpp").write_text("x")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="Target file.*not found"):
        repair.TranslationAgent().run(
            source_dir=source_dir,
            target_file="nonexistent.cpp",
            output_dir=output_dir,
        )


def test_run_returns_0_when_baseline_passes(monkeypatch, tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "lib.cpp").write_text("already good\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=True)
    monkeypatch.setattr(repair, "write_output", lambda d, s: d.mkdir(parents=True, exist_ok=True) or (d / "lib.cpp").write_text(s.files["lib.cpp"]))

    rc = repair.TranslationAgent().run(
        source_dir=source_dir,
        target_file="lib.cpp",
        output_dir=output_dir,
    )

    assert rc == 0
    assert (output_dir / "lib.cpp").read_text() == "already good\n"


def test_run_returns_1_after_max_steps(monkeypatch, tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "lib.cpp").write_text("broken\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=False)

    # _generate_valid_file always returns None (no progress)
    monkeypatch.setattr(
        repair.TranslationAgent,
        "_generate_valid_file",
        lambda self, messages, snapshot, file_name, workspaces, build_command: (
            None,
            ValidationResult(False, "compile", 1, "", "still broken"),
        ),
    )

    rc = repair.TranslationAgent().run(
        source_dir=source_dir,
        target_file="lib.cpp",
        output_dir=output_dir,
        max_steps=2,
    )

    assert rc == 1
    # Best-effort output is written even on failure
    assert output_dir.exists()


# ---------------------------------------------------------------------------
# Default build command test
# ---------------------------------------------------------------------------


def test_default_build_command_generates_compile_command():
    cmd = repair.default_build_command("ssw.c")
    assert "main.c ssw.c" in cmd
    assert "ssw_test" in cmd
    assert "-lm -lz" in cmd
    assert "demo/10M.fa" in cmd
    assert "Building zlib" in cmd
    assert "Compilation succeeded" in cmd
