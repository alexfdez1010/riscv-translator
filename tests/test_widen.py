"""Tests for the vector-width optimization pipeline (src/widen.py)."""

import pytest

from src import widen
from src.search_replace import search_replace_error_feedback
from src.validators import (
    DockerValidator,
    SSHValidator,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAMPLE_FILES = {
    "main.c": '#include "ssw.h"\nint main() { return 0; }\n',
    "ssw.h": "void foo();\n",
    "ssw.c": '#include "ssw.h"\n#include "sse2rvv.h"\nvoid foo() {}\n',
    "sse2rvv.h": "/* sse2rvv header stub */\n",
}


def _make_snapshot(
    target_code: str = '#include "ssw.h"\nvoid foo() {}\n',
    target_file: str = "ssw.c",
) -> widen.SourceSnapshot:
    return widen.SourceSnapshot(
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


def _patch_infra(monkeypatch, tmp_path, *, docker_ok=True):
    """Patch workspace creation and validators for unit tests."""
    monkeypatch.setattr(
        widen,
        "create_workspace",
        lambda source_dir, snapshot, test_data_dir=None: widen.WorkspaceSet(
            tmp_path / "root", tmp_path
        ),
    )
    monkeypatch.setattr(widen.shutil, "rmtree", lambda *a, **kw: None)
    monkeypatch.setattr(widen, "materialize_snapshot", lambda w, s: None)
    monkeypatch.setattr(
        DockerValidator,
        "validate",
        lambda self, workspace_dir, build_command: ValidationResult(
            docker_ok,
            "validation" if docker_ok else "compile",
            0 if docker_ok else 1,
            "" if docker_ok else "",
            "" if docker_ok else "error: broken",
        ),
    )
    monkeypatch.setattr(
        widen.SSHValidator,
        "_check_ssh",
        lambda self: False,
    )
    monkeypatch.setattr(
        widen.WidenAgent,
        "_get_intel_reference",
        lambda self, original_dir, test_data_dir: None,
    )
    monkeypatch.setattr(
        widen.WidenAgent,
        "_benchmark_on_ssh",
        lambda self, workspace_dir, ssh_compile_cmd, ssh_run_cmd, label: None,
    )


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


def test_apply_content_to_snapshot_updates_target():
    snapshot = widen.SourceSnapshot(files={"ssw.c": "old", "ssw.h": "keep"})
    updated = widen.apply_content_to_snapshot(snapshot, "ssw.c", "new")
    assert updated.files["ssw.c"] == "new"
    assert updated.files["ssw.h"] == "keep"


def test_apply_content_to_snapshot_rejects_unknown_file():
    snapshot = widen.SourceSnapshot(files={"ssw.c": "old"})
    with pytest.raises(ValueError, match="Unknown target file"):
        widen.apply_content_to_snapshot(snapshot, "other.c", "new")


def test_write_output_creates_files(tmp_path):
    snapshot = widen.SourceSnapshot(files={"ssw.c": "code", "ssw.h": "header"})
    out = tmp_path / "output"
    widen.write_output(out, snapshot)
    assert (out / "ssw.c").read_text() == "code"
    assert (out / "ssw.h").read_text() == "header"


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


def test_build_widen_system_prompt_structure():
    prompt = widen.build_widen_system_prompt("ssw.c")
    assert "ssw.c" in prompt
    assert "VLEN" in prompt
    assert "<<<<<<< SEARCH" in prompt
    assert ">>>>>>> REPLACE" in prompt
    assert "__riscv_vsetvlmax" in prompt
    assert "the header" in prompt and "ssw.c" in prompt


def test_build_widen_initial_prompt_structure():
    prompt = widen.build_widen_initial_prompt(
        "ssw.c", "int main() {}\n", "make all"
    )
    assert "ssw.c" in prompt
    assert "make all" in prompt
    assert "Widen" in prompt
    assert "<<<<<<< SEARCH" in prompt


def test_build_widen_initial_prompt_with_feedback():
    prompt = widen.build_widen_initial_prompt(
        "ssw.c", "int main() {}\n", "make all",
        validation_feedback="error: type mismatch"
    )
    assert "error: type mismatch" in prompt


def test_build_widen_repair_prompt_structure():
    prompt = widen.build_widen_repair_prompt(
        "ssw.c",
        "int main() {}\n",
        "Validation stage: compile\nReturn code: 1\nerror: broken",
    )
    assert "Fix the validation failure" in prompt
    assert "error: broken" in prompt
    assert "<<<<<<< SEARCH" in prompt


def test_build_widen_continue_prompt_structure():
    prompt = widen.build_widen_continue_prompt(
        "ssw.c", "int main() {}\n", "make all", pass_number=3
    )
    assert "pass 3" in prompt
    assert "ALL_WIDENED" in prompt
    assert "<<<<<<< SEARCH" in prompt


def test_build_widen_edit_format_feedback_structure():
    prompt = widen.build_widen_edit_format_feedback(
        "ssw.c", "int main() {}\n", "search text not found"
    )
    assert "search text not found" in prompt
    assert "<<<<<<< SEARCH" in prompt


# ---------------------------------------------------------------------------
# Build command tests
# ---------------------------------------------------------------------------


def test_default_docker_build_command():
    cmd = widen.default_docker_build_command()
    assert "main.c ssw.c" in cmd
    assert "ssw_test" in cmd
    assert "-lm" in cmd
    assert ">/dev/null" in cmd, "Execution output should be suppressed"


def test_default_ssh_compile_command():
    cmd = widen.default_ssh_compile_command()
    assert "main.c ssw.c" in cmd
    assert "rv64imafdcv" in cmd


def test_default_ssh_run_command():
    cmd = widen.default_ssh_run_command()
    assert "ssw_test" in cmd
    assert "demo/" in cmd


# ---------------------------------------------------------------------------
# _generate_single_pass tests
# ---------------------------------------------------------------------------


def test_generate_single_pass_applies_edits(monkeypatch, tmp_path):
    """Successful edit + Docker validation in one attempt."""
    monkeypatch.setattr(
        DockerValidator,
        "validate",
        lambda self, workspace_dir, build_command: ValidationResult(
            True, "validation", 0, "ok", ""
        ),
    )
    monkeypatch.setattr(widen, "materialize_snapshot", lambda w, s: None)

    response = (
        "Widen the inner loop.\n\n"
        "<<<<<<< SEARCH\nvoid foo() {}\n=======\n"
        "void foo() { /* widened */ }\n>>>>>>> REPLACE"
    )

    class FakeAgent(widen.WidenAgent):
        def __init__(self):
            super().__init__()
            self.llm = lambda messages: response

    agent = FakeAgent()
    snapshot = _make_snapshot("void foo() {}\n")
    workspaces = widen.WorkspaceSet(tmp_path, tmp_path)

    result, validation = agent._generate_single_pass(
        [{"role": "user", "content": "start"}],
        snapshot,
        "ssw.c",
        workspaces,
        "make all",
    )

    assert result is not None
    assert "widened" in result.files["ssw.c"]
    assert validation.ok is True


def test_generate_single_pass_detects_all_widened(monkeypatch, tmp_path):
    """LLM responds with ALL_WIDENED signal."""
    response = "ALL_WIDENED: No more SSE intrinsics to widen."

    class FakeAgent(widen.WidenAgent):
        def __init__(self):
            super().__init__()
            self.llm = lambda messages: response

    agent = FakeAgent()
    snapshot = _make_snapshot()
    workspaces = widen.WorkspaceSet(tmp_path, tmp_path)

    result, validation = agent._generate_single_pass(
        [{"role": "user", "content": "continue"}],
        snapshot,
        "ssw.c",
        workspaces,
        "make all",
    )

    assert result is None
    assert validation.ok is True
    assert validation.stage == "all-widened"


def test_generate_single_pass_returns_none_on_bad_edit_format(monkeypatch, tmp_path):
    """LLM sends malformed response — returns None with edit-failure."""

    class FakeAgent(widen.WidenAgent):
        def __init__(self):
            super().__init__()
            self.llm = lambda messages: "Here is the fix:\n\nSome text without blocks."

    agent = FakeAgent()
    snapshot = widen.SourceSnapshot(files={"ssw.c": "seed\n", "ssw.h": "h"})
    workspaces = widen.WorkspaceSet(tmp_path, tmp_path)

    result, validation = agent._generate_single_pass(
        [{"role": "user", "content": "start"}],
        snapshot, "ssw.c", workspaces, "make all",
    )

    assert result is None
    assert validation.stage == "edit-failure"


def test_generate_single_pass_returns_none_on_llm_exception(monkeypatch, tmp_path):
    """LLM raises an exception — should return None with internal-error."""

    class FakeAgent(widen.WidenAgent):
        def __init__(self):
            super().__init__()
            self.llm = lambda messages: (_ for _ in ()).throw(RuntimeError("API down"))

    agent = FakeAgent()
    snapshot = _make_snapshot()
    workspaces = widen.WorkspaceSet(tmp_path, tmp_path)

    result, validation = agent._generate_single_pass(
        [{"role": "user", "content": "start"}],
        snapshot, "ssw.c", workspaces, "make all",
    )

    assert result is None
    assert validation.stage == "internal-error"
    assert "API down" in validation.stderr


# ---------------------------------------------------------------------------
# run() integration tests
# ---------------------------------------------------------------------------


def test_run_fails_if_baseline_broken(monkeypatch, tmp_path):
    """If input code doesn't compile, run() returns 1 immediately."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "ssw.c").write_text("broken code\n")
    (source_dir / "ssw.h").write_text("h\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=False)

    rc = widen.WidenAgent().run(
        source_dir=source_dir,
        target_file="ssw.c",
        output_dir=output_dir,
    )

    assert rc == 1


def test_run_writes_output_on_success(monkeypatch, tmp_path):
    """Successful widening pass writes output and returns 0."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "ssw.c").write_text("seed\n")
    (source_dir / "ssw.h").write_text("h\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=True)
    monkeypatch.setattr(
        widen, "write_output",
        lambda d, s: d.mkdir(parents=True, exist_ok=True) or (d / "ssw.c").write_text(s.files["ssw.c"]),
    )

    monkeypatch.setattr(
        widen.WidenAgent,
        "_generate_single_pass",
        lambda self, messages, snapshot, file_name, workspaces, build_command: (
            widen.SourceSnapshot(files={**snapshot.files, "ssw.c": "widened code"}),
            ValidationResult(True, "validation", 0, "ok", ""),
        ),
    )

    rc = widen.WidenAgent().run(
        source_dir=source_dir,
        target_file="ssw.c",
        output_dir=output_dir,
        max_steps=1,
    )

    assert rc == 0
    assert (output_dir / "ssw.c").read_text() == "widened code"


def test_run_rejects_missing_target(monkeypatch, tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "ssw.c").write_text("x")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="Target file.*not found"):
        widen.WidenAgent().run(
            source_dir=source_dir,
            target_file="nonexistent.c",
            output_dir=output_dir,
        )


def test_run_stops_on_all_widened(monkeypatch, tmp_path):
    """When LLM signals ALL_WIDENED, run() writes output and returns 0."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "ssw.c").write_text("already widened\n")
    (source_dir / "ssw.h").write_text("h\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=True)
    monkeypatch.setattr(
        widen, "write_output",
        lambda d, s: d.mkdir(parents=True, exist_ok=True) or (d / "ssw.c").write_text(s.files["ssw.c"]),
    )

    monkeypatch.setattr(
        widen.WidenAgent,
        "_generate_single_pass",
        lambda self, messages, snapshot, file_name, workspaces, build_command: (
            None,
            ValidationResult(True, "all-widened", 0, "All done", ""),
        ),
    )

    rc = widen.WidenAgent().run(
        source_dir=source_dir,
        target_file="ssw.c",
        output_dir=output_dir,
    )

    assert rc == 0
    assert (output_dir / "ssw.c").read_text() == "already widened\n"


def test_run_stops_early_on_internal_error(monkeypatch, tmp_path):
    """Internal error (e.g. LLM failure) stops run() and returns 1."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "ssw.c").write_text("code\n")
    (source_dir / "ssw.h").write_text("h\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=True)

    monkeypatch.setattr(
        widen.WidenAgent,
        "_generate_single_pass",
        lambda self, messages, snapshot, file_name, workspaces, build_command: (
            None,
            ValidationResult(False, "internal-error", None, "", "API key expired"),
        ),
    )

    rc = widen.WidenAgent().run(
        source_dir=source_dir,
        target_file="ssw.c",
        output_dir=output_dir,
        max_steps=5,
    )

    assert rc == 1


def test_run_continues_past_edit_failure(monkeypatch, tmp_path):
    """Edit failure on one step doesn't stop the loop — continues to next."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "ssw.c").write_text("seed\n")
    (source_dir / "ssw.h").write_text("h\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=True)
    monkeypatch.setattr(
        widen, "write_output",
        lambda d, s: d.mkdir(parents=True, exist_ok=True) or (d / "ssw.c").write_text(s.files["ssw.c"]),
    )

    call_count = 0

    def fake_generate(self, messages, snapshot, file_name, workspaces, build_command):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None, ValidationResult(False, "edit-failure", None, "", "No edits")
        return (
            widen.SourceSnapshot(files={**snapshot.files, "ssw.c": "widened"}),
            ValidationResult(True, "validation", 0, "ok", ""),
        )

    monkeypatch.setattr(widen.WidenAgent, "_generate_single_pass", fake_generate)

    rc = widen.WidenAgent().run(
        source_dir=source_dir,
        target_file="ssw.c",
        output_dir=output_dir,
        max_steps=3,
    )

    assert rc == 0
    assert call_count >= 2


def test_run_writes_output_even_with_no_progress(monkeypatch, tmp_path):
    """All steps fail edits — still writes output (correctness skipped)."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "ssw.c").write_text("stuck\n")
    (source_dir / "ssw.h").write_text("h\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=True)
    monkeypatch.setattr(
        widen, "write_output",
        lambda d, s: d.mkdir(parents=True, exist_ok=True),
    )

    monkeypatch.setattr(
        widen.WidenAgent,
        "_generate_single_pass",
        lambda self, messages, snapshot, file_name, workspaces, build_command: (
            None,
            ValidationResult(False, "edit-failure", None, "", "No edits"),
        ),
    )

    rc = widen.WidenAgent().run(
        source_dir=source_dir,
        target_file="ssw.c",
        output_dir=output_dir,
        max_steps=2,
    )

    # Returns 0 because correctness check is skipped (no Intel reference)
    assert rc == 0


# ---------------------------------------------------------------------------
# truncate_for_log
# ---------------------------------------------------------------------------


def test_run_benchmarks_at_end(monkeypatch, tmp_path):
    """After all passes, baseline and final benchmarks are run."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "ssw.c").write_text("seed\n")
    (source_dir / "ssw.h").write_text("h\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=True)
    monkeypatch.setattr(
        widen, "write_output",
        lambda d, s: d.mkdir(parents=True, exist_ok=True),
    )

    bench_calls = []

    def fake_benchmark(self, workspace_dir, ssh_compile_cmd, ssh_run_cmd, label):
        bench_calls.append(label)
        if "baseline" in label:
            return 10.0
        return 5.0

    monkeypatch.setattr(widen.WidenAgent, "_benchmark_on_ssh", fake_benchmark)
    monkeypatch.setattr(
        widen.WidenAgent,
        "_generate_single_pass",
        lambda self, messages, snapshot, file_name, workspaces, build_command: (
            widen.SourceSnapshot(files={**snapshot.files, "ssw.c": "widened"}),
            ValidationResult(True, "validation", 0, "ok", ""),
        ),
    )

    widen.WidenAgent().run(
        source_dir=source_dir,
        target_file="ssw.c",
        output_dir=output_dir,
        max_steps=1,
    )

    assert len(bench_calls) == 2
    assert "baseline" in bench_calls[0]
    assert "final" in bench_calls[1]


def test_run_uses_original_source_for_intel_reference(monkeypatch, tmp_path):
    """Intel reference must use ORIGINAL_SOURCE_DIR (initial_code/), not
    the translated RVV source which won't compile on x86."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "ssw.c").write_text("seed\n")
    (source_dir / "ssw.h").write_text("h\n")
    output_dir = tmp_path / "out"

    _patch_infra(monkeypatch, tmp_path, docker_ok=True)
    monkeypatch.setattr(
        widen, "write_output",
        lambda d, s: d.mkdir(parents=True, exist_ok=True),
    )

    captured_dir = []

    def fake_get_intel_ref(self, original_dir, test_data_dir):
        captured_dir.append(original_dir)
        return None

    monkeypatch.setattr(widen.WidenAgent, "_get_intel_reference", fake_get_intel_ref)
    monkeypatch.setattr(
        widen.WidenAgent,
        "_generate_single_pass",
        lambda self, messages, snapshot, file_name, workspaces, build_command: (
            widen.SourceSnapshot(files={**snapshot.files, "ssw.c": "widened"}),
            ValidationResult(True, "validation", 0, "ok", ""),
        ),
    )

    widen.WidenAgent().run(
        source_dir=source_dir,
        target_file="ssw.c",
        output_dir=output_dir,
        max_steps=1,
    )

    assert len(captured_dir) == 1
    # Must be ORIGINAL_SOURCE_DIR (initial_code/), NOT source_dir
    assert captured_dir[0] == widen.ORIGINAL_SOURCE_DIR
    assert captured_dir[0] != source_dir


def test_truncate_for_log_short():
    assert widen.truncate_for_log("short", 100) == "short"


def test_truncate_for_log_long():
    result = widen.truncate_for_log("a" * 200, 50)
    assert len(result) < 200
    assert "truncated" in result
