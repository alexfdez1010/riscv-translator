import pytest

from src import repair, validators
from src.config import REFERENCE_FILE
from src.diff_utils import diff_error_feedback, validate_single_file_diff
from src.prompts import (
    build_diff_format_feedback,
    build_initial_user_prompt,
    build_repair_prompt,
    build_system_prompt,
    load_riscv_reference,
    preprocess_rvv_compat,
)
from src.validators import (
    InitialCodeValidator,
    SSHSSWValidator,
    ValidationResult,
    _infer_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_SNAPSHOT_FILES = {
    "ssw.c": "seed\n",
    "ssw.h": "h\n",
    "main.c": "m\n",
    "Makefile": "mk\n",
    "kseq.h": "k\n",
    "example.c": "e\n",
    "sse2rvv.h": "rvv\n",
}


def _make_snapshot(ssw_code: str = "seed\n") -> repair.SourceSnapshot:
    return repair.SourceSnapshot(files={**_FULL_SNAPSHOT_FILES, "ssw.c": ssw_code})


@pytest.fixture(autouse=True)
def _mock_ssh_validator_init(monkeypatch):
    """Prevent SSHSSWValidator from attempting real SSH connections in tests."""
    monkeypatch.setattr(
        SSHSSWValidator, "__init__", lambda self: setattr(self, "_available", False),
    )


def _patch_repair_infra(monkeypatch, tmp_path, *, initial_ok=False, ssw_ok=True, ssh_ok=True):
    """Patch load_initial_sources, create_workspace, shutil, and validators."""
    monkeypatch.setattr(
        repair, "load_initial_sources", lambda: _make_snapshot("baseline\n"),
    )
    monkeypatch.setattr(
        repair, "create_workspace",
        lambda snapshot: repair.WorkspaceSet(tmp_path / "root", tmp_path),
    )
    monkeypatch.setattr(repair.shutil, "rmtree", lambda *a, **kw: None)
    monkeypatch.setattr(
        InitialCodeValidator, "validate",
        lambda self, snap, wd, tf, qf: ValidationResult(
            initial_ok, "validation" if initial_ok else "compile",
            0 if initial_ok else 1, "", "" if initial_ok else "broken",
        ),
    )
    monkeypatch.setattr(
        repair.RepairAgent, "_passes_ssw_validation",
        lambda self, code: ssw_ok,
    )
    monkeypatch.setattr(
        SSHSSWValidator, "validate",
        lambda self, code: ValidationResult(
            ssh_ok, "ssh-validation" if ssh_ok else "ssh-runtime",
            0 if ssh_ok else 134, "", "" if ssh_ok else "munmap_chunk(): invalid pointer",
        ),
    )


def test_load_riscv_reference_matches_file_contents():
    expected = REFERENCE_FILE.read_text()
    assert load_riscv_reference() == expected


def test_build_system_prompt_includes_reference_path_and_contents():
    prompt = build_system_prompt("ssw.c")
    assert str(REFERENCE_FILE) in prompt
    assert "# RISC-V Vector (RVV) C Programming Concepts" in prompt
    assert "Make the smallest correct change needed." in prompt
    assert "Modify only ssw.c." in prompt
    assert "Return only:" in prompt
    assert "A short summary sentence." in prompt
    assert (
        "A single-file unified git diff patch in a fenced ```diff block for ssw.c."
        in prompt
    )
    assert "--- a/ssw.c" in prompt
    assert "+++ b/ssw.c" in prompt
    assert "@@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@" in prompt


def test_apply_content_to_snapshot_updates_only_requested_file():
    snapshot = repair.SourceSnapshot(files={"ssw.c": "old", "ssw.h": "keep"})
    updated = repair.apply_content_to_snapshot(snapshot, "ssw.c", "new")
    assert updated.files["ssw.c"] == "new"
    assert updated.files["ssw.h"] == "keep"


def test_extract_diff_from_fenced_response():
    response = """```diff
diff --git a/ssw.c b/ssw.c
--- a/ssw.c
+++ b/ssw.c
@@ -1 +1 @@
-old
+new
```"""
    assert repair.extract_diff(response) == "\n".join(response.splitlines()[1:-1])


def test_extract_diff_from_summary_and_fenced_response():
    response = """Small fix to keep the change localized.

```diff
diff --git a/ssw.c b/ssw.c
--- a/ssw.c
+++ b/ssw.c
@@ -1 +1 @@
-old
+new
```"""
    assert repair.extract_diff(response) == "\n".join(response.splitlines()[3:-1])


def test_validate_single_file_diff_accepts_expected_format():
    patch = """diff --git a/ssw.c b/ssw.c
--- a/ssw.c
+++ b/ssw.c
@@ -1 +1 @@
-old
+new
"""
    validate_single_file_diff(patch, "ssw.c")


def test_validate_single_file_diff_rejects_missing_hunk():
    patch = """diff --git a/ssw.c b/ssw.c
--- a/ssw.c
+++ b/ssw.c
"""
    with pytest.raises(ValueError, match="unified diff hunk"):
        validate_single_file_diff(patch, "ssw.c")


def test_validate_single_file_diff_rejects_wrong_headers():
    patch = """diff --git a/other.c b/other.c
--- a/other.c
+++ b/other.c
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(ValueError, match="Patch must target only ssw.c"):
        validate_single_file_diff(patch, "ssw.c")


def test_apply_patch_to_snapshot_updates_target_file(tmp_path):
    snapshot = repair.SourceSnapshot(files={"ssw.c": "old\n", "ssw.h": "keep\n"})
    patch = """diff --git a/ssw.c b/ssw.c
--- a/ssw.c
+++ b/ssw.c
@@ -1 +1 @@
-old
+new
"""
    updated = repair.apply_patch_to_snapshot(snapshot, "ssw.c", patch)
    assert updated.files["ssw.c"] == "new\n"
    assert updated.files["ssw.h"] == "keep\n"


def test_apply_patch_to_snapshot_rejects_missing_exact_headers(tmp_path):
    snapshot = repair.SourceSnapshot(files={"ssw.c": "old\n", "ssw.h": "keep\n"})
    patch = """--- ssw.c
+++ ssw.c
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(ValueError, match="Patch must include exact header"):
        repair.apply_patch_to_snapshot(snapshot, "ssw.c", patch)


def test_apply_patch_to_snapshot_rejects_cross_file_patch(tmp_path):
    snapshot = repair.SourceSnapshot(files={"ssw.c": "old\n", "ssw.h": "keep\n"})
    patch = """diff --git a/ssw.c b/ssw.c
--- a/ssw.c
+++ b/ssw.c
@@ -1 +1 @@
-old
+new
diff --git a/ssw.h b/ssw.h
--- a/ssw.h
+++ b/ssw.h
@@ -1 +1 @@
-keep
+changed
"""
    with pytest.raises(ValueError, match="Patch must target only ssw.c"):
        repair.apply_patch_to_snapshot(snapshot, "ssw.c", patch)


def test_build_initial_user_prompt_uses_structured_format():
    prompt = build_initial_user_prompt(
        "ssw.c", "int main(void) {}\n", "target.fa", "query.fa"
    )
    assert "Task: Repair this file for the RISC-V validation flow." in prompt
    assert "Goal:" in prompt
    assert "Repository context:" in prompt
    assert "What to change:" in prompt
    assert "What not to change:" in prompt
    assert "Acceptance criteria:" in prompt
    assert "Output:" in prompt
    assert "First, one short summary sentence." in prompt
    assert (
        "Then, a single fenced `diff` block containing only the unified diff for `ssw.c`."
        in prompt
    )


def test_build_repair_prompt_uses_structured_format():
    prompt = build_repair_prompt(
        "ssw.c",
        "int main(void) {}\n",
        "Validation stage: compile\nReturn code: 1\nFailure details:\nerror: broken",
    )
    assert "Task: Fix the current validation failure in this file." in prompt
    assert "Goal:" in prompt
    assert "Validation failure details:" in prompt
    assert "error: broken" in prompt
    assert "First, one short summary sentence." in prompt
    assert (
        "Then, a single fenced `diff` block containing only the unified diff for `ssw.c`."
        in prompt
    )


def test_build_diff_format_feedback_uses_structured_format():
    prompt = diff_error_feedback("ssw.c", "int main(void) {}\n", "missing header")
    assert "missing header" in prompt
    assert "int main(void) {}" in prompt
    assert "--- a/ssw.c" in prompt
    assert "+++ b/ssw.c" in prompt


def test_live_model_returns_summary_and_applicable_diff_for_small_change(tmp_path):
    code = "uint32_t foo(void) {\n    return 1;\n}\n"
    feedback = ValidationResult(
        False,
        "compile",
        1,
        "",
        "error: unknown type name 'uint32_t'\n"
        "Make the smallest correct change by adding `#include <stdint.h>` at the top "
        "of the file. Do not modify the function body.",
    )
    agent = repair.RepairAgent()
    original_validate = agent.validator.validate
    original_retries = repair.LLM_VALIDATION_RETRIES
    try:
        snapshot = repair.SourceSnapshot(files={"ssw.c": code})
        workspaces = repair.WorkspaceSet(tmp_path, tmp_path)
        agent.validator.validate = (
            lambda snapshot,
            workspace_dir,
            target_fasta,
            query_fasta: ValidationResult(True, "validation", 0, "ok", "")
        )
        original_retries = repair.LLM_VALIDATION_RETRIES
        repair.LLM_VALIDATION_RETRIES = 2
        result, validation = agent._generate_valid_file(
            [
                {"role": "system", "content": build_system_prompt("ssw.c")},
                {
                    "role": "user",
                    "content": build_repair_prompt(
                        "ssw.c", code, feedback.as_feedback()
                    ),
                },
            ],
            snapshot,
            "ssw.c",
            workspaces,
            "target.fa",
            "query.fa",
        )
    except Exception as exc:
        pytest.skip(f"Live repair model probe unavailable: {exc}")
    finally:
        repair.LLM_VALIDATION_RETRIES = original_retries
        agent.validator.validate = original_validate

    if result is None and validation.stage == "internal-error":
        pytest.skip(f"Live repair model probe unavailable: {validation.stderr}")

    assert result is not None
    assert validation.ok is True
    assert result.files["ssw.c"].startswith("#include <stdint.h>\n")
    assert "uint32_t foo(void) {\n    return 1;\n}\n" in result.files["ssw.c"]


def test_generate_valid_file_retries_with_feedback(monkeypatch, tmp_path):
    responses = iter(
        [
            """Add a required include before changing anything else.

```diff
diff --git a/ssw.c b/ssw.c
--- a/ssw.c
+++ b/ssw.c
@@ -1 +1,2 @@
-seed
+#include <stdio.h>
+seed
```""",
            """Replace the placeholder line with a minimal stub.

```diff
diff --git a/ssw.c b/ssw.c
--- a/ssw.c
+++ b/ssw.c
@@ -1,2 +1,2 @@
 #include <stdio.h>
-seed
+void sequence_alignment_wavefront(void) {}
```""",
        ]
    )
    validations = iter(
        [
            ValidationResult(False, "compile", 1, "", "error: compile failed"),
            ValidationResult(True, "validation", 0, "ok", ""),
        ]
    )
    calls = []

    class FakeAgent(repair.RepairAgent):
        def __init__(self):
            super().__init__()
            self.llm = lambda messages: calls.append(messages) or next(responses)

    monkeypatch.setattr(repair, "LLM_VALIDATION_RETRIES", 2)
    monkeypatch.setattr(
        InitialCodeValidator,
        "validate",
        lambda self, snapshot, workspace_dir, target_fasta, query_fasta: next(
            validations
        ),
    )

    agent = FakeAgent()
    snapshot = repair.SourceSnapshot(
        files={
            "ssw.c": "seed\n",
            "ssw.h": "h",
            "main.c": "m",
            "Makefile": "mk",
            "kseq.h": "k",
            "example.c": "e",
            "sse2rvv.h": "rvv",
        }
    )
    workspaces = repair.WorkspaceSet(tmp_path, tmp_path)
    result, validation = agent._generate_valid_file(
        [{"role": "user", "content": "start"}],
        snapshot,
        "ssw.c",
        workspaces,
        "target.fa",
        "query.fa",
    )

    assert result is not None
    assert (
        result.files["ssw.c"]
        == "#include <stdio.h>\nvoid sequence_alignment_wavefront(void) {}\n"
    )
    assert validation.ok is True
    assert len(calls) == 2
    assert "compile failed" in calls[1][-1].content


def test_run_writes_output_only_after_success(monkeypatch, tmp_path):
    input_file = tmp_path / "ssw.c"
    output_file = tmp_path / "repaired_ssw.c"
    input_file.write_text("seed")

    monkeypatch.setattr(
        repair,
        "load_initial_sources",
        lambda: repair.SourceSnapshot(
            files={
                "ssw.c": "baseline",
                "ssw.h": "h",
                "main.c": "m",
                "Makefile": "mk",
                "kseq.h": "k",
                "example.c": "e",
                "sse2rvv.h": "rvv",
            }
        ),
    )
    monkeypatch.setattr(
        repair,
        "create_workspace",
        lambda snapshot: repair.WorkspaceSet(tmp_path / "root", tmp_path),
    )
    monkeypatch.setattr(
        repair.shutil,
        "rmtree",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        InitialCodeValidator,
        "validate",
        lambda self,
        snapshot,
        workspace_dir,
        target_fasta,
        query_fasta: ValidationResult(False, "compile", 1, "", "broken"),
    )
    monkeypatch.setattr(
        repair.RepairAgent,
        "_generate_valid_file",
        lambda self,
        messages,
        snapshot,
        file_name,
        workspaces,
        target_fasta,
        query_fasta: (
            repair.SourceSnapshot(
                files={
                    **snapshot.files,
                    "ssw.c": "void sequence_alignment_wavefront(void) {}",
                }
            ),
            ValidationResult(True, "validation", 0, "ok", ""),
        ),
    )
    monkeypatch.setattr(
        repair.RepairAgent,
        "_passes_ssw_validation",
        lambda self, code: True,
    )

    rc = repair.RepairAgent().run(
        input_file=input_file,
        output_file=output_file,
        target_fasta="target.fa",
        query_fasta="query.fa",
    )

    assert rc == 0
    assert output_file.read_text() == "void sequence_alignment_wavefront(void) {}"


def test_run_rejects_unsupported_target(tmp_path):
    input_file = tmp_path / "not_supported.c"
    output_file = tmp_path / "out.c"
    input_file.write_text("x")

    with pytest.raises(ValueError, match="Unsupported repair target"):
        repair.RepairAgent().run(
            input_file=input_file,
            output_file=output_file,
            target_fasta="target.fa",
            query_fasta="query.fa",
        )


# ===================================================================
# Preprocessing tests
# ===================================================================


class TestPreprocessRvvCompat:
    def test_noop_without_m128i(self):
        code = "int main() { return 0; }\n"
        assert preprocess_rvv_compat(code) == code

    def test_replaces_sizeof_m128i(self):
        code = '__m128i* buf = calloc(n, sizeof(__m128i));\n'
        result = preprocess_rvv_compat(code)
        assert "sizeof(__m128i)" not in result
        assert "_SSW_VEC_BYTES" in result

    def test_multiple_sizeof_replaced(self):
        code = (
            "__m128i x;\n"
            "a = sizeof(__m128i);\n"
            "b = sizeof(__m128i);\n"
        )
        result = preprocess_rvv_compat(code)
        assert result.count("sizeof(__m128i)") == 0
        assert result.count("_SSW_VEC_BYTES") == 2

    def test_injects_helpers_after_kroundup32(self):
        code = (
            "#define kroundup32(x) (--(x), ++(x))\n"
            "size_t s = sizeof(__m128i);\n"
        )
        result = preprocess_rvv_compat(code)
        assert "_load_vec" in result
        assert "_store_vec" in result
        assert "_VEC_PTR" in result
        # Helpers should appear after the kroundup32 line
        kr_pos = result.find("kroundup32")
        helpers_pos = result.find("_load_vec")
        assert helpers_pos > kr_pos

    def test_no_double_helper_injection(self):
        code = (
            "#define _SSW_VEC_BYTES 16\n"
            "#define kroundup32(x) (--(x), ++(x))\n"
            "size_t s = sizeof(__m128i);\n"
        )
        result = preprocess_rvv_compat(code)
        # _SSW_VEC_BYTES was already present → helpers not re-injected
        assert result.count("_load_vec") == 0

    def test_pointer_arithmetic_replacement(self):
        code = "__m128i x; _mm_load_si128(pvE + j);\n"
        result = preprocess_rvv_compat(code)
        assert "_VEC_PTR(pvE, j)" in result

    def test_pointer_arithmetic_multiple_names(self):
        code = (
            "__m128i x;\n"
            "_mm_load_si128(pvHStore + j);\n"
            "_mm_store_si128(pvHLoad + k, v);\n"
        )
        result = preprocess_rvv_compat(code)
        assert "_VEC_PTR(pvHStore, j)" in result
        assert "_VEC_PTR(pvHLoad, k" in result

    def test_store_load_array_pattern(self):
        code = "__m128i x; pvHmax[j] = pvHStore[j];\n"
        result = preprocess_rvv_compat(code)
        assert "_store_vec(pvHmax, j, _load_vec(pvHStore, j))" in result

    def test_rvalue_indexing(self):
        code = "__m128i vH = pvHStore[segLen - 1];\n"
        result = preprocess_rvv_compat(code)
        assert "_load_vec(pvHStore, segLen - 1)" in result

    def test_does_not_touch_non_vec_variables(self):
        code = '__m128i x; int arr[10]; arr[0] = 1;\n'
        result = preprocess_rvv_compat(code)
        # arr is not in _vec_names, should remain unchanged
        assert "arr[0] = 1" in result

    def test_realistic_calloc_pattern(self):
        code = (
            "#define kroundup32(x) (--(x), ++(x))\n"
            '__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));\n'
            '__m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));\n'
        )
        result = preprocess_rvv_compat(code)
        # The calloc lines should use _SSW_VEC_BYTES instead of sizeof
        assert "calloc(segLen, _SSW_VEC_BYTES)" in result
        # Count only non-helper occurrences: the injected helpers have their
        # own sizeof(__m128i) in the #else fallback, which is expected.
        calloc_lines = [l for l in result.splitlines() if "calloc" in l]
        for line in calloc_lines:
            assert "sizeof(__m128i)" not in line


# ===================================================================
# Snapshot and workspace tests
# ===================================================================


class TestSnapshotOperations:
    def test_apply_content_rejects_unknown_file(self):
        snapshot = repair.SourceSnapshot(files={"ssw.c": "code"})
        with pytest.raises(ValueError, match="Unsupported target path"):
            repair.apply_content_to_snapshot(snapshot, "unknown.c", "new")

    def test_materialize_snapshot_writes_files(self, tmp_path):
        snapshot = _make_snapshot("hello world\n")
        repair.materialize_snapshot(tmp_path, snapshot)
        assert (tmp_path / "ssw.c").read_text() == "hello world\n"
        assert (tmp_path / "ssw.h").read_text() == "h\n"
        assert (tmp_path / "Makefile").read_text() == "mk\n"

    def test_load_initial_sources_returns_all_expected_files(self):
        snapshot = repair.load_initial_sources()
        for name in repair.TARGET_FILE_NAMES:
            assert name in snapshot.files, f"Missing {name}"
            assert len(snapshot.files[name]) > 0, f"{name} is empty"

    def test_create_workspace_copies_directory(self):
        snapshot = _make_snapshot()
        ws = repair.create_workspace(snapshot)
        try:
            assert ws.current_dir.exists()
            assert (ws.current_dir / "ssw.c").read_text() == "seed\n"
        finally:
            import shutil
            shutil.rmtree(ws.root, ignore_errors=True)


# ===================================================================
# ValidationResult tests
# ===================================================================


class TestValidationResult:
    def test_combined_output_empty_parts(self):
        result = ValidationResult(False, "compile", 1, "", "")
        assert result.combined_output == ""

    def test_combined_output_only_stdout(self):
        result = ValidationResult(False, "compile", 1, "out", "")
        assert result.combined_output == "out"

    def test_combined_output_only_stderr(self):
        result = ValidationResult(False, "compile", 1, "", "err")
        assert result.combined_output == "err"

    def test_as_feedback_includes_all_fields(self):
        result = ValidationResult(False, "compile", 1, "out", "err")
        feedback = result.as_feedback()
        assert "Validation stage: compile" in feedback
        assert "Return code: 1" in feedback
        assert "out" in feedback
        assert "err" in feedback

    def test_as_feedback_truncates_long_output(self):
        result = ValidationResult(False, "compile", 1, "x" * 20000, "")
        feedback = result.as_feedback(limit=100)
        assert "...[truncated]" in feedback

    def test_as_feedback_no_output(self):
        result = ValidationResult(False, "compile", 1, "", "")
        feedback = result.as_feedback()
        assert "No additional output was captured." in feedback


# ===================================================================
# InitialCodeValidator._infer_stage tests
# ===================================================================


class TestInferStage:
    def test_undefined_reference_is_compile(self):
        assert (
            _infer_stage(
                "", "undefined reference to `ssw_init'"
            )
            == "compile"
        )

    def test_error_colon_is_compile(self):
        assert (
            _infer_stage(
                "", "ssw.c:42: error: unknown type"
            )
            == "compile"
        )

    def test_validation_failed_is_correctness(self):
        assert (
            _infer_stage(
                "VALIDATION FAILED: score mismatch", ""
            )
            == "correctness"
        )

    def test_mismatch_is_correctness(self):
        assert (
            _infer_stage("mismatch at (1,2)", "")
            == "correctness"
        )

    def test_segfault_is_runtime(self):
        assert (
            _infer_stage("Segmentation fault", "")
            == "runtime"
        )

    def test_empty_is_runtime(self):
        assert _infer_stage("", "") == "runtime"


# ===================================================================
# Repair agent flow tests (fully mocked)
# ===================================================================


class TestRepairRunFlow:
    def test_baseline_passes_both_validators_writes_immediately(
        self, monkeypatch, tmp_path
    ):
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("good code")
        _patch_repair_infra(monkeypatch, tmp_path, initial_ok=True, ssw_ok=True)

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa",
        )
        assert rc == 0
        assert output_file.exists()

    def test_baseline_passes_initial_but_fails_ssw_continues_repair(
        self, monkeypatch, tmp_path
    ):
        """If baseline passes InitialCodeValidator but fails SSWValidator,
        repair should continue (not write output immediately)."""
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("code")

        # First call: initial_ok=True but ssw_ok=False (baseline)
        # Then repair loop runs, _generate_valid_file returns None → max_steps
        monkeypatch.setattr(
            repair, "load_initial_sources", lambda: _make_snapshot("baseline\n"),
        )
        monkeypatch.setattr(
            repair, "create_workspace",
            lambda snap: repair.WorkspaceSet(tmp_path / "root", tmp_path),
        )
        monkeypatch.setattr(repair.shutil, "rmtree", lambda *a, **kw: None)
        monkeypatch.setattr(
            InitialCodeValidator, "validate",
            lambda self, snap, wd, tf, qf: ValidationResult(
                True, "validation", 0, "ok", "",
            ),
        )
        monkeypatch.setattr(
            repair.RepairAgent, "_passes_ssw_validation",
            lambda self, code: False,
        )
        monkeypatch.setattr(
            repair.RepairAgent, "_generate_valid_file",
            lambda self, msgs, snap, fn, ws, tf, qf: (
                None,
                ValidationResult(
                    False, "internal-error", None, "",
                    "LLM unavailable",
                ),
            ),
        )

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa", max_steps=1,
        )
        assert rc == 1
        assert not output_file.exists()

    def test_repair_stops_on_internal_error(self, monkeypatch, tmp_path):
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("code")
        _patch_repair_infra(monkeypatch, tmp_path)

        monkeypatch.setattr(
            repair.RepairAgent, "_generate_valid_file",
            lambda self, msgs, snap, fn, ws, tf, qf: (
                None,
                ValidationResult(
                    False, "internal-error", None, "", "connection refused",
                ),
            ),
        )

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa", max_steps=5,
        )
        assert rc == 1
        assert not output_file.exists()

    def test_repair_exhausts_max_steps(self, monkeypatch, tmp_path):
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("code")
        _patch_repair_infra(monkeypatch, tmp_path)

        step_count = [0]

        def fake_generate(self, msgs, snap, fn, ws, tf, qf):
            step_count[0] += 1
            return (
                None,
                ValidationResult(False, "compile", 1, "", "still broken"),
            )

        monkeypatch.setattr(
            repair.RepairAgent, "_generate_valid_file", fake_generate,
        )

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa", max_steps=3,
        )
        assert rc == 1
        assert step_count[0] == 3
        assert not output_file.exists()

    def test_repair_partial_progress_then_succeeds(self, monkeypatch, tmp_path):
        """Step 1: candidate produced but validation fails.
        Step 2: candidate produced and passes everything."""
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("broken")

        monkeypatch.setattr(
            repair, "load_initial_sources", lambda: _make_snapshot("broken\n"),
        )
        monkeypatch.setattr(
            repair, "create_workspace",
            lambda snap: repair.WorkspaceSet(tmp_path / "root", tmp_path),
        )
        monkeypatch.setattr(repair.shutil, "rmtree", lambda *a, **kw: None)
        monkeypatch.setattr(
            InitialCodeValidator, "validate",
            lambda self, snap, wd, tf, qf: ValidationResult(
                False, "compile", 1, "", "error",
            ),
        )
        monkeypatch.setattr(
            repair.RepairAgent, "_passes_ssw_validation",
            lambda self, code: True,
        )

        step = [0]
        generate_results = [
            # Step 1: partial progress (validation fails)
            (
                _make_snapshot("partial\n"),
                ValidationResult(False, "correctness", 2, "mismatch", ""),
            ),
            # Step 2: success
            (
                _make_snapshot("fixed\n"),
                ValidationResult(True, "validation", 0, "ok", ""),
            ),
        ]

        def fake_generate(self, msgs, snap, fn, ws, tf, qf):
            idx = step[0]
            step[0] += 1
            return generate_results[idx]

        monkeypatch.setattr(
            repair.RepairAgent, "_generate_valid_file", fake_generate,
        )

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa", max_steps=5,
        )
        assert rc == 0
        assert output_file.read_text() == "fixed\n"
        assert step[0] == 2

    def test_repair_passes_initial_but_fails_ssw_continues(
        self, monkeypatch, tmp_path
    ):
        """Repair candidate passes InitialCodeValidator but fails SSWValidator.
        Should NOT write output, should continue trying."""
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("code")

        monkeypatch.setattr(
            repair, "load_initial_sources", lambda: _make_snapshot("code\n"),
        )
        monkeypatch.setattr(
            repair, "create_workspace",
            lambda snap: repair.WorkspaceSet(tmp_path / "root", tmp_path),
        )
        monkeypatch.setattr(repair.shutil, "rmtree", lambda *a, **kw: None)
        monkeypatch.setattr(
            InitialCodeValidator, "validate",
            lambda self, snap, wd, tf, qf: ValidationResult(
                False, "compile", 1, "", "broken",
            ),
        )

        ssw_calls = [0]

        def ssw_validate(self, code):
            ssw_calls[0] += 1
            return False  # always fails

        monkeypatch.setattr(
            repair.RepairAgent, "_passes_ssw_validation", ssw_validate,
        )

        monkeypatch.setattr(
            repair.RepairAgent, "_generate_valid_file",
            lambda self, msgs, snap, fn, ws, tf, qf: (
                _make_snapshot("attempt\n"),
                ValidationResult(True, "validation", 0, "ok", ""),
            ),
        )

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa", max_steps=2,
        )
        assert rc == 1
        assert not output_file.exists()
        assert ssw_calls[0] == 2  # called once per step


# ===================================================================
# SSH hardware validation gate tests
# ===================================================================


class TestSSHValidationGate:
    def test_baseline_passes_qemu_and_ssh_writes_immediately(
        self, monkeypatch, tmp_path
    ):
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("good code")
        _patch_repair_infra(
            monkeypatch, tmp_path, initial_ok=True, ssw_ok=True, ssh_ok=True,
        )

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa",
        )
        assert rc == 0
        assert output_file.exists()

    def test_baseline_passes_qemu_but_fails_ssh_continues_repair(
        self, monkeypatch, tmp_path
    ):
        """If baseline passes QEMU but fails SSH, repair should continue."""
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("code")
        _patch_repair_infra(
            monkeypatch, tmp_path, initial_ok=True, ssw_ok=True, ssh_ok=False,
        )
        monkeypatch.setattr(
            repair.RepairAgent, "_generate_valid_file",
            lambda self, msgs, snap, fn, ws, tf, qf: (
                None,
                ValidationResult(
                    False, "internal-error", None, "", "LLM unavailable",
                ),
            ),
        )

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa", max_steps=1,
        )
        assert rc == 1
        assert not output_file.exists()

    def test_repair_step_passes_qemu_but_fails_ssh_continues(
        self, monkeypatch, tmp_path
    ):
        """Repair candidate passes QEMU + SSW but fails SSH → continue with
        SSH error feedback."""
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("code")

        monkeypatch.setattr(
            repair, "load_initial_sources", lambda: _make_snapshot("code\n"),
        )
        monkeypatch.setattr(
            repair, "create_workspace",
            lambda snap: repair.WorkspaceSet(tmp_path / "root", tmp_path),
        )
        monkeypatch.setattr(repair.shutil, "rmtree", lambda *a, **kw: None)
        monkeypatch.setattr(
            InitialCodeValidator, "validate",
            lambda self, snap, wd, tf, qf: ValidationResult(
                False, "compile", 1, "", "broken",
            ),
        )
        monkeypatch.setattr(
            repair.RepairAgent, "_passes_ssw_validation",
            lambda self, code: True,
        )
        monkeypatch.setattr(
            SSHSSWValidator, "validate",
            lambda self, code: ValidationResult(
                False, "ssh-runtime", 134, "munmap_chunk(): invalid pointer", "",
            ),
        )

        step = [0]

        def fake_generate(self, msgs, snap, fn, ws, tf, qf):
            step[0] += 1
            return (
                _make_snapshot("attempt\n"),
                ValidationResult(True, "validation", 0, "ok", ""),
            )

        monkeypatch.setattr(
            repair.RepairAgent, "_generate_valid_file", fake_generate,
        )

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa", max_steps=2,
        )
        assert rc == 1
        assert not output_file.exists()
        assert step[0] == 2  # tried both steps

    def test_repair_step_passes_all_validations_writes_output(
        self, monkeypatch, tmp_path
    ):
        """Repair candidate passes QEMU + SSW + SSH → success."""
        input_file = tmp_path / "ssw.c"
        output_file = tmp_path / "out.c"
        input_file.write_text("broken")

        monkeypatch.setattr(
            repair, "load_initial_sources", lambda: _make_snapshot("broken\n"),
        )
        monkeypatch.setattr(
            repair, "create_workspace",
            lambda snap: repair.WorkspaceSet(tmp_path / "root", tmp_path),
        )
        monkeypatch.setattr(repair.shutil, "rmtree", lambda *a, **kw: None)
        monkeypatch.setattr(
            InitialCodeValidator, "validate",
            lambda self, snap, wd, tf, qf: ValidationResult(
                False, "compile", 1, "", "error",
            ),
        )
        monkeypatch.setattr(
            repair.RepairAgent, "_passes_ssw_validation",
            lambda self, code: True,
        )
        monkeypatch.setattr(
            SSHSSWValidator, "validate",
            lambda self, code: ValidationResult(
                True, "ssh-validation", 0, "ok", "",
            ),
        )
        monkeypatch.setattr(
            repair.RepairAgent, "_generate_valid_file",
            lambda self, msgs, snap, fn, ws, tf, qf: (
                _make_snapshot("fixed\n"),
                ValidationResult(True, "validation", 0, "ok", ""),
            ),
        )

        rc = repair.RepairAgent().run(
            input_file, output_file, "t.fa", "q.fa", max_steps=3,
        )
        assert rc == 0
        assert output_file.read_text() == "fixed\n"

    def test_ssh_unavailable_skips_gracefully(self, monkeypatch, tmp_path):
        """When SSH is unavailable, SSH validation is skipped (ok=True)."""
        validator = SSHSSWValidator()
        # The autouse fixture sets _available=False
        result = validator.validate("some code")
        assert result.ok is True
        assert result.stage == "ssh-skipped"


# ===================================================================
# SSHSSWValidator unit tests
# ===================================================================


class TestSSHSSWValidator:
    def test_check_ssh_returns_false_on_failure(self, monkeypatch):
        monkeypatch.setattr(
            validators.subprocess, "run",
            lambda *a, **kw: type("R", (), {"returncode": 1, "stdout": ""})(),
        )
        assert SSHSSWValidator._check_ssh() is False

    def test_validate_skips_when_unavailable(self):
        validator = SSHSSWValidator()
        # autouse fixture ensures _available=False
        result = validator.validate("code")
        assert result.ok is True
        assert result.stage == "ssh-skipped"


# ===================================================================
# _generate_valid_file edge cases
# ===================================================================


class TestGenerateValidFile:
    def _make_agent(self, monkeypatch, llm_fn, validate_fn):
        """Create a RepairAgent with fake LLM and validator."""

        class FakeAgent(repair.RepairAgent):
            def __init__(self):
                super().__init__()
                self.llm = llm_fn

        monkeypatch.setattr(
            InitialCodeValidator, "validate", validate_fn,
        )
        return FakeAgent()

    def test_llm_exception_returns_none(self, monkeypatch, tmp_path):
        def boom(messages):
            raise ConnectionError("LLM down")

        agent = self._make_agent(
            monkeypatch, boom,
            lambda self, s, w, t, q: None,
        )
        snapshot = _make_snapshot()
        ws = repair.WorkspaceSet(tmp_path, tmp_path)

        result, val = agent._generate_valid_file(
            [{"role": "user", "content": "fix"}],
            snapshot, "ssw.c", ws, "t.fa", "q.fa",
        )
        assert result is None
        assert val.stage == "internal-error"

    def test_no_diff_extracted_all_retries(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repair, "LLM_VALIDATION_RETRIES", 1)

        calls = []

        def bad_llm(messages):
            calls.append(1)
            return "This response has no diff at all."

        agent = self._make_agent(
            monkeypatch, bad_llm,
            lambda self, s, w, t, q: None,
        )
        snapshot = _make_snapshot()
        ws = repair.WorkspaceSet(tmp_path, tmp_path)

        result, val = agent._generate_valid_file(
            [{"role": "user", "content": "fix"}],
            snapshot, "ssw.c", ws, "t.fa", "q.fa",
        )
        assert result is None
        assert len(calls) == 2  # initial + 1 retry

    def test_patch_application_error_retries(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repair, "LLM_VALIDATION_RETRIES", 1)

        calls = []
        responses = iter([
            # First: valid-looking diff but wrong context
            "Fix applied.\n\n```diff\n--- a/ssw.c\n+++ b/ssw.c\n"
            "@@ -1 +1 @@\n-nonexistent line\n+new\n```",
            # Second: correct diff
            "Fix.\n\n```diff\n--- a/ssw.c\n+++ b/ssw.c\n"
            "@@ -1 +1 @@\n-seed\n+fixed\n```",
        ])

        def fake_llm(messages):
            calls.append(messages)
            return next(responses)

        agent = self._make_agent(
            monkeypatch, fake_llm,
            lambda self, snap, wd, tf, qf: ValidationResult(
                True, "validation", 0, "ok", "",
            ),
        )
        snapshot = _make_snapshot()
        ws = repair.WorkspaceSet(tmp_path, tmp_path)

        result, val = agent._generate_valid_file(
            [{"role": "user", "content": "fix"}],
            snapshot, "ssw.c", ws, "t.fa", "q.fa",
        )
        assert result is not None
        assert result.files["ssw.c"] == "fixed\n"
        assert len(calls) == 2

    def test_validation_failure_feedback_includes_stage(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(repair, "LLM_VALIDATION_RETRIES", 1)

        calls = []
        responses = iter([
            "Fix.\n\n```diff\n--- a/ssw.c\n+++ b/ssw.c\n"
            "@@ -1 +1 @@\n-seed\n+attempt1\n```",
            "Fix.\n\n```diff\n--- a/ssw.c\n+++ b/ssw.c\n"
            "@@ -1 +1 @@\n-attempt1\n+attempt2\n```",
        ])
        validations = iter([
            ValidationResult(
                False, "correctness", 2, "score mismatch", "",
            ),
            ValidationResult(True, "validation", 0, "ok", ""),
        ])

        def fake_llm(messages):
            calls.append(messages)
            return next(responses)

        agent = self._make_agent(
            monkeypatch, fake_llm,
            lambda self, snap, wd, tf, qf: next(validations),
        )
        snapshot = _make_snapshot()
        ws = repair.WorkspaceSet(tmp_path, tmp_path)

        result, val = agent._generate_valid_file(
            [{"role": "user", "content": "fix"}],
            snapshot, "ssw.c", ws, "t.fa", "q.fa",
        )
        assert result is not None
        assert val.ok is True
        # Second call should have feedback about correctness failure
        assert "score mismatch" in calls[1][-1].content


# ===================================================================
# SSW validation gate integration tests
# ===================================================================


class TestSSWValidationGate:
    def test_passes_ssw_validation_delegates_to_ssw_validator(
        self, monkeypatch
    ):
        from src import fitness

        called_with = []

        def fake_validate(self, ssw_code):
            called_with.append(ssw_code)
            return fitness.ValidationResult(
                True, "validation", 0, "ok", "",
            )

        monkeypatch.setattr(
            fitness.SSWValidator, "validate", fake_validate,
        )

        agent = repair.RepairAgent()
        assert agent._passes_ssw_validation("test code") is True
        assert called_with == ["test code"]

    def test_fails_ssw_validation_returns_false(self, monkeypatch):
        from src import fitness

        monkeypatch.setattr(
            fitness.SSWValidator, "validate",
            lambda self, code: fitness.ValidationResult(
                False, "compile", 1, "", "error: unknown type",
            ),
        )

        agent = repair.RepairAgent()
        assert agent._passes_ssw_validation("bad code") is False


# ===================================================================
# Truncation helper
# ===================================================================


class TestTruncateForLog:
    def test_short_text_unchanged(self):
        assert repair.truncate_for_log("hello", 100) == "hello"

    def test_long_text_truncated(self):
        result = repair.truncate_for_log("x" * 200, 50)
        assert len(result) < 200
        assert result.endswith("...[truncated]")

    def test_exact_limit_unchanged(self):
        text = "x" * 100
        assert repair.truncate_for_log(text, 100) == text


# ===================================================================
# Prompt construction tests
# ===================================================================


class TestPromptConstruction:
    def test_system_prompt_contains_rvv_rules(self):
        prompt = build_system_prompt("ssw.c")
        assert "sizeof(__m128i) is ILLEGAL" in prompt
        assert "Pointer arithmetic on __m128i*" in prompt
        assert "Array indexing" in prompt

    def test_initial_prompt_without_validation_feedback(self):
        prompt = build_initial_user_prompt(
            "ssw.c", "code", "t.fa", "q.fa",
        )
        assert "Current validation failure:" not in prompt
        assert "Current code:" in prompt
        assert "code" in prompt

    def test_initial_prompt_with_validation_feedback(self):
        prompt = build_initial_user_prompt(
            "ssw.c", "code", "t.fa", "q.fa",
            validation_feedback="error: broken",
        )
        assert "Current validation failure:" in prompt
        assert "error: broken" in prompt

    def test_repair_prompt_includes_code_and_failure(self):
        prompt = build_repair_prompt(
            "ssw.c", "int main() {}", "compile error",
        )
        assert "int main() {}" in prompt
        assert "compile error" in prompt
        assert "--- a/ssw.c" in prompt
        assert "+++ b/ssw.c" in prompt
