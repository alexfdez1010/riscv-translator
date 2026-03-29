"""Tests for src.benchmark — focused on local vs remote Intel execution paths."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark import (
    BenchmarkResult,
    benchmark,
    compare_outputs,
    normalize_output,
    prepare_local_dir,
    run_locally,
    run_on_host,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_result(host: str = "h", label: str = "L", stdout: str = "out") -> BenchmarkResult:
    return BenchmarkResult(host=host, label=label, ok=True, elapsed_seconds=1.0, stdout=stdout, stderr="")


def _fail_result(host: str = "h", label: str = "L") -> BenchmarkResult:
    return BenchmarkResult(host=host, label=label, ok=False, elapsed_seconds=0, stdout="", stderr="error")


def _make_subprocess_result(returncode: int = 0, stdout: str = "", stderr: str = ""):
    r = MagicMock()
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


# ---------------------------------------------------------------------------
# run_locally
# ---------------------------------------------------------------------------

class TestRunLocally:
    """Tests for the run_locally function."""

    @patch("src.benchmark.subprocess.run")
    def test_compile_and_run_success(self, mock_run):
        mock_run.side_effect = [
            _make_subprocess_result(0, ""),           # compile
            _make_subprocess_result(0, "output123"),  # run
        ]
        result = run_locally(Path("/tmp/work"), "gcc -O2 foo.c", "./a.out", "test")

        assert result.ok is True
        assert result.host == "localhost"
        assert result.stdout == "output123"
        assert result.elapsed_seconds > 0
        assert mock_run.call_count == 2
        # Verify compile call uses shell=True and cwd
        compile_call = mock_run.call_args_list[0]
        assert compile_call.kwargs["shell"] is True
        assert compile_call.kwargs["cwd"] == Path("/tmp/work")

    @patch("src.benchmark.subprocess.run")
    def test_compile_failure_returns_not_ok(self, mock_run):
        mock_run.return_value = _make_subprocess_result(1, "", "compile error")
        result = run_locally(Path("/tmp/work"), "gcc bad.c", "./a.out", "test")

        assert result.ok is False
        assert result.elapsed_seconds == 0
        assert result.stderr == "compile error"
        # Should not attempt to run after compile failure
        assert mock_run.call_count == 1

    @patch("src.benchmark.subprocess.run")
    def test_run_failure_returns_not_ok(self, mock_run):
        mock_run.side_effect = [
            _make_subprocess_result(0),              # compile ok
            _make_subprocess_result(1, "", "segfault"),  # run fails
        ]
        result = run_locally(Path("/tmp/work"), "gcc foo.c", "./a.out", "test")

        assert result.ok is False
        assert result.stderr == "segfault"
        assert result.elapsed_seconds > 0  # timing still recorded


# ---------------------------------------------------------------------------
# run_on_host (remote)
# ---------------------------------------------------------------------------

class TestRunOnHost:
    """Tests for the run_on_host function (SSH-based)."""

    @patch("src.benchmark.subprocess.run")
    def test_compile_and_run_success(self, mock_run):
        mock_run.side_effect = [
            _make_subprocess_result(0),              # compile
            _make_subprocess_result(0, "remote_out"),  # run
        ]
        result = run_on_host("jump", "/tmp/remote", "gcc foo.c", "./a.out", "remote-test")

        assert result.ok is True
        assert result.host == "jump"
        assert result.stdout == "remote_out"
        # Verify SSH command structure
        compile_call = mock_run.call_args_list[0]
        assert compile_call.args[0][0] == "ssh"
        assert compile_call.args[0][1] == "jump"

    @patch("src.benchmark.subprocess.run")
    def test_compile_failure(self, mock_run):
        mock_run.return_value = _make_subprocess_result(1, "", "error")
        result = run_on_host("jump", "/tmp/remote", "gcc bad.c", "./a.out", "remote-test")

        assert result.ok is False
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# prepare_local_dir
# ---------------------------------------------------------------------------

class TestPrepareLocalDir:
    """Tests for prepare_local_dir — copies sources and datasets to a temp dir."""

    def test_copies_source_files_and_datasets(self, tmp_path):
        # Set up source dir with files
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "main.c").write_text("int main(){}")
        (src_dir / "ssw.c").write_text("void ssw(){}")
        # Sub-directory should be skipped (only files)
        (src_dir / "subdir").mkdir()
        (src_dir / "subdir" / "nested.c").write_text("nested")

        # Set up dataset dir
        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        (ds_dir / "test.fa").write_text("FASTA")
        (ds_dir / "54mer_hap1_1.100.fa").write_text("REF")

        result = prepare_local_dir(src_dir, ds_dir, "test.fa")

        try:
            assert result.is_dir()
            # Source files copied
            assert (result / "main.c").read_text() == "int main(){}"
            assert (result / "ssw.c").read_text() == "void ssw(){}"
            # Subdirectory NOT copied
            assert not (result / "subdir").exists()
            # demo/ subdirectory with datasets
            assert (result / "demo" / "test.fa").read_text() == "FASTA"
            assert (result / "demo" / "54mer_hap1_1.100.fa").read_text() == "REF"
        finally:
            import shutil
            shutil.rmtree(result, ignore_errors=True)

    def test_missing_dataset_does_not_crash(self, tmp_path):
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "main.c").write_text("")

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        # No dataset files present

        result = prepare_local_dir(src_dir, ds_dir, "missing.fa")

        try:
            assert result.is_dir()
            assert (result / "demo").is_dir()
            # Files are missing but no crash
            assert not (result / "demo" / "missing.fa").exists()
        finally:
            import shutil
            shutil.rmtree(result, ignore_errors=True)


# ---------------------------------------------------------------------------
# benchmark() — local Intel path (SSH_JUMP_HOST unset)
# ---------------------------------------------------------------------------

class TestBenchmarkLocalIntel:
    """Test that benchmark() runs Intel locally when SSH_JUMP_HOST is empty."""

    @patch("src.benchmark.shutil.rmtree")
    @patch("src.benchmark.prepare_local_dir")
    @patch("src.benchmark.run_locally")
    @patch("src.benchmark.run_on_host")
    @patch("src.benchmark.upload_datasets")
    @patch("src.benchmark.upload_to_host")
    @patch("src.benchmark.check_ssh")
    @patch("src.benchmark.SSH_JUMP_HOST", "")
    def test_local_path_used_when_jump_host_empty(
        self, mock_check_ssh, mock_upload, mock_upload_ds,
        mock_run_on_host, mock_run_locally, mock_prepare, mock_rmtree,
        tmp_path,
    ):
        output = "alignment output line 1"
        mock_check_ssh.return_value = True  # for RISC-V host
        mock_upload.return_value = True
        mock_upload_ds.return_value = True
        mock_prepare.return_value = tmp_path / "local_work"
        (tmp_path / "local_work").mkdir()
        mock_run_locally.return_value = _ok_result("localhost", "Intel (original SSE)", output)
        mock_run_on_host.return_value = _ok_result("final", "RISC-V (translated RVV)", output)

        # Create minimal dirs for iterdir()
        orig = tmp_path / "orig"
        orig.mkdir()
        (orig / "main.c").write_text("")
        trans = tmp_path / "trans"
        trans.mkdir()
        (trans / "main.c").write_text("")
        ds = tmp_path / "datasets"
        ds.mkdir()

        rc = benchmark(
            dataset="test.fa",
            original_dir=orig,
            translated_dir=trans,
            dataset_dir=ds,
        )

        assert rc == 0
        # run_locally was called for Intel, NOT run_on_host for Intel
        mock_run_locally.assert_called_once()
        assert mock_run_locally.call_args.args[3] == "Intel (original SSE)"
        # run_on_host called only for RISC-V
        mock_run_on_host.assert_called_once()
        assert mock_run_on_host.call_args.args[4] == "RISC-V (translated RVV)"
        # check_ssh NOT called for jump host (only for final)
        calls = [c.args[0] for c in mock_check_ssh.call_args_list]
        assert "" not in calls  # no check_ssh("")
        # prepare_local_dir was called
        mock_prepare.assert_called_once()
        # temp dir cleaned up
        mock_rmtree.assert_called_once()

    @patch("src.benchmark.shutil.rmtree")
    @patch("src.benchmark.prepare_local_dir")
    @patch("src.benchmark.run_locally")
    @patch("src.benchmark.run_on_host")
    @patch("src.benchmark.upload_datasets")
    @patch("src.benchmark.upload_to_host")
    @patch("src.benchmark.check_ssh")
    @patch("src.benchmark.SSH_JUMP_HOST", "")
    def test_local_intel_failure_returns_1(
        self, mock_check_ssh, mock_upload, mock_upload_ds,
        mock_run_on_host, mock_run_locally, mock_prepare, mock_rmtree,
        tmp_path,
    ):
        mock_check_ssh.return_value = True
        mock_upload.return_value = True
        mock_upload_ds.return_value = True
        mock_prepare.return_value = tmp_path / "local_work"
        (tmp_path / "local_work").mkdir()
        mock_run_locally.return_value = _fail_result("localhost", "Intel (original SSE)")
        mock_run_on_host.return_value = _ok_result("final", "RISC-V (translated RVV)", "out")

        orig = tmp_path / "orig"
        orig.mkdir()
        (orig / "main.c").write_text("")
        trans = tmp_path / "trans"
        trans.mkdir()
        (trans / "main.c").write_text("")

        rc = benchmark(dataset="t.fa", original_dir=orig, translated_dir=trans, dataset_dir=tmp_path)
        assert rc == 1


# ---------------------------------------------------------------------------
# benchmark() — remote Intel path (SSH_JUMP_HOST set)
# ---------------------------------------------------------------------------

class TestBenchmarkRemoteIntel:
    """Test that benchmark() uses SSH when SSH_JUMP_HOST is set."""

    @patch("src.benchmark.run_on_host")
    @patch("src.benchmark.upload_datasets")
    @patch("src.benchmark.upload_to_host")
    @patch("src.benchmark.check_ssh")
    @patch("src.benchmark.SSH_JUMP_HOST", "jump-host")
    def test_remote_path_used_when_jump_host_set(
        self, mock_check_ssh, mock_upload, mock_upload_ds, mock_run_on_host,
        tmp_path,
    ):
        output = "alignment output"
        mock_check_ssh.return_value = True
        mock_upload.return_value = True
        mock_upload_ds.return_value = True
        mock_run_on_host.side_effect = [
            _ok_result("jump-host", "Intel (original SSE)", output),
            _ok_result("final", "RISC-V (translated RVV)", output),
        ]

        orig = tmp_path / "orig"
        orig.mkdir()
        (orig / "main.c").write_text("")
        trans = tmp_path / "trans"
        trans.mkdir()
        (trans / "main.c").write_text("")

        rc = benchmark(dataset="t.fa", original_dir=orig, translated_dir=trans, dataset_dir=tmp_path)

        assert rc == 0
        # run_on_host called twice (Intel + RISC-V)
        assert mock_run_on_host.call_count == 2
        # check_ssh called for both hosts
        assert mock_check_ssh.call_count == 2

    @patch("src.benchmark.check_ssh")
    @patch("src.benchmark.SSH_JUMP_HOST", "jump-host")
    def test_unreachable_jump_host_returns_1(self, mock_check_ssh, tmp_path):
        mock_check_ssh.return_value = False

        orig = tmp_path / "orig"
        orig.mkdir()
        trans = tmp_path / "trans"
        trans.mkdir()

        rc = benchmark(dataset="t.fa", original_dir=orig, translated_dir=trans, dataset_dir=tmp_path)
        assert rc == 1


# ---------------------------------------------------------------------------
# normalize_output / compare_outputs
# ---------------------------------------------------------------------------

class TestNormalizeOutput:
    def test_strips_cpu_time(self):
        raw = "score: 42  CPU time: 0.074732 seconds\n"
        assert "CPU time" not in normalize_output(raw)
        assert "score: 42" in normalize_output(raw)

    def test_strips_empty_lines_and_trailing_whitespace(self):
        raw = "  line1  \n\n  line2  \n\n"
        assert normalize_output(raw) == "line1\n  line2"

    def test_cpu_time_interleaved_midline(self):
        """CPU time from stderr can land mid-line when 2>&1 is used."""
        raw = "optimal_alignmCPU time: 5.88 seconds\nent_score: 30\tstrand: +\n"
        result = normalize_output(raw)
        assert "optimal_alignment_score: 30" in result
        assert "CPU time" not in result


class TestCompareOutputs:
    _SSW_REC = (
        "target_name: chr1\n"
        "query_name: read1\n"
        "optimal_alignment_score: 42\tsuboptimal_alignment_score: 30\t"
        "strand: +\ttarget_end: 100\tquery_end: 50\n\n"
    )
    _SSW_REC_NO_SUB = (
        "target_name: chr1\n"
        "query_name: read1\n"
        "optimal_alignment_score: 42\tstrand: +\ttarget_end: 100\tquery_end: 50\n\n"
    )
    _SSW_REC_DIFF_SUB = (
        "target_name: chr1\n"
        "query_name: read1\n"
        "optimal_alignment_score: 42\tsuboptimal_alignment_score: 25\t"
        "strand: +\ttarget_end: 100\tquery_end: 50\n\n"
    )

    def test_matching_outputs(self):
        a = _ok_result(stdout=self._SSW_REC)
        b = _ok_result(stdout=self._SSW_REC)
        match, details = compare_outputs(a, b)
        assert match is True

    def test_suboptimal_ignored_by_default(self):
        a = _ok_result(stdout=self._SSW_REC)
        b = _ok_result(stdout=self._SSW_REC_NO_SUB)
        match, _ = compare_outputs(a, b)
        assert match is True

    def test_different_suboptimal_ignored_by_default(self):
        a = _ok_result(stdout=self._SSW_REC)
        b = _ok_result(stdout=self._SSW_REC_DIFF_SUB)
        match, _ = compare_outputs(a, b)
        assert match is True

    def test_strict_suboptimal_catches_difference(self):
        a = _ok_result(stdout=self._SSW_REC)
        b = _ok_result(stdout=self._SSW_REC_DIFF_SUB)
        match, details = compare_outputs(a, b, strict_suboptimal=True)
        assert match is False
        assert "suboptimal" in details

    def test_mismatched_score(self):
        other = self._SSW_REC.replace("score: 42", "score: 99")
        a = _ok_result(stdout=self._SSW_REC)
        b = _ok_result(stdout=other)
        match, details = compare_outputs(a, b)
        assert match is False

    def test_mismatched_record_count(self):
        a = _ok_result(stdout=self._SSW_REC * 2)
        b = _ok_result(stdout=self._SSW_REC)
        match, details = compare_outputs(a, b)
        assert match is False
        assert "Record count" in details
