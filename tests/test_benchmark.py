"""Tests for src.benchmark — output parsing, validation, statistics, CSV."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark import (
    BenchmarkResult,
    _normalize,
    _parse_records,
    _percentile,
    _stats_row,
    compare_outputs,
    merge_csv,
    normalize_output,
    prepare_local_dir,
    read_csv,
    run_locally,
    run_on_host,
    validate_output,
    write_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_result(host: str = "h", label: str = "L", stdout: str = "out") -> BenchmarkResult:
    return BenchmarkResult(host=host, label=label, ok=True, elapsed_seconds=1.0, stdout=stdout, stderr="")


def _make_subprocess_result(returncode: int = 0, stdout: str = "", stderr: str = ""):
    r = MagicMock()
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


_SSW_REC = (
    "target_name: chr1\n"
    "query_name: read1\n"
    "optimal_alignment_score: 42\tsuboptimal_alignment_score: 30\t"
    "strand: +\ttarget_end: 100\tquery_end: 50\n\n"
)


# ---------------------------------------------------------------------------
# normalize / parse
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_strips_cpu_time(self):
        raw = "score: 42  CPU time: 0.074732 seconds\n"
        assert "CPU time" not in _normalize(raw)
        assert "score: 42" in _normalize(raw)

    def test_normalize_output_alias(self):
        raw = "hello\n"
        assert normalize_output(raw) == _normalize(raw)

    def test_cpu_time_interleaved_midline(self):
        raw = "optimal_alignmCPU time: 5.88 seconds\nent_score: 30\tstrand: +\n"
        result = _normalize(raw)
        assert "optimal_alignment_score: 30" in result


class TestParseRecords:
    def test_single_record(self):
        recs = _parse_records(_SSW_REC)
        assert len(recs) == 1
        assert recs[0]["target_name"] == "chr1"
        assert recs[0]["optimal_alignment_score"] == "42"

    def test_empty(self):
        assert _parse_records("") == []


# ---------------------------------------------------------------------------
# validate_output
# ---------------------------------------------------------------------------

class TestValidateOutput:
    def test_matching(self):
        ok, _ = validate_output(_SSW_REC, _SSW_REC)
        assert ok is True

    def test_score_mismatch(self):
        other = _SSW_REC.replace("score: 42", "score: 99")
        ok, details = validate_output(_SSW_REC, other)
        assert ok is False

    def test_record_count_mismatch(self):
        ok, details = validate_output(_SSW_REC * 2, _SSW_REC)
        assert ok is False
        assert "count" in details

    def test_compare_outputs_wrapper(self):
        a = _ok_result(stdout=_SSW_REC)
        b = _ok_result(stdout=_SSW_REC)
        ok, _ = compare_outputs(a, b)
        assert ok is True


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_median_odd(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_q1_q3(self):
        data = sorted([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        q1 = _percentile(data, 25)
        q3 = _percentile(data, 75)
        assert 2.0 < q1 < 4.0
        assert 7.0 < q3 < 9.0

    def test_single_value(self):
        assert _percentile([5.0], 50) == 5.0

    def test_empty(self):
        assert _percentile([], 50) == 0.0


class TestStatsRow:
    def test_basic(self):
        st = _stats_row([1.0, 2.0, 3.0, 4.0, 5.0], 5)
        assert st["n_runs"] == 5
        assert st["mean"] == 3.0
        assert st["median"] == 3.0
        assert st["min"] == 1.0
        assert st["max"] == 5.0
        assert st["stdev"] > 0
        assert st["q1"] < st["median"] < st["q3"]


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

class TestWriteCSV:
    def test_writes_csv(self, tmp_path):
        results = {
            ("variant-a", "1k.fa"): {
                "stats": _stats_row([1.0, 2.0, 3.0], 3),
                "correct": True,
            },
        }
        csv_path = tmp_path / "out.csv"
        write_csv(results, csv_path, 3)

        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "variant-a" in lines[1]
        assert "1k.fa" in lines[1]
        assert "True" in lines[1]


# ---------------------------------------------------------------------------
# prepare_local_dir
# ---------------------------------------------------------------------------

class TestPrepareLocalDir:
    def test_copies_source_files_and_datasets(self, tmp_path):
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "main.c").write_text("int main(){}")
        (src_dir / "subdir").mkdir()

        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        (ds_dir / "test.fa").write_text("FASTA")
        (ds_dir / "54mer_hap1_1.100.fa").write_text("REF")

        result = prepare_local_dir(src_dir, ds_dir, "test.fa")
        try:
            assert (result / "main.c").read_text() == "int main(){}"
            assert not (result / "subdir").exists()
            assert (result / "demo" / "test.fa").read_text() == "FASTA"
        finally:
            import shutil
            shutil.rmtree(result, ignore_errors=True)


# ---------------------------------------------------------------------------
# run_locally / run_on_host
# ---------------------------------------------------------------------------

class TestRunLocally:
    @patch("src.benchmark.subprocess.run")
    def test_success(self, mock_run):
        mock_run.side_effect = [
            _make_subprocess_result(0, ""),
            _make_subprocess_result(0, "output123"),
        ]
        result = run_locally(Path("/tmp/work"), "gcc foo.c", "./a.out", "test")
        assert result.ok is True
        assert result.stdout == "output123"

    @patch("src.benchmark.subprocess.run")
    def test_compile_failure(self, mock_run):
        mock_run.return_value = _make_subprocess_result(1, "", "error")
        result = run_locally(Path("/tmp/work"), "gcc bad.c", "./a.out", "test")
        assert result.ok is False


class TestRunOnHost:
    @patch("src.benchmark.run_once")
    @patch("src.benchmark.compile_on_host")
    def test_success(self, mock_compile, mock_run):
        mock_compile.return_value = True
        mock_run.return_value = (1.5, "output", True)
        result = run_on_host("host", "/dir", "gcc foo.c", "./a.out", "test")
        assert result.ok is True
        assert result.elapsed_seconds == 1.5

    @patch("src.benchmark.compile_on_host")
    def test_compile_failure(self, mock_compile):
        mock_compile.return_value = False
        result = run_on_host("host", "/dir", "gcc bad.c", "./a.out", "test")
        assert result.ok is False


# ---------------------------------------------------------------------------
# read_csv — incremental skip
# ---------------------------------------------------------------------------

class TestReadCSV:
    def test_reads_existing_keys(self, tmp_path):
        csv_path = tmp_path / "bench.csv"
        results = {
            ("variant-a", "1k.fa"): {
                "stats": _stats_row([1.0, 2.0], 2),
                "correct": True,
            },
            ("variant-b", "10k.fa"): {
                "stats": _stats_row([3.0], 1),
                "correct": False,
            },
        }
        write_csv(results, csv_path, 2)
        stored = read_csv(csv_path)
        assert stored == {("variant-a", "1k.fa"), ("variant-b", "10k.fa")}

    def test_empty_file(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        assert read_csv(csv_path) == set()

    def test_nonexistent_file(self, tmp_path):
        csv_path = tmp_path / "nope.csv"
        assert read_csv(csv_path) == set()


# ---------------------------------------------------------------------------
# merge_csv — merging new results with existing CSV
# ---------------------------------------------------------------------------

class TestMergeCSV:
    def test_merge_adds_new_rows(self, tmp_path):
        csv_path = tmp_path / "merge.csv"
        # Write initial data
        initial = {
            ("variant-a", "1k.fa"): {
                "stats": _stats_row([1.0, 2.0, 3.0], 3),
                "correct": True,
            },
        }
        write_csv(initial, csv_path, 3)

        # Merge new data
        new = {
            ("variant-b", "10k.fa"): {
                "stats": _stats_row([4.0, 5.0, 6.0], 3),
                "correct": True,
            },
        }
        merge_csv(new, csv_path, 3)

        stored = read_csv(csv_path)
        assert ("variant-a", "1k.fa") in stored
        assert ("variant-b", "10k.fa") in stored

    def test_merge_overwrites_existing_key(self, tmp_path):
        csv_path = tmp_path / "overwrite.csv"
        initial = {
            ("variant-a", "1k.fa"): {
                "stats": _stats_row([1.0], 1),
                "correct": True,
            },
        }
        write_csv(initial, csv_path, 3)

        # Merge with same key — should overwrite
        new = {
            ("variant-a", "1k.fa"): {
                "stats": _stats_row([9.0, 8.0, 7.0], 3),
                "correct": True,
            },
        }
        merge_csv(new, csv_path, 3)

        # Verify only one entry for that key
        import csv as csv_mod
        with open(csv_path) as f:
            rows = list(csv_mod.DictReader(f))
        matching = [r for r in rows if r["code_variant"] == "variant-a" and r["dataset"] == "1k.fa"]
        assert len(matching) == 1
        assert matching[0]["n_runs"] == "3"

    def test_merge_creates_file_if_missing(self, tmp_path):
        csv_path = tmp_path / "new.csv"
        new = {
            ("variant-x", "100k.fa"): {
                "stats": _stats_row([2.0, 3.0], 2),
                "correct": True,
            },
        }
        merge_csv(new, csv_path, 2)

        stored = read_csv(csv_path)
        assert ("variant-x", "100k.fa") in stored
