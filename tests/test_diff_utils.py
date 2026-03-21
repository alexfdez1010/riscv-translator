"""Comprehensive tests for src/diff_utils.py — the unified diff parser.

Covers edge cases that LLMs commonly produce: bare @@ headers, wrong line
counts, trailing-whitespace differences, non-contiguous context, overlapping
hunks, missing a/ b/ prefixes, multi-hunk patches, etc.
"""

import pytest

from src.diff_utils import (
    apply_patch,
    apply_unified_diff,
    diff_error_feedback,
    diff_format_example,
    diff_format_instructions,
    extract_diff,
    normalize_patch,
    number_lines,
    parse_hunk_header,
    patch_targets,
    validate_single_file_diff,
)


# =====================================================================
# extract_diff
# =====================================================================


class TestExtractDiff:
    def test_fenced_diff_block(self):
        response = (
            "Summary sentence.\n\n"
            "```diff\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "```"
        )
        result = extract_diff(response)
        assert result is not None
        assert "--- a/ssw.c" in result
        assert "-old" in result
        assert "+new" in result

    def test_fenced_patch_block(self):
        response = (
            "```patch\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "```"
        )
        assert extract_diff(response) is not None

    def test_generic_fenced_block_with_diff_git(self):
        response = (
            "Here is the fix:\n"
            "```\n"
            "diff --git a/ssw.c b/ssw.c\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "```"
        )
        result = extract_diff(response)
        assert result is not None
        assert "diff --git" in result

    def test_bare_diff_entire_response(self):
        response = (
            "diff --git a/ssw.c b/ssw.c\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        assert extract_diff(response) is not None

    def test_bare_triple_dash_entire_response(self):
        response = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        assert extract_diff(response) is not None

    def test_no_diff_returns_none(self):
        assert extract_diff("Just a plain text response.") is None

    def test_empty_response_returns_none(self):
        assert extract_diff("") is None

    def test_trailing_text_after_fence(self):
        response = (
            "Summary.\n\n"
            "```diff\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "```\n\n"
            "Let me know if this helps!"
        )
        result = extract_diff(response)
        assert result is not None
        assert "Let me know" not in result

    def test_multiple_fenced_blocks_merged(self):
        """LLM returns two ```diff blocks for the same file — merged into one."""
        response = (
            "Here is hunk 1:\n"
            "```diff\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-first\n"
            "+FIRST\n"
            "```\n\n"
            "And hunk 2:\n"
            "```diff\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -10 +10 @@\n"
            "-tenth\n"
            "+TENTH\n"
            "```"
        )
        result = extract_diff(response)
        assert result is not None
        # Both hunks should be present
        assert "-first" in result
        assert "+FIRST" in result
        assert "-tenth" in result
        assert "+TENTH" in result
        # Only one set of --- / +++ headers
        assert result.count("--- a/ssw.c") == 1
        assert result.count("+++ b/ssw.c") == 1

    def test_single_fenced_block_unchanged(self):
        """Single fenced block works as before."""
        response = (
            "```diff\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "```"
        )
        result = extract_diff(response)
        assert result is not None
        assert "-old" in result


# =====================================================================
# parse_hunk_header
# =====================================================================


class TestParseHunkHeader:
    def test_standard_header(self):
        old_s, old_c, new_s, new_c = parse_hunk_header("@@ -10,5 +10,7 @@")
        assert (old_s, old_c, new_s, new_c) == (10, 5, 10, 7)

    def test_no_counts(self):
        old_s, old_c, new_s, new_c = parse_hunk_header("@@ -10 +10 @@")
        assert (old_s, old_c, new_s, new_c) == (10, 1, 10, 1)

    def test_header_with_function_name(self):
        old_s, old_c, new_s, new_c = parse_hunk_header(
            "@@ -10,5 +10,7 @@ void foo()"
        )
        assert (old_s, old_c, new_s, new_c) == (10, 5, 10, 7)

    def test_bare_at_at(self):
        old_s, old_c, new_s, new_c = parse_hunk_header("@@")
        assert (old_s, old_c, new_s, new_c) == (1, None, 1, None)

    def test_bare_at_at_with_spaces(self):
        old_s, old_c, new_s, new_c = parse_hunk_header("@@ @@")
        assert (old_s, old_c, new_s, new_c) == (1, None, 1, None)

    def test_missing_new_part(self):
        """LLM sometimes writes @@ -10,5 @@ without the + part."""
        old_s, old_c, new_s, new_c = parse_hunk_header("@@ -10,5 @@")
        assert old_s == 10
        assert old_c == 5
        assert new_s == 10  # inferred
        assert new_c is None

    def test_single_line_change(self):
        old_s, old_c, new_s, new_c = parse_hunk_header("@@ -1 +1 @@")
        assert (old_s, old_c, new_s, new_c) == (1, 1, 1, 1)

    def test_invalid_header_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            parse_hunk_header("not a header")

    def test_zero_counts(self):
        """Additions-only hunk: @@ -5,0 +5,3 @@"""
        old_s, old_c, new_s, new_c = parse_hunk_header("@@ -5,0 +5,3 @@")
        assert (old_s, old_c, new_s, new_c) == (5, 0, 5, 3)


# =====================================================================
# validate_single_file_diff
# =====================================================================


class TestValidateSingleFileDiff:
    def test_valid_patch(self):
        patch = (
            "diff --git a/ssw.c b/ssw.c\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        validate_single_file_diff(patch, "ssw.c")  # should not raise

    def test_valid_without_diff_git(self):
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        validate_single_file_diff(patch, "ssw.c")

    def test_empty_patch_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_single_file_diff("", "ssw.c")

    def test_no_hunk_raises(self):
        patch = "--- a/ssw.c\n+++ b/ssw.c\n"
        with pytest.raises(ValueError, match="hunk"):
            validate_single_file_diff(patch, "ssw.c")

    def test_wrong_file_in_diff_git_raises(self):
        patch = (
            "diff --git a/other.c b/other.c\n"
            "--- a/other.c\n"
            "+++ b/other.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        with pytest.raises(ValueError, match="target only ssw.c"):
            validate_single_file_diff(patch, "ssw.c")

    def test_missing_old_header_raises(self):
        patch = (
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        with pytest.raises(ValueError, match="--- a/ssw.c"):
            validate_single_file_diff(patch, "ssw.c")

    def test_missing_new_header_raises(self):
        patch = (
            "--- a/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        with pytest.raises(ValueError, match="\\+\\+\\+ b/ssw.c"):
            validate_single_file_diff(patch, "ssw.c")


# =====================================================================
# normalize_patch
# =====================================================================


class TestNormalizePatch:
    def test_strips_and_adds_newline(self):
        assert normalize_patch("  hello  ") == "hello\n"

    def test_empty_returns_empty(self):
        assert normalize_patch("   ") == ""

    def test_already_normalized(self):
        assert normalize_patch("hello\n") == "hello\n"


# =====================================================================
# patch_targets
# =====================================================================


class TestPatchTargets:
    def test_diff_git_line(self):
        patch = "diff --git a/ssw.c b/ssw.c\n"
        assert patch_targets(patch) == {"ssw.c"}

    def test_triple_dash_lines(self):
        patch = "--- a/ssw.c\n+++ b/ssw.c\n"
        assert patch_targets(patch) == {"ssw.c"}

    def test_dev_null_excluded(self):
        patch = "--- /dev/null\n+++ b/new_file.c\n"
        assert patch_targets(patch) == {"new_file.c"}

    def test_multiple_files(self):
        patch = (
            "diff --git a/ssw.c b/ssw.c\n"
            "--- a/ssw.c\n+++ b/ssw.c\n"
            "diff --git a/ssw.h b/ssw.h\n"
            "--- a/ssw.h\n+++ b/ssw.h\n"
        )
        assert patch_targets(patch) == {"ssw.c", "ssw.h"}


# =====================================================================
# apply_unified_diff — basic cases
# =====================================================================


class TestApplyUnifiedDiffBasic:
    def test_simple_replacement(self):
        original = "line1\nline2\nline3\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+LINE2\n"
            " line3\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "line1\nLINE2\nline3\n"

    def test_simple_addition(self):
        original = "line1\nline3\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,2 +1,3 @@\n"
            " line1\n"
            "+line2\n"
            " line3\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "line1\nline2\nline3\n"

    def test_simple_deletion(self):
        original = "line1\nline2\nline3\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,3 +1,2 @@\n"
            " line1\n"
            "-line2\n"
            " line3\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "line1\nline3\n"

    def test_multi_hunk_patch(self):
        original = "a\nb\nc\nd\ne\nf\ng\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,3 +1,3 @@\n"
            " a\n"
            "-b\n"
            "+B\n"
            " c\n"
            "@@ -5,3 +5,3 @@\n"
            " e\n"
            "-f\n"
            "+F\n"
            " g\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "a\nB\nc\nd\ne\nF\ng\n"

    def test_no_hunk_raises(self):
        original = "line1\n"
        patch = "--- a/f.c\n+++ b/f.c\n"
        with pytest.raises(ValueError, match="hunk"):
            apply_unified_diff(original, patch)


# =====================================================================
# apply_unified_diff — LLM tolerance edge cases
# =====================================================================


class TestApplyUnifiedDiffTolerance:
    def test_trailing_whitespace_in_context(self):
        """LLM adds trailing spaces to context lines."""
        original = "line1\nline2\nline3\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,3 +1,3 @@\n"
            " line1  \n"  # trailing whitespace
            "-line2\n"
            "+LINE2\n"
            " line3\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "line1\nLINE2\nline3\n"

    def test_trailing_whitespace_in_removal(self):
        """LLM adds trailing spaces to removal lines."""
        original = "line1\nline2\nline3\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2   \n"  # trailing whitespace
            "+LINE2\n"
            " line3\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "line1\nLINE2\nline3\n"

    def test_bare_at_at_header(self):
        """LLM emits @@ with no line numbers."""
        original = "old_line\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@\n"
            "-old_line\n"
            "+new_line\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "new_line\n"

    def test_wrong_old_count_accepted(self):
        """LLM claims wrong number of old lines — still works."""
        original = "line1\nline2\nline3\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,99 +1,99 @@\n"  # counts are wrong
            " line1\n"
            "-line2\n"
            "+LINE2\n"
            " line3\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "line1\nLINE2\nline3\n"

    def test_fuzzy_hunk_offset(self):
        """Hunk line number is off by several lines — fuzzy match finds it."""
        original = "a\nb\nc\ntarget_line\ne\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,1 +1,1 @@\n"  # claims line 1, but really at line 4
            "-target_line\n"
            "+REPLACED\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "a\nb\nc\nREPLACED\ne\n"

    def test_non_contiguous_context_skip(self):
        """LLM omits some context lines between changes.

        The hunk header correctly points to line 1 and the context/removal
        lines match the source, but context jumps from ``a`` to ``e``
        (skipping c and d).  The parser should emit the skipped lines verbatim.
        """
        original = "a\nb\nc\nd\ne\nf\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,4 +1,4 @@\n"
            " a\n"
            "-b\n"
            "+B\n"
            " e\n"  # skips c and d — non-contiguous context
            "-f\n"
            "+F\n"
        )
        result = apply_unified_diff(original, patch)
        assert "B\n" in result
        assert "F\n" in result
        # c and d should still be present (emitted verbatim)
        assert "c\n" in result
        assert "d\n" in result

    def test_overlapping_hunks(self):
        """Two hunks that overlap in their context (LLM mistake).

        The second hunk's context references the same source line ``c``
        (which the first hunk already consumed).  The parser rewinds and
        replays to handle this.
        """
        original = "a\nb\nc\nd\ne\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -2,2 +2,2 @@\n"
            " b\n"
            "-c\n"
            "+C\n"
            "@@ -4,2 +4,2 @@\n"
            " d\n"
            "-e\n"
            "+E\n"
        )
        result = apply_unified_diff(original, patch)
        assert "C\n" in result
        assert "E\n" in result

    def test_addition_only_hunk(self):
        """Pure addition hunk with zero old lines."""
        original = "line1\nline2\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,0 +1,2 @@\n"
            "+added1\n"
            "+added2\n"
        )
        result = apply_unified_diff(original, patch)
        assert result.startswith("added1\nadded2\n")

    def test_no_newline_marker_ignored(self):
        r"""Lines starting with \ (no newline at end of file) are skipped."""
        original = "line1\nline2\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,2 +1,2 @@\n"
            " line1\n"
            "-line2\n"
            "+LINE2\n"
            "\\ No newline at end of file\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "line1\nLINE2\n"

    def test_diff_git_preamble_skipped(self):
        """diff --git header before hunks is fine."""
        original = "old\n"
        patch = (
            "diff --git a/ssw.c b/ssw.c\n"
            "index abc..def 100644\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "new\n"

    def test_tabs_vs_spaces_in_context(self):
        """LLM uses tabs where source has spaces (or vice versa)."""
        original = "\t\tint x = 1;\n\t\tint y = 2;\n\t\tint z = 3;\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,3 +1,3 @@\n"
            "         int x = 1;\n"  # 8 spaces instead of 2 tabs
            "-        int y = 2;\n"  # 8 spaces instead of 2 tabs
            "+        int y = 99;\n"
            "         int z = 3;\n"  # 8 spaces instead of 2 tabs
        )
        result = apply_unified_diff(original, patch)
        assert "\t\tint x = 1;\n" in result  # original preserved
        assert "int y = 99;\n" in result
        assert "\t\tint z = 3;\n" in result  # original preserved

    def test_mixed_tabs_spaces_in_context(self):
        """Source has mixed tabs/spaces, LLM normalizes to tabs."""
        original = "\t    \t\tvMaxColumn = foo();\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1 +1 @@\n"
            "-\t\t\t\tvMaxColumn = foo();\n"  # LLM uses 4 tabs
            "+\t\t\t\tvMaxColumn = bar();\n"
        )
        result = apply_unified_diff(original, patch)
        assert "bar()" in result

    def test_extra_interior_spaces_in_context(self):
        """LLM adds extra spaces in context: 'Swap the  2' vs 'Swap the 2'."""
        original = "\t\t/* Swap the 2 H buffers. */\n\t\tint x = 1;\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,2 +1,2 @@\n"
            " \t\t/* Swap the  2 H buffers. */\n"  # extra space
            "-\t\tint x = 1;\n"
            "+\t\tint x = 2;\n"
        )
        result = apply_unified_diff(original, patch)
        assert "int x = 2;\n" in result
        # Original comment preserved verbatim
        assert "/* Swap the 2 H buffers. */" in result

    def test_preserves_untouched_lines(self):
        """Lines not covered by any hunk are preserved verbatim."""
        original = "header\na\nb\nc\nfooter\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -2,3 +2,3 @@\n"
            " a\n"
            "-b\n"
            "+B\n"
            " c\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "header\na\nB\nc\nfooter\n"


# =====================================================================
# apply_unified_diff — error cases
# =====================================================================


class TestApplyUnifiedDiffErrors:
    def test_context_mismatch_raises(self):
        """Context line doesn't match source at all."""
        original = "real_line\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,1 +1,1 @@\n"
            " completely_wrong_context\n"
        )
        with pytest.raises(ValueError, match="context.*does not match"):
            apply_unified_diff(original, patch)

    def test_removal_mismatch_raises(self):
        """Removal line doesn't match source."""
        original = "actual_line\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1,1 +1,1 @@\n"
            "-wrong_removal\n"
            "+replacement\n"
        )
        with pytest.raises(ValueError, match="does not match"):
            apply_unified_diff(original, patch)

    def test_multi_file_patch_raises(self):
        """Patch contains markers for a second file."""
        original = "line1\n"
        patch = (
            "--- a/f.c\n"
            "+++ b/f.c\n"
            "@@ -1 +1 @@\n"
            "-line1\n"
            "+new1\n"
            "--- a/g.c\n"
            "+++ b/g.c\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        )
        with pytest.raises(ValueError, match="only one file"):
            apply_unified_diff(original, patch)


# =====================================================================
# apply_patch (high-level)
# =====================================================================


class TestApplyPatch:
    def test_full_pipeline(self):
        original = "old\n"
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        result = apply_patch(original, patch, "ssw.c")
        assert result == "new\n"

    def test_wrong_file_raises(self):
        original = "old\n"
        patch = (
            "--- a/other.c\n"
            "+++ b/other.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        with pytest.raises(ValueError):
            apply_patch(original, patch, "ssw.c")


# =====================================================================
# Realistic LLM-output scenarios
# =====================================================================


class TestRealisticLLMScenarios:
    """Tests based on actual patterns seen from LLM-generated diffs."""

    def test_multiple_diff_blocks_apply_correctly(self):
        """LLM returns two separate ```diff blocks — merged and applied."""
        original = "a\nb\nc\nd\ne\nf\ng\n"
        response = (
            "First fix:\n"
            "```diff\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -2,1 +2,1 @@\n"
            "-b\n"
            "+B\n"
            "```\n\n"
            "Second fix:\n"
            "```diff\n"
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -6,1 +6,1 @@\n"
            "-f\n"
            "+F\n"
            "```"
        )
        patch = extract_diff(response)
        assert patch is not None
        result = apply_unified_diff(original, patch)
        assert "B\n" in result
        assert "F\n" in result
        assert "b\n" not in result
        assert "f\n" not in result

    def test_llm_sizeof_replacement(self):
        """Replace sizeof(__m128i) with _SSW_VEC_BYTES — a common repair."""
        original = (
            "#include <stdint.h>\n"
            "\n"
            "void foo(void) {\n"
            "    int n = sizeof(__m128i);\n"
            "    bar(n);\n"
            "}\n"
        )
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -3,4 +3,4 @@\n"
            " void foo(void) {\n"
            "-    int n = sizeof(__m128i);\n"
            "+    int n = _SSW_VEC_BYTES;\n"
            "     bar(n);\n"
            " }\n"
        )
        result = apply_unified_diff(original, patch)
        assert "_SSW_VEC_BYTES" in result
        assert "sizeof(__m128i)" not in result

    def test_llm_adds_include_at_top(self):
        """LLM adds an #include at the very top of the file."""
        original = "void main(void) {}\n"
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1,2 @@\n"
            "+#include <stdint.h>\n"
            " void main(void) {}\n"
        )
        result = apply_unified_diff(original, patch)
        assert result.startswith("#include <stdint.h>\n")

    def test_llm_multiline_replacement(self):
        """LLM replaces multiple lines at once."""
        original = (
            "int a = 1;\n"
            "int b = 2;\n"
            "int c = 3;\n"
            "int d = 4;\n"
            "int e = 5;\n"
        )
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -2,3 +2,2 @@\n"
            " int b = 2;\n"
            "-int c = 3;\n"
            "-int d = 4;\n"
            "+int cd = 34;\n"
        )
        result = apply_unified_diff(original, patch)
        assert "int cd = 34;\n" in result
        assert "int c = 3;\n" not in result
        assert "int d = 4;\n" not in result
        # Surrounding lines preserved
        assert "int a = 1;\n" in result
        assert "int e = 5;\n" in result

    def test_llm_context_line_with_tabs_vs_spaces(self):
        """Source uses tabs but LLM sends spaces — trailing-whitespace tolerance."""
        original = "void f() {\n\treturn;\t\n}\n"
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1,3 +1,3 @@\n"
            " void f() {\n"
            "-\treturn;\t\n"
            "+\treturn 0;\n"
            " }\n"
        )
        result = apply_unified_diff(original, patch)
        assert "\treturn 0;\n" in result

    def test_large_fuzzy_offset(self):
        """Hunk claims line 1 but the target is deep in the file."""
        lines = [f"line{i}\n" for i in range(100)]
        lines[50] = "target\n"
        original = "".join(lines)
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -1 +1 @@\n"
            "-target\n"
            "+REPLACED\n"
        )
        result = apply_unified_diff(original, patch)
        assert "REPLACED\n" in result
        assert "target\n" not in result

    def test_hunk_at_end_of_file(self):
        """Hunk modifies the very last line."""
        original = "a\nb\nlast_line\n"
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -3 +3 @@\n"
            "-last_line\n"
            "+NEW_LAST\n"
        )
        result = apply_unified_diff(original, patch)
        assert result.endswith("NEW_LAST\n")

    def test_empty_original_with_addition(self):
        """Applying additions to an empty file."""
        original = ""
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -0,0 +1,2 @@\n"
            "+line1\n"
            "+line2\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "line1\nline2\n"

    def test_three_hunks_various_positions(self):
        """Three separate hunks at different positions."""
        original = "\n".join(f"L{i}" for i in range(20)) + "\n"
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@ -2,1 +2,1 @@\n"
            "-L1\n"
            "+FIRST\n"
            "@@ -10,1 +10,1 @@\n"
            "-L9\n"
            "+SECOND\n"
            "@@ -18,1 +18,1 @@\n"
            "-L17\n"
            "+THIRD\n"
        )
        result = apply_unified_diff(original, patch)
        assert "FIRST\n" in result
        assert "SECOND\n" in result
        assert "THIRD\n" in result
        assert "L1\n" not in result
        assert "L9\n" not in result
        assert "L17\n" not in result


# =====================================================================
# Prompt helpers
# =====================================================================


class TestPromptHelpers:
    def test_diff_format_instructions_contains_key_elements(self):
        result = diff_format_instructions("ssw.c")
        assert "--- a/ssw.c" in result
        assert "+++ b/ssw.c" in result
        assert "@@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@" in result
        assert "context" in result.lower()

    def test_diff_format_example_is_valid(self):
        result = diff_format_example("ssw.c")
        assert "```diff" in result
        assert "--- a/ssw.c" in result
        assert "@@ -10,7 +10,7 @@" in result

    def test_diff_error_feedback_includes_error_and_code(self):
        result = diff_error_feedback("ssw.c", "int x;", "hunk mismatch")
        assert "hunk mismatch" in result
        assert "int x;" in result
        assert "--- a/ssw.c" in result

    def test_diff_error_feedback_with_line_number_shows_snippet(self):
        code = "\n".join(f"line {i}" for i in range(1, 21))
        result = diff_error_feedback(
            "ssw.c", code,
            "Patch context line does not match source at line 10."
        )
        assert "line 10" in result
        # Should show a snippet around line 10
        assert ">>>" in result

    def test_number_lines_basic(self):
        code = "a\nb\nc\n"
        result = number_lines(code, every=1)
        assert "1| a\n" in result
        assert "2| b\n" in result
        assert "3| c\n" in result

    def test_number_lines_every_10(self):
        code = "\n".join(f"line{i}" for i in range(1, 22))
        result = number_lines(code, every=10)
        # Line 1 and multiples of 10 should have numbers
        assert " 1| line1" in result
        assert "10| line10" in result
        assert "20| line20" in result
        # Line 5 should have blank margin
        assert "  | line5" in result


# =====================================================================
# Regression: the exact failure from the user's log
# =====================================================================


class TestRegressionBareHunkHeader:
    """The user's log showed:
    'Bare hunk header detected (@@); inferring position from context'
    followed by 'Patch context does not match ... (hunk at line 1, searched ±200 lines)'
    This tests that scenario works when the context lines actually match.
    """

    def test_bare_header_with_matching_context(self):
        original = (
            "#include <stdio.h>\n"
            "int main() {\n"
            "    printf(\"hello\");\n"
            "    return 0;\n"
            "}\n"
        )
        # LLM emits bare @@ with no line numbers
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@\n"
            " #include <stdio.h>\n"
            " int main() {\n"
            "-    printf(\"hello\");\n"
            "+    printf(\"world\");\n"
            "     return 0;\n"
            " }\n"
        )
        result = apply_unified_diff(original, patch)
        assert 'printf("world")' in result
        assert 'printf("hello")' not in result

    def test_bare_header_no_context_just_replacement(self):
        """Bare @@ with only - and + lines (no context)."""
        original = "old_line\nsecond\n"
        patch = (
            "--- a/ssw.c\n"
            "+++ b/ssw.c\n"
            "@@\n"
            "-old_line\n"
            "+new_line\n"
        )
        result = apply_unified_diff(original, patch)
        assert result == "new_line\nsecond\n"
