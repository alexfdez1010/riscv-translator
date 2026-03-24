"""Tests for src/diff_utils.py — search/replace block parsing and application."""

import pytest

from src.diff_utils import (
    SearchReplaceBlock,
    apply_search_replace,
    extract_search_replace,
    number_lines,
    search_replace_error_feedback,
    search_replace_format_example,
)


# =====================================================================
# extract_search_replace
# =====================================================================


class TestExtractSearchReplace:
    def test_single_block(self):
        response = (
            "Summary line.\n\n"
            "<<<<<<< SEARCH\nold code\n=======\nnew code\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0].search == "old code"
        assert blocks[0].replace == "new code"

    def test_multiple_blocks(self):
        response = (
            "<<<<<<< SEARCH\naaa\n=======\nbbb\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nccc\n=======\nddd\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 2
        assert blocks[0].search == "aaa"
        assert blocks[1].search == "ccc"

    def test_multiline_search(self):
        response = (
            "<<<<<<< SEARCH\nline1\nline2\nline3\n=======\n"
            "new1\nnew2\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert blocks[0].search == "line1\nline2\nline3"
        assert blocks[0].replace == "new1\nnew2"

    def test_no_blocks_returns_none(self):
        assert extract_search_replace("no blocks here") is None

    def test_empty_response_returns_none(self):
        assert extract_search_replace("") is None

    def test_block_with_surrounding_text(self):
        response = (
            "Here is the fix for the __m128i issue:\n\n"
            "<<<<<<< SEARCH\n__m128i x;\n=======\nauto x = hn::Zero(du8);\n"
            ">>>>>>> REPLACE\n\nThis replaces the SSE type."
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 1
        assert "__m128i" in blocks[0].search

    def test_empty_replace(self):
        """Deleting code: replace with nothing."""
        response = "<<<<<<< SEARCH\ndelete me\n=======\n\n>>>>>>> REPLACE"
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert blocks[0].replace == ""

    def test_trailing_whitespace_on_markers(self):
        """Markers with trailing spaces/tabs should still be parsed."""
        response = (
            "<<<<<<< SEARCH   \n"
            "old code\n"
            "=======  \n"
            "new code\n"
            ">>>>>>> REPLACE \n"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert blocks[0].search == "old code"
        assert blocks[0].replace == "new code"

    def test_blocks_wrapped_in_markdown_code_fence(self):
        """LLMs sometimes wrap search/replace in ```."""
        response = (
            "Here is the fix:\n\n"
            "```\n"
            "<<<<<<< SEARCH\n"
            "old line\n"
            "=======\n"
            "new line\n"
            ">>>>>>> REPLACE\n"
            "```\n"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0].search == "old line"
        assert blocks[0].replace == "new line"

    def test_blocks_wrapped_in_language_code_fence(self):
        """LLMs sometimes use ```c or ```cpp fences."""
        response = (
            "```c\n"
            "<<<<<<< SEARCH\n"
            "#include <stdio.h>\n"
            "=======\n"
            "#include <stdlib.h>\n"
            ">>>>>>> REPLACE\n"
            "```\n"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert blocks[0].search == "#include <stdio.h>"

    def test_multiple_blocks_in_separate_fences(self):
        """Multiple blocks each wrapped in their own code fence."""
        response = (
            "Fix 1:\n"
            "```\n"
            "<<<<<<< SEARCH\naaa\n=======\nbbb\n>>>>>>> REPLACE\n"
            "```\n"
            "Fix 2:\n"
            "```\n"
            "<<<<<<< SEARCH\nccc\n=======\nddd\n>>>>>>> REPLACE\n"
            "```\n"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 2

    def test_block_with_indented_code(self):
        """Code inside blocks may have various indentation levels."""
        response = (
            "<<<<<<< SEARCH\n"
            "    if (x > 0) {\n"
            "        return x;\n"
            "    }\n"
            "=======\n"
            "    if (x > 0) {\n"
            "        return x + 1;\n"
            "    }\n"
            ">>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "return x;" in blocks[0].search
        assert "return x + 1;" in blocks[0].replace

    def test_noop_blocks_filtered_out(self):
        """Blocks where search == replace should be silently dropped."""
        response = (
            "<<<<<<< SEARCH\nold code\n=======\nnew code\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nsame\n=======\nsame\n>>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0].search == "old code"

    def test_all_noop_blocks_returns_none(self):
        """If every block is a no-op, return None."""
        response = (
            "<<<<<<< SEARCH\nsame\n=======\nsame\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nalso same\n=======\nalso same\n>>>>>>> REPLACE"
        )
        assert extract_search_replace(response) is None

    def test_block_with_blank_lines_in_code(self):
        """Code may contain blank lines."""
        response = (
            "<<<<<<< SEARCH\n"
            "int a;\n"
            "\n"
            "int b;\n"
            "=======\n"
            "int a;\n"
            "int b;\n"
            ">>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "\n\n" in blocks[0].search
        assert "\n\n" not in blocks[0].replace


# =====================================================================
# apply_search_replace
# =====================================================================


class TestApplySearchReplace:
    def test_simple_replacement(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [SearchReplaceBlock(search="bbb", replace="xxx")]
        assert apply_search_replace(original, blocks) == "aaa\nxxx\nccc\n"

    def test_multiline_replacement(self):
        original = "aaa\nbbb\nccc\nddd\n"
        blocks = [SearchReplaceBlock(search="bbb\nccc", replace="xxx\nyyy\nzzz")]
        assert apply_search_replace(original, blocks) == "aaa\nxxx\nyyy\nzzz\nddd\n"

    def test_multiple_blocks_applied_sequentially(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [
            SearchReplaceBlock(search="aaa", replace="AAA"),
            SearchReplaceBlock(search="ccc", replace="CCC"),
        ]
        assert apply_search_replace(original, blocks) == "AAA\nbbb\nCCC\n"

    def test_search_not_found_raises(self):
        original = "aaa\nbbb\n"
        blocks = [SearchReplaceBlock(search="zzz", replace="xxx")]
        with pytest.raises(ValueError, match="search text not found"):
            apply_search_replace(original, blocks)

    def test_ambiguous_search_raises(self):
        original = "aaa\naaa\n"
        blocks = [SearchReplaceBlock(search="aaa", replace="bbb")]
        with pytest.raises(ValueError, match="appears 2 times"):
            apply_search_replace(original, blocks)

    def test_deletion(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [SearchReplaceBlock(search="bbb\n", replace="")]
        assert apply_search_replace(original, blocks) == "aaa\nccc\n"

    def test_insertion_via_context(self):
        original = "aaa\nccc\n"
        blocks = [SearchReplaceBlock(search="aaa\nccc", replace="aaa\nbbb\nccc")]
        assert apply_search_replace(original, blocks) == "aaa\nbbb\nccc\n"

    def test_whitespace_tolerant_matching(self):
        """Trailing whitespace differences should be tolerated."""
        original = "aaa  \nbbb\n"
        blocks = [SearchReplaceBlock(search="aaa", replace="xxx")]
        # Exact match fails (original has trailing spaces), fuzzy should kick in
        result = apply_search_replace(original, blocks)
        assert "xxx" in result

    def test_second_block_sees_result_of_first(self):
        original = "old1\nold2\n"
        blocks = [
            SearchReplaceBlock(search="old1", replace="new1"),
            SearchReplaceBlock(search="new1", replace="final"),
        ]
        assert apply_search_replace(original, blocks) == "final\nold2\n"

    def test_fuzzy_match_mixed_indent_tabs_spaces(self):
        """Fuzzy matching should handle trailing whitespace, not leading."""
        original = "void foo() {\t\n\treturn 0;\n}\n"
        blocks = [SearchReplaceBlock(search="void foo() {\n\treturn 0;", replace="void foo() {\n\treturn 1;")]
        result = apply_search_replace(original, blocks)
        assert "return 1;" in result

    def test_empty_blocks_list(self):
        """Applying zero blocks returns original unchanged."""
        original = "hello\n"
        assert apply_search_replace(original, []) == "hello\n"

    def test_search_with_special_regex_chars(self):
        """Search text with regex metacharacters should work (literal match)."""
        original = "if (x > 0 && y < 10) { return (a + b); }\n"
        blocks = [SearchReplaceBlock(
            search="if (x > 0 && y < 10) { return (a + b); }",
            replace="if (x > 0 && y < 10) { return (a * b); }",
        )]
        result = apply_search_replace(original, blocks)
        assert "(a * b)" in result

    def test_realistic_sse_header_swap(self):
        original = (
            '#include <emmintrin.h>\n'
            '#ifdef __SSE2__\n'
            "void foo() {\n"
            "\t__m128i vZero = _mm_set1_epi32(0);\n"
            "}\n"
            "#endif\n"
        )
        blocks = [
            SearchReplaceBlock(
                search='#include <emmintrin.h>',
                replace='#include "sse2rvv.h"',
            ),
            SearchReplaceBlock(
                search='#ifdef __SSE2__',
                replace='#if defined(__SSE2__) || defined(__riscv_vector)',
            ),
        ]
        result = apply_search_replace(original, blocks)
        assert "sse2rvv.h" in result
        assert "__riscv_vector" in result
        assert "emmintrin.h" not in result


# =====================================================================
# number_lines
# =====================================================================


class TestNumberLines:
    def test_basic(self):
        code = "a\nb\nc\n"
        result = number_lines(code, every=1)
        assert "1|" in result
        assert "2|" in result
        assert "3|" in result

    def test_every_10(self):
        lines = "\n".join(f"line{i}" for i in range(1, 21)) + "\n"
        result = number_lines(lines, every=10)
        assert " 1|" in result
        assert "10|" in result
        assert "20|" in result


# =====================================================================
# Prompt helpers
# =====================================================================


class TestPromptHelpers:
    def test_search_replace_format_example_contains_markers(self):
        ex = search_replace_format_example()
        assert "<<<<<<< SEARCH" in ex
        assert "=======" in ex
        assert ">>>>>>> REPLACE" in ex

    def test_error_feedback_includes_error_and_file(self):
        feedback = search_replace_error_feedback(
            "lib.cpp", "int main() {}\n", "search text not found"
        )
        assert "search text not found" in feedback
        assert "lib.cpp" in feedback
        assert "<<<<<<< SEARCH" in feedback
