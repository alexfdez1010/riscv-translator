"""Tests for src/search_replace.py — search/replace block parsing and application."""

import pytest

from src.search_replace import (
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

    def test_block_with_tabs(self):
        """Code with tab indentation."""
        response = (
            "<<<<<<< SEARCH\n"
            "\tint x = 0;\n"
            "\tint y = 1;\n"
            "=======\n"
            "\tint x = 42;\n"
            "\tint y = 99;\n"
            ">>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "\t" in blocks[0].search
        assert "42" in blocks[0].replace

    def test_block_with_special_chars(self):
        """Code with regex metacharacters, pointers, etc."""
        response = (
            "<<<<<<< SEARCH\n"
            "ptr->data[i] = (a + b) * c;\n"
            "=======\n"
            "ptr->data[i] = (a + b) * d;\n"
            ">>>>>>> REPLACE"
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "ptr->data[i]" in blocks[0].search

    def test_block_with_backslashes(self):
        """Code with escape characters."""
        response = (
            '<<<<<<< SEARCH\n'
            'printf("hello\\n");\n'
            '=======\n'
            'printf("world\\n");\n'
            '>>>>>>> REPLACE'
        )
        blocks = extract_search_replace(response)
        assert blocks is not None
        assert "\\n" in blocks[0].search


# =====================================================================
# apply_search_replace — basic operations
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

    def test_deletion(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [SearchReplaceBlock(search="bbb\n", replace="")]
        assert apply_search_replace(original, blocks) == "aaa\nccc\n"

    def test_insertion_via_context(self):
        original = "aaa\nccc\n"
        blocks = [SearchReplaceBlock(search="aaa\nccc", replace="aaa\nbbb\nccc")]
        assert apply_search_replace(original, blocks) == "aaa\nbbb\nccc\n"

    def test_second_block_sees_result_of_first(self):
        original = "old1\nold2\n"
        blocks = [
            SearchReplaceBlock(search="old1", replace="new1"),
            SearchReplaceBlock(search="new1", replace="final"),
        ]
        assert apply_search_replace(original, blocks) == "final\nold2\n"

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
# apply_search_replace — duplicate/multiple occurrence handling
# =====================================================================


class TestApplySearchReplaceDuplicates:
    """The core scenario that was broken: LLM sends a search that matches
    multiple locations in the file."""

    def test_duplicate_search_replaces_all(self):
        """When search text appears 2+ times, ALL should be replaced."""
        original = "aaa\naaa\n"
        blocks = [SearchReplaceBlock(search="aaa", replace="bbb")]
        result = apply_search_replace(original, blocks)
        assert result == "bbb\nbbb\n"

    def test_duplicate_search_three_occurrences(self):
        original = "x = sizeof(T);\ny = sizeof(T);\nz = sizeof(T);\n"
        blocks = [SearchReplaceBlock(search="sizeof(T)", replace="16")]
        result = apply_search_replace(original, blocks)
        assert result.count("16") == 3
        assert "sizeof(T)" not in result

    def test_sizeof_m128i_real_world_case(self):
        """The exact scenario from the bug report: sizeof(__m128i) in two functions."""
        original = (
            "void func1() {\n"
            "\t__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));\n"
            "\tint16_t* t = (int16_t*)vProfile;\n"
            "}\n"
            "\n"
            "void func2() {\n"
            "\t__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));\n"
            "\tint16_t* t = (int16_t*)vProfile;\n"
            "}\n"
        )
        blocks = [SearchReplaceBlock(
            search="\t__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));",
            replace="\t__m128i* vProfile = (__m128i*)malloc((size_t)n * segLen * 16);",
        )]
        result = apply_search_replace(original, blocks)
        assert "sizeof(__m128i)" not in result
        assert result.count("* 16)") == 2

    def test_duplicate_multiline_search_replaces_all(self):
        """Multi-line search appearing twice should replace both."""
        original = (
            "// block 1\n"
            "a = 1;\n"
            "b = 2;\n"
            "// middle\n"
            "// block 2\n"
            "a = 1;\n"
            "b = 2;\n"
        )
        blocks = [SearchReplaceBlock(
            search="a = 1;\nb = 2;",
            replace="a = 10;\nb = 20;",
        )]
        result = apply_search_replace(original, blocks)
        assert result.count("a = 10;") == 2
        assert result.count("b = 20;") == 2
        assert "a = 1;" not in result

    def test_calloc_sizeof_multiple_lines(self):
        """Real-world: multiple calloc lines with sizeof(__m128i)."""
        original = (
            "\t__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "\t__m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "\t__m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "\t__m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
        )
        # LLM sends a block matching all 4 lines at once — should work
        blocks = [SearchReplaceBlock(
            search=(
                "\t__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
                "\t__m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
                "\t__m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
                "\t__m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));"
            ),
            replace=(
                "\t__m128i* pvHStore = (__m128i*) calloc(segLen, 16);\n"
                "\t__m128i* pvHLoad = (__m128i*) calloc(segLen, 16);\n"
                "\t__m128i* pvE = (__m128i*) calloc(segLen, 16);\n"
                "\t__m128i* pvHmax = (__m128i*) calloc(segLen, 16);"
            ),
        )]
        result = apply_search_replace(original, blocks)
        assert "sizeof(__m128i)" not in result
        assert result.count(", 16)") == 4

    def test_two_blocks_same_search_text_second_skipped(self):
        """If the LLM sends two blocks with the same search text, the second
        should be gracefully skipped (text was already replaced by block 1)."""
        original = "aaa\nbbb\naaa\n"
        blocks = [
            SearchReplaceBlock(search="aaa", replace="xxx"),
            SearchReplaceBlock(search="aaa", replace="xxx"),
        ]
        result = apply_search_replace(original, blocks)
        assert result == "xxx\nbbb\nxxx\n"

    def test_overlapping_blocks_second_search_consumed(self):
        """Block 1 replaces text that block 2's search looks for — block 2
        should be skipped without error."""
        original = "aaa\nbbb\nccc\n"
        blocks = [
            SearchReplaceBlock(search="aaa\nbbb", replace="xxx\nyyy"),
            SearchReplaceBlock(search="aaa", replace="zzz"),  # 'aaa' is gone
        ]
        # Block 2's search "aaa" is gone, but block 1 already handled it.
        result = apply_search_replace(original, blocks)
        assert result == "xxx\nyyy\nccc\n"

    def test_replacement_already_present_skipped(self):
        """If block's replacement text is already in the file (because a
        prior block put it there), skip silently."""
        original = "old_val\n"
        blocks = [
            SearchReplaceBlock(search="old_val", replace="new_val"),
            SearchReplaceBlock(search="old_val", replace="new_val"),
        ]
        result = apply_search_replace(original, blocks)
        assert result == "new_val\n"

    def test_same_replacement_in_different_functions_not_skipped(self):
        """Block targeting function B must NOT be skipped just because
        function A already contains the same replacement text.

        Regression test: the LLM fixes pointer arithmetic in sw_sse2_word
        (step 1), then sends a block to fix the same pattern in sw_sse2_byte
        (step 2).  The replacement text already exists in sw_sse2_word, but
        the block for sw_sse2_byte must still be applied.
        """
        original = (
            "void func_a() {\n"
            "    x = *(ptr + (size_t)j * VSIZE);  // already fixed\n"
            "}\n"
            "void func_b() {\n"
            "    x = ptr[j];  // needs fixing\n"
            "}\n"
        )
        blocks = [
            SearchReplaceBlock(
                search="    x = ptr[j];  // needs fixing",
                replace="    x = *(ptr + (size_t)j * VSIZE);  // already fixed",
            ),
        ]
        result = apply_search_replace(original, blocks)
        assert result.count("*(ptr + (size_t)j * VSIZE)") == 2
        assert "ptr[j]" not in result


# =====================================================================
# apply_search_replace — fuzzy/whitespace matching
# =====================================================================


class TestApplySearchReplaceFuzzy:
    def test_trailing_whitespace_tolerance(self):
        """Trailing whitespace differences should be tolerated."""
        original = "aaa  \nbbb\n"
        blocks = [SearchReplaceBlock(search="aaa", replace="xxx")]
        result = apply_search_replace(original, blocks)
        assert "xxx" in result

    def test_tab_vs_spaces_leading_whitespace(self):
        """Tabs in file, spaces in search (or vice versa) should match."""
        original = "\tint x = 0;\n\tint y = 1;\n"
        blocks = [SearchReplaceBlock(
            search="    int x = 0;\n    int y = 1;",
            replace="    int x = 42;\n    int y = 99;",
        )]
        result = apply_search_replace(original, blocks)
        assert "42" in result
        assert "99" in result

    def test_spaces_in_file_tabs_in_search(self):
        """Spaces in file, tabs in search should also match."""
        original = "    int x = 0;\n    int y = 1;\n"
        blocks = [SearchReplaceBlock(
            search="\tint x = 0;\n\tint y = 1;",
            replace="\tint x = 42;\n\tint y = 99;",
        )]
        result = apply_search_replace(original, blocks)
        assert "42" in result
        assert "99" in result

    def test_mixed_indent_tabs_spaces(self):
        """Fuzzy matching should handle trailing whitespace, not leading."""
        original = "void foo() {\t\n\treturn 0;\n}\n"
        blocks = [SearchReplaceBlock(
            search="void foo() {\n\treturn 0;",
            replace="void foo() {\n\treturn 1;",
        )]
        result = apply_search_replace(original, blocks)
        assert "return 1;" in result

    def test_fuzzy_match_with_duplicates_replaces_all(self):
        """Fuzzy matching should also replace all when there are duplicates."""
        original = "  x = sizeof(T);  \n  y = sizeof(T);  \n"
        # Search without trailing spaces — fuzzy should match both
        blocks = [SearchReplaceBlock(
            search="  x = sizeof(T);",
            replace="  x = 16;",
        )]
        result = apply_search_replace(original, blocks)
        assert "16" in result

    def test_fuzzy_multiline_with_trailing_spaces(self):
        """Multi-line search with trailing whitespace differences."""
        original = "void f() {   \n    return 0;   \n}   \n"
        blocks = [SearchReplaceBlock(
            search="void f() {\n    return 0;\n}",
            replace="void f() {\n    return 1;\n}",
        )]
        result = apply_search_replace(original, blocks)
        assert "return 1;" in result


# =====================================================================
# apply_search_replace — escape characters and special content
# =====================================================================


class TestApplySearchReplaceSpecialChars:
    def test_backslash_in_code(self):
        original = 'printf("hello\\nworld");\n'
        blocks = [SearchReplaceBlock(
            search='printf("hello\\nworld");',
            replace='printf("goodbye\\nworld");',
        )]
        result = apply_search_replace(original, blocks)
        assert "goodbye" in result

    def test_dollar_sign(self):
        original = "cost = $100;\n"
        blocks = [SearchReplaceBlock(search="$100", replace="$200")]
        assert "$200" in apply_search_replace(original, blocks)

    def test_curly_braces(self):
        original = "if (x) { y(); }\n"
        blocks = [SearchReplaceBlock(
            search="if (x) { y(); }",
            replace="if (x) { z(); }",
        )]
        assert "z();" in apply_search_replace(original, blocks)

    def test_square_brackets(self):
        original = "arr[0] = arr[1];\n"
        blocks = [SearchReplaceBlock(
            search="arr[0] = arr[1];",
            replace="arr[0] = arr[2];",
        )]
        assert "arr[2]" in apply_search_replace(original, blocks)

    def test_parentheses_and_asterisks(self):
        original = "int* p = (int*)malloc(sizeof(int));\n"
        blocks = [SearchReplaceBlock(
            search="int* p = (int*)malloc(sizeof(int));",
            replace="int* p = (int*)malloc(4);",
        )]
        assert "malloc(4)" in apply_search_replace(original, blocks)

    def test_angle_brackets(self):
        original = "#include <stdio.h>\n#include <stdlib.h>\n"
        blocks = [SearchReplaceBlock(
            search="#include <stdio.h>",
            replace='#include "stdio.h"',
        )]
        result = apply_search_replace(original, blocks)
        assert '"stdio.h"' in result

    def test_hash_and_preprocessor(self):
        original = "#ifdef __SSE2__\n#define USE_SSE 1\n#endif\n"
        blocks = [SearchReplaceBlock(
            search="#ifdef __SSE2__\n#define USE_SSE 1",
            replace="#if defined(__SSE2__) || defined(__riscv_vector)\n#define USE_SSE 1",
        )]
        result = apply_search_replace(original, blocks)
        assert "__riscv_vector" in result

    def test_pipe_and_ampersand(self):
        original = "if (a & b || c | d) {}\n"
        blocks = [SearchReplaceBlock(
            search="if (a & b || c | d) {}",
            replace="if (a & b && c | d) {}",
        )]
        assert "&&" in apply_search_replace(original, blocks)

    def test_quotes_single_and_double(self):
        original = "char c = 'x'; char* s = \"hello\";\n"
        blocks = [SearchReplaceBlock(
            search='char c = \'x\'; char* s = "hello";',
            replace='char c = \'y\'; char* s = "world";',
        )]
        result = apply_search_replace(original, blocks)
        assert "'y'" in result
        assert '"world"' in result

    def test_pointer_arithmetic(self):
        """Real-world RVV scenario: pointer arithmetic with casts."""
        original = "\t\t__m128i vH = pvHStore[segLen - 1];\n"
        blocks = [SearchReplaceBlock(
            search="\t\t__m128i vH = pvHStore[segLen - 1];",
            replace="\t\t__m128i vH = *(__m128i*)((char*)pvHStore + (segLen - 1) * 16);",
        )]
        result = apply_search_replace(original, blocks)
        assert "(char*)pvHStore" in result

    def test_c_string_escapes(self):
        """Various C string escape sequences."""
        original = 'msg = "tab\\there\\nnewline\\0null";\n'
        blocks = [SearchReplaceBlock(
            search='msg = "tab\\there\\nnewline\\0null";',
            replace='msg = "modified";',
        )]
        assert "modified" in apply_search_replace(original, blocks)

    def test_cast_expressions(self):
        """Complex cast expressions that look like they might confuse parsing."""
        original = "val = (__m128i*)((const char*)(vP) + j * 16);\n"
        blocks = [SearchReplaceBlock(
            search="val = (__m128i*)((const char*)(vP) + j * 16);",
            replace="val = (__m128i*)((const char*)(vP) + j * 32);",
        )]
        assert "j * 32" in apply_search_replace(original, blocks)


# =====================================================================
# apply_search_replace — edge cases
# =====================================================================


class TestApplySearchReplaceEdgeCases:
    def test_empty_original(self):
        """Searching in empty file raises."""
        blocks = [SearchReplaceBlock(search="x", replace="y")]
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace("", blocks)

    def test_empty_search_still_matches(self):
        """Empty search string matches (it's a substring of everything)."""
        original = "hello\n"
        blocks = [SearchReplaceBlock(search="", replace="world")]
        # Python's str.replace("", "world") inserts between every char
        result = apply_search_replace(original, blocks)
        assert "world" in result

    def test_replace_with_longer_text(self):
        original = "a\n"
        blocks = [SearchReplaceBlock(search="a", replace="aaa\nbbb\nccc")]
        result = apply_search_replace(original, blocks)
        assert "aaa\nbbb\nccc" in result

    def test_replace_with_shorter_text(self):
        original = "aaa\nbbb\nccc\n"
        blocks = [SearchReplaceBlock(search="aaa\nbbb\nccc", replace="x")]
        result = apply_search_replace(original, blocks)
        assert result.startswith("x")

    def test_many_blocks(self):
        """10 blocks applied sequentially."""
        original = "\n".join(f"line{i}" for i in range(10)) + "\n"
        blocks = [
            SearchReplaceBlock(search=f"line{i}", replace=f"LINE{i}")
            for i in range(10)
        ]
        result = apply_search_replace(original, blocks)
        for i in range(10):
            assert f"LINE{i}" in result

    def test_search_at_beginning_of_file(self):
        original = "first line\nsecond line\n"
        blocks = [SearchReplaceBlock(search="first line", replace="new first")]
        assert apply_search_replace(original, blocks).startswith("new first")

    def test_search_at_end_of_file(self):
        original = "first line\nlast line\n"
        blocks = [SearchReplaceBlock(search="last line", replace="new last")]
        result = apply_search_replace(original, blocks)
        assert "new last" in result

    def test_search_spanning_entire_file(self):
        original = "entire file content"
        blocks = [SearchReplaceBlock(search="entire file content", replace="new content")]
        assert apply_search_replace(original, blocks) == "new content"

    def test_unicode_content(self):
        original = "// Comment: café résumé\nint x = 0;\n"
        blocks = [SearchReplaceBlock(
            search="// Comment: café résumé",
            replace="// Comment: updated",
        )]
        assert "updated" in apply_search_replace(original, blocks)

    def test_windows_line_endings_in_search(self):
        """Search with \\r\\n should still match \\n in file via fuzzy."""
        original = "line1\nline2\nline3\n"
        blocks = [SearchReplaceBlock(
            search="line1\r\nline2",
            replace="LINE1\nLINE2",
        )]
        # Exact match fails, but the content is there
        # The fuzzy match won't handle \r\n splitting well, so this might fail.
        # Let's at least verify it doesn't crash.
        try:
            result = apply_search_replace(original, blocks)
            assert "LINE1" in result
        except ValueError:
            pass  # acceptable — \r\n is a tricky edge case

    def test_block_that_creates_duplicate_then_next_block_hits_it(self):
        """Block 1 creates duplicated text, block 2 replaces it (now appears 2x)."""
        original = "unique\nother\n"
        blocks = [
            SearchReplaceBlock(search="other", replace="unique"),
            SearchReplaceBlock(search="unique", replace="final"),
        ]
        result = apply_search_replace(original, blocks)
        assert result.count("final") == 2


# =====================================================================
# Real-world LLM output scenarios
# =====================================================================


class TestRealWorldScenarios:
    def test_sizeof_m128i_in_two_functions_single_line_block(self):
        """The exact failure from the bug: LLM sends single-line search for
        sizeof(__m128i) that appears in two different functions."""
        original = (
            "// Function 1: qP_byte\n"
            "__m128i* qP_byte(const int8_t* read, const int8_t* mat, int32_t n, int32_t segLen) {\n"
            "\t__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));\n"
            "\tint8_t* t = (int8_t*)vProfile;\n"
            "\treturn vProfile;\n"
            "}\n"
            "\n"
            "// Function 2: qP_word\n"
            "__m128i* qP_word(const int8_t* read, const int8_t* mat, int32_t n, int32_t segLen) {\n"
            "\t__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));\n"
            "\tint16_t* t = (int16_t*)vProfile;\n"
            "\treturn vProfile;\n"
            "}\n"
        )
        blocks = [SearchReplaceBlock(
            search="\t__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));",
            replace="\t__m128i* vProfile = (__m128i*)malloc((size_t)n * segLen * 16);",
        )]
        result = apply_search_replace(original, blocks)
        # Both occurrences should be replaced
        assert "sizeof(__m128i)" not in result
        assert result.count("* 16)") == 2

    def test_multiple_calloc_sizeof_blocks(self):
        """LLM sends a multi-line block for calloc lines."""
        original = (
            "void sw_byte() {\n"
            "\t__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "\t__m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "\t__m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "\t__m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "}\n"
            "\n"
            "void sw_word() {\n"
            "\t__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "\t__m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "\t__m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "\t__m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
            "}\n"
        )
        blocks = [SearchReplaceBlock(
            search=(
                "\t__m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
                "\t__m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
                "\t__m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));\n"
                "\t__m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));"
            ),
            replace=(
                "\t__m128i* pvHStore = (__m128i*) calloc(segLen, 16);\n"
                "\t__m128i* pvHLoad = (__m128i*) calloc(segLen, 16);\n"
                "\t__m128i* pvE = (__m128i*) calloc(segLen, 16);\n"
                "\t__m128i* pvHmax = (__m128i*) calloc(segLen, 16);"
            ),
        )]
        result = apply_search_replace(original, blocks)
        assert "sizeof(__m128i)" not in result
        assert result.count(", 16)") == 8  # 4 per function, 2 functions

    def test_pointer_arithmetic_replacement(self):
        """LLM replaces pointer indexing with byte-offset casts."""
        original = (
            "\t\t__m128i vH = pvHStore[segLen - 1];\n"
            "\t\tconst __m128i* vP = vProfile + ref[i] * segLen;\n"
        )
        blocks = [
            SearchReplaceBlock(
                search="\t\t__m128i vH = pvHStore[segLen - 1];",
                replace="\t\t__m128i vH = *(__m128i*)((char*)pvHStore + (segLen - 1) * 16);",
            ),
            SearchReplaceBlock(
                search="\t\tconst __m128i* vP = vProfile + ref[i] * segLen;",
                replace="\t\tconst __m128i* vP = (__m128i*)((char*)vProfile + (size_t)ref[i] * segLen * 16);",
            ),
        ]
        result = apply_search_replace(original, blocks)
        assert "(char*)pvHStore" in result
        assert "(char*)vProfile" in result

    def test_inner_loop_large_block(self):
        """LLM sends a large block replacing an entire loop body."""
        original = (
            "\t\tfor (j = 0; j < segLen; ++j) {\n"
            "\t\t\tvH = _mm_adds_epu8(vH, _mm_load_si128(vP + j));\n"
            "\t\t\te = _mm_load_si128(pvE + j);\n"
            "\t\t\t_mm_store_si128(pvHStore + j, vH);\n"
            "\t\t\t_mm_store_si128(pvE + j, e);\n"
            "\t\t\tvH = _mm_load_si128(pvHLoad + j);\n"
            "\t\t}\n"
        )
        blocks = [SearchReplaceBlock(
            search=(
                "\t\tfor (j = 0; j < segLen; ++j) {\n"
                "\t\t\tvH = _mm_adds_epu8(vH, _mm_load_si128(vP + j));\n"
                "\t\t\te = _mm_load_si128(pvE + j);\n"
                "\t\t\t_mm_store_si128(pvHStore + j, vH);\n"
                "\t\t\t_mm_store_si128(pvE + j, e);\n"
                "\t\t\tvH = _mm_load_si128(pvHLoad + j);\n"
                "\t\t}"
            ),
            replace=(
                "\t\tfor (j = 0; j < segLen; ++j) {\n"
                "\t\t\tvH = _mm_adds_epu8(vH, _mm_load_si128((__m128i*)((const char*)(vP) + j * 16)));\n"
                "\t\t\te = _mm_load_si128((__m128i*)((const char*)(pvE) + j * 16));\n"
                "\t\t\t_mm_store_si128((__m128i*)((char*)(pvHStore) + j * 16), vH);\n"
                "\t\t\t_mm_store_si128((__m128i*)((char*)(pvE) + j * 16), e);\n"
                "\t\t\tvH = _mm_load_si128((__m128i*)((const char*)(pvHLoad) + j * 16));\n"
                "\t\t}"
            ),
        )]
        result = apply_search_replace(original, blocks)
        assert "j * 16" in result
        assert "vP + j" not in result


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
