"""Unit tests for llm_utils.py – focuses on extract_code."""

from src.llm_utils import extract_code


class TestExtractCode:
    def test_extract_from_c_fenced_block(self):
        response = (
            "Here is the code:\n"
            "```c\n"
            "#include <stdint.h>\n"
            "void sequence_alignment_wavefront(int32_t *H) {}\n"
            "```\n"
        )
        code = extract_code(response)
        assert code is not None
        assert "sequence_alignment_wavefront" in code
        assert "#include <stdint.h>" in code

    def test_extract_from_C_fenced_block(self):
        response = "```C\nvoid sequence_alignment_wavefront(int32_t *H) {}\n```\n"
        code = extract_code(response)
        assert code is not None
        assert "sequence_alignment_wavefront" in code

    def test_extract_from_unfenced_block(self):
        response = "```\nvoid sequence_alignment_wavefront(int32_t *H) {}\n```\n"
        code = extract_code(response)
        assert code is not None
        assert "sequence_alignment_wavefront" in code

    def test_extract_no_fences_with_include(self):
        response = (
            "#include <stdint.h>\nvoid sequence_alignment_wavefront(int32_t *H) {}\n"
        )
        code = extract_code(response)
        assert code is not None
        assert "sequence_alignment_wavefront" in code

    def test_extract_no_fences_with_void(self):
        response = "void sequence_alignment_wavefront(int32_t *H) {}\n"
        code = extract_code(response)
        assert code is not None

    def test_returns_none_for_no_function(self):
        response = "```c\nint main() { return 0; }\n```"
        code = extract_code(response)
        assert code is None

    def test_returns_none_for_empty(self):
        assert extract_code("") is None

    def test_returns_none_for_random_text(self):
        assert extract_code("Hello, this is just a comment.") is None

    def test_strips_whitespace(self):
        response = (
            "```c\n  \nvoid sequence_alignment_wavefront(int32_t *H) {}\n  \n```\n"
        )
        code = extract_code(response)
        assert code is not None
        assert not code.startswith(" ")
        assert not code.endswith(" ")

    def test_multiple_fenced_blocks_picks_correct(self):
        response = (
            "```c\nint helper() { return 0; }\n```\n"
            "```c\nvoid sequence_alignment_wavefront(int32_t *H) {}\n```\n"
        )
        code = extract_code(response)
        assert code is not None
        assert "sequence_alignment_wavefront" in code

    def test_backtick_lines_stripped_in_fallback(self):
        response = (
            "```\n"
            "some random text\n"
            "```\n"
            "void sequence_alignment_wavefront(int32_t *H) {}\n"
            "#include <riscv_vector.h>\n"
        )
        code = extract_code(response)
        assert code is not None
        assert "```" not in code
