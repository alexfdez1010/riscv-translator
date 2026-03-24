"""Integration test: verify the LLM can produce valid search/replace blocks.

This test calls the real LLM (via OpenRouter or local endpoint) with a small
code snippet and a trivial compiler error, then checks that:

1. The response contains parseable search/replace blocks.
2. The blocks apply cleanly to the original code.
3. The resulting code contains the expected fix.

Skipped when no LLM backend is reachable (CI, offline, etc.).
"""

import pytest

from src.diff_utils import apply_search_replace, extract_search_replace
from src.llm_types import Message
from src.prompts import build_system_prompt, build_initial_translation_prompt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_CODE = """\
#include <emmintrin.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
    __m128i a = _mm_set1_epi32(42);
    int result = _mm_extract_epi16(a, 0);
    printf("result = %d\\n", result);
    return 0;
}
"""

_COMPILER_ERROR = """\
Validation stage: compile
Return code: 1
Failure details:
test.c:1:10: fatal error: emmintrin.h: No such file or directory
    1 | #include <emmintrin.h>
      |          ^~~~~~~~~~~~~
compilation terminated.
"""


def _try_create_llm():
    """Try to create an OpenRouter LLM; return None if unavailable."""
    from src.config import OPENROUTER_API_KEY
    if not OPENROUTER_API_KEY:
        return None
    try:
        from src.llm_utils import create_llm
        llm = create_llm()
        # Quick smoke test
        result = llm([Message(role="user", content="Reply with only the word OK.")])
        if result and len(result.strip()) < 100:
            return llm
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm():
    """Provide a working LLM or skip the entire module."""
    backend = _try_create_llm()
    if backend is None:
        pytest.skip("No LLM backend reachable")
    return backend


def test_llm_produces_valid_search_replace_blocks(llm):
    """The LLM should return parseable search/replace blocks for a trivial fix."""
    system = build_system_prompt("test.c")
    user = build_initial_translation_prompt(
        "test.c", _SMALL_CODE, "gcc -o test test.c", _COMPILER_ERROR,
    )
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]

    response = llm(messages)
    assert response, "LLM returned an empty response"

    blocks = extract_search_replace(response)
    assert blocks is not None, (
        f"Could not parse search/replace blocks from LLM response:\n{response[:500]}"
    )
    assert len(blocks) >= 1


def test_llm_blocks_apply_cleanly(llm):
    """Extracted search/replace blocks should apply to the original code."""
    system = build_system_prompt("test.c")
    user = build_initial_translation_prompt(
        "test.c", _SMALL_CODE, "gcc -o test test.c", _COMPILER_ERROR,
    )
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]

    response = llm(messages)
    blocks = extract_search_replace(response)
    assert blocks is not None, (
        f"No blocks extracted from:\n{response[:500]}"
    )

    # Should not raise
    result = apply_search_replace(_SMALL_CODE, blocks)
    assert result != _SMALL_CODE, "Blocks produced no change"
    # The fix should replace emmintrin.h with sse2rvv.h
    assert "sse2rvv.h" in result, (
        f"Expected sse2rvv.h in result but got:\n{result[:500]}"
    )
    assert "emmintrin.h" not in result


def test_llm_handles_repair_prompt(llm):
    """The LLM should also produce valid blocks for a repair (follow-up) prompt."""
    from src.prompts import build_repair_prompt

    # Code that already has sse2rvv.h but has a different minor error
    code_with_error = """\
#include "sse2rvv.h"
#include <stdlib.h>

int main() {
    __m128i* buf = (__m128i*)malloc(4 * sizeof(__m128i));
    return 0;
}
"""
    error_feedback = """\
Validation stage: compile
Return code: 1
Failure details:
test.c:5:49: error: RVV type '__m128i' {aka 'vint32m1_t'} does not have a fixed size
    5 |     __m128i* buf = (__m128i*)malloc(4 * sizeof(__m128i));
      |                                                 ^~~~~~~~~~~~~~~
"""
    system = build_system_prompt("test.c")
    user = build_repair_prompt("test.c", code_with_error, error_feedback)
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]

    response = llm(messages)
    assert response, "LLM returned an empty response"

    blocks = extract_search_replace(response)
    assert blocks is not None, (
        f"No blocks extracted from repair response:\n{response[:500]}"
    )

    result = apply_search_replace(code_with_error, blocks)
    assert "sizeof(__m128i)" not in result, (
        f"sizeof(__m128i) should have been replaced:\n{result[:500]}"
    )
