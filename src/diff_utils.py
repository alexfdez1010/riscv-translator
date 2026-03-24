"""Search/replace block parsing and application for LLM-driven code edits.

The LLM returns edits as search/replace blocks — exact old-text / new-text
pairs that are far more reliable than unified diffs for LLM output.

Key design goals
----------------
* Simple, unambiguous format that LLMs produce correctly.
* Exact string matching with whitespace-tolerant fallback.
* Clear, actionable error messages so the LLM can self-correct.
* Stay dependency-free (stdlib only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Search/replace block format
# ---------------------------------------------------------------------------

# Primary pattern: exact markers on their own lines.
# Tolerates optional trailing whitespace on marker lines and an optional
# markdown code-fence wrapper (```).
_SR_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH[ \t]*\n(.*?)\n=======[ \t]*\n(.*?)\n>>>>>>> REPLACE[ \t]*",
    re.DOTALL,
)


@dataclass(slots=True)
class SearchReplaceBlock:
    """A single search→replace edit."""
    search: str
    replace: str


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that some LLMs wrap around the blocks."""
    return re.sub(r"^```[a-zA-Z]*\s*\n", "", text, flags=re.MULTILINE).replace(
        "\n```\n", "\n"
    ).replace("\n```", "\n")


def extract_search_replace(response: str) -> list[SearchReplaceBlock] | None:
    """Extract search/replace blocks from an LLM response.

    Format::

        <<<<<<< SEARCH
        old code exactly as it appears
        =======
        new replacement code
        >>>>>>> REPLACE

    Tolerates:
    - Trailing whitespace on marker lines.
    - Markdown code fences wrapping the blocks.

    Returns ``None`` if no blocks are found.
    """
    # First try the raw response
    matches = _SR_BLOCK_RE.findall(response)
    if not matches:
        # Retry after stripping markdown fences
        cleaned = _strip_markdown_fences(response)
        matches = _SR_BLOCK_RE.findall(cleaned)
    if not matches:
        return None
    # Filter out no-op blocks where search == replace (LLM mistake)
    blocks = [
        SearchReplaceBlock(search=s, replace=r)
        for s, r in matches
        if s != r
    ]
    if not blocks:
        logger.warning("All search/replace blocks were no-ops (search == replace)")
        return None
    return blocks


def apply_search_replace(
    original: str, blocks: list[SearchReplaceBlock]
) -> str:
    """Apply search/replace blocks sequentially to *original*.

    Each block's *search* text must appear exactly once in the current
    state of the file (after previous blocks have been applied).

    Raises ``ValueError`` with an actionable message if a search string
    is not found or is ambiguous (appears more than once).
    """
    result = original
    for i, block in enumerate(blocks, 1):
        count = result.count(block.search)
        if count == 0:
            # Try whitespace-tolerant matching
            found = _fuzzy_search_replace(result, block.search, block.replace)
            if found is not None:
                result = found
                continue
            # Show a snippet of what we searched for
            preview = block.search[:120].replace("\n", "\\n")
            raise ValueError(
                f"Search/replace block {i}: search text not found in file. "
                f"Searched for: {preview!r}"
            )
        if count > 1:
            preview = block.search[:120].replace("\n", "\\n")
            raise ValueError(
                f"Search/replace block {i}: search text appears {count} times "
                f"(must be unique). Text: {preview!r}"
            )
        result = result.replace(block.search, block.replace, 1)
    return result


def _fuzzy_search_replace(
    text: str, search: str, replace: str
) -> str | None:
    """Try to match *search* against *text* with whitespace tolerance.

    Normalises trailing whitespace on each line before comparing.
    Returns the modified text or ``None`` if no match.
    """
    def _norm_lines(s: str) -> list[str]:
        return [ln.rstrip() for ln in s.splitlines()]

    search_norm = _norm_lines(search)
    text_lines = text.splitlines(keepends=True)
    text_norm = [ln.rstrip() for ln in text_lines]

    # Slide search window over text
    slen = len(search_norm)
    for start in range(len(text_norm) - slen + 1):
        if text_norm[start : start + slen] == search_norm:
            # Found — replace the original lines
            before = text_lines[:start]
            after = text_lines[start + slen:]
            # Preserve trailing newline style from original
            replace_lines = replace.splitlines(keepends=True)
            if replace_lines and not replace_lines[-1].endswith("\n"):
                if after or (text_lines and text_lines[-1].endswith("\n")):
                    replace_lines[-1] += "\n"
            return "".join(before) + "".join(replace_lines) + "".join(after)
    return None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def number_lines(code: str, every: int = 10) -> str:
    """Add line numbers to code for LLM context.

    Numbers every *every*-th line and the first line.  Unnumbered lines
    get a blank margin so the alignment stays consistent.
    """
    lines = code.splitlines(keepends=True)
    width = len(str(len(lines)))
    result: list[str] = []
    for i, line in enumerate(lines, 1):
        if i == 1 or i % every == 0:
            result.append(f"{i:>{width}}| {line}")
        else:
            result.append(f"{' ' * width}| {line}")
    return "".join(result)


def _extract_error_line(error_message: str) -> int | None:
    """Try to extract a line number from an error message."""
    m = re.search(r"at line (\d+)", error_message)
    return int(m.group(1)) if m else None


def _source_context_snippet(code: str, line_no: int, radius: int = 5) -> str:
    """Return a numbered snippet of *code* around *line_no* (1-based)."""
    lines = code.splitlines()
    start = max(0, line_no - 1 - radius)
    end = min(len(lines), line_no + radius)
    width = len(str(end))
    snippet_lines = []
    for i in range(start, end):
        marker = ">>>" if i == line_no - 1 else "   "
        snippet_lines.append(f"{marker} {i + 1:>{width}}| {lines[i]}")
    return "\n".join(snippet_lines)


def search_replace_format_example() -> str:
    """Return a concrete search/replace example for the LLM."""
    return """\
## Search/replace block format

Each block MUST have the SEARCH section (old code) and a DIFFERENT REPLACE
section (new code).  The REPLACE section must NOT be identical to SEARCH —
that would be a no-op.  Every block must change something.

Example — single-line change:

<<<<<<< SEARCH
#include <emmintrin.h>
=======
#include "sse2rvv.h"
>>>>>>> REPLACE

Example — multi-line change (include enough context to be unique):

<<<<<<< SEARCH
    __m128i* buf = (__m128i*) calloc(segLen, sizeof(__m128i));
    __m128i* tmp = (__m128i*) calloc(segLen, sizeof(__m128i));
=======
    __m128i* buf = (__m128i*) calloc(segLen, 16);
    __m128i* tmp = (__m128i*) calloc(segLen, 16);
>>>>>>> REPLACE

IMPORTANT:
- SEARCH must match the file EXACTLY (same whitespace, same indentation).
- SEARCH must appear exactly once in the file — include surrounding lines
  to make it unique if needed.
- REPLACE must be DIFFERENT from SEARCH — every block must make a change.
"""


def search_replace_error_feedback(
    file_name: str, code: str, error_message: str
) -> str:
    """Build feedback for the LLM when search/replace blocks failed."""
    context_section = ""
    line_no = _extract_error_line(error_message)
    if line_no:
        snippet = _source_context_snippet(code, line_no)
        context_section = (
            f"\nLines around the error location (line {line_no}):\n"
            f"```\n{snippet}\n```\n"
        )

    return (
        f"Your previous edit could NOT be applied.  Read the error carefully "
        f"and try again.\n\n"
        f"Error: {error_message}\n{context_section}\n"
        f"RULES — read these before responding:\n"
        f"1. The SEARCH text must be copied EXACTLY from the current file "
        f"(same indentation, same whitespace, character for character).\n"
        f"2. Each SEARCH text must appear exactly once in the file.  If the "
        f"text you chose appears multiple times, include more surrounding "
        f"lines to make it unique.\n"
        f"3. The REPLACE text must be DIFFERENT from the SEARCH text.  "
        f"Every block must change something — do not emit no-op blocks.\n\n"
        f"Current {file_name} (with line numbers):\n"
        f"```c\n{number_lines(code)}\n```\n\n"
        f"{search_replace_format_example()}"
    )
