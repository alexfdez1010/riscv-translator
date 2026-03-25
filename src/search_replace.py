"""Search/replace block parsing and application for LLM-driven code edits.

The LLM returns edits as search/replace blocks — exact old-text / new-text
pairs that are far more reliable than unified diffs for LLM output.

Key design goals
----------------
* Simple, unambiguous format that LLMs produce correctly.
* Exact string matching with whitespace-tolerant fallback.
* Graceful handling of duplicate matches (replace all occurrences).
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
    """A single search->replace edit."""
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


# ---------------------------------------------------------------------------
# Apply search/replace blocks
# ---------------------------------------------------------------------------


def apply_search_replace(
    original: str, blocks: list[SearchReplaceBlock]
) -> str:
    """Apply search/replace blocks sequentially to *original*.

    Strategy for each block:
    1. Exact match, unique -> replace the single occurrence.
    2. Exact match, multiple occurrences -> replace ALL occurrences.
       (The LLM almost always wants the same mechanical fix everywhere.)
    3. No exact match -> try fuzzy matching (whitespace tolerance).
    4. Still no match -> check if the search text was already replaced
       by a previous block (skip silently).
    5. Otherwise -> raise ``ValueError`` with an actionable message.
    """
    result = original
    applied_replacements: list[tuple[str, str]] = []

    for i, block in enumerate(blocks, 1):
        count = result.count(block.search)

        if count == 1:
            # Unique exact match — ideal case
            result = result.replace(block.search, block.replace, 1)
            applied_replacements.append((block.search, block.replace))
            continue

        if count > 1:
            # Multiple occurrences — replace ALL of them.
            logger.info(
                "Search/replace block %d: search text appears %d times; "
                "replacing all occurrences",
                i, count,
            )
            result = result.replace(block.search, block.replace)
            applied_replacements.append((block.search, block.replace))
            continue

        # count == 0: try fuzzy matching
        found = _fuzzy_search_replace(result, block.search, block.replace)
        if found is not None:
            result = found
            applied_replacements.append((block.search, block.replace))
            continue

        # Check if a previous block already replaced this text
        already_applied = _was_already_applied(
            block, applied_replacements, result
        )
        if already_applied:
            logger.info(
                "Search/replace block %d: search text already handled "
                "by a prior block; skipping",
                i,
            )
            continue

        # Truly not found
        preview = block.search[:120].replace("\n", "\\n")
        raise ValueError(
            f"Search/replace block {i}: search text not found in file. "
            f"Searched for: {preview!r}"
        )
    return result


def _was_already_applied(
    block: SearchReplaceBlock,
    applied_replacements: list[tuple[str, str]],
    current_text: str,
) -> bool:
    """Check if a block's search text was already consumed by a prior block."""
    for prev_search, prev_replace in applied_replacements:
        # Same search text was used before
        if prev_search == block.search:
            return True
        # The search text is a substring of a previous search
        if block.search in prev_search:
            return True
        # The search text appears in a previous replacement
        # (the prior block already transformed it)
        if block.search in prev_replace:
            return True
    return False


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


def _norm_line(line: str) -> str:
    """Normalise a single line for fuzzy comparison.

    Strips trailing whitespace and normalises leading whitespace
    (tabs <-> spaces) so minor indentation differences don't prevent
    matching.
    """
    stripped = line.rstrip()
    leading = len(stripped) - len(stripped.lstrip())
    prefix = stripped[:leading].replace("\t", "    ")
    return prefix + stripped[leading:]


def _find_all_line_matches(
    text_norm: list[str], search_norm: list[str]
) -> list[int]:
    """Return start indices of all non-overlapping line-based matches."""
    slen = len(search_norm)
    if slen == 0:
        return []
    matches: list[int] = []
    i = 0
    while i <= len(text_norm) - slen:
        if text_norm[i : i + slen] == search_norm:
            matches.append(i)
            i += slen  # skip past to avoid overlaps
        else:
            i += 1
    return matches


def _replace_line_matches(
    text_lines: list[str],
    match_starts: list[int],
    search_len: int,
    replace: str,
) -> str:
    """Replace all matched line regions in *text_lines*."""
    replace_lines = replace.splitlines(keepends=True)

    result_lines = list(text_lines)
    # Work backwards so indices stay valid
    for start in reversed(match_starts):
        end = start + search_len
        patched = list(replace_lines)
        if patched and not patched[-1].endswith("\n"):
            has_trailing = (
                end < len(result_lines)
                or (result_lines and result_lines[-1].endswith("\n"))
            )
            if has_trailing:
                patched[-1] += "\n"
        result_lines[start:end] = patched

    return "".join(result_lines)


def _fuzzy_search_replace(
    text: str, search: str, replace: str
) -> str | None:
    """Try to match *search* against *text* with whitespace tolerance.

    Tries the following strategies in order:
    1. Trailing-whitespace normalisation (rstrip each line).
    2. Tab <-> spaces normalisation (leading whitespace).

    If matches are found (even multiple), replaces all of them.
    Returns the modified text or ``None`` if no match.
    """
    search_lines_raw = search.splitlines()
    text_lines = text.splitlines(keepends=True)

    if not search_lines_raw:
        return None

    search_len = len(search_lines_raw)

    # --- Strategy 1: trailing-whitespace-only normalisation ---
    search_norm = [ln.rstrip() for ln in search_lines_raw]
    text_norm = [ln.rstrip() for ln in text_lines]
    matches = _find_all_line_matches(text_norm, search_norm)
    if matches:
        return _replace_line_matches(text_lines, matches, search_len, replace)

    # --- Strategy 2: full whitespace normalisation (tabs <-> spaces) ---
    search_norm2 = [_norm_line(ln) for ln in search_lines_raw]
    text_norm2 = [_norm_line(ln) for ln in text_lines]
    matches = _find_all_line_matches(text_norm2, search_norm2)
    if matches:
        return _replace_line_matches(text_lines, matches, search_len, replace)

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
- If the same text appears in multiple places and you want to change ALL
  of them, a single block is enough — all occurrences will be replaced.
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
        f"2. If the same text appears multiple times, a single search/replace "
        f"block is enough — all occurrences will be replaced.\n"
        f"3. The REPLACE text must be DIFFERENT from the SEARCH text.  "
        f"Every block must change something — do not emit no-op blocks.\n\n"
        f"Current {file_name} (with line numbers):\n"
        f"```c\n{number_lines(code)}\n```\n\n"
        f"{search_replace_format_example()}"
    )
