"""Robust unified-diff parsing, extraction, and application.

This module centralises all Git/unified-diff handling so that both the
repair agent and the evolutionary operators share the same flexible parser.

Key design goals
----------------
* Tolerate the many formatting mistakes LLMs make: bare ``@@`` headers,
  wrong line counts, trailing-whitespace differences, non-contiguous
  context lines, overlapping hunks, missing ``a/``/``b/`` prefixes, etc.
* Provide clear, actionable error messages so the LLM can self-correct.
* Stay dependency-free (stdlib only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Hunk:
    """A single hunk parsed from a unified diff."""

    old_start: int
    old_count: int | None
    new_start: int
    new_count: int | None
    lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedPatch:
    """A fully parsed single-file unified diff."""

    old_path: str | None
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction: pull a diff out of free-form LLM text
# ---------------------------------------------------------------------------

_FENCED_DIFF_RE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL)
_FENCED_GENERIC_RE = re.compile(r"```\s*\n((?:diff --git|---) .*?)```", re.DOTALL)
_BARE_DIFF_RE = re.compile(r"(?:^|\n)((?:diff --git |--- ).*)", re.DOTALL)


def _merge_diff_blocks(blocks: list[str]) -> str:
    """Merge multiple diff blocks that target the same file into one patch.

    LLMs (especially during crossover) sometimes return multiple separate
    ```diff blocks instead of one patch with multiple hunks.  This function
    concatenates them, keeping only one set of file headers.
    """
    if len(blocks) == 1:
        return blocks[0].strip()

    merged_lines: list[str] = []
    seen_headers = False

    for block in blocks:
        for line in block.strip().splitlines():
            if line.startswith(("diff --git ", "--- ", "+++ ")):
                if seen_headers:
                    continue  # skip duplicate file headers
                if line.startswith("+++ "):
                    seen_headers = True
            merged_lines.append(line)

    return "\n".join(merged_lines)


def extract_diff(response: str) -> str | None:
    """Extract a unified diff from an LLM response.

    Handles:
    * Single or multiple fenced ``\\`\\`\\`diff`` blocks (merged into one patch).
    * Generic fenced blocks containing diff markers.
    * Bare trailing diffs without fencing.
    * Entire response being a diff.

    Returns ``None`` when no diff-like content is found.
    """
    # 1. Try fenced ```diff / ```patch blocks — collect ALL of them and merge.
    fenced = _FENCED_DIFF_RE.findall(response)
    if fenced:
        non_empty = [b.strip() for b in fenced if b.strip()]
        if non_empty:
            return _merge_diff_blocks(non_empty)

    # 2. Generic fenced block that starts with diff markers.
    m = _FENCED_GENERIC_RE.search(response)
    if m:
        patch = m.group(1).strip()
        if patch:
            return patch

    # 3. Unboxed diff at the end of the response.
    m = _BARE_DIFF_RE.search(response)
    if m:
        patch = m.group(1).strip()
        if patch:
            return patch

    # 4. Last resort: the entire response *is* a diff.
    stripped = response.strip()
    if stripped.startswith("diff --git ") or (
        stripped.startswith("--- ") and "\n+++ " in stripped
    ):
        return stripped
    return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_single_file_diff(patch: str, file_name: str) -> None:
    """Raise ``ValueError`` if *patch* is not a well-formed single-file diff.

    Checks:
    * At least one ``@@`` hunk header.
    * ``diff --git`` line (if present) targets *file_name*.
    * ``--- a/<file>`` and ``+++ b/<file>`` headers present.
    """
    lines = patch.splitlines()
    if not lines:
        raise ValueError("Patch is empty.")

    if not any(_is_hunk_header_str(ln) for ln in lines):
        raise ValueError("Patch must contain at least one unified diff hunk (@@ ... @@).")

    # diff --git header (optional but must match if present)
    allowed_diff_git = {
        f"diff --git a/{file_name} b/{file_name}",
        f'diff --git "a/{file_name}" "b/{file_name}"',
    }
    diff_git_lines = [ln for ln in lines if ln.startswith("diff --git ")]
    if diff_git_lines and any(ln not in allowed_diff_git for ln in diff_git_lines):
        raise ValueError(f"Patch must target only {file_name}.")

    # --- / +++ headers
    header_old = f"--- a/{file_name}"
    header_new = f"+++ b/{file_name}"
    if not any(ln == header_old or ln.startswith(header_old + "\t") for ln in lines):
        raise ValueError(f"Patch must include exact header: {header_old}")
    if not any(ln == header_new or ln.startswith(header_new + "\t") for ln in lines):
        raise ValueError(f"Patch must include exact header: {header_new}")


def normalize_patch(patch: str) -> str:
    """Strip and ensure the patch ends with a single newline."""
    normalized = patch.strip()
    if not normalized:
        return normalized
    return normalized + "\n"


def patch_targets(patch: str) -> set[str]:
    """Return all file paths mentioned in ``diff --git``, ``---``, ``+++`` lines."""
    targets: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                for candidate in (parts[2], parts[3]):
                    normalized = candidate.removeprefix("a/").removeprefix("b/")
                    if normalized != "/dev/null":
                        targets.add(normalized)
        elif line.startswith("+++ ") or line.startswith("--- "):
            candidate = line[4:].strip().split("\t", 1)[0]
            normalized = candidate.removeprefix("a/").removeprefix("b/")
            if normalized != "/dev/null":
                targets.add(normalized)
    return targets


# ---------------------------------------------------------------------------
# Hunk-header parsing
# ---------------------------------------------------------------------------

# Standard format: @@ -OLD_START[,OLD_COUNT] +NEW_START[,NEW_COUNT] @@[...]
_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@.*$"
)

# Relaxed: allow missing +NEW or other LLM oddities like "@@ -10 @@"
_HUNK_HEADER_RELAXED_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?(?:\s+\+(\d+)(?:,(\d+))?)?\s*@@.*$"
)


def _is_hunk_header_str(line: str) -> bool:
    """Return True if *line* looks like a hunk header."""
    s = line.rstrip("\n").rstrip()
    return s.startswith("@@ ") or s == "@@"


def _is_hunk_header(line: str) -> bool:
    """Return True if *line* (with possible trailing newline) is a hunk header."""
    return _is_hunk_header_str(line.rstrip("\n"))


def parse_hunk_header(line: str) -> tuple[int, int | None, int, int | None]:
    """Parse ``@@ -OLD,CNT +NEW,CNT @@`` into (old_start, old_count, new_start, new_count).

    Returns ``None`` for counts the LLM omitted.  Falls back gracefully for
    bare ``@@`` headers (returns ``(1, None, 1, None)``).
    """
    stripped = line.rstrip("\n").strip()

    # Try standard pattern first
    m = _HUNK_HEADER_RE.match(stripped)
    if m:
        return (
            int(m.group(1)),
            int(m.group(2)) if m.group(2) else 1,
            int(m.group(3)),
            int(m.group(4)) if m.group(4) else 1,
        )

    # Relaxed pattern (e.g. "@@ -10 @@" without + part)
    m = _HUNK_HEADER_RELAXED_RE.match(stripped)
    if m:
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) else None
        new_start = int(m.group(3)) if m.group(3) else old_start
        new_count = int(m.group(4)) if m.group(4) else None
        logger.debug(
            "Relaxed hunk header parsed (%s); new_start inferred as %d",
            stripped, new_start,
        )
        return old_start, old_count, new_start, new_count

    # Bare @@ or @@ ... @@ with no parseable numbers
    if stripped.startswith("@@"):
        logger.debug(
            "Bare hunk header detected (%s); inferring position from context",
            stripped,
        )
        return 1, None, 1, None

    raise ValueError(f"Invalid unified diff hunk header: {line.rstrip()}")


# ---------------------------------------------------------------------------
# Fuzzy matching helpers
# ---------------------------------------------------------------------------


def _normalize_whitespace(line: str) -> str:
    """Normalize a line for fuzzy comparison.

    LLMs frequently confuse tabs and spaces, add extra spaces, or change
    indentation when reproducing context lines.  This normalizer:

    * Expands tabs to 4-space equivalents.
    * Collapses runs of interior whitespace to a single space.
    * Strips trailing whitespace.
    """
    expanded = line.expandtabs(4).rstrip()
    # Collapse interior whitespace runs (preserve leading indent structure).
    # Split into leading whitespace + rest, then collapse only the rest.
    stripped = expanded.lstrip()
    if not stripped:
        return ""
    indent = expanded[: len(expanded) - len(stripped)]
    collapsed = re.sub(r"  +", " ", stripped)
    return indent + collapsed


def _lines_match(src: str, patch: str) -> bool:
    """Check if two lines match, tolerating whitespace differences.

    Tolerates:
    * Trailing whitespace differences
    * Tabs vs spaces (expanded to 4-space equivalents)
    * Extra interior spaces (``a  b`` matches ``a b``)
    """
    if src == patch:
        return True
    if src.rstrip() == patch.rstrip():
        return True
    return _normalize_whitespace(src) == _normalize_whitespace(patch)


def _extract_hunk_old_lines(
    patch_lines: list[str], hunk_start: int
) -> list[str]:
    """Extract the leading contiguous context/removal lines from a hunk.

    Only returns the prefix of old lines before the first addition line.
    This ensures the lines are contiguous in the source, which is required
    for fuzzy position matching.  Non-contiguous context (after additions)
    is handled later during line-by-line application.
    """
    old_lines: list[str] = []
    j = hunk_start
    seen_addition = False
    while j < len(patch_lines):
        line = patch_lines[j]
        # Stop at next hunk header or file-level marker.
        if _is_hunk_header(line):
            break
        if line.startswith(("diff --git ", "--- ", "+++ ")):
            break
        if not line:
            break
        if line.startswith("\\"):
            j += 1
            continue
        if line.startswith("+"):
            seen_addition = True
            j += 1
            continue
        marker = line[0]
        if marker in (" ", "-"):
            if seen_addition:
                # After an addition, context/removal lines may be
                # non-contiguous with the prefix.  Stop collecting.
                break
            old_lines.append(line[1:])
        j += 1
    return old_lines


def _find_hunk_offset(
    source_lines: list[str],
    old_lines: list[str],
    nominal_start: int,
    min_start: int,
    fuzz: int = 200,
) -> int | None:
    """Search near *nominal_start* for a position where *old_lines* match *source_lines*.

    Searches ±*fuzz* lines around the nominal position.  Returns ``None``
    if no match is found.
    """
    if not old_lines:
        return nominal_start

    for delta in range(fuzz + 1):
        for candidate in (nominal_start + delta, nominal_start - delta):
            if candidate < min_start or candidate + len(old_lines) > len(source_lines):
                continue
            if all(
                _lines_match(source_lines[candidate + k], old_lines[k])
                for k in range(len(old_lines))
            ):
                if candidate != nominal_start:
                    logger.debug(
                        "Fuzzy match: hunk shifted from line %d to %d (delta %+d)",
                        nominal_start + 1,
                        candidate + 1,
                        candidate - nominal_start,
                    )
                return candidate
    return None


# ---------------------------------------------------------------------------
# Core: apply a unified diff to source text
# ---------------------------------------------------------------------------

# Maximum number of lines to skip forward when a context or removal line
# does not match at the expected position.
_MAX_CONTEXT_SKIP = 500


def apply_unified_diff(original: str, patch: str) -> str:
    """Apply a unified diff (*patch*) to *original* and return the result.

    This implementation is deliberately tolerant of LLM mistakes:

    * Trailing whitespace differences between patch context and source.
    * Non-contiguous context lines (skips forward up to 500 lines).
    * Overlapping hunks (rewinds and replays already-emitted lines).
    * Bare ``@@`` headers with no line numbers.
    * Mismatched hunk line counts (warns but accepts).
    * Fuzzy hunk positioning with ±200 line search, expanding to full file.
    """
    source_lines = original.splitlines(keepends=True)
    patch_lines = normalize_patch(patch).splitlines(keepends=True)

    # Advance past preamble (diff --git, ---, +++) to the first hunk.
    start_index = 0
    while start_index < len(patch_lines) and not _is_hunk_header(patch_lines[start_index]):
        start_index += 1
    if start_index >= len(patch_lines):
        raise ValueError("Patch must contain at least one unified diff hunk (@@ ... @@).")

    result: list[str] = []
    source_index = 0
    i = start_index

    while i < len(patch_lines):
        header_line = patch_lines[i]
        if not _is_hunk_header(header_line):
            raise ValueError(
                f"Expected unified diff hunk header, got: {header_line.rstrip()}"
            )

        old_start, old_count, _new_start, new_count = parse_hunk_header(
            header_line.rstrip("\n")
        )

        # Determine nominal position in the source.
        if old_count is None:
            # Bare @@ — search forward from current position.
            nominal_index = source_index
        else:
            nominal_index = max(old_start - 1, 0)

        # Extract old lines from the hunk for fuzzy matching.
        old_lines = _extract_hunk_old_lines(patch_lines, i + 1)

        # --- Two-pass fuzzy search ---
        found_index = _find_hunk_offset(
            source_lines, old_lines, nominal_index, source_index
        )
        # Retry from the top of the file if needed (overlapping context).
        if found_index is None and source_index > 0:
            found_index = _find_hunk_offset(
                source_lines, old_lines, nominal_index, 0, fuzz=len(source_lines)
            )
        if found_index is None:
            raise ValueError(
                f"Patch context does not match the current in-memory file "
                f"(hunk at line {old_start}, searched ±200 lines). "
                f"First expected context line: {old_lines[0].rstrip() if old_lines else '<empty>'}. "
                f"Make sure the context lines in the diff match the actual file content."
            )
        target_index = found_index

        # Handle overlapping hunks by rewinding.
        if target_index < source_index:
            replayed = 0
            while source_index > target_index:
                source_index -= 1
                if result and result[-1] == source_lines[source_index]:
                    result.pop()
                    replayed += 1
            if replayed:
                logger.debug(
                    "Overlapping hunk: rewound %d line(s) to source line %d",
                    replayed, target_index + 1,
                )

        # Emit source lines between previous hunk end and this hunk start.
        result.extend(source_lines[source_index:target_index])
        source_index = target_index
        i += 1

        # Process hunk body lines.
        consumed_old = 0
        produced_new = 0

        while i < len(patch_lines) and not _is_hunk_header(patch_lines[i]):
            line = patch_lines[i]

            # Stop if we hit another file's markers.
            if line.startswith(("diff --git ", "--- ", "+++ ")):
                raise ValueError("Patch must modify only one file.")

            # Skip no-newline markers.
            if line.startswith("\\"):
                i += 1
                continue

            if not line:
                raise ValueError("Unexpected empty patch line.")

            marker = line[0]
            content = line[1:]

            if marker == " ":
                # Context line — must match source (with tolerance).
                if source_index >= len(source_lines):
                    raise ValueError(
                        "Patch context extends past end of file."
                    )

                if not _lines_match(source_lines[source_index], content):
                    # Try skipping forward (non-contiguous LLM context).
                    found = False
                    for skip in range(1, _MAX_CONTEXT_SKIP):
                        if source_index + skip >= len(source_lines):
                            break
                        if _lines_match(source_lines[source_index + skip], content):
                            result.extend(
                                source_lines[source_index : source_index + skip]
                            )
                            source_index += skip
                            found = True
                            logger.debug(
                                "Context skip: advanced %d line(s) to source line %d",
                                skip, source_index + 1,
                            )
                            break
                    if not found:
                        raise ValueError(
                            f"Patch context line does not match source at line {source_index + 1}. "
                            f"Expected: {content.rstrip()!r}, "
                            f"got: {source_lines[source_index].rstrip()!r}. "
                            f"Make sure context lines match the actual file."
                        )

                result.append(source_lines[source_index])  # keep original
                source_index += 1
                consumed_old += 1
                produced_new += 1

            elif marker == "-":
                # Removal line.
                if source_index >= len(source_lines):
                    raise ValueError(
                        "Patch removal extends past end of file."
                    )

                if not _lines_match(source_lines[source_index], content):
                    # Try skipping forward for removal lines too.
                    found = False
                    for skip in range(1, _MAX_CONTEXT_SKIP):
                        if source_index + skip >= len(source_lines):
                            break
                        if _lines_match(source_lines[source_index + skip], content):
                            result.extend(
                                source_lines[source_index : source_index + skip]
                            )
                            source_index += skip
                            found = True
                            break
                    if not found:
                        raise ValueError(
                            f"Patch removal line does not match source at line {source_index + 1}. "
                            f"Expected to remove: {content.rstrip()!r}, "
                            f"got: {source_lines[source_index].rstrip()!r}. "
                            f"Make sure removal lines (lines starting with -) match the actual file."
                        )

                source_index += 1
                consumed_old += 1

            elif marker == "+":
                result.append(content)
                produced_new += 1

            else:
                raise ValueError(
                    f"Invalid unified diff line marker '{marker}': {line.rstrip()}"
                )
            i += 1

        # Warn on count mismatches but accept.
        if old_count is not None and consumed_old != old_count:
            logger.debug(
                "Hunk header claimed %d old line(s) but consumed %d; accepting anyway",
                old_count, consumed_old,
            )
        if new_count is not None and produced_new != new_count:
            logger.debug(
                "Hunk header claimed %d new line(s) but produced %d; accepting anyway",
                new_count, produced_new,
            )

    # Emit remaining source lines.
    result.extend(source_lines[source_index:])
    return "".join(result)


# ---------------------------------------------------------------------------
# High-level: apply a patch to a named file (snapshot-style)
# ---------------------------------------------------------------------------


def apply_patch(original: str, patch: str, file_name: str) -> str:
    """Normalize, validate, and apply *patch* to *original* for *file_name*.

    Combines ``normalize_patch``, ``validate_single_file_diff``,
    ``patch_targets``, and ``apply_unified_diff`` into a single call.

    Raises ``ValueError`` with an actionable message on failure.
    """
    patch = normalize_patch(patch)
    validate_single_file_diff(patch, file_name)
    targets = patch_targets(patch)
    if not targets:
        raise ValueError("Patch does not target any file.")
    if targets != {file_name}:
        raise ValueError(
            f"Patch must modify only {file_name}; got: {', '.join(sorted(targets))}"
        )
    return apply_unified_diff(original, patch)


# ---------------------------------------------------------------------------
# Prompt helpers: instructions for the LLM on how to format diffs
# ---------------------------------------------------------------------------


def number_lines(code: str, every: int = 10) -> str:
    """Add line numbers to code for LLM context.

    Numbers every *every*-th line and the first line.  Unnumbered lines
    get a blank margin so the alignment stays consistent.  This helps the
    LLM produce correct ``@@ -LINE ... @@`` headers.
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


def diff_format_instructions(file_name: str) -> str:
    """Return concise instructions for LLMs to produce well-formed diffs."""
    return (
        f"\n\nResponse format:\n"
        f"1) A short summary sentence.\n"
        f"2) A single fenced ```diff block for {file_name}.\n"
        f"   Use `--- a/{file_name}` and `+++ b/{file_name}`.\n"
        f"   Each hunk MUST have a proper header: `@@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@`\n"
        f"   where OLD_START is the 1-based line number in the original file where the context begins,\n"
        f"   OLD_COUNT is the number of lines from the original (context + removed),\n"
        f"   NEW_START is the corresponding line in the new file, and\n"
        f"   NEW_COUNT is the number of lines in the new version (context + added).\n"
        f"   Include 3 lines of unchanged context before and after each change.\n"
        f"   Context lines (unchanged) start with a SPACE character.\n"
        f"   Removed lines start with `-`.\n"
        f"   Added lines start with `+`.\n"
        f"   Keep the patch small and focused.\n"
        f"   Do not rewrite the whole file.\n"
    )


def diff_format_example(file_name: str) -> str:
    """Return a concrete diff example for the LLM."""
    return (
        f"Example of a correct diff:\n"
        f"```diff\n"
        f"--- a/{file_name}\n"
        f"+++ b/{file_name}\n"
        f"@@ -10,7 +10,7 @@\n"
        f" // unchanged context line 1\n"
        f" // unchanged context line 2\n"
        f" // unchanged context line 3\n"
        f"-    old_code_here();\n"
        f"+    new_code_here();\n"
        f" // unchanged context line 4\n"
        f" // unchanged context line 5\n"
        f" // unchanged context line 6\n"
        f"```\n"
    )


def _extract_error_line(error_message: str) -> int | None:
    """Try to extract a line number from a diff application error message."""
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


def diff_error_feedback(file_name: str, code: str, error_message: str) -> str:
    """Build feedback for the LLM when a diff failed to parse or apply."""
    # If the error references a specific line, show a snippet around it
    # so the LLM can see what lines actually exist in the file.
    context_section = ""
    line_no = _extract_error_line(error_message)
    if line_no:
        snippet = _source_context_snippet(code, line_no)
        context_section = (
            f"\nLines around the error location (line {line_no}):\n"
            f"```\n{snippet}\n```\n"
        )

    return (
        f"The previous diff could not be applied.\n\n"
        f"Error: {error_message}\n{context_section}\n"
        f"Please re-emit the fix as a proper unified diff that applies cleanly "
        f"to the current version of {file_name}.\n\n"
        f"IMPORTANT RULES:\n"
        f"- Context lines (starting with a space) MUST be copied EXACTLY from the "
        f"current file, character for character.\n"
        f"- Removal lines (starting with -) MUST exactly match lines in the current file.\n"
        f"- Include 3 lines of unchanged context before and after each change.\n"
        f"- Use correct line numbers: @@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@\n"
        f"- Do NOT include lines that are not in the file.\n"
        f"- Return a SINGLE fenced ```diff block with all hunks.\n\n"
        f"Current {file_name} (with line numbers):\n"
        f"```c\n{number_lines(code)}\n```\n\n"
        f"{diff_format_example(file_name)}"
    )
