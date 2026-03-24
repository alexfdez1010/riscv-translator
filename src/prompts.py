"""Prompt construction for the generic SSE→RISC-V translation pipeline.

All prompts are library-agnostic: they guide the LLM to port x86 SSE/SSE2
code to RISC-V using the sse2rvv.h drop-in compatibility header, and to fix
compiler errors based on feedback — without any hardcoded, library-specific fixes.
"""

from src.config import REFERENCE_FILE, RVV_REFERENCE
from src.search_replace import search_replace_error_feedback, search_replace_format_example
from src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM prompt builders
# ---------------------------------------------------------------------------


def build_system_prompt(target_file: str) -> str:
    """Build the system prompt for SSE→sse2rvv translation/repair."""
    return f"""\
You are an expert systems programmer specialising in SIMD portability.
Your task is to incrementally repair C/C++ code that uses x86 SSE/SSE2
intrinsics so that it compiles and runs correctly on RISC-V using the
**sse2rvv.h** drop-in compatibility header.

## sse2rvv.h overview

`sse2rvv.h` is a header-only translation layer that re-implements SSE/SSE2
intrinsics using RISC-V Vector (RVV) instructions.  It is analogous to
`sse2neon.h` for ARM NEON.

- Include `"sse2rvv.h"` instead of any x86 SSE headers.
- All standard SSE types (`__m128i`, `__m128`, `__m128d`) and intrinsic
  functions (`_mm_*`) are provided by sse2rvv.h — no API changes needed.
- The existing SSE code should compile with minimal modifications once the
  include is swapped.

## Translation strategy

Because sse2rvv.h is a drop-in replacement, the translation requires only
**minor, localised changes**:

1. Replace x86 SSE `#include` directives (`<emmintrin.h>`, `<xmmintrin.h>`,
   `<smmintrin.h>`, `<immintrin.h>`, etc.) with `#include "sse2rvv.h"`.
2. Remove or guard any `#ifdef __SSE2__` / `#ifdef __SSE__` preprocessor
   conditionals that would disable the SIMD code paths on non-x86 targets.
3. Fix any remaining compiler errors — these are typically minor issues such
   as missing includes, type mismatches, or platform-specific assumptions.

## CRITICAL: Keep changes minimal

- Do NOT rewrite SSE intrinsics into a different API — sse2rvv.h already
  provides them.
- Do NOT change algorithmic logic, data structures, or function signatures.
- Only touch code that the compiler actually complains about.
- Each search/replace block should be as small as possible — a few lines at
  most.  Prefer many small blocks over one large block.

## Rules

- Modify only `{target_file}`.
- Focus on the specific compiler error(s) shown in the feedback.
- Make the smallest correct change that fixes each error.
- The repair loop will call you again with updated code and remaining errors.
- Preserve existing style unless the fix requires otherwise.
- Do not change unrelated code.
- Do not invent APIs, functions, files, or build steps.
- Do not rewrite unchanged lines just to restyle them.

## Output format

Return only:
1) A short summary sentence.
2) One or more search/replace blocks.

Each block has this exact format:

<<<<<<< SEARCH
exact lines from the current file to find
=======
replacement lines
>>>>>>> REPLACE

{search_replace_format_example()}

## Edit requirements

- The SEARCH section must be copied EXACTLY from the current file
  (same indentation, same whitespace, character for character).
- The REPLACE section must be DIFFERENT from SEARCH — every block must
  actually change something.  Never emit a block where search == replace.
- If the same text appears in multiple places and you want to change ALL
  of them, a single block is enough — all occurrences will be replaced.
- You may use multiple search/replace blocks for changes in different
  parts of the file.
- Do NOT modify unrelated code.
- Keep changes focused and small — only fix what the compiler reports.

## RISC-V Vector (RVV) reference

Reference material from: {REFERENCE_FILE}

{RVV_REFERENCE}
""".strip()


def build_initial_translation_prompt(
    target_file: str,
    source_code: str,
    build_command: str,
    validation_feedback: str | None = None,
) -> str:
    """Build the first user prompt that asks the LLM to translate or fix the code."""
    validation_section = ""
    if validation_feedback:
        validation_section = f"""

Current validation failure:
{validation_feedback}
""".rstrip()

    return f"""\
Task: Fix this file so it compiles and runs on RISC-V using sse2rvv.h.

Goal:
Make the smallest correct change so the project compiles and runs successfully.

Context:
- Target file: {target_file}
- Build & validation command: {build_command}

What to change:
- SSE headers have already been replaced with `#include "sse2rvv.h"` automatically.
- Fix compiler errors shown in the validation feedback below.
- Keep changes minimal and localised — sse2rvv.h provides all SSE intrinsics.

What not to change:
- Do not modify any file other than {target_file}.
- Do not refactor unrelated logic.
- Do not introduce new dependencies or files.
- Do not rewrite SSE intrinsics — sse2rvv.h handles them.

Current code:
```cpp
{source_code}
```
{validation_section}

Output:
- First, one short summary sentence.
- Then, one or more search/replace blocks with the changes.
{search_replace_format_example()}
""".strip()


def build_repair_prompt(
    target_file: str,
    code: str,
    validation_feedback: str,
) -> str:
    """Build follow-up prompts when previous attempts still have errors."""
    return f"""\
Task: Fix the current validation failure in this file.

Goal:
Make the smallest correct change needed so the project moves closer to passing
the RISC-V build/runtime validation.

Context:
- Target file: {target_file}
- The code uses sse2rvv.h as a drop-in SSE→RISC-V compatibility layer.

What to change:
- Address the failure indicated below.
- Prefer the smallest possible patch that fixes the reported failure.
- Do NOT rewrite SSE intrinsics — sse2rvv.h already provides them.

What not to change:
- Do not modify any file other than {target_file}.
- Do not refactor unrelated code.

Current code:
```cpp
{code}
```

Validation failure details:
{validation_feedback}

Output:
- First, one short summary sentence.
- Then, one or more search/replace blocks with the changes.
{search_replace_format_example()}
""".strip()


def build_edit_format_feedback(
    file_name: str, code: str, error_message: str
) -> str:
    return search_replace_error_feedback(file_name, code, error_message)
