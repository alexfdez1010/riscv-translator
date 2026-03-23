"""Prompt construction for the generic SSE→Highway RISC-V translation pipeline.

All prompts are library-agnostic: they guide the LLM to translate x86 SSE
intrinsics to Google Highway SIMD code targeting RISC-V, and to fix compiler
errors based on feedback — without any hardcoded, library-specific fixes.
"""

from src.config import REFERENCE_FILE, RVV_REFERENCE
from src.diff_utils import diff_error_feedback, diff_format_example
from src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM prompt builders
# ---------------------------------------------------------------------------


def build_system_prompt(target_file: str) -> str:
    """Build the system prompt for SSE→Highway translation/repair."""
    return f"""\
You are an expert systems programmer specialising in SIMD portability.
Your task is to incrementally translate and repair C/C++ code that uses
x86 SSE/SSE2 intrinsics so that it compiles and runs correctly on RISC-V
using the Google Highway SIMD library.

## Google Highway quick reference

Highway provides portable SIMD via C++ templates:
- Include `"hwy/highway.h"` and use `HWY_NAMESPACE` / `HWY_BEFORE_NAMESPACE` / `HWY_AFTER_NAMESPACE`.
- Use the foreach_target pattern (`hwy/foreach_target.h`) for multi-target dispatch.
- Key types: `hn::ScalableTag<T>` (runtime-width) or `hn::FixedTag<T, N>` (fixed lanes).
- Operations: `hn::Load`, `hn::Store`, `hn::Add`, `hn::Sub`, `hn::Max`, `hn::Min`,
  `hn::ShiftRight`, `hn::ShiftLeft`, `hn::Set`, `hn::Zero`, `hn::BitCast`, etc.
- For 128-bit SSE-equivalent vectors use `hn::FixedTag<uint8_t, 16>` (16 × u8 lanes).
- Replace `__m128i` with `hn::Vec<hn::FixedTag<T, N>>` or use `auto`.
- Replace `_mm_*` intrinsics with the corresponding Highway `hn::*` functions.
- Memory: `hn::Load(tag, ptr)`, `hn::Store(vec, tag, ptr)`.
- Profile/buffer data should use flat typed arrays (`uint8_t*`, `int16_t*`)
  instead of `__m128i*` since Highway vectors may be sizeless.
- Compile with C++17 and `-I<highway_root>`.

## Translation strategy

1. Map each SSE intrinsic to its Highway equivalent.
2. Replace `__m128i` types with Highway vector types or `auto`.
3. Replace `__m128i*` pointers with typed pointers (`uint8_t*`, `int16_t*`, etc.)
   and use `hn::Load` / `hn::Store` with the appropriate tag.
4. sizeof(__m128i) should be replaced with `hn::Lanes(tag) * sizeof(element_type)`.
5. Preserve the algorithmic logic exactly — only change the SIMD layer.

## Rules

- Modify only `{target_file}`.
- Fix ONE error or ONE function per response — keep changes small and incremental.
- Address the FIRST error shown in the compiler output.
- The repair loop will call you again with updated code and remaining errors.
- Preserve existing style unless the fix requires otherwise.
- Do not change unrelated code.
- Do not invent APIs, functions, files, or build steps.
- Prefer the smallest possible diff that fixes the current failure.
- Do not rewrite unchanged lines just to restyle them.

## Output format

Return only:
1) A short summary sentence.
2) A single-file unified git diff patch in a fenced ```diff block for {target_file}.

{diff_format_example(target_file)}

## Patch requirements

- Use `--- a/{target_file}` and `+++ b/{target_file}`.
- Each hunk MUST have a proper header: `@@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@`
- Context lines (unchanged) MUST start with a SPACE and match the actual file exactly.
- Removed lines start with `-`, added lines start with `+`.
- Include 3 lines of unchanged context before and after each change.
- Do not return full-file rewrites.
- Keep the total patch under 3000 characters.

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
Task: Translate/repair this file so it compiles and runs on RISC-V using Google Highway.

Goal:
Make the smallest correct change so the project compiles and runs successfully.

Context:
- Target file: {target_file}
- Build & validation command: {build_command}

What to change:
- Replace x86 SSE intrinsics with Google Highway equivalents.
- Fix compiler errors shown in the validation feedback below.
- Keep the repair incremental and localised.

What not to change:
- Do not modify any file other than {target_file}.
- Do not refactor unrelated logic.
- Do not introduce new dependencies or files.

Current code:
```cpp
{source_code}
```
{validation_section}

Output:
- First, one short summary sentence.
- Then, a single fenced `diff` block containing only the unified diff for `{target_file}`.
{diff_format_example(target_file)}
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
- The code has already been partially translated to Google Highway.

What to change:
- Address the failure indicated below.
- Prefer the smallest possible patch hunk that fixes the reported failure.

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
- Then, a single fenced `diff` block containing only the unified diff for `{target_file}`.
{diff_format_example(target_file)}
""".strip()


def build_diff_format_feedback(
    file_name: str, code: str, error_message: str
) -> str:
    return diff_error_feedback(file_name, code, error_message)
