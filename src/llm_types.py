"""Local LLM protocol types.

Provides the same interface as ``llm_evolution.ai.interfaces.llm`` so
that the rest of the codebase can import from here without requiring
the external package.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    content: str


@runtime_checkable
class LLM(Protocol):
    """Protocol for Large Language Models."""

    def __call__(self, messages: list[Message]) -> str: ...


def llm_fn(fn):
    """Decorator to convert a function into an LLM protocol implementation."""

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, messages: list[Message]) -> str:
            return self.func(messages)

    return Wrapper(fn)
