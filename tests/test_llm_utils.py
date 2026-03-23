"""Unit tests for llm_utils.py."""

from src.llm_utils import create_llm


def test_create_llm_returns_callable():
    """Smoke test that create_llm returns something callable."""
    llm = create_llm()
    assert callable(llm)
