"""Integration test for OpenAI-compatible inference over the SSH tunnel."""

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
from src.llm_types import Message

from src.config import LLM_BASE_URL
from src.llm_utils import create_llm


def _inference_endpoint_reachable() -> bool:
    """Best-effort check to avoid hard failing when tunnel/server is down."""
    health_url = f"{LLM_BASE_URL.rstrip('/')}/models"
    req = Request(health_url, method="GET")

    try:
        with urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 500
    except HTTPError as exc:
        # Server responded (e.g. 401, 403) — endpoint is reachable
        return 400 <= exc.code < 500
    except (URLError, TimeoutError, OSError):
        return False
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _inference_endpoint_reachable(),
    reason="Local OpenAI-compatible endpoint is not reachable. Start SSH tunnel first.",
)


class TestSSHTunnelLLM:
    def test_chat_completion_via_openai_client(self):
        """Use the exact same production code-path used by the app LLM calls."""
        llm = create_llm()
        response = llm(
            [
                Message(
                    role="system",
                    content="Eres un asistente de investigación impecable y conciso.",
                ),
                Message(
                    role="user",
                    content="¿Por qué es tan crítico gestionar bien la VRAM en modelos de 120B de parámetros?",
                ),
            ]
        )

        assert isinstance(response, str)
        print(response)
