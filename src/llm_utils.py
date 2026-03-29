import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.llm_types import LLM, Message, llm_fn

from src.config import (
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------


def _call_openrouter(payload: dict, max_retries: int = 5) -> dict:
    """Send a request to the OpenRouter API with retry on 429."""
    endpoint = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    payload = {**payload, "model": OPENROUTER_MODEL}
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries + 1):
        req = Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                wait = 2 ** attempt * 15
                logger.info("OpenRouter 429; retrying in %ds (%d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("OpenRouter request failed after retries")


def _extract_content(body: dict) -> str:
    """Extract the assistant message content from an OpenAI-compatible response."""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def create_llm() -> LLM:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Set it in .env or as an environment variable."
        )

    @llm_fn
    def _inference_llm(messages: list[Message]) -> str:
        payload = {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": LLM_TEMPERATURE,
        }
        return _extract_content(_call_openrouter(payload))

    return _inference_llm
