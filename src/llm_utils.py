import os
import json
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.llm_types import LLM, Message, llm_fn

from src.config import (
    LLM_BASE_URL,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_REASONING_EFFORT,
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


def _wait_for_endpoint(base_url: str, timeout_seconds: float = 10.0) -> bool:
    health_url = f"{base_url.rstrip('/')}/models"
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        req = Request(health_url, method="GET")
        try:
            with urlopen(req, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return True
        except (HTTPError, URLError, TimeoutError, OSError):
            time.sleep(0.5)

    return False


def _ensure_ssh_tunnel(base_url: str) -> bool:
    try:
        subprocess.Popen(
            ["ssh", "spark"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False

    return _wait_for_endpoint(base_url)


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
    endpoint = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    ssh_tunnel_attempted = False
    using_openrouter = False

    @llm_fn
    def _inference_llm(messages: list[Message]) -> str:
        nonlocal ssh_tunnel_attempted, using_openrouter
        payload = {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": LLM_TEMPERATURE,
            "max_completion_tokens": LLM_MAX_COMPLETION_TOKENS,
        }

        # If we already fell back to OpenRouter in a previous call, keep using it.
        if using_openrouter:
            return _extract_content(_call_openrouter(payload))

        req = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (URLError, TimeoutError, OSError) as exc:
            if not ssh_tunnel_attempted:
                ssh_tunnel_attempted = True
                if _ensure_ssh_tunnel(LLM_BASE_URL):
                    try:
                        with urlopen(req, timeout=600) as resp:
                            body = json.loads(resp.read().decode("utf-8"))
                        return _extract_content(body)
                    except (URLError, TimeoutError, OSError):
                        pass  # fall through to OpenRouter

            # Fallback to OpenRouter if the key is available.
            if OPENROUTER_API_KEY:
                logger.info(
                    "Local/SSH endpoint unavailable; falling back to OpenRouter (%s)",
                    OPENROUTER_MODEL,
                )
                using_openrouter = True
                return _extract_content(_call_openrouter(payload))

            raise exc

        return _extract_content(body)

    return _inference_llm
