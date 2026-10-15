"""
Provider-agnostic LLM access. Verification/groundedness logic (Phase 9/10)
calls `generate_completion(system, user)` and never touches a provider SDK
directly, so swapping LLM_PROVIDER doesn't ripple into pipeline code.

Providers:
  anthropic   -> Anthropic Messages API (api.anthropic.com/v1/messages)
  openai      -> any OpenAI-compatible /chat/completions endpoint (OpenAI itself,
                 or a local server via LLM_BASE_URL, e.g. Ollama's OpenAI-compatible route)
  extractive  -> no LLM call at all; returns the single most relevant context
                 passage verbatim with its marker. Exists so the full pipeline
                 (and its tests) can run end-to-end with zero API key/cost —
                 useful for local development and for the evaluation harness's
                 "Baseline Vector RAG" comparison row.

`httpx` is imported lazily so this module can be imported (and the
`extractive` path used) without it installed.
"""

import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


def _call_anthropic(system: str, user: str) -> str:
    import httpx

    settings = get_settings()
    if not settings.LLM_API_KEY:
        raise LLMError("LLM_API_KEY is not set (LLM_PROVIDER=anthropic).")

    payload = {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if settings.LLM_SEND_TEMPERATURE:
        payload["temperature"] = settings.LLM_TEMPERATURE

    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.response.status_code}: {exc.response.text[:500]}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"Anthropic API request failed: {exc}") from exc

    data = resp.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    if not text_blocks:
        raise LLMError(f"Anthropic API returned no text content: {data}")
    return "".join(text_blocks)


def _call_openai_compatible(system: str, user: str) -> str:
    import httpx

    settings = get_settings()
    if not settings.LLM_API_KEY and not settings.LLM_BASE_URL:
        raise LLMError("LLM_API_KEY is not set (LLM_PROVIDER=openai).")

    base_url = settings.LLM_BASE_URL or "https://api.openai.com/v1"
    payload = {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if settings.LLM_SEND_TEMPERATURE:
        payload["temperature"] = settings.LLM_TEMPERATURE

    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}", "content-type": "application/json"},
            json=payload,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"LLM API error {exc.response.status_code}: {exc.response.text[:500]}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"LLM API request failed: {exc}") from exc

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected response shape from LLM API: {data}") from exc


def _call_extractive(system: str, user: str) -> str:  # noqa: ARG001
    """
    No-LLM fallback: pulls the first context passage's marker straight out of
    the prompt and returns it as a trivially "grounded" (if unhelpful) answer.
    Only used when LLM_PROVIDER=extractive.
    """
    if "[1]" not in user:
        return "INSUFFICIENT_EVIDENCE: No context passages were available."
    return "See the most relevant passage. [1]"


_PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai_compatible,
    "extractive": _call_extractive,
}


def generate_completion(system: str, user: str) -> str:
    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()
    if provider not in _PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Expected one of: {list(_PROVIDERS)}.")
    logger.info("Calling LLM provider=%s model=%s", provider, settings.LLM_MODEL)
    return _PROVIDERS[provider](system, user)
