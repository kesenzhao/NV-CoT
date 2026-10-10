"""Small OpenAI-chat-compatible adapter for a Gemini ``generateContent`` API.

The reward code historically talks to a local vLLM server through
``client.chat.completions.create``.  This adapter deliberately exposes only
that narrow surface so the existing prompts, response parsing, and reward
formula remain unchanged when the remote Gemini backend is selected.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from urllib.parse import quote, urlparse

import httpx


class GeminiAPIError(RuntimeError):
    """Raised when the remote judge returns an unusable response."""


def _message_text(message) -> str:
    """Extract text while preserving the caller's prompt verbatim."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            else:
                raise TypeError("Gemini judge supports text message parts only")
        return "\n".join(text_parts)
    raise TypeError(f"unsupported judge message content: {type(content)!r}")


def _messages_to_gemini(messages):
    """Convert OpenAI text messages to native Gemini message fields."""
    contents = []
    system_parts = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        text = _message_text(message)
        if role == "system":
            system_parts.append({"text": text})
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported judge message role: {role!r}")
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            }
        )
    if not contents:
        raise ValueError("Gemini judge requires at least one non-system message")
    return contents, system_parts


def _response_text(payload: dict) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as error:
        raise GeminiAPIError("Gemini judge response has no candidate content") from error
    text = "".join(
        part.get("text", "")
        for part in parts
        if (
            isinstance(part, dict)
            and not part.get("thought", False)
            and isinstance(part.get("text", ""), str)
        )
    ).strip()
    if not text:
        raise GeminiAPIError("Gemini judge response contains no text")
    return text


class _GeminiCompletions:
    def __init__(self, owner: "GeminiJudgeClient") -> None:
        self._owner = owner

    def create(self, *, model: str, messages, temperature=None, **kwargs):
        seed = kwargs.pop("seed", None)
        request_timeout = kwargs.pop("timeout", None)
        # The compatibility surface intentionally ignores other OpenAI-only options.
        return self._owner.create_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            request_timeout=request_timeout,
        )


class _GeminiChat:
    def __init__(self, owner: "GeminiJudgeClient") -> None:
        self.completions = _GeminiCompletions(owner)


class GeminiJudgeClient:
    """Call a bearer-authenticated Gemini ``generateContent`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        max_output_tokens: int = 128,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("JUDGE_API_KEY is required for the Gemini judge backend")
        base_url = base_url.rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Gemini judge base URL must be an absolute HTTPS URL")
        if max_retries < 0:
            raise ValueError("Gemini judge max_retries must be non-negative")
        if max_output_tokens < 1:
            raise ValueError("Gemini judge max_output_tokens must be positive")

        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._client = http_client or httpx.Client(timeout=timeout, trust_env=False)
        self.chat = _GeminiChat(self)

    def create_completion(
        self,
        *,
        model: str,
        messages,
        temperature=None,
        seed=None,
        request_timeout=None,
    ):
        if not model:
            raise ValueError("Gemini judge model must not be empty")
        safe_model = quote(model, safe="-._")
        url = f"{self._base_url}/models/{safe_model}:generateContent"
        generation_config = {
            "candidateCount": 1,
            "maxOutputTokens": self._max_output_tokens,
        }
        if temperature is not None:
            generation_config["temperature"] = float(temperature)
        if seed is not None:
            generation_config["seed"] = int(seed)
        contents, system_parts = _messages_to_gemini(messages)
        body = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        request_kwargs = {"headers": headers, "json": body}
        if request_timeout is not None:
            request_kwargs["timeout"] = request_timeout

        response = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(url, **request_kwargs)
                response.raise_for_status()
                text = _response_text(response.json())
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
                )
            except (httpx.HTTPError, ValueError, GeminiAPIError) as error:
                retryable = (
                    isinstance(error, httpx.TransportError)
                    or (
                        isinstance(error, httpx.HTTPStatusError)
                        and error.response.status_code
                        in {408, 409, 429, 500, 502, 503, 504}
                    )
                )
                if attempt >= self._max_retries or not retryable:
                    status = (
                        response.status_code if response is not None else "unavailable"
                    )
                    raise GeminiAPIError(
                        f"Gemini judge request failed after {attempt + 1} attempt(s); "
                        f"status={status}: {error}"
                    ) from error
                time.sleep(min(2**attempt, 8))
