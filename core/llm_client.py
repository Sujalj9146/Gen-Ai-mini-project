"""Thin Gemini REST client.

Uses the free Google AI Studio REST API so the project needs no extra SDK.
Only stdlib + urllib are used, which keeps `requirements.txt` small.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "gemini-3.8-flash"
_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# Models to try in order when the requested one is unavailable or overloaded.
# Names this account cannot use return 404 and are skipped immediately.
_FALLBACK_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.8-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
)


class LLMError(RuntimeError):
    """Raised when the model call cannot be completed."""


_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
_KEY_NAME = "GEMINI_API_KEY"


def _read_env_lines() -> list[str]:
    if not os.path.exists(_ENV_PATH):
        return []
    with open(_ENV_PATH, encoding="utf-8", errors="ignore") as handle:
        return handle.read().splitlines()


def saved_key_from_env_file() -> str:
    """Return the key stored in the project's .env file, or ''."""
    for line in _read_env_lines():
        line = line.strip()
        if line.startswith(f"{_KEY_NAME}="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value and value != "your-api-key-here":
                return value
    return ""


def load_api_key(explicit: str | None = None) -> str:
    """Read the API key from the argument, the environment, or .env."""
    if explicit and explicit.strip():
        return explicit.strip()
    env_value = os.environ.get(_KEY_NAME, "").strip()
    if env_value:
        return env_value
    return saved_key_from_env_file()


def save_api_key(api_key: str) -> None:
    """Store the key in the local .env file (gitignored) so it is entered once."""
    api_key = api_key.strip()
    lines = [ln for ln in _read_env_lines() if not ln.strip().startswith(f"{_KEY_NAME}=")]
    lines.append(f"{_KEY_NAME}={api_key}")
    with open(_ENV_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def forget_api_key() -> None:
    """Remove the saved key from the local .env file."""
    lines = [ln for ln in _read_env_lines() if not ln.strip().startswith(f"{_KEY_NAME}=")]
    if lines:
        with open(_ENV_PATH, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    elif os.path.exists(_ENV_PATH):
        os.remove(_ENV_PATH)


def _post(model: str, api_key: str, payload: dict, timeout: int) -> dict:
    """POST to the Gemini generateContent endpoint and return parsed JSON."""
    # The key goes in a header rather than the URL so it never shows up in
    # proxy logs or error messages.
    url = f"{_API_ROOT}/{model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_text(response: dict) -> str:
    """Pull the first text block out of a Gemini response."""
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                return text.strip()
    raise LLMError("The model returned an empty response.")


def generate(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    timeout: int = 60,
    retries: int = 3,
    json_mode: bool = False,
) -> str:
    """Send a prompt to Gemini and return the raw text response.

    Retries transient failures (429 rate limit, 500/503 overloaded, timeouts)
    with a backoff, and falls back to other flash models if the requested one
    is unavailable or busy. With ``json_mode`` the API is asked to return a
    JSON document instead of free text.
    """
    if not api_key:
        raise LLMError("No Gemini API key found. Add one in the sidebar or in .env.")

    generation_config: dict = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    candidates = [model] + [m for m in _FALLBACK_MODELS if m != model]
    last_error: Exception | None = None

    for attempt in range(retries):
        for candidate_model in candidates:
            try:
                print(f"[gemini] attempt {attempt + 1}: {candidate_model}", flush=True)
                response = _post(candidate_model, api_key, payload, timeout)
                return _extract_text(response)
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="ignore")[:200]
                except Exception:  # noqa: BLE001
                    pass
                print(f"[gemini] {candidate_model} -> HTTP {exc.code}", flush=True)

                # 404 means this model is not available to the account.
                # 429/503 are transient - try the next fallback model now,
                # and retry the whole list after a pause if all are busy.
                if exc.code == 404:
                    last_error = LLMError(f"Model {candidate_model} unavailable: {body}")
                    continue
                if exc.code in (429, 500, 503):
                    last_error = LLMError(f"Gemini returned {exc.code}: {body}")
                    continue
                raise LLMError(f"Gemini request failed ({exc.code}): {body}") from exc
            except Exception as exc:  # noqa: BLE001
                # Timeouts and connection errors: try the next model as well.
                print(f"[gemini] {candidate_model} -> {type(exc).__name__}: {exc}", flush=True)
                last_error = LLMError(f"Network error while calling Gemini: {exc}")
                continue

        if attempt < retries - 1:
            time.sleep(2 * 2 ** attempt)

    raise last_error or LLMError("Gemini call failed for an unknown reason.")
