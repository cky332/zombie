"""DeepSeek-V3.2 client via SiliconFlow's OpenAI-compatible API.

Adapted from the user's snippet:
- API key read from environment variable SILICONFLOW_API_KEY (never hardcoded).
- Adds `json_mode` flag so callers can require JSON output (used by the agent loop).
- Adds `call_deepseek_with_messages` for multi-turn calls (Verbal Reflection).
"""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "deepseek-ai/DeepSeek-V3.2"


def _get_client() -> OpenAI:
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is not set. Export it before running, e.g.\n"
            "  export SILICONFLOW_API_KEY=sk-xxxxxxxx"
        )
    return OpenAI(api_key=key, base_url=BASE_URL)


def _is_rate_limit(err_text: str) -> bool:
    low = err_text.lower()
    return (
        "429" in err_text
        or "503" in err_text
        or any(k in low for k in ["rate", "quota", "overload", "busy", "capacity", "limit"])
        or "饱和" in err_text
        or "繁忙" in err_text
    )


def call_deepseek_with_messages(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.5,
    max_tokens: int = 4000,
    top_p: float = 1.0,
    model: str = MODEL,
    max_retries: int = 60,
    json_mode: bool = False,
    verbose_retry: bool = False,
) -> tuple[str, dict[str, int]]:
    """Generic multi-turn call. Returns (content, usage)."""

    client = _get_client()
    retries = 0
    err = ""
    while retries < max_retries:
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            content = msg.content or ""
            usage = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }
            return content, usage
        except Exception as e:
            err = str(e)
            wait = min(120, 30 + retries * 10) if _is_rate_limit(err) else min(60, 2 ** min(retries, 6))
            if verbose_retry:
                print(f"[deepseek] error: {err[:160]}; retry {retries + 1}/{max_retries} after {wait}s")
            time.sleep(wait)
            retries += 1
    raise RuntimeError(f"Max retries ({max_retries}) reached. Last error: {err}")


def call_deepseek(
    prompt: str,
    system: str = "You are a helpful assistant.",
    *,
    temperature: float = 0.5,
    max_tokens: int = 4000,
    top_p: float = 1.0,
    model: str = MODEL,
    max_retries: int = 60,
    json_mode: bool = False,
    verbose_retry: bool = False,
) -> tuple[str, dict[str, int]]:
    """Single-turn helper preserving the original snippet's signature."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return call_deepseek_with_messages(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        model=model,
        max_retries=max_retries,
        json_mode=json_mode,
        verbose_retry=verbose_retry,
    )


if __name__ == "__main__":
    content, usage = call_deepseek("What is 2 + 2? Answer briefly.")
    print(content)
    print(usage)
