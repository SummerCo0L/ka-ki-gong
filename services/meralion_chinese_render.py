#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MERaLiON-3-10B English -> Singapore Chinese rendering client.

This model is used only for
the narrow transformation: a verified grounded English answer -> conversational
Singapore Chinese written in Chinese characters. It must never be asked to
answer the user's question again or introduce new facts.

Talks to the MERaLiON chat-completions endpoint
(https://api.meralion.ai/v1/chat/completions), the same host used by
MERaLiONSpeechTranslationService.
"""

import httpx
from loguru import logger


class MERaLiONChineseRenderClient:
    """Thin async HTTP client for the MERaLiON-3-10B Singapore Chinese rendering step."""

    def __init__(
        self,
        *,
        api_key: str,
        prompt_template: str,
        base_url: str = "https://api.meralion.ai",
        model: str = "MERaLiON/MERaLiON-3-10B",
        temperature: float = 0.2,
        request_timeout: float = 60.0,
    ):
        """Initialize the Singapore Chinese render client.

        Args:
            api_key: MERaLiON API key.
            prompt_template: Full renderer prompt containing a ``{{answer_en}}``
                placeholder (see prompt/english_to_singapore-chinese.md).
            base_url: MERaLiON API base URL.
            model: MERaLiON rendering model name.
            temperature: Sampling temperature for the rendering call.
            request_timeout: HTTP request timeout in seconds.
        """
        self._api_key = api_key
        self._prompt_template = prompt_template
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._client = httpx.AsyncClient(timeout=request_timeout)

    async def render(self, english_text: str) -> str:
        """Render a verified English answer as Singapore Chinese.

        Args:
            english_text: The complete, buffered grounded English answer.

        Returns:
            The rendered Singapore Chinese text, or an empty string if the
            model returned no usable content.
        """
        if not english_text.strip():
            return ""

        prompt = self._prompt_template.replace("{{answer_en}}", english_text)

        response = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
            },
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise ValueError(f"Unexpected MERaLiON render response schema: {data}")

        content = choices[0]["message"]["content"]
        if isinstance(content, list):
            # Defensive: some OpenAI-compatible backends may return content parts.
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

        text = (content or "").strip()
        if not text:
            logger.warning("MERaLiON Singapore Chinese renderer returned empty content")
        return text

    async def aclose(self):
        """Release the underlying HTTP client."""
        await self._client.aclose()
