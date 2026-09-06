#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenRouter embeddings client for ``nvidia/nemotron-3-embed-1b:free``.

Uses the same async ``httpx`` style as ``services/meralion_speech_translation.py``
rather than the synchronous ``requests``-based snippet on OpenRouter's model page,
since embedding calls happen inside the live Pipecat event loop at query time (in
addition to the offline ``scripts/build_index.py`` batch run).

Note on prefixes: unlike ``intfloat/multilingual-e5-small`` (the model originally
suggested in STREAMLINED_PIPELINE_CONTEXT.md), nemotron-3-embed-1b's asymmetric
query/passage handling is not documented on OpenRouter's model page as of this
writing. ``query_prefix``/``passage_prefix`` default to empty strings; set them
via env vars if evaluation shows retrieval quality improves with explicit
prefixes (see STREAMLINED_PIPELINE_CONTEXT.md section 15.2 evaluation plan).
"""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

_MAX_RETRY_DELAY_SECONDS = 5.0


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """Backoff before retrying: honor a 429's Retry-After header, else exponential."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after = exc.response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(float(retry_after), _MAX_RETRY_DELAY_SECONDS)
            except ValueError:
                pass
        return _MAX_RETRY_DELAY_SECONDS
    return min(0.5 * (2**attempt), _MAX_RETRY_DELAY_SECONDS)


class OpenRouterEmbeddingClient:
    """Async client for OpenRouter's OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "nvidia/nemotron-3-embed-1b:free",
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        request_timeout: float = 60.0,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = max(1, batch_size)
        self._client = httpx.AsyncClient(timeout=request_timeout)

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed whole-scheme documents (one embedding per scheme)."""
        prefixed = [f"{self._passage_prefix}{text}" for text in texts]
        return await self._embed_batched(prefixed)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single user query."""
        vectors = await self._embed_batched([f"{self._query_prefix}{text}"])
        return vectors[0]

    async def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            results.extend(await self._embed_request(batch))
        return results

    async def _embed_request(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }

        last_exc: Exception | None = None
        for attempt in range(3):  # transient failures + one 429 backoff-and-retry
            try:
                response = await self._client.post(
                    f"{self._base_url}/embeddings", json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()
                items = data.get("data")
                if not isinstance(items, list) or len(items) != len(texts):
                    raise ValueError(
                        f"Unexpected embeddings response shape: {data!r}"[:500]
                    )
                # Preserve request order via each item's "index" field, per the
                # OpenAI embeddings response schema.
                ordered = sorted(items, key=lambda item: item.get("index", 0))
                return [item["embedding"] for item in ordered]
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last_exc = exc
                logger.warning(
                    "OpenRouter embeddings request failed (attempt {}/3): {}",
                    attempt + 1,
                    exc,
                )
                delay = _retry_delay_seconds(exc, attempt)
                if delay:
                    await asyncio.sleep(delay)

        raise RuntimeError(f"OpenRouter embeddings request failed: {last_exc}") from last_exc
