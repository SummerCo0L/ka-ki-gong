#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenRouter rerank client (``POST /rerank``), a cross-encoder second stage over
hybrid dense+BM25 candidates: retrieve top_k cheaply, then rerank to the few
actually shown to the LLM. See eval/README or run
``scripts/evaluate_retrieval.py --reranker-model ...`` to measure the lift
before wiring this into the live bot (it adds one serial API round trip per
turn, which matters for the voice pipeline's latency budget).
"""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from retrieval.embeddings import _retry_delay_seconds


class OpenRouterRerankClient:
    """Async client for OpenRouter's ``/rerank`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "cohere/rerank-v3.5",
        request_timeout: float = 30.0,
        max_attempts: int = 3,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_attempts = max(1, max_attempts)
        self._client = httpx.AsyncClient(timeout=request_timeout)

    @property
    def model(self) -> str:
        return self._model

    async def rerank(self, query: str, documents: list[str], *, top_n: int | None = None) -> list[tuple[int, float]]:
        """Return ``(original_index, relevance_score)`` pairs, sorted by relevance desc."""
        if not documents:
            return []
        payload = {"model": self._model, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n

        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._client.post(
                    f"{self._base_url}/rerank",
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                results = response.json()["results"]
                return [(r["index"], r["relevance_score"]) for r in results]
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt + 1 >= self._max_attempts:
                    break
                delay = _retry_delay_seconds(exc, attempt)
                logger.warning("Rerank request failed (attempt {}), retrying in {:.1f}s: {}", attempt + 1, delay, exc)
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def aclose(self) -> None:
        await self._client.aclose()
