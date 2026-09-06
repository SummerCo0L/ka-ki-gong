#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Formatting helpers and the optional follow-up ``search_senior_support`` tool.

Per STREAMLINED_PIPELINE_CONTEXT.md section 8:

- Initial-turn policy: retrieval runs deterministically (no LLM/tool call).
- Follow-up policy: the LLM *may* call ``search_senior_support`` again if it
  decides it needs new evidence.

See ``bots/bot.py`` for why the follow-up tool is gated behind
``ENABLE_LLM_RETRIEVAL_TOOL`` (default off): Pipecat's own CHANGELOG documents
a broken-streaming-tool-call bug specific to ``openai/gpt-oss-120b`` through at
least one OpenRouter upstream (Nebius) as of v1.4.0, and a related class of
"tool_call id/name split across deltas" bugs affecting other OpenAI-compatible
providers (pipecat-ai/pipecat#4987). OpenRouter can route this model to
several different backing providers, and Pipecat has no way to know in
advance which one will serve a given request, so this failure mode cannot be
ruled out reliably. The deterministic per-turn retrieval below does not
depend on tool calling at all and is the safe default.
"""

from __future__ import annotations

import re

from retrieval.index import SchemeIndex
from retrieval.schemas import RetrievedScheme

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _strip_markdown_links(text: str) -> str:
    """Drop [label](url) -> label; evidence is read aloud, not clicked."""
    return _MD_LINK_RE.sub(r"\1", text)


def format_evidence_block(results: list[RetrievedScheme]) -> str:
    """Render retrieved schemes as evidence for the grounded-answer LLM turn.

    Includes every scraped section (not just phone/how-to-apply/where-to-go) so
    the LLM can weigh eligibility criteria, full descriptions, etc. when
    deciding which of the top-k schemes to surface — the Output Contract in
    the prompt template is what keeps the spoken reply itself short and voice
    appropriate, not the size of this evidence block.
    """
    if not results:
        return "No verified scheme evidence was found for this turn."

    lines = []
    for i, result in enumerate(results, start=1):
        scheme = result.scheme
        lines.append(f"{i}. {scheme.title} (relevance={result.hybrid_score:.2f})")
        if scheme.description:
            lines.append(f"   Summary: {scheme.description}")
        for highlight in scheme.highlights:
            lines.append(f"   - {highlight}")

        for heading, body in scheme.sections.items():
            if body:
                lines.append(f"   {heading}: {_strip_markdown_links(body).strip()}")

        updated = scheme.source_updated_at or "unknown"
        lines.append(f"   Source: {scheme.source_url} (last updated: {updated})")
    return "\n".join(lines)


def format_tool_result(results: list[RetrievedScheme]) -> dict:
    """Shape retrieved schemes into the tool-result JSON from section 8 of the spec."""
    return {
        "results": [
            {
                "scheme_id": result.scheme.scheme_id,
                "title": result.scheme.title,
                "relevant_passage": result.scheme.description,
                "source_url": result.scheme.source_url,
                "source_updated_at": result.scheme.source_updated_at,
                "retrieval_score": round(result.hybrid_score, 4),
            }
            for result in results
        ],
        "retrieval_confident": bool(results) and results[0].hybrid_score >= 0.5,
    }


def make_search_senior_support_tool(
    index: SchemeIndex,
    embed_client,
    *,
    top_k: int = 3,
    min_score: float = 0.0,
    alpha: float = 0.7,
):
    """Build the direct-function tool handler for follow-up retrieval.

    Returns an async function suitable for ``LLMContext(tools=[...])`` — Pipecat
    derives the tool's JSON schema from this function's signature and docstring
    (see the "direct functions" pattern in Pipecat's function-calling docs).
    """

    async def search_senior_support(params, query: str, category: str | None = None):
        """Search verified Singapore senior-support scheme records.

        Args:
            query: A concise English description of the user's support need.
            category: Optional category such as healthcare, caregiving, mobility,
                financial_support, or long_term_care.
        """
        query_embedding = await embed_client.embed_query(query)
        results = index.search(
            query,
            query_embedding,
            top_k=top_k,
            min_score=min_score,
            category=category,
            alpha=alpha,
        )
        await params.result_callback(format_tool_result(results))

    return search_senior_support
