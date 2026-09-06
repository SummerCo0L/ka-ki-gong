#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Formatting helpers for the deterministic per-turn retrieval evidence block.

Retrieval runs deterministically before every LLM call (see
``bots/bot.py``'s ``_update_retrieval_evidence``) rather than via LLM tool
calling, so the grounded-answer call never depends on gpt-oss-120b
successfully emitting a tool call.
"""

from __future__ import annotations

import re

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
