#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Data structures shared by the retrieval package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchemeRecord:
    """One scheme, embedded as a single whole-document passage.

    ``scheme_id`` is deduplicated across ``knowledge_base/<category>/`` folders:
    the same scheme (e.g. MediFund) commonly appears under several categories on
    SupportGoWhere; only the first occurrence's record is kept.
    """

    scheme_id: str
    title: str
    description: str
    domains: list[str] = field(default_factory=list)
    profiles: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    source_url: str = ""
    source_updated_at: str | None = None
    embedding_text: str = ""
    """Full Markdown content that was embedded (one embedding per scheme)."""
    sections: dict[str, str] = field(default_factory=dict)
    """Raw label -> Markdown body, as scraped (e.g. "How to apply?", "Phone number")."""

    def to_json(self) -> dict:
        return {
            "scheme_id": self.scheme_id,
            "title": self.title,
            "description": self.description,
            "domains": self.domains,
            "profiles": self.profiles,
            "highlights": self.highlights,
            "source_url": self.source_url,
            "source_updated_at": self.source_updated_at,
            "embedding_text": self.embedding_text,
            "sections": self.sections,
        }

    @classmethod
    def from_json(cls, data: dict) -> "SchemeRecord":
        return cls(
            scheme_id=data["scheme_id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            domains=data.get("domains", []),
            profiles=data.get("profiles", []),
            highlights=data.get("highlights", []),
            source_url=data.get("source_url", ""),
            source_updated_at=data.get("source_updated_at"),
            embedding_text=data.get("embedding_text", ""),
            sections=data.get("sections") or {},
        )


@dataclass
class RetrievedScheme:
    """A scheme returned by hybrid search, with its component scores."""

    scheme: SchemeRecord
    dense_score: float
    lexical_score: float
    hybrid_score: float
