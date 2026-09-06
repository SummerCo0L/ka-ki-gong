#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""In-memory hybrid (dense + lexical) index over ``knowledge_base/``.

Each scheme is embedded once, as its whole Markdown page (per
STREAMLINED_PIPELINE_CONTEXT.md section 3.3: "Embed one complete structured
record per short scheme"). The same scheme commonly appears under several
``knowledge_base/<category>/schemes.json`` files (SupportGoWhere tags one
scheme under multiple topics), so records are deduplicated by ``id``, keeping
only the first occurrence.

No vector database: embeddings are a single in-memory NumPy matrix, built
offline by ``scripts/build_index.py`` and loaded at bot startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

from retrieval.embeddings import OpenRouterEmbeddingClient
from retrieval.lexical import BM25Index
from retrieval.schemas import RetrievedScheme, SchemeRecord

EMBEDDINGS_FILENAME = "embeddings.npy"
SCHEMES_FILENAME = "schemes.json"
MANIFEST_FILENAME = "manifest.json"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _fallback_embedding_text(entry: dict) -> str:
    """Reconstruct a Markdown-ish text from schemes.json when no md file exists."""
    lines = [entry.get("title", "").strip(), entry.get("description", "").strip()]
    for highlight in entry.get("highlights") or []:
        lines.append(f"- {highlight}")
    for heading, body in (entry.get("sections") or {}).items():
        lines.append(f"## {heading}\n{body}")
    return "\n".join(line for line in lines if line).strip()


def _find_md_text(md_dir: Path, scheme_id: str, entry: dict) -> str:
    if md_dir.is_dir():
        for path in md_dir.glob(f"*_{scheme_id}.md"):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    return _fallback_embedding_text(entry)


def _with_topic_context(record: SchemeRecord) -> str:
    """Prepend the scheme's own scraped ``domains`` tags to embedding_text.

    This topic signal (e.g. "Financial support and benefits") previously only
    lived in SchemeRecord metadata, unused by dense/lexical scoring. Baking it
    into the text that actually gets embedded and BM25-indexed lets a query
    like "caregiving" or "financial support" match on topic terms, not just on
    words appearing in the scheme's own prose.
    """
    if not record.domains:
        return record.embedding_text
    return f"Domains: {', '.join(record.domains)}" + "\n\n" + record.embedding_text


def load_records(kb_dir: Path) -> list[SchemeRecord]:
    """Load and deduplicate scheme records from every ``knowledge_base/<category>/``."""
    records: dict[str, SchemeRecord] = {}

    for category_dir in sorted(p for p in kb_dir.iterdir() if p.is_dir()):
        schemes_path = category_dir / SCHEMES_FILENAME
        if not schemes_path.exists():
            continue

        md_dir = category_dir / "md"
        entries = json.loads(schemes_path.read_text(encoding="utf-8"))

        for entry in entries:
            scheme_id = entry.get("id")
            if not scheme_id:
                continue

            if scheme_id in records:
                continue

            records[scheme_id] = SchemeRecord(
                scheme_id=scheme_id,
                title=(entry.get("title") or "").strip(),
                description=(entry.get("description") or "").strip(),
                domains=entry.get("domains") or [],
                profiles=entry.get("profiles") or [],
                highlights=entry.get("highlights") or [],
                source_url=entry.get("url", ""),
                source_updated_at=entry.get("info_last_updated"),
                embedding_text=_find_md_text(md_dir, scheme_id, entry),
                sections=entry.get("sections") or {},
            )

    for record in records.values():
        record.embedding_text = _with_topic_context(record)

    return list(records.values())


class SchemeIndex:
    """Hybrid dense+lexical retrieval over the deduplicated scheme corpus."""

    def __init__(
        self, records: list[SchemeRecord], embeddings: np.ndarray, manifest: dict
    ):
        self._records = records
        self._embeddings = embeddings
        self._manifest = manifest
        self._bm25 = BM25Index([r.embedding_text or r.description for r in records])

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def scheme_count(self) -> int:
        return len(self._records)

    @classmethod
    async def build(
        cls, kb_dir: Path, embed_client: OpenRouterEmbeddingClient
    ) -> "SchemeIndex":
        records = load_records(kb_dir)
        if not records:
            raise ValueError(f"No scheme records found under {kb_dir}")

        logger.info("Embedding {} scheme records via OpenRouter...", len(records))
        texts = [r.embedding_text or r.description for r in records]
        vectors = await embed_client.embed_passages(texts)
        embeddings = _l2_normalize(np.array(vectors, dtype=np.float32))

        manifest = {
            "embedding_model": embed_client.model,
            "dim": int(embeddings.shape[1]) if embeddings.size else 0,
            "scheme_count": len(records),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return cls(records, embeddings, manifest)

    def save(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / EMBEDDINGS_FILENAME, self._embeddings)
        (data_dir / SCHEMES_FILENAME).write_text(
            json.dumps([r.to_json() for r in self._records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / MANIFEST_FILENAME).write_text(
            json.dumps(self._manifest, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, data_dir: Path) -> "SchemeIndex":
        embeddings = np.load(data_dir / EMBEDDINGS_FILENAME)
        records = [
            SchemeRecord.from_json(d)
            for d in json.loads((data_dir / SCHEMES_FILENAME).read_text(encoding="utf-8"))
        ]
        manifest = json.loads((data_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        return cls(records, embeddings, manifest)

    def search(
        self,
        query_text: str,
        query_embedding: list[float],
        *,
        top_k: int = 3,
        min_score: float = 0.0,
        category: str | None = None,
        alpha: float = 0.7,
    ) -> list[RetrievedScheme]:
        """Hybrid search: ``hybrid_score = alpha * dense_cosine + (1 - alpha) * lexical``."""
        if not self._records:
            return []

        query_vec = _l2_normalize(np.array([query_embedding], dtype=np.float32))[0]
        dense_scores = self._embeddings @ query_vec
        lexical_scores = self._bm25.score_normalized(query_text)
        hybrid_scores = alpha * dense_scores + (1 - alpha) * lexical_scores

        def matches_category(record: SchemeRecord) -> bool:
            if not category:
                return True
            needle = category.strip().lower().replace("_", " ")
            haystack = " ".join(record.domains).lower()
            return needle in haystack

        candidates = [
            RetrievedScheme(
                scheme=record,
                dense_score=float(dense_scores[i]),
                lexical_score=float(lexical_scores[i]),
                hybrid_score=float(hybrid_scores[i]),
            )
            for i, record in enumerate(self._records)
            if matches_category(record) and float(hybrid_scores[i]) >= min_score
        ]
        candidates.sort(key=lambda c: c.hybrid_score, reverse=True)
        return candidates[:top_k]
