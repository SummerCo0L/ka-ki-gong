#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Build the local hybrid retrieval index from ``knowledge_base/``.

Embeds one complete Markdown page per scheme (not chunked) via OpenRouter's
``openai/text-embedding-3-small`` embedding model, then saves the result to
``data/`` for the bot to load at startup without needing network access.

Usage::

    uv run scripts/build_index.py

Reruns are cheap: this script always re-embeds every scheme from scratch, since
the corpus is small (a few hundred schemes) and OpenRouter's ``:free`` model
tier has no cost.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from retrieval.embeddings import OpenRouterEmbeddingClient  # noqa: E402
from retrieval.index import SchemeIndex  # noqa: E402

load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)


async def main() -> None:
    kb_dir = _PROJECT_ROOT / os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
    data_dir = _PROJECT_ROOT / os.getenv("RETRIEVAL_DATA_DIR", "data")

    api_key = os.environ["OPENROUTER_API_KEY"]
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    query_prefix = os.getenv("OPENROUTER_EMBED_QUERY_PREFIX", "")
    passage_prefix = os.getenv("OPENROUTER_EMBED_PASSAGE_PREFIX", "")

    if (data_dir / "embeddings.npy").exists():
        logger.warning(
            "Existing embeddings found at {} — they will be overwritten.",
            data_dir / "embeddings.npy",
        )

    logger.info("Building retrieval index from {} using {}", kb_dir, model)

    embed_client = OpenRouterEmbeddingClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    try:
        index = await SchemeIndex.build(kb_dir, embed_client)
    finally:
        await embed_client.aclose()

    index.save(data_dir)
    logger.info(
        "Saved {} scheme embeddings (dim={}) to {}",
        index.scheme_count,
        index.manifest.get("dim"),
        data_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
