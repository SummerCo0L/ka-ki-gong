#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local hybrid retrieval over the curated senior-support knowledge base.

No vector database is used: embeddings and BM25 statistics are precomputed by
``scripts/build_index.py`` and loaded into memory at bot startup (see
``retrieval.index.SchemeIndex``).
"""
