#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Evaluate local hybrid retrieval quality against a small hand-labeled query set.

Computes standard IR metrics (Recall@k, MRR) against a labeled eval set (default:
eval/retrieval_eval_set.json), so retrieval quality can be tracked as you tune
RETRIEVAL_TOP_K / RETRIEVAL_HYBRID_ALPHA / RETRIEVAL_MIN_SCORE or swap the
embedding model. Also supports an opt-in LLM-as-judge pass that scores the
top-3 results' relevance for each query (useful as a supplementary signal, not
a substitute for the labeled Recall/MRR metrics above).

Usage::

    uv run scripts/evaluate_retrieval.py
    uv run scripts/evaluate_retrieval.py --alpha 0.5 --top-k 5
    uv run scripts/evaluate_retrieval.py --llm-judge

    # Evaluate the eval set in daily batches, accumulating into one report file.
    # Each run only evaluates queries not already present in the resume file, then
    # recomputes summary metrics over the full accumulated set:
    uv run scripts/evaluate_retrieval.py --resume eval/results/incremental.json --limit 50
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from retrieval.embeddings import OpenRouterEmbeddingClient  # noqa: E402
from retrieval.index import SchemeIndex  # noqa: E402
from retrieval.reranker import OpenRouterRerankClient  # noqa: E402

load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

DEFAULT_EVAL_SET = _PROJECT_ROOT / "eval" / "retrieval_eval_set.json"
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"
RECALL_KS = (1, 3, 5, 10)


class _RateLimiter:
    """Sliding-window limiter: blocks until fewer than max_calls occurred in the last period_seconds."""

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self._max_calls = max_calls
        self._period = period_seconds
        self._timestamps: deque[float] = deque()

    async def acquire(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= self._period:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max_calls:
            sleep_for = self._period - (now - self._timestamps[0])
            if sleep_for > 0:
                logger.info("Rate limit: sleeping {:.1f}s to stay under {}/{:.0f}s", sleep_for, self._max_calls, self._period)
                await asyncio.sleep(sleep_for)
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] >= self._period:
                self._timestamps.popleft()
        self._timestamps.append(time.monotonic())


def load_eval_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_summary(rows: list[dict], recall_ks: tuple[int, ...]) -> dict:
    """Compute Recall@k / MRR over a list of per-query result rows.

    Reads each row's already-computed ``rank_of_first_match`` rather than
    re-deriving it, so this can be applied to a merged set of rows spanning
    multiple evaluation runs (see ``--resume``).
    """
    n = len(rows)
    hits_at_k = {k: 0 for k in recall_ks}
    reciprocal_ranks: list[float] = []
    for row in rows:
        rank = row.get("rank_of_first_match")
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in recall_ks:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
    return {
        "num_queries": n,
        **{f"recall@{k}": hits_at_k[k] / n for k in recall_ks},
        "mrr": sum(reciprocal_ranks) / n,
    }


async def judge_relevance_llm(
    *, api_key: str, base_url: str, model: str, query: str, candidates: list[tuple[str, str]]
) -> list[int]:
    """Ask an LLM to rate each candidate's relevance to ``query`` on a 0-2 scale.

    ``candidates`` is a list of (title, description) pairs, same order as
    retrieved. Returns one int per candidate (0=not relevant, 1=somewhat,
    2=highly relevant); entries that fail to parse are -1.
    """
    listing = "\n".join(f"{i + 1}. {title}: {desc}" for i, (title, desc) in enumerate(candidates))
    prompt = (
        "You are judging search relevance for a Singapore senior-support scheme "
        "retrieval system. Given the user's need and a numbered list of candidate "
        "schemes, rate each candidate's relevance to the need on this scale:\n"
        "0 = not relevant, 1 = somewhat relevant, 2 = highly relevant / directly answers the need.\n\n"
        f"User need: {query}\n\nCandidates:\n{listing}\n\n"
        "Respond with ONLY a JSON array of integers, one per candidate, in order. "
        "Example: [2, 0, 1]"
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

    try:
        # Models sometimes wrap the array in prose or a code fence; grab the
        # first '[' ... ']' span rather than requiring an exact-match response.
        start, end = content.index("["), content.rindex("]") + 1
        scores = json.loads(content[start:end])
        return [int(s) for s in scores]
    except (ValueError, json.JSONDecodeError):
        logger.warning("Could not parse LLM-judge response, skipping: {}", content[:200])
        return [-1] * len(candidates)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--top-k", type=int, default=10, help="Max results fetched per query")
    parser.add_argument(
        "--alpha", type=float, default=None, help="Override RETRIEVAL_HYBRID_ALPHA for this run"
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Also score top-3 relevance via LLM-as-judge (costs one chat completion per query)",
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL") or "cohere/rerank-4-fast",
        help=(
            "OpenRouter rerank model slug (e.g. cohere/rerank-4-fast, voyageai/rerank-2.5-lite). "
            "Retrieves --rerank-candidates hybrid results, reranks them, and truncates to --top-k "
            "before computing metrics. Defaults to RERANKER_MODEL from .env, or cohere/rerank-4-fast. "
            "Pass 'none' to disable reranking and evaluate hybrid search alone."
        ),
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=15,
        help="Number of hybrid-search candidates fetched per query before reranking (ignored without --reranker-model)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only evaluate the first N not-yet-evaluated queries this run (useful for a quick "
            "smoke test, or for processing the eval set in daily batches with --resume)"
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many not-yet-evaluated queries before applying --limit (manual batching control)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Path to an existing report JSON to accumulate into. Queries already present there "
            "(matched by exact query text) are skipped this run; newly evaluated queries are "
            "merged in and summary metrics are recomputed over the full accumulated set. Also "
            "used as the save path unless --output is given explicitly. If the file doesn't exist "
            "yet, starts a fresh accumulation there."
        ),
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=None,
        help=(
            "Throttle combined embed+rerank API calls to at most this many per rolling 60s window "
            "(e.g. 20 to respect OpenRouter's :free-tier cap, which is shared across embed and rerank calls). "
            "Omit for no throttling (default)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to save the full report (per-query results + summary metrics) as JSON. "
            f"Defaults to a timestamped file under {DEFAULT_RESULTS_DIR.relative_to(_PROJECT_ROOT)}/. "
            "Pass 'none' to skip saving."
        ),
    )
    args = parser.parse_args()
    if args.reranker_model and args.reranker_model.strip().lower() == "none":
        args.reranker_model = None

    data_dir = _PROJECT_ROOT / os.getenv("RETRIEVAL_DATA_DIR", "data")
    api_key = os.environ["OPENROUTER_API_KEY"]
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    embedding_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    judge_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
    alpha = args.alpha if args.alpha is not None else float(os.getenv("RETRIEVAL_HYBRID_ALPHA", "0.7"))

    index = SchemeIndex.load(data_dir)
    embed_client = OpenRouterEmbeddingClient(api_key=api_key, base_url=base_url, model=embedding_model)
    rerank_client = (
        OpenRouterRerankClient(api_key=api_key, base_url=base_url, model=args.reranker_model)
        if args.reranker_model
        else None
    )
    rate_limiter = _RateLimiter(args.max_rpm) if args.max_rpm else None

    all_eval_items = load_eval_set(args.eval_set)

    existing_rows_by_query: dict[str, dict] = {}
    if args.resume is not None and args.resume.exists():
        existing_report = json.loads(args.resume.read_text(encoding="utf-8"))
        existing_rows_by_query = {row["query"]: row for row in existing_report.get("per_query", [])}
        logger.info("Resuming from {}: {} queries already evaluated", args.resume, len(existing_rows_by_query))

    # only evaluate queries not already present in the resume file, then slice by offset/limit
    eval_items = [item for item in all_eval_items if item["query"] not in existing_rows_by_query]
    eval_items = eval_items[args.offset :]
    if args.limit is not None:
        eval_items = eval_items[: args.limit]
    logger.info(
        "Evaluating {} new queries from {} ({} already done, {} remaining after this run)",
        len(eval_items),
        args.eval_set,
        len(existing_rows_by_query),
        len(all_eval_items) - len(existing_rows_by_query) - len(eval_items),
    )

    per_query_rows: list[dict] = []

    for item in eval_items:
        query = item["query"]
        expected = set(item.get("expected_scheme_ids", []))

        if rate_limiter:
            await rate_limiter.acquire()
        query_embedding = await embed_client.embed_query(query)
        fetch_k = max(args.top_k, args.rerank_candidates) if rerank_client else args.top_k
        results = index.search(query, query_embedding, top_k=fetch_k, alpha=alpha)

        rerank_scores: list[float] | None = None
        if rerank_client:
            documents = [r.scheme.embedding_text or r.scheme.description for r in results]
            if rate_limiter:
                await rate_limiter.acquire()
            reranked = await rerank_client.rerank(query, documents, top_n=args.top_k)
            results = [results[i] for i, _ in reranked]
            rerank_scores = [round(score, 3) for _, score in reranked]
        else:
            results = results[: args.top_k]

        retrieved_ids = [r.scheme.scheme_id for r in results]

        rank = next((i + 1 for i, sid in enumerate(retrieved_ids) if sid in expected), None)

        row = {
            "query": query,
            "expected": sorted(expected),
            "top_results": [
                (r.scheme.scheme_id, r.scheme.title, round(r.hybrid_score, 3)) for r in results[:5]
            ],
            "rank_of_first_match": rank,
        }
        if rerank_scores is not None:
            row["rerank_scores"] = rerank_scores[:5]

        if args.llm_judge:
            candidates = [(r.scheme.title, r.scheme.description) for r in results[:3]]
            row["llm_judge_scores"] = await judge_relevance_llm(
                api_key=api_key, base_url=base_url, model=judge_model, query=query, candidates=candidates
            )

        per_query_rows.append(row)

    await embed_client.aclose()
    if rerank_client:
        await rerank_client.aclose()

    reranker_suffix = f", reranker={args.reranker_model}" if args.reranker_model else ""
    print(
        f"\n=== Retrieval evaluation ({len(eval_items)} new queries, alpha={alpha}, "
        f"top_k={args.top_k}{reranker_suffix}) ===\n"
    )
    if args.top_k < max(RECALL_KS):
        # results are truncated to top_k before rank is computed, so recall@k
        # for k > top_k has nothing beyond rank top_k to count and is not meaningful.
        print(f"Note: --top-k {args.top_k} caps rank; Recall@k for k > {args.top_k} will equal Recall@{args.top_k}.\n")
    for row in per_query_rows:
        status = "HIT" if row["rank_of_first_match"] else "MISS"
        print(f"[{status}] rank={row['rank_of_first_match']}  {row['query']}")
        rerank_scores = row.get("rerank_scores")
        for i, (sid, title, score) in enumerate(row["top_results"]):
            marker = "*" if sid in row["expected"] else " "
            score_str = f"{score:.3f}" if rerank_scores is None else f"{score:.3f} rerank={rerank_scores[i]:.3f}"
            print(f"    {marker} {sid:20s} {score_str}  {title}")
        if args.llm_judge:
            print(f"    LLM-judge scores (top-3): {row['llm_judge_scores']}")
        print()

    # merge this run's rows with any prior --resume rows, keeping the eval set's original order
    merged_rows_by_query = {**existing_rows_by_query, **{row["query"]: row for row in per_query_rows}}
    combined_rows = [
        merged_rows_by_query[item["query"]] for item in all_eval_items if item["query"] in merged_rows_by_query
    ]
    summary = compute_summary(combined_rows, RECALL_KS)

    print(f"--- Summary (cumulative over {summary['num_queries']}/{len(all_eval_items)} queries) ---")
    for k in RECALL_KS:
        print(f"Recall@{k}: {summary[f'recall@{k}']:.2%}")
    print(f"MRR: {summary['mrr']:.3f}")

    if args.output is None or str(args.output).lower() != "none":
        output_path = args.output if args.output is not None else args.resume
        if output_path is None:
            DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = DEFAULT_RESULTS_DIR / f"retrieval_eval_{timestamp}.json"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "eval_set": str(args.eval_set),
                "top_k": args.top_k,
                "alpha": alpha,
                "embedding_model": embedding_model,
                "reranker_model": args.reranker_model,
                "rerank_candidates": args.rerank_candidates if args.reranker_model else None,
                "max_rpm": args.max_rpm,
                "llm_judge": args.llm_judge,
                "llm_judge_model": judge_model if args.llm_judge else None,
            },
            "summary": summary,
            "per_query": combined_rows,
        }
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved full report to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
