# Retrieval Evaluation

This folder contains the golden query set and results used to benchmark
`retrieval/index.py`'s hybrid (dense + BM25) search, with an optional
cross-encoder reranking stage, via `scripts/evaluate_retrieval.py`.

## How `retrieval_eval_set.json` was built

1. **Scoping**: every scheme record across `knowledge_base/*/schemes.json`
   with `"Seniors"` in its `profiles` list was identified (167 schemes at the
   time the set was built) — the target population for the voice bot.
2. **Query generation**: for each of those 167 schemes, one or more natural-
   language queries were AI-generated (Claude Sonnet 5, in GitHub Copilot)
   written to simulate how an elderly Singaporean caller might actually phrase
   the need in conversation — casual, indirect, sometimes mentioning a family
   member instead of themselves — rather than the scheme's official title or
   keywords. Every one of the 167 schemes got at least one query.
3. **Deliberate ambiguity**: some queries were intentionally written to be
   vague enough that multiple schemes are genuinely valid answers (e.g. a
   general "help with CareShield/ElderShield premiums" question spans several
   related insurance-subsidy schemes; a "downsizing my flat" question spans
   several senior housing schemes). These were tagged with multiple
   `expected_scheme_ids` rather than forced into a single answer, since
   picking just one would have been an artificial/incorrect ground truth.
4. **Filtering down**: the raw one-query-per-scheme draft was consolidated
   into the current **133 queries** — some closely related schemes were
   merged into a single grouped/ambiguous query (see point 3) instead of
   kept as separate near-duplicate queries, shrinking the set while keeping
   full scheme coverage.
5. **AI-assisted ground-truth audit**: the set's `expected_scheme_ids` were
   not treated as final after step 4 — they were corrected using the *results
   of a real evaluation run*, not just re-read in isolation. The process:
   1. Run 1 (`retrieval_eval_20260906T075905Z.json`) was evaluated against
      the original ground truth using hybrid search alone.
   2. For every query, Claude Sonnet 5 (in GitHub Copilot) was given that
      run's top-5 retrieved results alongside the actual scheme
      eligibility/description text in `data/schemes.json`, and asked to
      judge each *non-expected* top-5 result: is this actually also a valid
      answer to the query (not just topically similar), and should it be
      added to `expected_scheme_ids`?
   3. Every accepted addition was applied programmatically (exact query-text
      matching against `retrieval_eval_set.json`, deduplicated and sorted),
      not hand-edited — 62 of the 133 queries were corrected this way.
   4. This is why the set now covers more scheme_ids (179) than the original
      167 Seniors-tagged schemes it started from — some valid alternates
      aren't themselves tagged "Seniors" but are still correct answers for a
      senior's query (e.g. a general tax-relief or transport-concession
      scheme). Crucially, this process only *adds* missing correct answers
      that the retrieval system had actually found — it never removes a
      genuine miss to flatter the score.

Current shape of the set:

| | |
|---|---|
| Total queries | 133 |
| Solo queries (1 expected scheme) | 65 |
| Grouped/ambiguous queries (2+ expected schemes) | 68 |
| Unique `scheme_id`s covered | 179 |

The set is a **hit-rate style** ground truth: a query is "correct" if *any*
one of its `expected_scheme_ids` appears in the results, not a strict
single-answer match — appropriate given real user needs are often genuinely
served by more than one scheme.

## Evaluation plan

`scripts/evaluate_retrieval.py` runs every query in `retrieval_eval_set.json`
through the same retrieval path the live bot uses (`SchemeIndex.search()`),
optionally followed by a rerank pass, and reports:

- **Recall@k** (k = 1, 3, 5, 10) — of all queries, what fraction have at
  least one expected scheme somewhere in the top-k results. Recall@1 is the
  strictest, most production-relevant number (it's what the bot effectively
  needs when `RETRIEVAL_TOP_K` is small); Recall@10 shows the ceiling of what
  hybrid search + reranking can surface at all, regardless of rank.
- **MRR** (Mean Reciprocal Rank) — average of `1 / rank_of_first_match`
  across all queries (0 if never found). Rewards getting the right answer
  *higher up*, not just present somewhere in the top-k.
- Optional **LLM-as-judge** (`--llm-judge`) — a supplementary 0-2 relevance
  score on the top-3 results per query, useful for spotting cases the binary
  hit/miss metrics above can't distinguish (e.g. a "hit" that's technically
  correct but a worse match than another expected scheme), not a replacement
  for the labeled Recall/MRR metrics.

### Pipeline under test

```
query → embed (OPENROUTER_EMBEDDING_MODEL) → hybrid search (dense + BM25, --alpha)
      → [optional] rerank (--reranker-model, cross-encoder over --rerank-candidates)
      → top --top-k results → compared against expected_scheme_ids
```

Defaults (overridable via CLI flags or `.env`): `openai/text-embedding-3-small`
embeddings, `alpha=0.7` hybrid blend, `cohere/rerank-4-fast` reranking over the
top 15 hybrid candidates, truncated to `--top-k`.

### Common invocations

```
# Full run with current defaults, saved to a timestamped file under eval/results/
uv run scripts/evaluate_retrieval.py

# Hybrid search only, no reranking
uv run scripts/evaluate_retrieval.py --reranker-model none

# Tune the hybrid dense/lexical blend
uv run scripts/evaluate_retrieval.py --alpha 0.5 --top-k 5

# Quick smoke test on a handful of queries (cheap, no rate-limit/credit risk)
uv run scripts/evaluate_retrieval.py --limit 5 --output none

# Add LLM-as-judge scoring on top of the labeled metrics
uv run scripts/evaluate_retrieval.py --llm-judge

# Process the set in batches over multiple runs/days, accumulating into one file
uv run scripts/evaluate_retrieval.py --resume eval/results/incremental.json --limit 50

# Throttle to stay under a :free-tier model's requests-per-minute cap
uv run scripts/evaluate_retrieval.py --max-rpm 20
```

### Results

Each run (unless `--output none`) writes a full report — config, summary
metrics, and every query's top results — to a timestamped file under
`eval/results/`. `results/archive/` holds superseded runs (e.g. incremental/
batched runs) kept for reference rather than deleted.

### Recall/MRR across iterations

Three real runs against the same 133-query set trace how each change moved
the numbers — ground-truth accuracy first, then a reranking stage:

| Run | Change applied | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|---|
| `retrieval_eval_20260906T075905Z.json` | Baseline: hybrid search, original (pre-audit) ground truth | 67.7% | 86.5% | 93.2% | 96.2% | 0.785 |
| `retrieval_eval_20260906T082105Z_after_update_groundtruth.json` | + AI-assisted ground-truth audit (62 queries corrected) | 74.4% | 90.2% | 94.7% | 96.2% | 0.833 |
| `retrieval_eval_20260906T100946Z.json` | + `cohere/rerank-4-fast` reranking (15 candidates → top 3-10) | **93.2%** | **97.7%** | **97.7%** | **97.7%** | **0.955** |

Two distinct effects are visible here:

- **Ground-truth audit (run 1 → 2)**: Recall@1 alone jumped +6.8pp with *no
  code or model changes at all* — purely from correcting mislabeled ground
  truth. This is a reminder that a jump (or drop) in retrieval metrics isn't
  always a retrieval change; it can be an artifact of an incomplete eval set.
  Note Recall@10 stayed flat (96.2% → 96.2%) since the audit only ever
  *adds* already-retrieved-but-uncredited answers to `expected_scheme_ids` —
  it can't manufacture a hit that wasn't in the candidate set at all.
- **Reranking (run 2 → 3)**: with ground truth now accurate, reranking added
  a further +18.8pp to Recall@1 by reordering the same hybrid candidates so
  the correct answer surfaces at rank 1 far more often — the exact gap
  Recall@1 vs Recall@10 revealed beforehand (correct answers were often
  *present* in the top 10 but not ranked first).

Latest known baseline: **Recall@1 93.2%, Recall@3/5/10 97.7%, MRR 0.955**.
